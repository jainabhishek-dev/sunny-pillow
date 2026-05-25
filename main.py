import asyncio
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from functools import partial
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import auth
import db
from checker import run_vision_check, run_vision_review, run_document_check, run_document_review, generate_workflow_content
from commenter import post_selected_comments
from reader import get_file_as_pdf, get_pdf_bytes_by_id, create_drive_subfolder, upload_jpeg_to_drive

load_dotenv()

app = FastAPI(title="CheckPoint")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", ""),
    session_cookie="checkpoint_session",
    max_age=3600,
    # True in production (Render always serves over HTTPS); False for local HTTP dev.
    https_only=os.getenv("ENV") == "production",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Job storage directory
_JOBS_DIR = Path(tempfile.gettempdir()) / "checkpoint_jobs"
_JOBS_DIR.mkdir(exist_ok=True)


def _ensure_job_dir(job_id: str) -> Path:
    """Create and return the job directory."""
    job_dir = _JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    return job_dir


def _save_job(job_id: str, data: dict) -> None:
    """Save job metadata to job.json."""
    job_dir = _ensure_job_dir(job_id)
    (job_dir / "job.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_job(job_id: str) -> dict | None:
    """Load job metadata from job.json."""
    job_dir = _JOBS_DIR / job_id
    path = job_dir / "job.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ── Startup: load checkpoints and workflows ───────────────────────────────────

def _slugify(name: str) -> str:
    """Generate a lowercase, underscore-separated ID from a workflow name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s]", "", slug)   # strip punctuation
    slug = re.sub(r"\s+", "_", slug)       # spaces → underscore
    slug = re.sub(r"_+", "_", slug)        # collapse consecutive underscores
    return slug.strip("_")


def _group_by_category(checkpoints: list[dict]) -> dict[str, list[dict]]:
    categories: dict[str, list[dict]] = {}
    for cp in checkpoints:
        cat = cp["category"]
        categories.setdefault(cat, []).append(cp)
    return categories


def _filter_checkpoints_by_workflow(checkpoints: list[dict], workflow_id: str) -> list[dict]:
    return [cp for cp in checkpoints if workflow_id in cp.get("workflows", [])]


def _reload_checkpoints() -> None:
    """Reload checkpoints from Supabase into global state."""
    global CHECKPOINTS, CHECKPOINT_MAP, CATEGORIES
    CHECKPOINTS = db.fetch_all_checkpoints()
    CHECKPOINT_MAP = {cp["id"]: cp for cp in CHECKPOINTS}
    CATEGORIES = _group_by_category(CHECKPOINTS)


def _reload_workflows() -> None:
    """Reload workflows from Supabase into global state."""
    global WORKFLOWS
    WORKFLOWS = db.fetch_all_workflows()


WORKFLOWS: list[dict] = []
CHECKPOINTS: list[dict] = []
CHECKPOINT_MAP: dict[str, dict] = {}
CATEGORIES: dict[str, list[dict]] = {}
_reload_workflows()
_reload_checkpoints()

# Jobs currently being streamed. Prevents multiple concurrent stream connections
# for the same job (e.g. from EventSource auto-reconnects) from each loading and
# processing the full PDF simultaneously, which causes OOM crashes.
_ACTIVE_JOBS: set[str] = set()

# ── Role-based access ─────────────────────────────────────────────────────────

SUPER_ADMIN = "abhishek.jain@leadschool.in"
ADMINS: set[str] = set()


def _reload_admins() -> None:
    global ADMINS
    ADMINS = {a["email"] for a in db.fetch_all_admins()}


_reload_admins()


def _is_super_admin(user: dict | None) -> bool:
    return bool(user) and user.get("email") == SUPER_ADMIN


def _is_admin(user: dict | None) -> bool:
    return bool(user) and (
        user.get("email") == SUPER_ADMIN or user.get("email") in ADMINS
    )


def _cascade_delete_workflow(wf_id: str) -> None:
    """
    Delete a workflow and handle its checkpoints:
    - Checkpoints belonging ONLY to this workflow are deleted entirely.
    - Checkpoints shared with other workflows have this workflow removed from
      their workflows array (they remain intact for the other workflows).
    Reloads both CHECKPOINTS and WORKFLOWS globals after completion.
    """
    checkpoints = db.fetch_checkpoints_by_workflow(wf_id)
    for cp in checkpoints:
        remaining = [w for w in cp.get("workflows", []) if w != wf_id]
        if remaining:
            db.update_checkpoint(cp["id"], {"workflows": remaining})
        else:
            db.delete_checkpoint(cp["id"])
    db.delete_workflow(wf_id)
    _reload_checkpoints()
    _reload_workflows()


def _ctx(request: Request, user: dict | None, **kwargs) -> dict:
    """Base template context including role flags."""
    return {
        "request": request,
        "user": user,
        "is_admin": _is_admin(user),
        "is_super_admin": _is_super_admin(user),
        **kwargs,
    }


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    user = auth.get_current_user(request)
    if user:
        return RedirectResponse(url="/")
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/login/google")
async def login_google(request: Request):
    return await auth.login(request)


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    return await auth.auth_callback(request)


@app.get("/logout")
def logout(request: Request):
    return auth.logout(request)


# ── Main app routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    error = request.query_params.get("error")
    workflow_id = request.query_params.get("workflow")

    # Filter checkpoints based on selected workflow
    if workflow_id:
        filtered_checkpoints = _filter_checkpoints_by_workflow(CHECKPOINTS, workflow_id)
        filtered_categories = _group_by_category(filtered_checkpoints)
    else:
        filtered_categories = {}

    # Find the selected workflow object
    selected_workflow = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)

    return templates.TemplateResponse("index.html", _ctx(
        request, user,
        workflows=WORKFLOWS,
        selected_workflow=selected_workflow,
        categories=filtered_categories,
        error=error,
    ))


@app.post("/check", response_class=HTMLResponse)
async def run_check(
    request: Request,
    drive_url: Annotated[str, Form()],
    workflow_id: Annotated[str, Form()],
    checkpoint_ids: Annotated[list[str], Form()] = [],
):
    """Validate input and create a job, then redirect to the processing page."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    # Validate that a workflow was selected
    if not workflow_id or workflow_id not in [w["id"] for w in WORKFLOWS]:
        return templates.TemplateResponse("index.html", _ctx(
            request, user,
            workflows=WORKFLOWS,
            selected_workflow=None,
            categories={},
            error="Please select a workflow before running the check.",
        ))

    # Validate that at least one checkpoint was selected
    if not checkpoint_ids:
        filtered_checkpoints = _filter_checkpoints_by_workflow(CHECKPOINTS, workflow_id)
        filtered_categories = _group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", _ctx(
            request, user,
            workflows=WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error="Please select at least one checkpoint before running the check.",
        ))

    # Validate that the URL field is not empty
    if not drive_url.strip():
        filtered_checkpoints = _filter_checkpoints_by_workflow(CHECKPOINTS, workflow_id)
        filtered_categories = _group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", _ctx(
            request, user,
            workflows=WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error="Please enter a Google Drive file URL.",
        ))

    selected_checkpoints = [
        CHECKPOINT_MAP[cid] for cid in checkpoint_ids if cid in CHECKPOINT_MAP
    ]

    # Try to get file metadata (to validate the URL early)
    loop = asyncio.get_running_loop()
    try:
        file_data = await loop.run_in_executor(
            None, partial(get_file_as_pdf, token, drive_url.strip())
        )
    except ValueError as exc:
        filtered_checkpoints = _filter_checkpoints_by_workflow(CHECKPOINTS, workflow_id)
        filtered_categories = _group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", _ctx(
            request, user,
            workflows=WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error=str(exc),
        ))
    except Exception as exc:
        error_msg = str(exc)
        if "invalid_grant" in error_msg or "Token has been expired" in error_msg:
            request.session.clear()
            return RedirectResponse(url="/login", status_code=303)
        filtered_checkpoints = _filter_checkpoints_by_workflow(CHECKPOINTS, workflow_id)
        filtered_categories = _group_by_category(filtered_checkpoints)
        selected_workflow = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)
        return templates.TemplateResponse("index.html", _ctx(
            request, user,
            workflows=WORKFLOWS,
            selected_workflow=selected_workflow,
            categories=filtered_categories,
            error=f"Could not read the file: {error_msg}",
        ))

    # Create a job and save metadata
    job_id = uuid.uuid4().hex
    _save_job(job_id, {
        "drive_url": drive_url.strip(),
        "checked_by": user["email"],
        "file_id": file_data["file_id"],
        "file_type": file_data["file_type"],
        "title": file_data["title"],
        "workflow_id": workflow_id,
        "checkpoint_ids": [cp["id"] for cp in selected_checkpoints],
        "status": "processing",
    })

    # Redirect to the processing page
    return RedirectResponse(url=f"/process/{job_id}", status_code=303)


@app.get("/process/{job_id}", response_class=HTMLResponse)
async def show_process(request: Request, job_id: str, retry_from: int = None):
    """Serve the live processing page with card-based layout."""
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    job = _load_job(job_id)
    if not job:
        return RedirectResponse(url="/?error=Job+not+found.+Please+run+a+new+check.")

    checkpoint_names = {cp["id"]: cp["category"] for cp in CHECKPOINTS}

    return templates.TemplateResponse("process.html", _ctx(
        request, user,
        job_id=job_id,
        title=job["title"],
        retry_from=retry_from,
        checkpoint_names=checkpoint_names,
    ))


async def _save_run_to_history(
    job_id: str,
    job: dict,
    all_findings: list,
    total_pages: int,
    token: dict,
    job_dir: Path,
) -> None:
    """
    Persists a completed run to Google Drive (page images) and Supabase
    (run metadata + findings). Runs as an independent asyncio.Task so that
    a client disconnect cannot cancel it.
    """
    loop = asyncio.get_running_loop()
    runs_folder_id = os.getenv("DRIVE_RUNS_FOLDER_ID")
    drive_folder_id = None
    page_records: list[dict] = []

    if runs_folder_id:
        try:
            drive_folder_id = await loop.run_in_executor(
                None, partial(create_drive_subfolder, token, runs_folder_id, job_id)
            )
            for img_path in sorted(job_dir.glob("page_*.jpg")):
                pg = int(img_path.stem.split("_")[1])
                img_data = img_path.read_bytes()
                file_id = await loop.run_in_executor(
                    None, partial(upload_jpeg_to_drive, token, drive_folder_id, img_path.name, img_data)
                )
                page_records.append({"run_id": job_id, "page_num": pg, "drive_file_id": file_id})
        except Exception as e:
            print(f"[history] Drive upload failed for {job_id}: {e}")

    try:
        wf = next((w for w in WORKFLOWS if w["id"] == job.get("workflow_id")), {})
        db.insert_run({
            "id": job_id,
            "workflow_id": job.get("workflow_id", ""),
            "workflow_name": wf.get("name", job.get("workflow_id", "")),
            "checked_by": job.get("checked_by", ""),
            "document_name": job.get("title"),
            "drive_url": job.get("drive_url"),
            "file_type": job.get("file_type"),
            "drive_folder_id": drive_folder_id,
            "checkpoint_ids": job.get("checkpoint_ids", []),
            "total_pages": total_pages,
            "total_findings": len(all_findings),
            "valid_findings": sum(1 for f in all_findings if f.get("review_status") == "valid"),
            "invalid_findings": sum(1 for f in all_findings if f.get("review_status") == "invalid"),
        })
        if page_records:
            db.insert_run_pages(page_records)
        if all_findings:
            db.insert_run_findings([
                {
                    "run_id": job_id,
                    "page_num": f.get("page_num"),
                    "checkpoint_id": f.get("checkpoint_id"),
                    "quote": f.get("quote"),
                    "location": f.get("location"),
                    "issue": f.get("issue"),
                    "suggestion": f.get("suggestion"),
                    "review_status": f.get("review_status"),
                    "review_comment": f.get("review_comment"),
                }
                for f in all_findings
            ])
        print(f"[history] Run {job_id} saved: {total_pages} pages, {len(all_findings)} findings.")
    except Exception as e:
        print(f"[history] Supabase save failed for {job_id}: {e}")


async def _stream_processing(job_id: str, token: dict, retry_from: int = None) -> None:
    """
    SSE endpoint that processes a PDF page-by-page.
    Renders images, calls vision AI, and streams results.

    Args:
        job_id: The job ID
        token: User's OAuth token
        retry_from: Optional page number to resume from (for error recovery)
    """
    import fitz  # PyMuPDF
    from io import BytesIO

    job = _load_job(job_id)
    if not job:
        yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
        return

    # Acquire job lock — released in the finally block below regardless of how
    # this generator exits (normal completion, exception, or client disconnect).
    _ACTIVE_JOBS.add(job_id)
    try:
        # Resolve workflow name — fail fast if deleted between job creation and now.
        workflow_id = job.get("workflow_id", "")
        workflow = next((w for w in WORKFLOWS if w["id"] == workflow_id), None)
        if not workflow:
            yield f"event: error\ndata: {json.dumps({'message': f'Workflow \"{workflow_id}\" not found. It may have been deleted.'})}\n\n"
            return
        workflow_name = workflow["name"]

        loop = asyncio.get_running_loop()
        job_dir = _ensure_job_dir(job_id)

        # ── Load PDF ──────────────────────────────────────────────────────────
        if retry_from is None:
            try:
                pdf_data = await loop.run_in_executor(
                    None, partial(get_pdf_bytes_by_id, token, job["file_id"])
                )
                pdf_bytes = pdf_data.get("pdf_bytes")
                if not pdf_bytes:
                    raise ValueError("No PDF bytes retrieved")
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not export file as PDF: {str(e)}'})}\n\n"
                return

            try:
                pdf_document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not parse PDF: {str(e)}'})}\n\n"
                return

            total_pages = len(pdf_document)
            all_findings = []
            finding_id_counter = 0
            start_page = 1

            yield f"event: start\ndata: {json.dumps({'total_pages': total_pages, 'title': job['title']})}\n\n"
        else:
            try:
                pdf_data = await loop.run_in_executor(
                    None, partial(get_pdf_bytes_by_id, token, job["file_id"])
                )
                pdf_bytes = pdf_data.get("pdf_bytes")
                pdf_document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not re-open PDF: {str(e)}'})}\n\n"
                return

            total_pages = len(pdf_document)
            start_page = retry_from

            findings_file = job_dir / "findings.json"
            if findings_file.exists():
                all_findings = json.loads(findings_file.read_text(encoding="utf-8"))
                finding_id_counter = max([f.get("id", 0) for f in all_findings] + [0]) + 1
            else:
                all_findings = []
                finding_id_counter = 0

            yield f"event: retry_start\ndata: {json.dumps({'starting_page': start_page, 'total_pages': total_pages})}\n\n"

        selected_checkpoints = [
            CHECKPOINT_MAP[cid] for cid in job["checkpoint_ids"] if cid in CHECKPOINT_MAP
        ]

        # ── Page-by-page processing ───────────────────────────────────────────
        for page_num in range(start_page, total_pages + 1):
            try:
                page = pdf_document[page_num - 1]
                mat = fitz.Matrix(2, 2)  # 2× zoom
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes(output="jpeg")

                img_path = job_dir / f"page_{page_num:03d}.jpg"
                img_path.write_bytes(img_bytes)

                yield f"event: page_ready\ndata: {json.dumps({'page': page_num, 'total_pages': total_pages})}\n\n"

                findings = await loop.run_in_executor(
                    None, partial(run_vision_check, img_bytes, selected_checkpoints, page_num, workflow_name)
                )

                for finding in findings:
                    finding["id"] = finding_id_counter
                    finding["page_num"] = page_num
                    if "location" not in finding or finding["location"] == "":
                        finding["location"] = f"Page {page_num}"
                    all_findings.append(finding)
                    finding_id_counter += 1

                yield f"event: page_findings\ndata: {json.dumps({'page': page_num, 'findings': findings})}\n\n"

                if findings:
                    reviews = await loop.run_in_executor(
                        None, partial(run_vision_review, img_bytes, findings, page_num)
                    )
                    yield f"event: page_review\ndata: {json.dumps({'page': page_num, 'reviews': reviews})}\n\n"

                    # Merge review verdicts into findings (which are also referenced in all_findings)
                    review_map = {r["finding_id"]: r for r in reviews}
                    for f in findings:
                        rev = review_map.get(f["id"])
                        if rev:
                            f["review_status"] = rev["verdict"]
                            f["review_comment"] = rev["reason"]

                # Explicitly free the page image — it is no longer needed and
                # can be several MB; do not wait for GC.
                del img_bytes

            except Exception as e:
                job["last_successful_page"] = page_num - 1
                _save_job(job_id, job)
                try:
                    (job_dir / "findings.json").write_text(
                        json.dumps(all_findings, ensure_ascii=False), encoding="utf-8"
                    )
                except Exception as save_error:
                    yield f"event: error\ndata: {json.dumps({'message': f'Could not save findings: {str(save_error)}'})}\n\n"

                yield f"event: partial_complete\ndata: {json.dumps({'last_successful_page': page_num - 1, 'total_pages': total_pages, 'error_message': str(e)})}\n\n"
                return

        # Free fitz document after page loop — pdf_bytes is still needed for
        # the document-level check but the decoded document object is not.
        pdf_document.close()
        del pdf_document

        # ── Save page findings ────────────────────────────────────────────────
        try:
            (job_dir / "findings.json").write_text(
                json.dumps(all_findings, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not save findings: {str(e)}'})}\n\n"

        yield f"event: done\ndata: {json.dumps({'total_findings': len(all_findings)})}\n\n"

        # ── Document-level check ──────────────────────────────────────────────
        doc_checkpoints = [cp for cp in selected_checkpoints if cp.get("scope") == "document"]

        if doc_checkpoints:
            yield f"event: document_start\ndata: {json.dumps({})}\n\n"

            try:
                doc_findings = await loop.run_in_executor(
                    None, partial(run_document_check, pdf_bytes, doc_checkpoints)
                )

                for finding in doc_findings:
                    finding["id"] = finding_id_counter
                    finding.setdefault("location", "Document")
                    all_findings.append(finding)
                    finding_id_counter += 1

                yield f"event: document_findings\ndata: {json.dumps({'findings': doc_findings})}\n\n"

                if doc_findings:
                    doc_reviews = await loop.run_in_executor(
                        None, partial(run_document_review, pdf_bytes, doc_findings)
                    )
                    yield f"event: document_review\ndata: {json.dumps({'reviews': doc_reviews})}\n\n"

                    # Merge review verdicts into doc_findings (also referenced in all_findings)
                    doc_review_map = {r["finding_id"]: r for r in doc_reviews}
                    for f in doc_findings:
                        rev = doc_review_map.get(f["id"])
                        if rev:
                            f["review_status"] = rev["verdict"]
                            f["review_comment"] = rev["reason"]

                (job_dir / "findings.json").write_text(
                    json.dumps(all_findings, ensure_ascii=False), encoding="utf-8"
                )

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'Document-level check failed: {str(e)}'})}\n\n"

        # Free PDF bytes — document-level check is complete.
        del pdf_bytes

        # Schedule history save BEFORE yielding all_done. The task is already
        # queued on the event loop, so a client disconnect after all_done
        # cannot cancel it. Snapshots are passed so mutations below are safe.
        asyncio.create_task(_save_run_to_history(
            job_id=job_id,
            job=dict(job),
            all_findings=list(all_findings),
            total_pages=total_pages,
            token=token,
            job_dir=job_dir,
        ))

        yield f"event: all_done\ndata: {json.dumps({'total_findings': len(all_findings)})}\n\n"

        job["status"] = "completed"
        job.pop("last_successful_page", None)
        _save_job(job_id, job)

    finally:
        # Always release the job lock so future stream requests are accepted.
        _ACTIVE_JOBS.discard(job_id)


@app.get("/stream/{job_id}")
async def stream_processing(request: Request, job_id: str, retry_from: int = None):
    """SSE endpoint for streaming page processing."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login")

    # Reject duplicate connections for the same job. EventSource auto-reconnects
    # on any network hiccup; without this guard each reconnect would load and
    # process the full PDF again in parallel, causing OOM crashes.
    if job_id in _ACTIVE_JOBS:
        async def _already_processing():
            yield (
                f"event: error\ndata: {json.dumps({'message': 'This job is already being processed. '
                'Please wait for the current run to complete.'})}\n\n"
            )
        return StreamingResponse(
            _already_processing(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_processing(job_id, token, retry_from),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/job/{job_id}/page/{page_num:int}")
async def serve_page_image(job_id: str, page_num: int):
    """Serve a rendered page image (JPEG) from disk."""
    job_dir = _JOBS_DIR / job_id
    page_file = job_dir / f"page_{page_num:03d}.jpg"

    if not page_file.exists():
        raise HTTPException(status_code=404, detail="Page image not found")

    return FileResponse(page_file, media_type="image/jpeg")


@app.post("/retry-check/{job_id}")
async def retry_check(request: Request, job_id: str):
    """Initiate retry of processing from the last failed page."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    job = _load_job(job_id)
    if not job:
        return RedirectResponse(url="/?error=Job+not+found.", status_code=303)

    # Get the last successful page from job metadata
    from_page = job.get("last_successful_page", 0) + 1

    # Redirect to process page with retry flag
    return RedirectResponse(
        url=f"/process/{job_id}?retry_from={from_page}",
        status_code=303
    )


@app.post("/insert-comments/{job_id}")
async def insert_comments(
    request: Request,
    job_id: str,
    finding_ids: Annotated[list[str], Form()] = [],
):
    """Insert selected findings as Drive comments."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    job = _load_job(job_id)
    if not job:
        return {"error": "Job not found"}

    # Load findings
    job_dir = _JOBS_DIR / job_id
    findings_file = job_dir / "findings.json"
    if not findings_file.exists():
        return {"error": "Findings not found"}

    all_findings = json.loads(findings_file.read_text(encoding="utf-8"))

    # Filter to selected finding IDs
    selected_finding_ids = [int(fid) for fid in finding_ids if fid.isdigit()]
    selected_findings = [f for f in all_findings if f.get("id") in selected_finding_ids]

    # Prepare file data (needed by post_selected_comments)
    file_data = {
        "file_id": job["file_id"],
        "type": job["file_type"],
        "title": job["title"],
    }

    # Post comments in executor
    loop = asyncio.get_running_loop()
    try:
        posted = await loop.run_in_executor(
            None, partial(post_selected_comments, token, file_data, selected_findings, CHECKPOINT_MAP)
        )
        return {"posted": posted, "total_selected": len(selected_findings)}
    except Exception as exc:
        return {"error": f"Could not post comments: {str(exc)}"}


# ── Checkpoint management routes ──────────────────────────────────────────────

def _next_checkpoint_id() -> str:
    """Generate the next sequential checkpoint ID (e.g. cp_042)."""
    existing = [cp["id"] for cp in CHECKPOINTS if cp["id"].startswith("cp_")]
    nums = []
    for cid in existing:
        try:
            nums.append(int(cid.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums, default=0) + 1
    return f"cp_{next_num:03d}"


@app.get("/checkpoints", response_class=HTMLResponse)
async def manage_checkpoints(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    default_wf_id = WORKFLOWS[0]["id"] if WORKFLOWS else ""
    workflow_id = request.query_params.get("workflow", default_wf_id)
    selected_workflow = next((w for w in WORKFLOWS if w["id"] == workflow_id), WORKFLOWS[0] if WORKFLOWS else None)
    filtered = _filter_checkpoints_by_workflow(CHECKPOINTS, selected_workflow["id"])

    return templates.TemplateResponse("checkpoints.html", _ctx(
        request, user,
        workflows=WORKFLOWS,
        selected_workflow=selected_workflow,
        categories=_group_by_category(filtered),
        success=request.query_params.get("success"),
        error=request.query_params.get("error"),
    ))


@app.post("/checkpoints/add", response_class=HTMLResponse)
async def add_checkpoint(
    request: Request,
    category: Annotated[str, Form()],
    instructions: Annotated[str, Form()],
    type: Annotated[str, Form()],
    scope: Annotated[str, Form()],
    workflows: Annotated[list[str], Form()],
):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    new_id = _next_checkpoint_id()
    sort_order = max((cp["sort_order"] for cp in CHECKPOINTS), default=0) + 1

    workflow_param = request.query_params.get("workflow", workflows[0] if workflows else "")
    base = f"/checkpoints?workflow={workflow_param}"
    try:
        db.insert_checkpoint({
            "id": new_id,
            "category": category.strip(),
            "instructions": instructions.strip(),
            "type": type,
            "scope": scope,
            "workflows": workflows,
            "sort_order": sort_order,
        })
        _reload_checkpoints()
        return RedirectResponse(url=f"{base}&success=Checkpoint+added.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"{base}&error={exc}", status_code=303)


@app.post("/checkpoints/{cp_id}/edit", response_class=HTMLResponse)
async def edit_checkpoint(
    request: Request,
    cp_id: str,
    instructions: Annotated[str, Form()],
    type: Annotated[str, Form()],
    scope: Annotated[str, Form()],
    workflows: Annotated[list[str], Form()],
):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    workflow_param = request.query_params.get("workflow", workflows[0] if workflows else "")
    base = f"/checkpoints?workflow={workflow_param}"
    try:
        db.update_checkpoint(cp_id, {
            "instructions": instructions.strip(),
            "type": type,
            "scope": scope,
            "workflows": workflows,
        })
        _reload_checkpoints()
        return RedirectResponse(url=f"{base}&success=Checkpoint+updated.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"{base}&error={exc}", status_code=303)


@app.post("/checkpoints/{cp_id}/delete", response_class=HTMLResponse)
async def delete_checkpoint(request: Request, cp_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    workflow_param = request.query_params.get("workflow", "")
    base = f"/checkpoints?workflow={workflow_param}" if workflow_param else "/checkpoints"
    try:
        db.delete_checkpoint(cp_id)
        _reload_checkpoints()
        return RedirectResponse(url=f"{base}&success=Checkpoint+deleted.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"{base}&error={exc}", status_code=303)


# ── Admin management routes ────────────────────────────────────────────────────

@app.get("/admins", response_class=HTMLResponse)
async def manage_admins(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.")

    admins = db.fetch_all_admins()
    return templates.TemplateResponse("admins.html", _ctx(
        request, user,
        admins=admins,
        success=request.query_params.get("success"),
        error=request.query_params.get("error"),
    ))


@app.post("/admins/add", response_class=HTMLResponse)
async def add_admin(
    request: Request,
    email: Annotated[str, Form()],
):
    user = auth.get_current_user(request)
    if not user or not _is_admin(user):
        return RedirectResponse(url="/login", status_code=303)

    try:
        db.insert_admin({"email": email.strip().lower(), "added_by": user["email"]})
        _reload_admins()
        return RedirectResponse(url="/admins?success=Admin+added.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/admins?error={exc}", status_code=303)


@app.post("/admins/{email}/delete", response_class=HTMLResponse)
async def delete_admin_route(request: Request, email: str):
    user = auth.get_current_user(request)
    if not user or not _is_super_admin(user):
        return RedirectResponse(url="/login", status_code=303)

    try:
        db.delete_admin(email)
        _reload_admins()
        return RedirectResponse(url="/admins?success=Admin+removed.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/admins?error={exc}", status_code=303)


# ── Workflow management routes ─────────────────────────────────────────────────

@app.get("/workflows", response_class=HTMLResponse)
async def manage_workflows(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.")

    # Compute checkpoint count per workflow from the in-memory global
    checkpoint_counts: dict[str, int] = {}
    for cp in CHECKPOINTS:
        for wf_id in cp.get("workflows", []):
            checkpoint_counts[wf_id] = checkpoint_counts.get(wf_id, 0) + 1

    return templates.TemplateResponse("workflows.html", _ctx(
        request, user,
        workflows=WORKFLOWS,
        checkpoint_counts=checkpoint_counts,
        success=request.query_params.get("success"),
        error=request.query_params.get("error"),
    ))


@app.post("/workflows/add", response_class=HTMLResponse)
async def add_workflow(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    ai_notes: Annotated[str, Form()] = "",
    action: Annotated[str, Form()] = "manual",
):
    """Create a new workflow, optionally generating checkpoints with AI."""
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    name = name.strip()
    description = description.strip()

    if not name:
        return RedirectResponse(url="/workflows?error=Workflow+name+is+required.", status_code=303)

    wf_id = _slugify(name)
    if not wf_id:
        return RedirectResponse(url="/workflows?error=Could+not+generate+a+valid+ID+from+that+name.", status_code=303)

    # Ensure ID is unique
    if any(w["id"] == wf_id for w in WORKFLOWS):
        return RedirectResponse(
            url=f"/workflows?error=A+workflow+with+id+\"{wf_id}\"+already+exists.+Choose+a+different+name.",
            status_code=303,
        )

    sort_order = max((w.get("sort_order", 0) for w in WORKFLOWS), default=0) + 1

    if action == "generate":
        if not ai_notes.strip():
            return RedirectResponse(
                url="/workflows?error=AI+Generation+Notes+are+required+when+using+Create+with+AI.",
                status_code=303,
            )
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, partial(generate_workflow_content, name, ai_notes.strip())
            )
        except Exception as exc:
            return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)

        # Insert workflow row
        try:
            db.insert_workflow({
                "id": wf_id,
                "name": name,
                "description": description,
                "sort_order": sort_order,
                "created_by": user["email"],
            })
        except Exception as exc:
            return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)

        # Batch-insert generated checkpoints
        existing_nums = [
            int(cp["id"].split("_")[1])
            for cp in CHECKPOINTS
            if cp["id"].startswith("cp_") and cp["id"].split("_")[1].isdigit()
        ]
        next_num = max(existing_nums, default=0) + 1
        next_sort = max((cp.get("sort_order", 0) for cp in CHECKPOINTS), default=0) + 1

        insert_errors = []
        for cp_data in result.get("checkpoints", []):
            try:
                db.insert_checkpoint({
                    "id": f"cp_{next_num:03d}",
                    "category": cp_data["category"].strip(),
                    "instructions": cp_data["instructions"].strip(),
                    "type": cp_data["type"],
                    "scope": cp_data["scope"],
                    "workflows": [wf_id],
                    "sort_order": next_sort,
                })
                next_num += 1
                next_sort += 1
            except Exception as exc:
                insert_errors.append(str(exc))

        _reload_checkpoints()
        _reload_workflows()

        cp_count = len(result.get("checkpoints", [])) - len(insert_errors)
        msg = f"Workflow+created+with+{cp_count}+AI-generated+checkpoints."
        if insert_errors:
            msg += f"+({len(insert_errors)}+checkpoint+inserts+failed)"
        return RedirectResponse(url=f"/workflows?success={msg}", status_code=303)

    else:
        # Manual creation — no checkpoints yet; admin adds them via /checkpoints
        try:
            db.insert_workflow({
                "id": wf_id,
                "name": name,
                "description": description,
                "sort_order": sort_order,
                "created_by": user["email"],
            })
            _reload_workflows()
            return RedirectResponse(
                url="/workflows?success=Workflow+created.+Add+a+system+prompt+and+checkpoints+to+activate+it.",
                status_code=303,
            )
        except Exception as exc:
            return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)


@app.post("/workflows/{wf_id}/edit", response_class=HTMLResponse)
async def edit_workflow(
    request: Request,
    wf_id: str,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    name = name.strip()
    if not name:
        return RedirectResponse(url="/workflows?error=Workflow+name+is+required.", status_code=303)

    try:
        db.update_workflow(wf_id, {
            "name": name,
            "description": description.strip(),
        })
        _reload_workflows()
        return RedirectResponse(url="/workflows?success=Workflow+updated.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)


@app.post("/workflows/{wf_id}/delete", response_class=HTMLResponse)
async def delete_workflow(request: Request, wf_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not _is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    # Only super admin or the admin who created this workflow may delete it
    workflow = next((w for w in WORKFLOWS if w["id"] == wf_id), None)
    if not workflow:
        return RedirectResponse(url="/workflows?error=Workflow+not+found.", status_code=303)

    if not _is_super_admin(user) and workflow.get("created_by") != user.get("email"):
        return RedirectResponse(
            url="/workflows?error=Only+the+super+admin+or+the+workflow+creator+can+delete+this+workflow.",
            status_code=303,
        )

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(_cascade_delete_workflow, wf_id))
        return RedirectResponse(url="/workflows?success=Workflow+and+its+exclusive+checkpoints+deleted.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)


# ── Run history routes ─────────────────────────────────────────────────────────

_IST = timezone(timedelta(hours=5, minutes=30))

def _to_ist(dt_str: str | None) -> str | None:
    """Convert a UTC datetime string from Supabase to IST, keeping the same
    'YYYY-MM-DD HH:MM:SS' format so template string-slicing still works."""
    if not dt_str:
        return dt_str
    try:
        dt = datetime.fromisoformat(dt_str.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt_str


@app.get("/history", response_class=HTMLResponse)
async def view_history(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    selected_workflow_id = request.query_params.get("workflow") or None
    runs = db.fetch_runs(workflow_id=selected_workflow_id)
    for run in runs:
        run["created_at"] = _to_ist(run.get("created_at"))

    return templates.TemplateResponse("history.html", _ctx(
        request, user,
        runs=runs,
        workflows=WORKFLOWS,
        selected_workflow_id=selected_workflow_id,
        error=request.query_params.get("error"),
    ))


@app.get("/history/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    run = db.fetch_run(run_id)
    if not run:
        return RedirectResponse(url="/history?error=Run+not+found.")
    run["created_at"] = _to_ist(run.get("created_at"))

    pages = db.fetch_run_pages(run_id)
    findings = db.fetch_run_findings(run_id)

    # Split into page findings (grouped by page_num) and document-level findings
    page_findings: dict[int, list] = {}
    doc_findings: list = []
    for f in findings:
        if f["page_num"] is None:
            doc_findings.append(f)
        else:
            page_findings.setdefault(f["page_num"], []).append(f)

    checkpoint_map = {cp["id"]: cp["category"] for cp in CHECKPOINTS}
    page_image_map = {p["page_num"]: p["drive_file_id"] for p in pages}

    return templates.TemplateResponse("run_detail.html", _ctx(
        request, user,
        run=run,
        page_findings=page_findings,
        doc_findings=doc_findings,
        checkpoint_map=checkpoint_map,
        page_image_map=page_image_map,
        total_pages=run["total_pages"],
    ))
