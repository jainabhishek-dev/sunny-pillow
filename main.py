import asyncio
import json
import os
import tempfile
import uuid
from functools import partial
from pathlib import Path
from typing import Annotated

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import auth
from checker import run_vision_check
from commenter import post_selected_comments
from reader import get_file_as_pdf, get_pdf_bytes_by_id

load_dotenv()

app = FastAPI(title="Sunny Pillow")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", ""),
    session_cookie="sunny_pillow_session",
    max_age=3600,
    # True in production (Render always serves over HTTPS); False for local HTTP dev.
    https_only=os.getenv("ENV") == "production",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Job storage directory
_JOBS_DIR = Path(tempfile.gettempdir()) / "sunny_pillow_jobs"
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


# ── Startup: load checkpoints once ───────────────────────────────────────────

def _load_checkpoints() -> list[dict]:
    path = Path(__file__).parent / "checkpoints.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["checkpoints"]


def _group_by_category(checkpoints: list[dict]) -> dict[str, list[dict]]:
    categories: dict[str, list[dict]] = {}
    for cp in checkpoints:
        cat = cp["category"]
        categories.setdefault(cat, []).append(cp)
    return categories


CHECKPOINTS: list[dict] = _load_checkpoints()
CHECKPOINT_MAP: dict[str, dict] = {cp["id"]: cp for cp in CHECKPOINTS}
CATEGORIES: dict[str, list[dict]] = _group_by_category(CHECKPOINTS)


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
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "categories": CATEGORIES,
        "error": error,
    })


@app.post("/check", response_class=HTMLResponse)
async def run_check(
    request: Request,
    drive_url: Annotated[str, Form()],
    checkpoint_ids: Annotated[list[str], Form()] = [],
):
    """Validate input and create a job, then redirect to the processing page."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login", status_code=303)

    # Validate that at least one checkpoint was selected
    if not checkpoint_ids:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user": user,
            "categories": CATEGORIES,
            "error": "Please select at least one checkpoint before running the check.",
        })

    # Validate that the URL field is not empty
    if not drive_url.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user": user,
            "categories": CATEGORIES,
            "error": "Please enter a Google Drive file URL.",
        })

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
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user": user,
            "categories": CATEGORIES,
            "error": str(exc),
        })
    except Exception as exc:
        error_msg = str(exc)
        if "invalid_grant" in error_msg or "Token has been expired" in error_msg:
            request.session.clear()
            return RedirectResponse(url="/login", status_code=303)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user": user,
            "categories": CATEGORIES,
            "error": f"Could not read the file: {error_msg}",
        })

    # Create a job and save metadata
    job_id = uuid.uuid4().hex
    _save_job(job_id, {
        "file_id": file_data["file_id"],
        "file_type": file_data["file_type"],
        "title": file_data["title"],
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

    return templates.TemplateResponse("process.html", {
        "request": request,
        "user": user,
        "job_id": job_id,
        "title": job["title"],
        "retry_from": retry_from,
    })


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

    loop = asyncio.get_running_loop()
    job_dir = _ensure_job_dir(job_id)

    # If not retrying, get PDF; if retrying, use existing job state
    if retry_from is None:
        # Get the PDF bytes using file_id from the job
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

        # Open PDF with PyMuPDF
        try:
            pdf_document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': f'Could not parse PDF: {str(e)}'})}\n\n"
            return

        total_pages = len(pdf_document)
        all_findings = []
        finding_id_counter = 0
        start_page = 1

        # Send start event
        yield f"event: start\ndata: {json.dumps({'total_pages': total_pages, 'title': job['title']})}\n\n"
    else:
        # Retry mode: load existing PDF and findings
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

        # Load existing findings
        findings_file = job_dir / "findings.json"
        if findings_file.exists():
            all_findings = json.loads(findings_file.read_text(encoding="utf-8"))
            finding_id_counter = max([f.get("id", 0) for f in all_findings] + [0]) + 1
        else:
            all_findings = []
            finding_id_counter = 0

        # Send retry_start event
        yield f"event: retry_start\ndata: {json.dumps({'starting_page': start_page, 'total_pages': total_pages})}\n\n"

    # Get selected checkpoints
    selected_checkpoints = [
        CHECKPOINT_MAP[cid] for cid in job["checkpoint_ids"] if cid in CHECKPOINT_MAP
    ]

    # Process each page
    for page_num in range(start_page, total_pages + 1):
        try:
            # Render page to image (2× zoom for readability)
            page = pdf_document[page_num - 1]
            mat = fitz.Matrix(2, 2)  # 2× zoom
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes(output="jpeg")

            # Save image to disk
            img_path = job_dir / f"page_{page_num:03d}.jpg"
            img_path.write_bytes(img_bytes)

            # Send page_ready event
            yield f"event: page_ready\ndata: {json.dumps({'page': page_num, 'total_pages': total_pages})}\n\n"

            # Call vision AI in executor (blocking operation)
            findings = await loop.run_in_executor(
                None, partial(run_vision_check, img_bytes, selected_checkpoints, page_num)
            )

            # Assign finding IDs and add page reference
            for finding in findings:
                finding["id"] = finding_id_counter
                if "location" not in finding or finding["location"] == "":
                    finding["location"] = f"Page {page_num}"
                all_findings.append(finding)
                finding_id_counter += 1

            # Send page_findings event
            yield f"event: page_findings\ndata: {json.dumps({'page': page_num, 'findings': findings})}\n\n"

        except Exception as e:
            # On error: save state, send partial_complete event, and stop processing
            job["last_successful_page"] = page_num - 1
            _save_job(job_id, job)

            # Save findings accumulated so far
            try:
                (job_dir / "findings.json").write_text(
                    json.dumps(all_findings, ensure_ascii=False),
                    encoding="utf-8"
                )
            except Exception as save_error:
                yield f"event: error\ndata: {json.dumps({'message': f'Could not save findings: {str(save_error)}'})}\n\n"

            # Send partial_complete event with retry info
            yield f"event: partial_complete\ndata: {json.dumps({
                'last_successful_page': page_num - 1,
                'total_pages': total_pages,
                'error_message': str(e)
            })}\n\n"

            # Stop processing
            return

    # Save findings to findings.json
    try:
        (job_dir / "findings.json").write_text(
            json.dumps(all_findings, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'message': f'Could not save findings: {str(e)}'})}\n\n"

    # Send done event
    yield f"event: done\ndata: {json.dumps({'total_findings': len(all_findings)})}\n\n"

    # Update job status
    job["status"] = "completed"
    job.pop("last_successful_page", None)  # Clear retry state on success
    _save_job(job_id, job)


@app.get("/stream/{job_id}")
async def stream_processing(request: Request, job_id: str, retry_from: int = None):
    """SSE endpoint for streaming page processing."""
    user = auth.get_current_user(request)
    token = auth.get_token(request)

    if not user or not token:
        return RedirectResponse(url="/login")

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
