"""History routes: run list, review run detail, CIC run detail, finding review."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

import auth
import db
import state
from utils import ctx, templates, to_ist

router = APIRouter()


@router.get("/history", response_class=HTMLResponse)
async def view_history(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    active_tab = request.query_params.get("tab", "review")
    if active_tab not in ("review", "cic"):
        active_tab = "review"

    selected_workflow_id = request.query_params.get("workflow") or None

    if active_tab == "review":
        runs = db.fetch_runs(workflow_id=selected_workflow_id)
        for run in runs:
            run["created_at"] = to_ist(run.get("created_at"))
        cic_runs = []
    else:
        runs = []
        cic_runs = db.fetch_cic_runs(workflow_id=selected_workflow_id)
        for run in cic_runs:
            run["created_at"] = to_ist(run.get("created_at"))

    review_workflows = [w for w in state.WORKFLOWS if w.get("type", "review") == "review"]
    cic_workflows = [w for w in state.WORKFLOWS if w.get("type") == "cic"]

    return templates.TemplateResponse("history.html", ctx(
        request, user,
        runs=runs,
        cic_runs=cic_runs,
        active_tab=active_tab,
        review_workflows=review_workflows,
        cic_workflows=cic_workflows,
        selected_workflow_id=selected_workflow_id,
        error=request.query_params.get("error"),
    ))


@router.get("/history/cic/{run_id}", response_class=HTMLResponse)
async def view_cic_run(request: Request, run_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    run = db.fetch_cic_run(run_id)
    if not run:
        return RedirectResponse(url="/history?tab=cic&error=CIC+run+not+found.")
    run["created_at"] = to_ist(run.get("created_at"))

    pages = db.fetch_cic_run_pages(run_id)
    comments = db.fetch_cic_comments(run_id)

    f1_image_map: dict[int, str] = {}
    f2_image_map: dict[int, str] = {}
    for p in pages:
        if p["file_version"] == "commented":
            f1_image_map[p["page_num"]] = p["drive_file_id"]
        else:
            f2_image_map[p["page_num"]] = p["drive_file_id"]

    page_comments_map: dict[int | None, list] = {}
    for c in comments:
        pr = c.get("page_resolved")
        page_comments_map.setdefault(pr, []).append(c)

    total_pages = run.get("total_pages", 0)

    return templates.TemplateResponse("cic_run_detail.html", ctx(
        request, user,
        run=run,
        total_pages=total_pages,
        f1_image_map=f1_image_map,
        f2_image_map=f2_image_map,
        page_comments_map=page_comments_map,
    ))


@router.get("/history/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    run = db.fetch_run(run_id)
    if not run:
        return RedirectResponse(url="/history?error=Run+not+found.")
    run["created_at"] = to_ist(run.get("created_at"))

    pages = db.fetch_run_pages(run_id)
    findings = db.fetch_run_findings(run_id)

    page_findings: dict[int, list] = {}
    doc_findings: list = []
    for f in findings:
        if f["page_num"] is None:
            doc_findings.append(f)
        else:
            page_findings.setdefault(f["page_num"], []).append(f)

    checkpoint_map = {cp["id"]: cp["category"] for cp in state.CHECKPOINTS}
    page_image_map = {p["page_num"]: p["drive_file_id"] for p in pages}

    return templates.TemplateResponse("run_detail.html", ctx(
        request, user,
        run=run,
        page_findings=page_findings,
        doc_findings=doc_findings,
        checkpoint_map=checkpoint_map,
        page_image_map=page_image_map,
        total_pages=run["total_pages"],
    ))


@router.post("/findings/{finding_id}/review")
async def update_finding_review(request: Request, finding_id: str):
    user = auth.get_current_user(request)
    if not user:
        return {"error": "Unauthorized"}, 401

    body = await request.json()
    review_status = body.get("review_status", "").strip()
    review_comment = body.get("review_comment", "").strip()

    if review_status not in ("valid", "invalid"):
        return {"error": "review_status must be 'valid' or 'invalid'"}
    if not review_comment:
        return {"error": "review_comment is required"}

    try:
        db.update_finding_review(finding_id, review_status, review_comment)
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}
