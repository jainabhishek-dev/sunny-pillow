"""Dashboard route: GET /"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

import auth
import state
from utils import ctx, filter_by_workflow, group_by_category, templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    error = request.query_params.get("error")
    workflow_id = request.query_params.get("workflow")

    if workflow_id:
        filtered_checkpoints = filter_by_workflow(state.CHECKPOINTS, workflow_id)
        filtered_categories = group_by_category(filtered_checkpoints)
    else:
        filtered_categories = {}

    selected_workflow = next((w for w in state.WORKFLOWS if w["id"] == workflow_id), None)
    review_workflows = [w for w in state.WORKFLOWS if w.get("type", "review") == "review"]
    cic_workflows = [w for w in state.WORKFLOWS if w.get("type") == "cic"]

    return templates.TemplateResponse("index.html", ctx(
        request, user,
        workflows=state.WORKFLOWS,
        review_workflows=review_workflows,
        cic_workflows=cic_workflows,
        selected_workflow=selected_workflow,
        categories=filtered_categories,
        error=error,
    ))
