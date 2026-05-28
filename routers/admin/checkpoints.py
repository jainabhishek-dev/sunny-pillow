"""Admin checkpoint CRUD routes."""

from typing import Annotated

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

import auth
import db
import state
from utils import ctx, filter_by_workflow, group_by_category, next_checkpoint_id, templates

router = APIRouter()


@router.get("/checkpoints", response_class=HTMLResponse)
async def manage_checkpoints(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    default_wf_id = state.WORKFLOWS[0]["id"] if state.WORKFLOWS else ""
    workflow_id = request.query_params.get("workflow", default_wf_id)
    selected_workflow = next(
        (w for w in state.WORKFLOWS if w["id"] == workflow_id),
        state.WORKFLOWS[0] if state.WORKFLOWS else None,
    )
    filtered = filter_by_workflow(state.CHECKPOINTS, selected_workflow["id"]) if selected_workflow else []

    return templates.TemplateResponse("checkpoints.html", ctx(
        request, user,
        workflows=state.WORKFLOWS,
        selected_workflow=selected_workflow,
        categories=group_by_category(filtered),
        success=request.query_params.get("success"),
        error=request.query_params.get("error"),
    ))


@router.post("/checkpoints/add", response_class=HTMLResponse)
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
    if not state.is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    new_id = next_checkpoint_id()
    sort_order = max((cp["sort_order"] for cp in state.CHECKPOINTS), default=0) + 1
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
        state.reload_checkpoints()
        return RedirectResponse(url=f"{base}&success=Checkpoint+added.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"{base}&error={exc}", status_code=303)


@router.post("/checkpoints/{cp_id}/edit", response_class=HTMLResponse)
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
    if not state.is_admin(user):
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
        state.reload_checkpoints()
        return RedirectResponse(url=f"{base}&success=Checkpoint+updated.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"{base}&error={exc}", status_code=303)


@router.post("/checkpoints/{cp_id}/delete", response_class=HTMLResponse)
async def delete_checkpoint(request: Request, cp_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not state.is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    workflow_param = request.query_params.get("workflow", "")
    base = f"/checkpoints?workflow={workflow_param}" if workflow_param else "/checkpoints"
    try:
        db.delete_checkpoint(cp_id)
        state.reload_checkpoints()
        return RedirectResponse(url=f"{base}&success=Checkpoint+deleted.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"{base}&error={exc}", status_code=303)
