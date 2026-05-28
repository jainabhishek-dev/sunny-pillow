"""Workflow CRUD routes: /workflows."""

import asyncio
from functools import partial
from typing import Annotated

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

import auth
import db
import state
from utils import ctx, slugify, templates
from services.workflow_gen import generate_workflow_content

router = APIRouter()


@router.get("/workflows", response_class=HTMLResponse)
async def manage_workflows(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")
    if not state.is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.")

    checkpoint_counts: dict[str, int] = {}
    for cp in state.CHECKPOINTS:
        for wf_id in cp.get("workflows", []):
            checkpoint_counts[wf_id] = checkpoint_counts.get(wf_id, 0) + 1

    return templates.TemplateResponse("workflows.html", ctx(
        request, user,
        workflows=state.WORKFLOWS,
        checkpoint_counts=checkpoint_counts,
        success=request.query_params.get("success"),
        error=request.query_params.get("error"),
    ))


@router.post("/workflows/add", response_class=HTMLResponse)
async def add_workflow(
    request: Request,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
    ai_notes: Annotated[str, Form()] = "",
    action: Annotated[str, Form()] = "manual",
    type: Annotated[str, Form()] = "review",
):
    """Create a new workflow, optionally generating checkpoints with AI."""
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not state.is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    name = name.strip()
    description = description.strip()

    if not name:
        return RedirectResponse(url="/workflows?error=Workflow+name+is+required.", status_code=303)

    wf_id = slugify(name)
    if not wf_id:
        return RedirectResponse(url="/workflows?error=Could+not+generate+a+valid+ID+from+that+name.", status_code=303)

    if any(w["id"] == wf_id for w in state.WORKFLOWS):
        return RedirectResponse(
            url=f"/workflows?error=A+workflow+with+id+\"{wf_id}\"+already+exists.+Choose+a+different+name.",
            status_code=303,
        )

    sort_order = max((w.get("sort_order", 0) for w in state.WORKFLOWS), default=0) + 1

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

        try:
            db.insert_workflow({
                "id": wf_id,
                "name": name,
                "description": description,
                "sort_order": sort_order,
                "created_by": user["email"],
                "type": type if type in ("review", "cic") else "review",
            })
        except Exception as exc:
            return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)

        existing_nums = [
            int(cp["id"].split("_")[1])
            for cp in state.CHECKPOINTS
            if cp["id"].startswith("cp_") and cp["id"].split("_")[1].isdigit()
        ]
        next_num = max(existing_nums, default=0) + 1
        next_sort = max((cp.get("sort_order", 0) for cp in state.CHECKPOINTS), default=0) + 1

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

        state.reload_checkpoints()
        state.reload_workflows()

        cp_count = len(result.get("checkpoints", [])) - len(insert_errors)
        msg = f"Workflow+created+with+{cp_count}+AI-generated+checkpoints."
        if insert_errors:
            msg += f"+({len(insert_errors)}+checkpoint+inserts+failed)"
        return RedirectResponse(url=f"/workflows?success={msg}", status_code=303)

    else:
        try:
            db.insert_workflow({
                "id": wf_id,
                "name": name,
                "description": description,
                "sort_order": sort_order,
                "created_by": user["email"],
                "type": type if type in ("review", "cic") else "review",
            })
            state.reload_workflows()
            return RedirectResponse(
                url="/workflows?success=Workflow+created.+Add+a+system+prompt+and+checkpoints+to+activate+it.",
                status_code=303,
            )
        except Exception as exc:
            return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)


@router.post("/workflows/{wf_id}/edit", response_class=HTMLResponse)
async def edit_workflow(
    request: Request,
    wf_id: str,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not state.is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    name = name.strip()
    if not name:
        return RedirectResponse(url="/workflows?error=Workflow+name+is+required.", status_code=303)

    try:
        db.update_workflow(wf_id, {"name": name, "description": description.strip()})
        state.reload_workflows()
        return RedirectResponse(url="/workflows?success=Workflow+updated.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)


@router.post("/workflows/{wf_id}/delete", response_class=HTMLResponse)
async def delete_workflow(request: Request, wf_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if not state.is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.", status_code=303)

    workflow = next((w for w in state.WORKFLOWS if w["id"] == wf_id), None)
    if not workflow:
        return RedirectResponse(url="/workflows?error=Workflow+not+found.", status_code=303)

    if not state.is_super_admin(user) and workflow.get("created_by") != user.get("email"):
        return RedirectResponse(
            url="/workflows?error=Only+the+super+admin+or+the+workflow+creator+can+delete+this+workflow.",
            status_code=303,
        )

    try:
        from utils import cascade_delete_workflow
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(cascade_delete_workflow, wf_id))
        return RedirectResponse(url="/workflows?success=Workflow+and+its+exclusive+checkpoints+deleted.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/workflows?error={exc}", status_code=303)
