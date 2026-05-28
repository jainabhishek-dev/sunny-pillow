"""Admin management routes: /admins."""

from typing import Annotated

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

import auth
import db
import state
from utils import ctx, templates

router = APIRouter()


@router.get("/admins", response_class=HTMLResponse)
async def manage_admins(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")
    if not state.is_admin(user):
        return RedirectResponse(url="/?error=Admin+access+required.")

    admins = db.fetch_all_admins()
    return templates.TemplateResponse("admins.html", ctx(
        request, user,
        admins=admins,
        success=request.query_params.get("success"),
        error=request.query_params.get("error"),
    ))


@router.post("/admins/add", response_class=HTMLResponse)
async def add_admin(
    request: Request,
    email: Annotated[str, Form()],
):
    user = auth.get_current_user(request)
    if not user or not state.is_admin(user):
        return RedirectResponse(url="/login", status_code=303)

    try:
        db.insert_admin({"email": email.strip().lower(), "added_by": user["email"]})
        state.reload_admins()
        return RedirectResponse(url="/admins?success=Admin+added.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/admins?error={exc}", status_code=303)


@router.post("/admins/{email}/delete", response_class=HTMLResponse)
async def delete_admin_route(request: Request, email: str):
    user = auth.get_current_user(request)
    if not user or not state.is_super_admin(user):
        return RedirectResponse(url="/login", status_code=303)

    try:
        db.delete_admin(email)
        state.reload_admins()
        return RedirectResponse(url="/admins?success=Admin+removed.", status_code=303)
    except Exception as exc:
        return RedirectResponse(url=f"/admins?error={exc}", status_code=303)
