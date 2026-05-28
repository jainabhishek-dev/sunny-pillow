"""Authentication routes: /login, /login/google, /auth/callback, /logout."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.requests import Request

import auth
from utils import templates

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = auth.get_current_user(request)
    if user:
        return RedirectResponse(url="/")
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/login/google")
async def login_google(request: Request):
    return await auth.login(request)


@router.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    return await auth.auth_callback(request)


@router.get("/logout")
def logout(request: Request):
    return auth.logout(request)
