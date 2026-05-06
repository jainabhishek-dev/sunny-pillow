import os
from dotenv import load_dotenv
from authlib.integrations.starlette_client import OAuth

load_dotenv()

from starlette.requests import Request
from fastapi.responses import RedirectResponse

oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": (
            "openid email profile "
            "https://www.googleapis.com/auth/drive "
            "https://www.googleapis.com/auth/documents.readonly "
            "https://www.googleapis.com/auth/presentations.readonly"
        ),
        "prompt": "select_account",
    },
)


async def login(request: Request) -> RedirectResponse:
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


async def auth_callback(request: Request) -> RedirectResponse:
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:
        return RedirectResponse(url=f"/?error=Authentication+failed:+{str(exc)[:80]}")

    user_info = token.get("userinfo") or {}
    request.session["token"] = dict(token)
    request.session["user"] = {
        "email": user_info.get("email", ""),
        "name": user_info.get("name", ""),
        "picture": user_info.get("picture", ""),
    }
    return RedirectResponse(url="/")


def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login")


def get_current_user(request: Request) -> dict | None:
    return request.session.get("user")


def get_token(request: Request) -> dict | None:
    return request.session.get("token")
