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
        "prompt": "select_account consent",
        "access_type": "offline",
    },
)


async def login(request: Request) -> RedirectResponse:
    redirect_uri = str(request.url_for("auth_callback"))
    # Render terminates TLS at its proxy — the app sees http:// internally.
    # Force https:// so the redirect_uri matches Google's registered URI.
    if os.getenv("ENV") == "production" and redirect_uri.startswith("http://"):
        redirect_uri = "https://" + redirect_uri[7:]
    return await oauth.google.authorize_redirect(request, redirect_uri)


async def auth_callback(request: Request) -> RedirectResponse:
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:
        error_str = str(exc)
        # Surface the error clearly so it can be diagnosed.
        request.session.clear()
        return RedirectResponse(url=f"/?error=Authentication+failed:+{error_str[:120]}")

    user_info = token.get("userinfo") or {}
    request.session["token"] = dict(token)
    request.session["user"] = {
        "email": user_info.get("email", ""),
        "name": user_info.get("name", ""),
        "picture": user_info.get("picture", ""),
    }
    # If the React SPA frontend is deployed, redirect there after login.
    # Falls back to the Jinja2 dashboard for local / Render-only setups.
    frontend_url = os.getenv("FRONTEND_URL", "")
    redirect_to = f"{frontend_url}/" if frontend_url else "/"
    return RedirectResponse(url=redirect_to)


def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    frontend_url = os.getenv("FRONTEND_URL", "")
    return RedirectResponse(url=f"{frontend_url}/login" if frontend_url else "/login")


def get_current_user(request: Request) -> dict | None:
    return request.session.get("user")


def get_token(request: Request) -> dict | None:
    return request.session.get("token")
