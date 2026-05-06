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
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

import auth
from checker import run_checks
from commenter import post_comments
from reader import get_file_content

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

# File-based store for check results — keyed by a short UUID.
# Using temp files instead of an in-memory dict so that results survive
# uvicorn --reload restarts (which wipe module-level state).
_RESULTS_DIR = Path(tempfile.gettempdir()) / "sunny_pillow_results"
_RESULTS_DIR.mkdir(exist_ok=True)


def _save_result(result_id: str, data: dict) -> None:
    path = _RESULTS_DIR / f"{result_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_result(result_id: str) -> dict | None:
    path = _RESULTS_DIR / f"{result_id}.json"
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

    loop = asyncio.get_running_loop()

    try:
        file_data = await loop.run_in_executor(
            None, partial(get_file_content, token, drive_url.strip())
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

    try:
        findings, prompts = await loop.run_in_executor(
            None, partial(run_checks, file_data["full_text"], selected_checkpoints)
        )
    except Exception as exc:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user": user,
            "categories": CATEGORIES,
            "error": f"AI check failed: {str(exc)}",
        })

    comments_posted = 0
    comment_error = None

    if file_data["type"] != "pdf" and findings:
        try:
            comments_posted = await loop.run_in_executor(
                None, partial(post_comments, token, file_data, findings, CHECKPOINT_MAP)
            )
        except Exception as exc:
            comment_error = (
                f"Findings were detected but comments could not be posted to Drive: {str(exc)}"
            )

    result_id = uuid.uuid4().hex
    _save_result(result_id, {
        "user": user,
        "file_title": file_data["title"],
        "file_type": file_data["type"],
        "findings": findings,
        "prompts": prompts,
        "comments_posted": comments_posted,
        "comment_error": comment_error,
        "is_pdf": file_data["type"] == "pdf",
    })
    return RedirectResponse(url=f"/results/{result_id}", status_code=303)


@app.get("/results/{result_id}", response_class=HTMLResponse)
async def show_results(request: Request, result_id: str):
    user = auth.get_current_user(request)
    if not user:
        return RedirectResponse(url="/login")

    result = _load_result(result_id)
    if not result:
        return RedirectResponse(url="/?error=Results+not+found.+Please+run+a+new+check.")

    return templates.TemplateResponse("results.html", {
        "request": request,
        **result,
        "checkpoint_map": CHECKPOINT_MAP,
    })
