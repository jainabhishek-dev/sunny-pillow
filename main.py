"""CheckPoint FastAPI application entry point.

All route logic lives in routers/. This file only wires up the app.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import state
from routers import auth_routes, dashboard, jobs, cic, history
from routers.admin import checkpoints, workflows, admins
from routers import api as api_router

load_dotenv()

_is_prod = os.getenv("ENV") == "production"
_frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

app = FastAPI(title="CheckPoint")

# CORS — required for the React SPA (Vercel) to talk to this backend (Render).
# Credentials (session cookie) need explicit origin + allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session cookie — SameSite=None + Secure required for cross-origin cookies in production.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", ""),
    session_cookie="checkpoint_session",
    max_age=3600,
    https_only=_is_prod,
    same_site="none" if _is_prod else "lax",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routers ────────────────────────────────────────────────────────────────────

# JSON API (used by the React SPA)
app.include_router(api_router.router, prefix="/api")

# HTML routes (Jinja2 — kept as fallback during React rollout)
app.include_router(auth_routes.router)
app.include_router(dashboard.router)
app.include_router(jobs.router)
app.include_router(cic.router)
app.include_router(history.router)
app.include_router(checkpoints.router)
app.include_router(workflows.router)
app.include_router(admins.router)
