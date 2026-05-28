"""CheckPoint FastAPI application entry point.

All route logic lives in routers/. This file only wires up the app.
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import state
from routers import auth_routes, dashboard, jobs, cic, history
from routers.admin import checkpoints, workflows, admins

load_dotenv()

app = FastAPI(title="CheckPoint")

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", ""),
    session_cookie="checkpoint_session",
    max_age=3600,
    https_only=os.getenv("ENV") == "production",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(auth_routes.router)
app.include_router(dashboard.router)
app.include_router(jobs.router)
app.include_router(cic.router)
app.include_router(history.router)
app.include_router(checkpoints.router)
app.include_router(workflows.router)
app.include_router(admins.router)
