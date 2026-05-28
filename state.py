"""Global in-memory state: caches for workflows, checkpoints, admins, active jobs."""

import tempfile
from pathlib import Path

import db

# ── Constants ─────────────────────────────────────────────────────────────────

SUPER_ADMIN = "abhishek.jain@leadschool.in"

# Temp directory for per-job files (page images, findings.json, job.json)
_JOBS_DIR = Path(tempfile.gettempdir()) / "checkpoint_jobs"
_JOBS_DIR.mkdir(exist_ok=True)

# Job IDs currently being streamed — prevents duplicate concurrent SSE connections
# (EventSource auto-reconnects can re-trigger processing in parallel → OOM)
_ACTIVE_JOBS: set[str] = set()

# ── In-memory caches (populated at startup, refreshed after each mutation) ────

WORKFLOWS: list[dict] = []
CHECKPOINTS: list[dict] = []
CHECKPOINT_MAP: dict[str, dict] = {}
CATEGORIES: dict[str, list[dict]] = {}
ADMINS: set[str] = set()


# ── Reload helpers ─────────────────────────────────────────────────────────────

def _group_by_cat(checkpoints: list[dict]) -> dict[str, list[dict]]:
    """Internal — avoids importing utils (which imports state) and causing a cycle."""
    result: dict[str, list[dict]] = {}
    for cp in checkpoints:
        result.setdefault(cp["category"], []).append(cp)
    return result


def reload_checkpoints() -> None:
    global CHECKPOINTS, CHECKPOINT_MAP, CATEGORIES
    CHECKPOINTS = db.fetch_all_checkpoints()
    CHECKPOINT_MAP = {cp["id"]: cp for cp in CHECKPOINTS}
    CATEGORIES = _group_by_cat(CHECKPOINTS)


def reload_workflows() -> None:
    global WORKFLOWS
    WORKFLOWS = db.fetch_all_workflows()


def reload_admins() -> None:
    global ADMINS
    ADMINS = {a["email"] for a in db.fetch_all_admins()}


# ── Role helpers ───────────────────────────────────────────────────────────────

def is_super_admin(user: dict | None) -> bool:
    return bool(user) and user.get("email") == SUPER_ADMIN


def is_admin(user: dict | None) -> bool:
    return bool(user) and (
        user.get("email") == SUPER_ADMIN or user.get("email") in ADMINS
    )


# ── Populate caches at import time ────────────────────────────────────────────
reload_workflows()
reload_checkpoints()
reload_admins()
