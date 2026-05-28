"""Small pure helpers shared across routers and services."""

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

import db
import state

# Shared Jinja2 template renderer — imported by every router
templates = Jinja2Templates(directory="templates")

_IST = timezone(timedelta(hours=5, minutes=30))


# ── String / ID helpers ───────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Convert a display name to a lowercase underscore-separated ID."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def next_checkpoint_id() -> str:
    """Generate the next sequential checkpoint ID (e.g. cp_042)."""
    existing = [cp["id"] for cp in state.CHECKPOINTS if cp["id"].startswith("cp_")]
    nums = []
    for cid in existing:
        try:
            nums.append(int(cid.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return f"cp_{max(nums, default=0) + 1:03d}"


# ── Checkpoint filtering ───────────────────────────────────────────────────────

def group_by_category(checkpoints: list[dict]) -> dict[str, list[dict]]:
    categories: dict[str, list[dict]] = {}
    for cp in checkpoints:
        categories.setdefault(cp["category"], []).append(cp)
    return categories


def filter_by_workflow(checkpoints: list[dict], workflow_id: str) -> list[dict]:
    return [cp for cp in checkpoints if workflow_id in cp.get("workflows", [])]


# ── Datetime ───────────────────────────────────────────────────────────────────

def to_ist(dt_str: str | None) -> str | None:
    """Convert a UTC datetime string from Supabase to IST (same format for template slicing)."""
    if not dt_str:
        return dt_str
    try:
        dt = datetime.fromisoformat(dt_str.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return dt_str


# ── Template context ───────────────────────────────────────────────────────────

def ctx(request: Request, user: dict | None, **kwargs) -> dict:
    """Base template context including auth/role flags."""
    return {
        "request": request,
        "user": user,
        "is_admin": state.is_admin(user),
        "is_super_admin": state.is_super_admin(user),
        **kwargs,
    }


# ── Job file helpers ───────────────────────────────────────────────────────────

def ensure_job_dir(job_id: str) -> Path:
    job_dir = state._JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    return job_dir


def save_job(job_id: str, data: dict) -> None:
    job_dir = ensure_job_dir(job_id)
    (job_dir / "job.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_job(job_id: str) -> dict | None:
    path = state._JOBS_DIR / job_id / "job.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


# ── Workflow cascade delete ────────────────────────────────────────────────────

def cascade_delete_workflow(wf_id: str) -> None:
    """
    Delete a workflow and handle its checkpoints:
    - Checkpoints exclusive to this workflow are deleted entirely.
    - Shared checkpoints have this workflow removed from their workflows array.
    Reloads both globals after completion.
    """
    checkpoints = db.fetch_checkpoints_by_workflow(wf_id)
    for cp in checkpoints:
        remaining = [w for w in cp.get("workflows", []) if w != wf_id]
        if remaining:
            db.update_checkpoint(cp["id"], {"workflows": remaining})
        else:
            db.delete_checkpoint(cp["id"])
    db.delete_workflow(wf_id)
    state.reload_checkpoints()
    state.reload_workflows()
