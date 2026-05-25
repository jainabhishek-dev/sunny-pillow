"""Supabase data-access helpers using httpx REST API directly."""

import os
import httpx

_HEADERS_CACHE: dict | None = None


def _headers() -> dict:
    global _HEADERS_CACHE
    if _HEADERS_CACHE is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        _HEADERS_CACHE = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
    return _HEADERS_CACHE


def _base_url() -> str:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    return f"{url}/rest/v1"


def fetch_all_checkpoints() -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"order": "sort_order"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def insert_checkpoint(row: dict) -> dict:
    resp = httpx.post(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        json=row,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def update_checkpoint(cp_id: str, fields: dict) -> dict:
    resp = httpx.patch(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"id": f"eq.{cp_id}"},
        json=fields,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def delete_checkpoint(cp_id: str) -> None:
    resp = httpx.delete(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"id": f"eq.{cp_id}"},
        timeout=10,
    )
    resp.raise_for_status()


# ── Workflow helpers ──────────────────────────────────────────────────────────

def fetch_all_workflows() -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/workflows",
        headers=_headers(),
        params={"order": "sort_order"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def insert_workflow(row: dict) -> dict:
    resp = httpx.post(
        f"{_base_url()}/workflows",
        headers=_headers(),
        json=row,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def update_workflow(wf_id: str, fields: dict) -> dict:
    resp = httpx.patch(
        f"{_base_url()}/workflows",
        headers=_headers(),
        params={"id": f"eq.{wf_id}"},
        json=fields,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def delete_workflow(wf_id: str) -> None:
    resp = httpx.delete(
        f"{_base_url()}/workflows",
        headers=_headers(),
        params={"id": f"eq.{wf_id}"},
        timeout=10,
    )
    resp.raise_for_status()


def fetch_checkpoints_by_workflow(wf_id: str) -> list[dict]:
    """Fetch all checkpoints whose workflows array contains wf_id."""
    resp = httpx.get(
        f"{_base_url()}/checkpoints",
        headers=_headers(),
        params={"workflows": f'cs.{{"{wf_id}"}}'},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ── Admin helpers ─────────────────────────────────────────────────────────────

def fetch_all_admins() -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/admins",
        headers=_headers(),
        params={"order": "added_at"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def insert_admin(row: dict) -> dict:
    resp = httpx.post(
        f"{_base_url()}/admins",
        headers=_headers(),
        json=row,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def delete_admin(email: str) -> None:
    resp = httpx.delete(
        f"{_base_url()}/admins",
        headers=_headers(),
        params={"email": f"eq.{email}"},
        timeout=10,
    )
    resp.raise_for_status()


# ── Run history helpers ───────────────────────────────────────────────────────

def insert_run(row: dict) -> None:
    resp = httpx.post(
        f"{_base_url()}/runs",
        headers=_headers(),
        json=row,
        timeout=15,
    )
    resp.raise_for_status()


def insert_run_pages(rows: list[dict]) -> None:
    """Batch-insert all page image records for a run in one call."""
    resp = httpx.post(
        f"{_base_url()}/run_pages",
        headers=_headers(),
        json=rows,
        timeout=15,
    )
    resp.raise_for_status()


def insert_run_findings(rows: list[dict]) -> None:
    """Batch-insert all findings for a run in one call."""
    resp = httpx.post(
        f"{_base_url()}/run_findings",
        headers=_headers(),
        json=rows,
        timeout=15,
    )
    resp.raise_for_status()


def fetch_runs(workflow_id: str | None = None) -> list[dict]:
    """Fetch all runs ordered by most recent first, optionally filtered by workflow."""
    params: dict = {"order": "created_at.desc"}
    if workflow_id:
        params["workflow_id"] = f"eq.{workflow_id}"
    resp = httpx.get(
        f"{_base_url()}/runs",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()




def fetch_run(run_id: str) -> dict | None:
    resp = httpx.get(
        f"{_base_url()}/runs",
        headers=_headers(),
        params={"id": f"eq.{run_id}", "limit": "1"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


def fetch_run_pages(run_id: str) -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/run_pages",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "order": "page_num"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_run_findings(run_id: str) -> list[dict]:
    resp = httpx.get(
        f"{_base_url()}/run_findings",
        headers=_headers(),
        params={"run_id": f"eq.{run_id}", "order": "page_num.asc.nullslast,id.asc"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def update_finding_review(finding_id: str, review_status: str, review_comment: str) -> None:
    """Overwrite the review verdict and comment on a single finding."""
    resp = httpx.patch(
        f"{_base_url()}/run_findings",
        headers=_headers(),
        params={"id": f"eq.{finding_id}"},
        json={"review_status": review_status, "review_comment": review_comment},
        timeout=10,
    )
    resp.raise_for_status()
