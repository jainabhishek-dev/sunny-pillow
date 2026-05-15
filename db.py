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
