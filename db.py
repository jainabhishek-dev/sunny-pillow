"""Supabase client and checkpoint data-access helpers."""

import os
from supabase import create_client, Client

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        _client = create_client(url, key)
    return _client


def fetch_all_checkpoints() -> list[dict]:
    """Return all checkpoints ordered by sort_order."""
    result = get_client().table("checkpoints").select("*").order("sort_order").execute()
    return result.data


def insert_checkpoint(row: dict) -> dict:
    result = get_client().table("checkpoints").insert(row).execute()
    return result.data[0]


def update_checkpoint(cp_id: str, fields: dict) -> dict:
    result = get_client().table("checkpoints").update(fields).eq("id", cp_id).execute()
    return result.data[0]


def delete_checkpoint(cp_id: str) -> None:
    get_client().table("checkpoints").delete().eq("id", cp_id).execute()
