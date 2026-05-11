"""
One-time seed script: reads checkpoints.yaml and upserts all checkpoints
into the Supabase `checkpoints` table.

Run from the checkpoint/ directory:
    python seed_checkpoints.py
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

client = create_client(SUPABASE_URL, SUPABASE_KEY)

yaml_path = Path(__file__).parent / "checkpoints.yaml"
with open(yaml_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

rows = []
for i, cp in enumerate(data["checkpoints"]):
    rows.append({
        "id":          cp["id"],
        "category":    cp["category"],
        "name":        cp["name"],
        "description": cp["description"].strip(),
        "type":        cp["type"],
        "workflows":   cp.get("workflows", ["edit"]),
        "sort_order":  i,
    })

result = client.table("checkpoints").upsert(rows).execute()
print(f"Seeded {len(rows)} checkpoints into Supabase.")
