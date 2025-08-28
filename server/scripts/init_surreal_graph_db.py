#!/usr/bin/env python3
"""
Initialize (or re-create) the SurrealDB graph schema for Slowcat.

Usage:
  cd server
  source .venv/bin/activate
  # Ensure env points to the target DB (e.g., memory_graph)
  export SURREALDB_DATABASE=memory_graph
  python scripts/init_surreal_graph_db.py

Env vars (with defaults):
  SURREALDB_URL=ws://127.0.0.1:8000/rpc
  SURREALDB_USER=root
  SURREALDB_PASS=slowcat_secure_2024
  SURREALDB_NAMESPACE=slowcat
  SURREALDB_DATABASE=memory_graph
  SURREAL_SCHEMA_FILE=scripts/surreal_define_graph_schema.surql
"""

import asyncio
import os
from pathlib import Path
from typing import Optional


async def apply_schema(url: str, ns: str, db: str, user: str, pwd: str, schema_path: Path) -> None:
    from surrealdb import AsyncSurreal  # type: ignore

    client = AsyncSurreal(url)
    # Best-effort signin
    try:
        await client.signin({"user": user, "pass": pwd})
    except Exception:
        try:
            await client.signin({"username": user, "password": pwd})
        except Exception:
            # Proceed unauthenticated if server permits
            pass
    await client.use(ns, db)

    sql = schema_path.read_text(encoding="utf-8")
    # SurrealDB accepts multi-statement strings in one call
    await client.query(sql)
    await client.close()


def main() -> int:
    url = os.getenv("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    ns = os.getenv("SURREALDB_NAMESPACE", "slowcat")
    db = os.getenv("SURREALDB_DATABASE", "memory_graph")
    user = os.getenv("SURREALDB_USER", "root")
    pwd = os.getenv("SURREALDB_PASS", "slowcat_secure_2024")
    schema_file = Path(os.getenv("SURREAL_SCHEMA_FILE", "scripts/surreal_define_graph_schema.surql")).resolve()

    if not schema_file.exists():
        print(f"❌ Schema file not found: {schema_file}")
        return 2

    print(f"Applying schema to SurrealDB: ns={ns} db={db} url={url}")
    try:
        asyncio.run(apply_schema(url, ns, db, user, pwd, schema_file))
        print("✅ Graph schema applied successfully")
        return 0
    except Exception as e:
        print(f"❌ Failed to apply schema: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

