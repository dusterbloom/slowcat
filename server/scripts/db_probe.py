#!/usr/bin/env python3
"""
SurrealDB DB probe: prints raw results for session and knowledge queries.

Usage:
  cd server
  source .venv/bin/activate
  python scripts/db_probe.py unit_user_smoke
"""

import asyncio
import os
import sys
from typing import Any


async def run(user: str) -> int:
    try:
        from surrealdb import AsyncSurreal  # type: ignore
    except Exception as e:
        print(f"Surreal client not available: {e}")
        return 2

    url = os.getenv("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    ns = os.getenv("SURREALDB_NAMESPACE", "unit")
    db = os.getenv("SURREALDB_DATABASE", "memory_graph")
    user_id_str = f"user:{user}"

    client = AsyncSurreal(url)
    # Best-effort auth
    try:
        await client.signin({"user": os.getenv("SURREALDB_USER", "root"), "pass": os.getenv("SURREALDB_PASS", "slowcat_secure_2024")})
    except Exception:
        try:
            await client.signin({"username": os.getenv("SURREALDB_USER", "root"), "password": os.getenv("SURREALDB_PASS", "slowcat_secure_2024")})
        except Exception:
            pass
    await client.use(ns, db)

    async def q(sql: str, params: dict | None = None) -> Any:
        try:
            res = await client.query(sql, params or {})
            return res
        except Exception as e:
            print(f"ERR: {e}\nSQL: {sql}\nPARAMS: {params}")
            return None

    print("--- Sessions (raw) ---")
    print(await q("SELECT id, started_at, user_id FROM session ORDER BY started_at DESC LIMIT 5"))

    print("\n--- Sessions filter (user_id = $s) ---")
    print(await q("SELECT id, started_at, user_id FROM session WHERE user_id = $s ORDER BY started_at DESC LIMIT 5", {"s": user_id_str}))

    print("\n--- Sessions filter (string::concat('', user_id) = $s) ---")
    print(await q("SELECT id, started_at, user_id FROM session WHERE string::concat('', user_id) = $s ORDER BY started_at DESC LIMIT 5", {"s": user_id_str}))

    print("\n--- Knows edges (raw) ---")
    print(await q("SELECT id, in, out, relationship, strength FROM knows ORDER BY strength DESC LIMIT 5"))

    print("\n--- Knows by IN (record) ---")
    print(await q("SELECT id, in, out, relationship, strength FROM knows WHERE in = type::thing('user', $k) ORDER BY strength DESC LIMIT 5", {"k": user}))

    print("\n--- Knows by IN (string) ---")
    print(await q("SELECT id, in, out, relationship, strength FROM knows WHERE string::concat('', in) = $s ORDER BY strength DESC LIMIT 5", {"s": user_id_str}))

    await client.close()
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/db_probe.py <user_key>")
        return 2
    user = sys.argv[1]
    return asyncio.run(run(user))


if __name__ == "__main__":
    raise SystemExit(main())

