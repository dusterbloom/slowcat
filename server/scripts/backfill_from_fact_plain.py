#!/usr/bin/env python3
"""
Backfill graph memory from fact_plain (idempotent, batched).

Usage:
  cd server
  source .venv/bin/activate
  python scripts/backfill_from_fact_plain.py --user unit_user_smoke --limit 500

If --user is omitted, processes all users found in fact_plain.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure the server package root is on sys.path regardless of CWD
SERVER_ROOT = Path(__file__).resolve().parent.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


async def backfill(user: str | None, limit: int, dry_run: bool) -> int:
    from memory.graph_surreal_memory import GraphSurrealMemory, GraphFact

    mem = GraphSurrealMemory()
    await mem.connect()
    db = mem.db  # underlying AsyncSurreal

    # Gather users to process
    users: List[str] = []
    if user:
        users = [user]
    else:
        try:
            res = await db.query(
                """
                SELECT distinct user_id FROM fact_plain GROUP ALL
                LIMIT 1000
                """
            )
            rows = mem._rows_from_query(res)
            users = [str(r.get('user_id')).split(':', 1)[1] if isinstance(r.get('user_id'), str) and ':' in str(r.get('user_id')) else str(r.get('user_id')) for r in rows]
        except Exception:
            users = []

    total = 0
    for u in users:
        uid = f"user:{u}"
        try:
            res = await db.query(
                """
                SELECT subject, predicate, value, fidelity, strength, source_text
                FROM fact_plain
                WHERE user_id = $uid
                LIMIT $lim
                """,
                {'uid': uid, 'lim': limit},
            )
            rows = mem._rows_from_query(res)
        except Exception:
            rows = []
        for r in rows:
            subject = r.get('subject') or u
            predicate = r.get('predicate') or 'related_to'
            value = r.get('value') or ''
            strength = float(r.get('strength', 0.6))
            fidelity = int(r.get('fidelity', 3))
            if dry_run:
                total += 1
                continue
            # Upsert relation edge
            try:
                gf = GraphFact(subject=subject, predicate=predicate, value=value, strength=strength, fidelity=fidelity, source_text=r.get('source_text',''))
                await mem.store_fact(u, gf)
            except Exception:
                pass
            # Upsert fragment
            try:
                frag = {
                    'type': 'semantic',
                    'content': {'subject': subject, 'predicate': predicate, 'value': value},
                    'context_tags': [predicate.lower(), str(value).lower(), u.lower()],
                    'strength': strength,
                }
                await mem.upsert_fragment(u, frag)  # type: ignore[attr-defined]
            except Exception:
                pass
            total += 1
    await mem.close()
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description='Backfill fragments/knows from fact_plain')
    ap.add_argument('--user', help='User key (without user:) to backfill; process all if omitted')
    ap.add_argument('--limit', type=int, default=1000, help='Max facts per user to process')
    ap.add_argument('--dry-run', action='store_true', help='Do not write, just count items')
    args = ap.parse_args()

    os.environ.setdefault('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc')
    os.environ.setdefault('SURREALDB_NAMESPACE', 'slowcat')
    os.environ.setdefault('SURREALDB_DATABASE', 'memory_graph')

    try:
        total = asyncio.run(backfill(args.user, args.limit, args.dry_run))
        print(f"Processed {total} fact_plain rows{' (dry-run)' if args.dry_run else ''}")
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
