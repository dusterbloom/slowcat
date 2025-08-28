#!/usr/bin/env python3
"""
Validate graph DB contents: counts and simple traversals.

Fails (exit 1) if core counts are unexpectedly low.
"""

import argparse
import os
from loguru import logger

try:
    from surrealdb import AsyncSurreal
except ImportError:
    raise SystemExit("SurrealDB client not available. pip install surrealdb")


async def validate(url: str, ns: str, db: str, user: str, password: str) -> int:
    client = AsyncSurreal(url)
    await client.connect()
    try:
        # Prefer username/password form first (your server accepts this reliably)
        try:
            await client.signin({"username": user, "password": password})
        except Exception:
            await client.signin({"user": user, "pass": password})
        await client.use(ns, db)
        # Log intended context (Surreal 1.0.6 lacks db::ns/db::name)
        logger.info(f"Active context (intended): ns={ns} db={db}")

        def rows(res):
            if not res:
                return []
            if isinstance(res, list) and res and isinstance(res[0], dict) and 'result' in res[0]:
                return res[0]['result'] or []
            return res

        # Counts
        counts = {}
        for tbl in ("user", "session", "message", "concept", "knows", "contains", "mentions"):
            try:
                r = await client.query(f"SELECT count() AS c FROM {tbl}")
                rr = rows(r)
                if rr and isinstance(rr[0], dict):
                    counts[tbl] = rr[0].get('c') or rr[0].get('count') or next(iter(rr[0].values()))
                else:
                    counts[tbl] = 0
            except Exception:
                counts[tbl] = 0
        logger.info("Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

        # Basic traversal sample: pick any user
        users = await client.query("SELECT id FROM user LIMIT 1")
        urows = rows(users)
        if urows:
            uid = urows[0]['id']
            knows = await client.query("SELECT out.name as concept FROM $uid->knows LIMIT 5", {"uid": uid})
            krows = rows(knows)
            logger.info(f"User {uid} knows: {[k.get('concept') for k in krows]}")

        # Pick any session and show 3 messages
        sess = await client.query("SELECT id FROM session LIMIT 1")
        srows = rows(sess)
        if srows:
            sid = srows[0]['id']
            # Order must be applied on the relation (contains.sequence_num)
            q = "SELECT out.content AS content FROM (SELECT * FROM $sid->contains ORDER BY sequence_num ASC LIMIT 3)"
            msgs = await client.query(q, {"sid": sid})
            mrows = rows(msgs)
            logger.info(f"Session {sid} first messages: {[m.get('content') for m in mrows if isinstance(m, dict)]}")

        # Sanity thresholds
        failures = []
        if counts.get('message', 0) == 0:
            failures.append('No messages created')
        if counts.get('session', 0) == 0:
            failures.append('No sessions created')
        if counts.get('user', 0) == 0:
            failures.append('No users created')

        await client.close()
        if failures:
            for f in failures:
                logger.error(f)
            return 1
        return 0
    finally:
        try:
            await client.close()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Validate graph DB contents")
    p.add_argument('--url', default=os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc'))
    p.add_argument('--ns', default=os.getenv('SURREALDB_NAMESPACE', 'slowcat'))
    p.add_argument('--db', default=os.getenv('SURREALDB_DATABASE', 'memory_graph'))
    p.add_argument('--user', default=os.getenv('SURREALDB_USER', 'root'))
    p.add_argument('--pass', dest='password', default=os.getenv('SURREALDB_PASS', 'slowcat_secure_2024'))
    return p.parse_args()


if __name__ == '__main__':
    import asyncio, sys
    args = parse_args()
    code = asyncio.run(validate(args.url, args.ns, args.db, args.user, args.password))
    sys.exit(code)
