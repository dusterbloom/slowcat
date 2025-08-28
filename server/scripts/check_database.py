#!/usr/bin/env python3
"""
Check Database Contents

Direct database connection to inspect what tables and data exist.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger
import os

def _rows(res):
    if not res:
        return []
    if isinstance(res, list) and res and isinstance(res[0], dict) and 'result' in res[0]:
        return res[0]['result'] or []
    return res

async def check_one(ns: str, dbname: str, url: str, user: str, password: str):
    logger.info(f"🔍 Checking database: {ns}.{dbname}")
    db = AsyncSurreal(url)
    await db.connect()
    try:
        try:
            await db.signin({"user": user, "pass": password})
        except Exception as e1:
            logger.warning(f"Signin with user/pass failed ({repr(e1)}), trying username/password")
            await db.signin({"username": user, "password": password})
        await db.use(ns, dbname)
        info = await db.query("INFO FOR DB")
        try:
            keys = list(info[0].keys()) if info and isinstance(info[0], dict) else 'n/a'
        except Exception:
            keys = 'n/a'
        logger.info(f"DB Info keys: {keys}")
        # Counts summary
        counts = {}
        for tbl in ("user", "session", "message", "concept", "knows", "contains", "reflects"):
            try:
                r = await db.query(f"SELECT count() AS c FROM {tbl}")
                rows = _rows(r)
                val = 0
                if rows:
                    first = rows[0]
                    if isinstance(first, dict):
                        # Try common keys
                        for k in ('c', 'count', 'count()'):
                            if k in first and isinstance(first[k], (int, float)):
                                val = int(first[k])
                                break
                        if val == 0:
                            # Try any numeric in dict
                            for v in first.values():
                                if isinstance(v, (int, float)):
                                    val = int(v)
                                    break
                    elif isinstance(first, (int, float)):
                        val = int(first)
                counts[tbl] = val
            except Exception:
                counts[tbl] = 0
        logger.info("Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

        # Sample a few records to confirm visibility
        try:
            msgs = _rows(await db.query("SELECT id, content FROM message LIMIT 3"))
            logger.info(f"message sample count={len(msgs)} ids={[m.get('id') for m in msgs] if msgs and isinstance(msgs[0], dict) else msgs}")
        except Exception as e:
            logger.warning(f"message sample failed: {repr(e)}")
        try:
            users = _rows(await db.query("SELECT id, name FROM user LIMIT 3"))
            logger.info(f"user sample count={len(users)} ids={[u.get('id') for u in users] if users and isinstance(users[0], dict) else users}")
        except Exception:
            pass
        # Schema spot-checks
        try:
            mi = await db.query("INFO FOR TABLE message")
            logger.info(f"message table info present: {bool(mi)}")
            ci = await db.query("INFO FOR TABLE contains")
            ki = await db.query("INFO FOR TABLE knows")
            logger.info(f"contains info present: {bool(ci)}, knows info present: {bool(ki)}")
        except Exception:
            pass
    finally:
        await db.close()

async def check_database():
    """Check both legacy and graph DBs"""
    url = os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc')
    user = os.getenv('SURREALDB_USER', 'root')
    password = os.getenv('SURREALDB_PASS', 'slowcat_secure_2024')
    ns = os.getenv('SURREALDB_NAMESPACE', 'slowcat')
    dbs = [os.getenv('LEGACY_DB', 'memory'), os.getenv('GRAPH_DB', 'memory_graph')]
    for dbname in dbs:
        try:
            await check_one(ns, dbname, url, user, password)
        except Exception as e:
            logger.error(f"Failed to check {ns}.{dbname}: {repr(e)}")
    logger.info("✅ Database check completed")

if __name__ == "__main__":
    asyncio.run(check_database())
