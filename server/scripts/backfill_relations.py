#!/usr/bin/env python3
"""
Backfill graph relations for parity:
- mentions: message -> mentions -> concept
- reflects: session -> reflects -> thought (using backup mapping)

Usage examples:
  python server/scripts/backfill_relations.py --mentions
  python server/scripts/backfill_relations.py --reflects --backup server/memory/localslowcat-2025-08-27.surql
"""

import argparse
import os
from loguru import logger

try:
    from surrealdb import AsyncSurreal
except ImportError:
    raise SystemExit("SurrealDB client not available. pip install surrealdb")

try:
    from surrealdb.types import RecordID  # type: ignore
    HAVE_RECORDID = True
except Exception:
    HAVE_RECORDID = False

from extract_backup_data import SurrealQLParser


def rows(res):
    if not res:
        return []
    if isinstance(res, list) and res and isinstance(res[0], dict) and 'result' in res[0]:
        return res[0]['result'] or []
    return res


async def connect(url: str, ns: str, db: str, user: str, password: str) -> AsyncSurreal:
    client = AsyncSurreal(url)
    await client.connect()
    try:
        await client.signin({"username": user, "password": password})
    except Exception:
        await client.signin({"user": user, "pass": password})
    await client.use(ns, db)
    return client


async def backfill_mentions(db: AsyncSurreal):
    cres = await db.query("SELECT id, name FROM concept")
    concepts = rows(cres)
    created = 0
    for c in concepts:
        cid = c.get('id'); name = c.get('name')
        if not cid or not name:
            continue
        # Prefer FTS
        try:
            mres = await db.query("SELECT id FROM message WHERE content @@ $name", {"name": name})
            msgs = rows(mres)
        except Exception:
            mres = await db.query(
                "SELECT id FROM message WHERE string::contains(string::lowercase(content), string::lowercase($name))",
                {"name": name},
            )
            msgs = rows(mres)
        for m in msgs:
            mid = m.get('id') if isinstance(m, dict) else m
            if not mid:
                continue
            ex = rows(await db.query("SELECT count() AS c FROM mentions WHERE in=$m AND out=$c", {"m": mid, "c": cid}))
            if ex and isinstance(ex[0], dict) and int(ex[0].get('c') or 0) > 0:
                continue
            try:
                if HAVE_RECORDID:
                    m_table, m_id = (mid.split(':',1) if isinstance(mid,str) and ':' in mid else ('message', str(mid)))
                    c_table, c_id = (cid.split(':',1) if isinstance(cid,str) and ':' in cid else ('concept', str(cid)))
                    await db.relate(RecordID(m_table, m_id), 'mentions', RecordID(c_table, c_id), {
                        'relevance': 0.5,
                        'extracted_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc),
                    })
                else:
                    await db.query("RELATE $m->mentions->$c SET relevance=0.5, extracted_at=time::now()", {"m": mid, "c": cid})
                created += 1
            except Exception:
                pass
    logger.info(f"Mentions created: {created}")


async def backfill_reflects(db: AsyncSurreal, backup: str):
    parser = SurrealQLParser(backup)
    data = parser.parse_backup()
    # Build mapping of old session_id to current session record id via first message match
    session_map = {}
    seen = set()
    for tr in data.get('tape', []):
        sid = tr.get('session_id')
        if not sid or sid in seen:
            continue
        seen.add(sid)
        ts = tr.get('ts')
        content = tr.get('content', '')
        if isinstance(ts, (int,float)):
            q = "SELECT id, session_id FROM message WHERE content=$content AND timestamp=time::from::secs($ts) LIMIT 1"
            params = {"content": content, "ts": int(ts)}
        else:
            q = "SELECT id, session_id FROM message WHERE content=$content AND timestamp=$ts LIMIT 1"
            params = {"content": content, "ts": ts or ''}
        m = rows(await db.query(q, params))
        if m:
            session_map[sid] = m[0].get('session_id')
    created_count = 0
    for table_name in ('thought', 'emergent_event'):
        for t in data.get(table_name, []):
            sid = t.get('session_id')
            if not sid or sid not in session_map:
                continue
            sess = session_map[sid]
            content = t.get('content') or t.get('content_snippet') or ''
            ts = t.get('ts')
            # Locate or create a thought node
            if isinstance(ts, (int,float)):
                tq = "SELECT id FROM thought WHERE content=$content AND timestamp=time::from::secs($ts) LIMIT 1"
                tparams = {"content": content, "ts": int(ts)}
            else:
                tq = "SELECT id FROM thought WHERE content=$content AND timestamp=$ts LIMIT 1"
                tparams = {"content": content, "ts": ts or ''}
            thr = rows(await db.query(tq, tparams))
            if thr:
                thid = thr[0].get('id')
            else:
                # Create a thought from emergent_event if needed
                if table_name == 'emergent_event':
                    # Choose a type from kind or default
                    ttype = t.get('kind', 'observation')
                    if isinstance(ts, (int,float)):
                        cq = "CREATE thought SET agent_id='slowcat', session_id=$sess, thought_type=$type, content=$content, timestamp=time::from::secs($ts), visibility='private', confidence=0.5 RETURN id"
                        cparams = {"sess": sess, "type": ttype, "content": content, "ts": int(ts)}
                    else:
                        cq = "CREATE thought SET agent_id='slowcat', session_id=$sess, thought_type=$type, content=$content, timestamp=$ts, visibility='private', confidence=0.5 RETURN id"
                        cparams = {"sess": sess, "type": ttype, "content": content, "ts": ts or ''}
                    created_rows = rows(await db.query(cq, cparams))
                    if not created_rows:
                        continue
                    first = created_rows[0]
                    thid = first.get('id') if isinstance(first, dict) else first
            else:
                # Skip if we cannot locate a corresponding thought
                continue
            ex = rows(await db.query("SELECT count() AS c FROM reflects WHERE in=$s AND out=$t", {"s": sess, "t": thid}))
            if ex and isinstance(ex[0], dict) and int(ex[0].get('c') or 0) > 0:
                continue
            try:
                if HAVE_RECORDID:
                    s_table, s_id = (sess.split(':',1) if isinstance(sess,str) and ':' in sess else ('session', str(sess)))
                    t_table, t_id = (thid.split(':',1) if isinstance(thid,str) and ':' in thid else ('thought', str(thid)))
                    await db.relate(RecordID(s_table, s_id), 'reflects', RecordID(t_table, t_id), {
                        'generated_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc),
                        'trigger_event': 'migration_backfill',
                    })
                else:
                    await db.query("RELATE $s->reflects->$t SET generated_at=time::now(), trigger_event='migration_backfill'", {"s": sess, "t": thid})
                # Do not update thought.session_id here; drivers may treat record ids as strings.
                created_count += 1
            except Exception as e:
                logger.warning(f"reflects link failed: {repr(e)}")
    logger.info(f"Reflects created: {created_count}")


def parse_args():
    p = argparse.ArgumentParser(description="Backfill relations for graph DB")
    p.add_argument('--url', default=os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc'))
    p.add_argument('--ns', default=os.getenv('SURREALDB_NAMESPACE', 'slowcat'))
    p.add_argument('--db', default=os.getenv('SURREALDB_DATABASE', 'memory_graph'))
    p.add_argument('--user', default=os.getenv('SURREALDB_USER', 'root'))
    p.add_argument('--pass', dest='password', default=os.getenv('SURREALDB_PASS', 'slowcat_secure_2024'))
    p.add_argument('--mentions', action='store_true')
    p.add_argument('--reflects', action='store_true')
    p.add_argument('--backup', default=os.getenv('BACKUP_FILE'))
    return p.parse_args()


if __name__ == '__main__':
    import asyncio
    args = parse_args()
    async def main():
        db = await connect(args.url, args.ns, args.db, args.user, args.password)
        try:
            if args.mentions:
                await backfill_mentions(db)
            if args.reflects:
                backup = args.backup or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory', 'localslowcat-2025-08-27.surql'))
                await backfill_reflects(db, backup)
        finally:
            await db.close()
    asyncio.run(main())
