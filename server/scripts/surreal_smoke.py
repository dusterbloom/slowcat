#!/usr/bin/env python3
"""
SurrealDB smoke test (no pytest).

Verifies GraphSurrealMemory with:
- Session increment semantics (start_session, update_session)
- Fact store/search (bounded)
- Tape add/get_recent ordering

Usage:
  cd server
  source .venv/bin/activate
  python scripts/surreal_smoke.py

Environment (optional):
  SURREALDB_URL=ws://127.0.0.1:8000/rpc
  SURREALDB_NAMESPACE=unit
  SURREALDB_DATABASE=memory
"""

import asyncio
import os
import socket
import sys
from pathlib import Path
from typing import Any

# Ensure the server package root is on sys.path regardless of CWD
SERVER_ROOT = Path(__file__).resolve().parent.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


def server_available() -> bool:
    # Quick TCP check on default port; if health route is present, server is usually up
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=1):
            return True
    except Exception:
        return False


async def main() -> int:
    os.environ.setdefault("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    os.environ.setdefault("SURREALDB_NAMESPACE", "unit")
    os.environ.setdefault("SURREALDB_DATABASE", "memory_graph")

    try:
        from memory.graph_surreal_memory import GraphSurrealMemory, GraphFact
    except Exception as e:
        print(f"❌ Failed to import GraphSurrealMemory: {e}")
        return 2

    if not server_available():
        print("❌ SurrealDB server not reachable on 127.0.0.1:8000")
        return 2

    mem = GraphSurrealMemory()
    try:
        await mem.connect()
    except Exception as e:
        print(f"❌ Connect failed: {e}")
        return 2

    failures = 0

    # 1) Session semantics
    try:
        user = "unit_user_smoke"
        before = await mem.get_user_sessions(user_name=user, limit=100)
        before_count = len(before)
        sid = await mem.start_session(user)
        after = await mem.get_user_sessions(user_name=user, limit=100)
        ok = len(after) == before_count + 1 and isinstance(sid, str) and bool(sid)
        print(f"[session] increment: {'OK' if ok else 'FAIL'} (before={before_count}, after={len(after)})")
        if not ok:
            failures += 1

        await mem.update_session(user)
        await mem.update_session(user)
        latest = await mem.get_user_sessions(user_name=user, limit=1)
        turns = (latest[0].get('turn_count', 0) if latest else 0)
        ok = turns >= 2
        print(f"[session] turn_count: {'OK' if ok else 'FAIL'} (turns={turns})")
        if not ok:
            failures += 1
    except Exception as e:
        print(f"[session] ❌ Exception: {e}")
        failures += 1

    # 2) Facts store/search
    try:
        # Ensure search_facts routes to this user
        os.environ['USER_ID'] = user
        fact = GraphFact(
            subject=user,
            predicate="preference",
            value="coffee",
            fidelity=4,
            strength=0.9,
            source_text="User prefers coffee",
        )
        await mem.store_fact(user, fact)
        facts = await mem.search_facts("coffee", limit=3)
        ok = any(getattr(f, 'value', None) == 'coffee' for f in facts)
        print(f"[facts] search coffee: {'OK' if ok else 'FAIL'} (n={len(facts)})")
        if not ok:
            failures += 1
    except Exception as e:
        print(f"[facts] ❌ Exception: {e}")
        failures += 1

    # 3) Tape recent ordering
    try:
        await mem.add_entry(role='user', content='Hello from smoke test', speaker_id=user)
        await mem.add_entry(role='assistant', content='Hi there!', speaker_id=user)
        recent = await mem.get_recent(limit=2)
        ok = (
            isinstance(recent, list)
            and len(recent) == 2
            and recent[0].get('role') == 'assistant'
            and recent[1].get('role') == 'user'
        )
        print(f"[tape] recent ordering: {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1
    except Exception as e:
        print(f"[tape] ❌ Exception: {e}")
        failures += 1

    await asyncio.sleep(0)  # let pending tasks settle
    await mem.db.close()  # type: ignore[attr-defined]

    if failures:
        print(f"❌ Smoke test failed with {failures} issue(s)")
        return 1
    print("✅ Smoke test passed")
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)
