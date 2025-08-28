#!/usr/bin/env python3
"""
GraphSurrealMemory session semantics tests (skip if SurrealDB not available).
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from surrealdb import AsyncSurreal  # type: ignore
    SURREALDB_AVAILABLE = True
except Exception:
    SURREALDB_AVAILABLE = False


def _server_running() -> bool:
    # Prefer quick HTTP health if available; otherwise fall back to TCP port check
    try:
        r = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:8000/health"],
            capture_output=True,
            timeout=1,
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass
    # Fallback: check TCP port open
    try:
        import socket
        with socket.create_connection(("127.0.0.1", 8000), timeout=1):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not SURREALDB_AVAILABLE or not _server_running(),
    reason="SurrealDB client/server not available",
)


@pytest.mark.asyncio
async def test_session_increment_and_turns():
    os.environ.setdefault("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    os.environ.setdefault("SURREALDB_NAMESPACE", "unit")
    os.environ.setdefault("SURREALDB_DATABASE", "memory")

    from memory.graph_surreal_memory import GraphSurrealMemory

    mem = GraphSurrealMemory()
    await mem.connect()

    user = "unit_user_sessions"

    # Count sessions before
    before = await mem.get_user_sessions(user_name=user, limit=100)
    before_count = len(before)

    # Start a new session (increments count by creating a new session row)
    sid = await mem.start_session(user)
    assert isinstance(sid, str) and sid, "start_session should return a session id"

    after = await mem.get_user_sessions(user_name=user, limit=100)
    assert len(after) == before_count + 1, "session count should increase by 1"

    # Update session: increments turn_count on the active session
    await mem.update_session(user)
    await mem.update_session(user)

    latest = await mem.get_user_sessions(user_name=user, limit=1)
    assert latest, "should have at least one session"
    turn_count = latest[0].get("turn_count", 0)
    assert turn_count >= 2, f"turn_count expected >=2, got {turn_count}"

    # Clean close
    # (DB is durable; no deletion here)
