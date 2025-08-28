#!/usr/bin/env python3
"""
GraphSurrealMemory retrieval quality tests (skip if SurrealDB not available).

Covers:
- search_facts bounded recall via store_fact
- get_recent bounded and ordering
- optional knn_tape (skip if not implemented or encoder unavailable)
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
    # Prefer HTTP health; fallback to TCP port check
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
async def test_facts_and_recent_are_bounded_and_useful():
    os.environ.setdefault("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    os.environ.setdefault("SURREALDB_NAMESPACE", "unit")
    os.environ.setdefault("SURREALDB_DATABASE", "memory")

    from memory.graph_surreal_memory import GraphSurrealMemory, GraphFact

    mem = GraphSurrealMemory()
    await mem.connect()

    user = "unit_user_retrieval"

    # Store a high-signal fact
    fact = GraphFact(
        subject=user,
        predicate="preference",
        value="coffee",
        fidelity=4,
        strength=0.9,
        source_text="User said they prefer coffee.",
    )
    stored_reinforced = await mem.store_fact(user, fact)
    # First insert returns False (new), subsequent True (reinforced) — either is acceptable here
    assert stored_reinforced in (False, True)

    # Search facts (bounded, useful)
    results = await mem.search_facts("coffee", limit=3)
    assert len(results) >= 1 and len(results) <= 3
    assert any(getattr(r, "value", None) == "coffee" for r in results)

    # Add recent tape entries and assert ordering + bounds
    await mem.add_entry(role="user", content="Hello from unit test", speaker_id=user)
    await mem.add_entry(role="assistant", content="Hi there!", speaker_id=user)

    recent = await mem.get_recent(limit=2)
    assert len(recent) == 2
    # Most recent first; assistant should be first
    assert recent[0]["role"] == "assistant"
    assert recent[1]["role"] == "user"

    # Single-item bound
    one = await mem.get_recent(limit=1)
    assert len(one) == 1


@pytest.mark.asyncio
async def test_knn_tape_optional():
    """If knn_tape is available (doc path), verify bounds; otherwise skip."""
    os.environ.setdefault("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    os.environ.setdefault("SURREALDB_NAMESPACE", "unit")
    os.environ.setdefault("SURREALDB_DATABASE", "memory")

    # Prefer memory system factory (may return adapter exposing knn_tape)
    os.environ["USE_SURREALDB"] = "true"
    try:
        from memory import create_smart_memory_system
        ms = create_smart_memory_system()
        knn = getattr(ms, "knn_tape", None)
        # If not present, test is not applicable
        if knn is None:
            pytest.skip("knn_tape not implemented in graph path")

        # Insert some content
        await ms.tape_store.add_entry(role="user", content="anchor phrase apple banana carrot", speaker_id="unit_knn")
        await ms.tape_store.add_entry(role="assistant", content="response with banana", speaker_id="unit_knn")

        # KNN query
        results = await ms.knn_tape("banana", limit=3, scan=10, speaker_id="unit_knn")
        assert isinstance(results, list)
        assert len(results) <= 3
    finally:
        if "USE_SURREALDB" in os.environ:
            del os.environ["USE_SURREALDB"]
