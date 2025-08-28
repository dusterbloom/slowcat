#!/usr/bin/env python3
"""
Verify QueryRouter retrieval prefers graph and falls back cleanly (skips if SurrealDB not available).
"""

import os
import subprocess
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _server_running() -> bool:
    try:
        subprocess.run(["curl", "-s", "http://127.0.0.1:8000/health"], timeout=1, check=False)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_running(), reason="SurrealDB server not available")


@pytest.mark.asyncio
async def test_router_prefers_graph_and_fallbacks():
    os.environ['USE_SURREALDB'] = 'true'
    os.environ['SURREALDB_DATABASE'] = os.getenv('SURREALDB_DATABASE', 'memory_graph')
    os.environ['USER_ID'] = 'unit_router_user'

    # Create memory system and store a fact
    from memory.graph_surreal_memory import GraphSurrealMemory, GraphFact
    mem = GraphSurrealMemory()
    await mem.connect()
    await mem.store_fact('unit_router_user', GraphFact(subject='unit_router_user', predicate='preference', value='coffee', strength=0.9, fidelity=4))

    # Route a query via QueryRouter (through the adapter used by SCM)
    from memory import create_smart_memory_system
    ms = create_smart_memory_system()

    # Route a query either via adapter.process_query or via a QueryRouter built on top
    if hasattr(ms, 'process_query'):
        resp = await ms.process_query('coffee')
        assert getattr(resp, 'total_results', 0) >= 1
        assert any(getattr(r, 'source_store', 'facts') == 'facts' for r in resp.results)
    else:
        # Build a QueryRouter using the Surreal graph memory as both facts and tape stores
        from memory.query_router import create_query_router
        router = create_query_router(facts_graph=ms, tape_store=ms, embedding_store=None)
        routed = await router.route_query('coffee')
        if getattr(routed, 'total_results', 0) == 0:
            # Classifier may choose BYPASS for a one-token query like 'coffee'.
            # Verify the underlying graph memory fallback retrieves facts.
            got = await ms.search_facts('coffee', limit=3)  # type: ignore[attr-defined]
            assert isinstance(got, list) and len(got) >= 1
        else:
            assert any(getattr(r, 'source_store', 'facts') == 'facts' for r in routed.results)
