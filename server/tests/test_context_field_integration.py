"""
Integration-style test for SmartContextManager with ContextField plan budgets.

Avoids heavy libraries by injecting a lightweight fake `memory` module before
importing SmartContextManager. Compares behavior with USE_CONTEXT_FIELD=false
vs true, asserting that budgets trim blocks deterministically.
"""

import os
import sys
import types


def _install_fake_memory_module():
    """Install a fake `memory` module into sys.modules to avoid heavy deps."""
    mem = types.ModuleType("memory")

    class FakeFact:
        def __init__(self, subject, predicate, value, fidelity=3, last_seen=0):
            self.subject = subject
            self.predicate = predicate
            self.value = value
            self.fidelity = fidelity
            self.last_seen = last_seen

    class FakeFactsGraph:
        async def get_top_facts(self, limit=10):
            # Return many facts to create a large facts block
            items = []
            # Include a clearly relevant fact for the query
            items.append(FakeFact("user", "dog_name", "Potola", fidelity=4, last_seen=1200.0))
            for i in range(20):
                items.append(FakeFact("user", f"pref_{i}", f"value_{i}", fidelity=3, last_seen=1000 - i))
            return items[:limit]

        async def search_facts(self, query: str, limit: int = 20):
            # Return additional facts to enlarge context
            items = []
            for i in range(10):
                items.append(FakeFact("topic", f"tag_{i}", f"val_{i}", fidelity=2, last_seen=800 - i))
            return items[:limit]

        async def start_session(self, speaker_id: str):
            return None

        async def get_session_info(self, speaker_id: str):
            return {"session_count": 1, "first_seen": 0, "last_interaction": 0, "total_turns": 1}

    class FakeTapeStore:
        async def get_last_summary(self):
            return None

        async def add_entry(self, role: str, content: str, speaker_id: str):
            return True

        async def add_summary(self, *args, **kwargs):
            return True

    class FakeMemorySystem:
        def __init__(self):
            self.facts_graph = FakeFactsGraph()
            self.tape_store = FakeTapeStore()

        async def update_session(self, key: str):
            return None

        async def process_query(self, query: str):
            # Simulate returning conversation snippets when asked
            class R:
                def __init__(self):
                    class C:
                        def __init__(self):
                            self.intent = type("I", (), {"name": "CONVERSATION_HISTORY"})
                    self.classification = C()
                    class Res:
                        def __init__(self, text):
                            self.content = text
                            self.source_store = "tape"
                    self.results = [Res("We talked about Potola."), Res("You mentioned San Francisco.")]
            return R()

    def create_smart_memory_system(db_path: str = "data/facts.db"):
        return FakeMemorySystem()

    def extract_facts_from_text(text: str):
        return []

    mem.create_smart_memory_system = create_smart_memory_system
    mem.extract_facts_from_text = extract_facts_from_text
    sys.modules["memory"] = mem


def _make_manager(use_cf: bool):
    # Toggle feature flag
    os.environ["USE_CONTEXT_FIELD"] = "true" if use_cf else "false"

    # Install fake memory before import to avoid heavy deps
    _install_fake_memory_module()
    from processors.smart_context_manager import SmartContextManager

    class MockContext:
        def __init__(self):
            self._messages = []
        def set_messages(self, messages):
            self._messages = messages

    ctx = MockContext()
    mgr = SmartContextManager(context=ctx, facts_db_path="/tmp/fake.db", max_tokens=1024)

    # Provide a long summary to exercise system trimming
    mgr.summary_text = " ".join(["summary"] * 500)  # long

    return mgr


def _count_tokens(messages):
    from processors.token_counter import get_token_counter
    counter = get_token_counter()
    total = 0
    for m in messages:
        total += counter.count_tokens(m.get("content", ""))
    return total


def test_integration_compare_before_after(monkeypatch):
    import logging
    log = logging.getLogger("ctxfield.integration")
    # Build manager without ContextField
    mgr_off = _make_manager(use_cf=False)
    import asyncio
    msgs_off = asyncio.run(mgr_off._build_fixed_context("What's my dog's name?"))
    tokens_off = _count_tokens(msgs_off)
    off_metrics = getattr(mgr_off, "_last_context_build_metrics", {})
    log.info("OFF: total=%s system=%s summary=%s facts=%s snippets=%s recent=%s current=%s", \
             off_metrics.get("block_tokens", {}).get("total"), \
             off_metrics.get("block_tokens", {}).get("system"), \
             off_metrics.get("block_tokens", {}).get("summary"), \
             off_metrics.get("block_tokens", {}).get("facts"), \
             off_metrics.get("block_tokens", {}).get("snippets"), \
             off_metrics.get("block_tokens", {}).get("recent"), \
             off_metrics.get("block_tokens", {}).get("current"))
    log.info("OFF: facts_lines=%s snippets_lines=%s dth_lines=%s", \
             off_metrics.get("counts", {}).get("facts_lines"), \
             off_metrics.get("counts", {}).get("snippets_lines"), \
             off_metrics.get("counts", {}).get("dth_lines"))
    log.info("OFF: system_head=%r", msgs_off[0]["content"][:240])

    # Build manager with ContextField and log real activations/canonical
    mgr_on = _make_manager(use_cf=True)
    msgs_on_real = asyncio.run(mgr_on._build_fixed_context("What's my dog's name?"))
    on_metrics_real = getattr(mgr_on, "_last_context_build_metrics", {})
    real_plan = on_metrics_real.get("plan", {})
    log.info("ON(real): activ=%s canonical.q=%r facts=%s recents=%s dth=%s", \
             real_plan.get("activations"), \
             (real_plan.get("canonical_inputs", {}) or {}).get("query"), \
             len((real_plan.get("canonical_inputs", {}) or {}).get("facts", [])), \
             len((real_plan.get("canonical_inputs", {}) or {}).get("recents", [])), \
             len((real_plan.get("canonical_inputs", {}) or {}).get("dth", [])))
    log.info("ON(real): system_head=%r", msgs_on_real[0]["content"][:240])

    # Fake plan: tiny budgets to force trimming of summary/facts/snippets
    class FakePlan:
        def __init__(self):
            self.tier_activations = {"core": 0.5, "dynamic": 0.3, "retrieval": 0.1, "tools": 0.1}
            self.token_budgets = {
                "system": 300,      # likely smaller than system_prompt; summary will be trimmed to 0
                "facts": 60,
                "snippets": 30,
                "dth": 0,
                "recent": 200,
                "current": 80,
                "buffer": 50,
            }
            self.token_budget_total = 1024

    class FakeMetrics:
        def __init__(self):
            self.reproducibility_hash = "deadbeef"
            self.total_tokens_budgeted = 1024
            self.tier_activations = {}
            self.token_budgets = {}

    class FakeCF:
        def create_plan(self, *args, **kwargs):
            return FakePlan(), FakeMetrics()

    # Replace the planner instance with a fake one to force strict budgets
    mgr_on._context_field = FakeCF()

    msgs_on = asyncio.run(mgr_on._build_fixed_context("What's my dog's name?"))
    tokens_on = _count_tokens(msgs_on)
    on_metrics = getattr(mgr_on, "_last_context_build_metrics", {})
    log.info("ON: plan.hash=%s budgets=%s", on_metrics.get("plan", {}).get("hash"), on_metrics.get("plan", {}).get("budgets"))
    log.info("ON: total=%s system=%s summary=%s facts=%s snippets=%s recent=%s current=%s", \
             on_metrics.get("block_tokens", {}).get("total"), \
             on_metrics.get("block_tokens", {}).get("system"), \
             on_metrics.get("block_tokens", {}).get("summary"), \
             on_metrics.get("block_tokens", {}).get("facts"), \
             on_metrics.get("block_tokens", {}).get("snippets"), \
             on_metrics.get("block_tokens", {}).get("recent"), \
             on_metrics.get("block_tokens", {}).get("current"))
    log.info("ON: facts_lines=%s snippets_lines=%s dth_lines=%s", \
             on_metrics.get("counts", {}).get("facts_lines"), \
             on_metrics.get("counts", {}).get("snippets_lines"), \
             on_metrics.get("counts", {}).get("dth_lines"))
    log.info("ON: system_head=%r", msgs_on[0]["content"][:240])

    # The ContextField-enabled build should not exceed the disabled build tokens
    assert tokens_on <= tokens_off

    # The system message should have summary trimmed when enabled (budget too small)
    sys_off = msgs_off[0]["content"]
    sys_on = msgs_on[0]["content"]
    assert "[Conversation Summary]" in sys_off
    # Trimmed summary should be significantly smaller under ContextField budgets
    assert sys_on.count("summary") < sys_off.count("summary")

    # Verify per-block token and line count reductions using SCM metrics
    off_metrics = getattr(mgr_off, "_last_context_build_metrics", {})
    on_metrics = getattr(mgr_on, "_last_context_build_metrics", {})

    # Facts/snippets tokens and line counts should be <= when planner is on
    off_facts_toks = off_metrics.get("block_tokens", {}).get("facts", 0)
    on_facts_toks = on_metrics.get("block_tokens", {}).get("facts", 0)
    off_snip_toks = off_metrics.get("block_tokens", {}).get("snippets", 0)
    on_snip_toks = on_metrics.get("block_tokens", {}).get("snippets", 0)
    assert on_facts_toks <= off_facts_toks
    assert on_snip_toks <= off_snip_toks

    off_facts_lines = off_metrics.get("counts", {}).get("facts_lines", 0)
    on_facts_lines = on_metrics.get("counts", {}).get("facts_lines", 0)
    off_snip_lines = off_metrics.get("counts", {}).get("snippets_lines", 0)
    on_snip_lines = on_metrics.get("counts", {}).get("snippets_lines", 0)
    assert on_facts_lines <= off_facts_lines
    assert on_snip_lines <= off_snip_lines

    # Budgets applied: facts/snippets under their respective plan budgets
    plan = on_metrics.get("plan", {})
    budgets = plan.get("budgets", {}) or {}
    if budgets:
        assert on_facts_toks <= budgets.get("facts", 10**9)
        assert on_snip_toks <= budgets.get("snippets", 10**9)
        # Basic sanity: hash present from planner
        assert plan.get("hash")

    # Simple relevance heuristic: ensure key terms remain present
    def relevance_score(text: str, terms: list[str]) -> int:
        t = text.lower()
        return sum(1 for term in terms if term in t)

    sys_off = msgs_off[0]["content"]
    sys_on = msgs_on[0]["content"]
    terms = ["dog", "name", "potola"]
    score_off = relevance_score(sys_off, terms)
    score_on = relevance_score(sys_on, terms)
    log.info("Relevance: OFF=%s ON=%s terms=%s", score_off, score_on, terms)
    assert score_on >= score_off
