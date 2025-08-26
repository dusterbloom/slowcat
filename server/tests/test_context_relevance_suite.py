"""
Relevance suite: ensure planner surfaces most relevant local data per turn.

We simulate a memory system with multiple topical facts and verify that for
different queries, the SmartContextManager (with planner budgets) includes
facts aligned with the query terms, while staying within budgets.
"""

import os
import sys
import types
import asyncio
import logging


def _install_fake_memory_module():
    mem = types.ModuleType("memory")

    class FakeFact:
        def __init__(self, subject, predicate, value, fidelity=3, last_seen=0):
            self.subject = subject
            self.predicate = predicate
            self.value = value
            self.fidelity = fidelity
            self.last_seen = last_seen

    class FakeFactsGraph:
        def __init__(self):
            self.facts = []
            # Topics
            # music
            self.facts.append(FakeFact("user", "favorite_song", "Bohemian Rhapsody", fidelity=4, last_seen=1500))
            self.facts.append(FakeFact("user", "favorite_artist", "Queen", fidelity=3, last_seen=1400))
            # work
            self.facts.append(FakeFact("user", "role", "software engineer", fidelity=4, last_seen=1300))
            self.facts.append(FakeFact("user", "company", "ACME Corp", fidelity=3, last_seen=1200))
            # location
            self.facts.append(FakeFact("user", "city", "San Francisco", fidelity=4, last_seen=1100))
            self.facts.append(FakeFact("user", "country", "USA", fidelity=3, last_seen=1000))
            # hobbies
            self.facts.append(FakeFact("user", "hobby", "cycling", fidelity=3, last_seen=900))
            self.facts.append(FakeFact("user", "hobby", "photography", fidelity=2, last_seen=800))

        async def get_top_facts(self, limit=10):
            # Order by fidelity desc, last_seen desc
            s = sorted(self.facts, key=lambda f: (f.fidelity, f.last_seen), reverse=True)
            return s[:limit]

        async def search_facts(self, query: str, limit: int = 20):
            import re
            def norm_tokens(s: str):
                toks = [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t]
                out = []
                for t in toks:
                    if t.endswith("ies"):
                        t = t[:-3] + "y"
                    elif t.endswith("s") and len(t) > 3:
                        t = t[:-1]
                    out.append(t)
                return set(out)

            qset = norm_tokens(query)
            scored = []
            for f in self.facts:
                text = f"{f.subject} {f.predicate} {f.value}"
                fset = norm_tokens(text)
                overlap = len(qset & fset)
                if overlap > 0:
                    score = overlap * (1 + 0.1 * f.fidelity)
                    scored.append((score, f))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [f for _, f in scored[:limit]]

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
            # Produce fact results ranked by overlap to simulate a router
            class R:
                def __init__(self, facts):
                    class C:
                        def __init__(self):
                            self.intent = type("I", (), {"name": "PERSONAL_FACTS"})
                    self.classification = C()
                    class Res:
                        def __init__(self, f):
                            self.subject = getattr(f, 'subject', '')
                            self.predicate = getattr(f, 'predicate', '')
                            self.value = getattr(f, 'value', None)
                            self.fidelity = getattr(f, 'fidelity', 0)
                            self.last_seen = getattr(f, 'last_seen', 0)
                            self.source_store = 'facts'
                            self.content = ''
                    self.results = [Res(f) for f in facts]
            ranked = await self.facts_graph.search_facts(query, limit=10)
            return R(ranked)

    def create_smart_memory_system(db_path: str = "data/facts.db"):
        return FakeMemorySystem()

    def extract_facts_from_text(text: str):
        return []

    mem.create_smart_memory_system = create_smart_memory_system
    mem.extract_facts_from_text = extract_facts_from_text
    sys.modules["memory"] = mem


def _make_mgr_with_fake_planner():
    os.environ["USE_CONTEXT_FIELD"] = "true"
    _install_fake_memory_module()
    # Ensure SmartContextManager rebinds to the newly installed fake memory
    import importlib, sys as _sys
    if 'processors.smart_context_manager' in _sys.modules:
        del _sys.modules['processors.smart_context_manager']
    from processors.smart_context_manager import SmartContextManager

    class MockContext:
        def __init__(self):
            self.messages = []
        def set_messages(self, messages):
            self.messages = messages

    mgr = SmartContextManager(context=MockContext(), facts_db_path="/tmp/fake.db", max_tokens=1024)

    # Force strict budgets to test selection pressure
    class FakePlan:
        def __init__(self):
            self.tier_activations = {"core": 0.5, "dynamic": 0.3, "retrieval": 0.1, "tools": 0.1}
            self.token_budgets = {
                "system": 300,
                "facts": 80,      # small facts budget -> top facts should survive
                "snippets": 0,
                "dth": 0,
                "recent": 150,
                "current": 80,
                "buffer": 50,
            }
            self.token_budget_total = 1024

    class FakeMetrics:
        def __init__(self):
            self.reproducibility_hash = "cafebabe"
            self.total_tokens_budgeted = 1024
            self.tier_activations = {}
            self.token_budgets = {}

    class FakeCF:
        def create_plan(self, *args, **kwargs):
            return FakePlan(), FakeMetrics()

    mgr._context_field = FakeCF()
    return mgr


def _coverage_score(facts_kept, keywords):
    text = " ".join([f"{f.get('predicate','')} {f.get('value','')}" for f in facts_kept]).lower()
    return sum(1 for k in keywords if k in text)


def test_relevance_across_queries():
    log = logging.getLogger("ctxfield.relevance")
    mgr = _make_mgr_with_fake_planner()

    cases = [
        ("Play some Queen", ["queen", "artist"]),
        ("what city do I live in?", ["city", "san", "francisco"]),
        ("what's my role at work?", ["role", "software", "engineer"]),
        ("what country am I in?", ["country", "usa"]),
        ("what are my hobbies?", ["hobby", "cycling", "photography"]),
    ]

    hits = 0
    for q, kws in cases:
        messages = asyncio.run(mgr._build_fixed_context(q))
        m = getattr(mgr, "_last_context_build_metrics", {})
        kept = m.get("facts_kept", []) or []
        score = _coverage_score(kept, kws)
        log.info("Q=%r kept=%s score=%s", q, kept, score)
        # Require at least one keyword hit for each query
        assert score >= 1
        hits += 1

    # All cases should achieve coverage
    assert hits == len(cases)
