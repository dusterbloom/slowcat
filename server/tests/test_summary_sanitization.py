import asyncio
from types import SimpleNamespace

import pytest

from processors.smart_context_manager import SmartContextManager


class DummyTape:
    def __init__(self, entries):
        self._entries = entries[:]  # list of dicts with role, content, ts
        self.summaries = []

    async def get_recent(self, limit=20):
        return self._entries[-limit:]

    async def add_summary(self, session_id, summary, keywords_json='[]', turns=0, duration_s=0):
        self.summaries.append({
            'session_id': session_id,
            'summary': summary,
            'turns': turns,
            'duration_s': duration_s,
        })


@pytest.mark.asyncio
async def test_sanitized_summary_filters_boilerplate():
    # Build fake entries including assistant boilerplate we want to drop
    entries = [
        {'role': 'assistant', 'content': 'Ah, I see. Let me clarify: I\'m the one processing and responding to your question.', 'ts': 1.0},
        {'role': 'user', 'content': 'Tell me about the history of Chinese writing.', 'ts': 2.0},
        {'role': 'assistant', 'content': 'A concise overview: Oracle bones to modern script.', 'ts': 3.0},
        {'role': 'user', 'content': 'Thanks!', 'ts': 4.0},
    ]

    # Minimal SmartContextManager with dummy context and tape
    class DummyContext:
        def set_messages(self, _):
            pass

    scm = SmartContextManager(DummyContext())
    scm.session.session_start = 0
    scm.tape_store = DummyTape(entries)
    scm._use_abstract_summary = False
    scm._summary_last_turns = 10

    # Use internal helper to prepare lines and verify boilerplate is removed
    lines = scm._prepare_summary_lines(entries)
    text = "\n".join(lines)
    assert 'Let me clarify' not in text
    assert "I'm the one processing" not in text

    # Finalize and ensure summary stored and sanitized
    out = await scm.finalize_summary()
    assert out is not None and len(out) > 0
    assert 'Let me clarify' not in out
    assert "I'm the one processing" not in out
    assert len(scm.tape_store.summaries) == 1

