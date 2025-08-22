import asyncio
import os
import types
import pytest

from processors.smart_context_manager import SmartContextManager


class DummyContext:
    def __init__(self):
        self.messages = None
    def set_messages(self, messages):
        self.messages = messages


class FakeEntry:
    def __init__(self, role, content):
        self.role = role
        self.content = content
        self.ts = 0.0


class FakeTape:
    def __init__(self, entries):
        self._entries = entries
        self.saved = None
    def get_entries_since(self, since_ts: float):
        # Return as list of objects (SCM expects attributes)
        return self._entries
    def add_summary(self, session_id: str, summary: str, keywords_json='[]', turns: int = 0, duration_s: int = 0):
        self.saved = (session_id, summary, keywords_json, turns, duration_s)


@pytest.mark.asyncio
async def test_scm_uses_abstract_summary(monkeypatch):
    # Force abstract summary path
    monkeypatch.setenv('SC_USE_ABSTRACT_SUMMARY', 'true')
    monkeypatch.setenv('SC_SUMMARY_EVERY_N', '1')

    # Stub summarizer
    from utils import abstract_summarizer as summ
    monkeypatch.setattr(summ, 'summarize_dialogue', lambda msgs, **kw: 'Stub summary.')

    ctx = DummyContext()
    scm = SmartContextManager(context=ctx, facts_db_path='data/test_facts.db', max_tokens=512)

    # Inject fake tape store
    entries = [
        FakeEntry('user', 'I want your honest review on the system prompt.'),
        FakeEntry('assistant', 'I can help. Which parts concern you most?'),
        FakeEntry('user', "Be specific about what's not working and what you'd change."),
        FakeEntry('assistant', 'Okay, I will focus on clarity and redundancy.'),
    ]
    tape = FakeTape(entries)
    scm.tape_store = tape

    # Set session state
    scm.session.turn_count = 2
    scm.session.session_start = 0.0

    await scm._maybe_update_running_summary()

    assert scm.summary_text == 'Stub summary.'
    assert tape.saved is not None
    assert 'Stub summary.' in tape.saved[1]

