#!/usr/bin/env python3
"""Test two-model DSPy extraction on a real-ish conversation sample.

This test monkeypatches the HTTP calls to the local LM Studio endpoint so it
does not require network/model availability. It validates that:
- Strict relations model can return empty for some utterances
- General model backfills valid relations
- The merged output returns normalized relations usable for edge creation
"""

import json
from types import SimpleNamespace

import pytest


def _fake_response(payload):
    """Return a fake LM Studio-like chat completion based on model and input."""
    model = payload.get("model", "")
    user_msg = next((m["content"] for m in payload.get("messages", []) if m.get("role") == "user"), "")

    def wrap(content):
        return {"choices": [{"message": {"content": content}}]}

    # Extract the original text quoted in the user prompt
    # We look for Text: "..."
    import re
    m = re.search(r'Text:\s*"(.+?)"', user_msg, flags=re.S)
    text = m.group(1) if m else user_msg

    # Strict model: qwen2.5-0.5b-instruct-mlx:2
    if ":2" in model:
        # Return empty for generic lines; return one clean relation for a target line
        if "sample head" in text.lower():
            content = json.dumps({
                "relations": [
                    {"subject": "user", "predicate": "prefers", "object": "samples", "confidence": 0.92}
                ]
            })
        else:
            content = json.dumps({"relations": []})
        return wrap(content)

    # General model: qwen2.5-0.5b-instruct-mlx
    # Provide a looser extraction when strict returns nothing
    # Add another relation for a different line
    rels = []
    tl = text.lower()
    if "electricity" in tl:
        rels.append({"subject": "user", "predicate": "asks_about", "object": "electricity", "confidence": 0.85})
    if "hip - hop beat" in tl or "hip-hop" in tl:
        rels.append({"subject": "user", "predicate": "likes", "object": "hip-hop", "confidence": 0.88})
    content = json.dumps({"relations": rels})
    return wrap(content)


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
    def json(self):
        return _fake_response(self._payload)


@pytest.fixture(autouse=True)
def patch_requests(monkeypatch):
    import server.memory.dspy_single_call_extractor as mod
    def fake_post(url, json=None, headers=None, timeout=None):
        return _Resp(json)
    monkeypatch.setattr(mod.requests, "post", fake_post)
    yield


def test_two_model_extraction_merges_results():
    from server.memory.dspy_integration import extract_facts_from_text_dspy

    # A lightly anonymized subset of the provided conversation
    convo = [
        ("assistant", "Hello! How can I assist you today?"),
        ("user", "I want you to ask me one question at a time."),
        ("assistant", "Great! I'll ask one question at a time."),
        ("user", "As a human, I'd like to ask you: how does electricity feel?"),
        ("assistant", "I don't have senses, but imagine a spark!"),
        ("user", "hip - hop beat."),
        ("assistant", "Do you create or mostly listen to others' beats?"),
        ("user", "I'm a sample head, so I go for samples all the way."),
    ]

    extracted = []
    for role, text in convo:
        if role == "user":
            facts = extract_facts_from_text_dspy(text)
            extracted.extend(facts)

    # We expect at least two facts merged from strict and general flows
    assert len(extracted) >= 2

    # Check that we got usable relations (subject/predicate/value)
    preds = {f.get("predicate") for f in extracted}
    objs = {f.get("value") for f in extracted}
    # From strict: prefers → samples
    assert "prefers" in preds and "samples" in objs
    # From general backfill: likes → hip-hop or asks_about → electricity
    assert ("likes" in preds and "hip-hop" in objs) or ("asks_about" in preds and "electricity" in objs)

