"""
Tests for LLM-backed abstract summarization.

These tests are skipped unless a local provider is configured and reachable.
Set env:
  SUMMARIZER_PROVIDER=lmstudio|ollama
  SUMMARIZER_MODEL=<model id>
Optionally:
  LMSTUDIO_BASE_URL=http://localhost:1234/v1
  OLLAMA_BASE_URL=http://localhost:11434
"""

import os
import pytest

from utils.abstract_summarizer import summarize_dialogue


sample_dialogue = [
    {"role": "user", "content": "I want your honest review on the system prompt."},
    {"role": "assistant", "content": "I'm happy to help. Which parts concern you most?"},
    {"role": "user", "content": "Be specific about what's not working and what you'd change."},
    {"role": "assistant", "content": "Understood. I'll focus on clarity, redundancy, and tone."},
]


def _provider_available():
    return os.getenv('SUMMARIZER_PROVIDER') is not None


@pytest.mark.skipif(not _provider_available(), reason="No SUMMARIZER_PROVIDER configured")
def test_abstract_summary_is_concise_and_not_verbatim():
    summary = summarize_dialogue(sample_dialogue, max_tokens=120)
    assert isinstance(summary, str) and len(summary) > 0
    # Check it's not just repeating last user line verbatim
    assert sample_dialogue[-1]['content'][:20] not in summary
    # Expect a compact result (not too long and not trivial)
    assert 30 <= len(summary) <= 600


@pytest.mark.skipif(not _provider_available(), reason="No SUMMARIZER_PROVIDER configured")
def test_abstract_summary_mentions_review_signal():
    summary = summarize_dialogue(sample_dialogue, max_tokens=120)
    s = summary.lower()
    # Expect it references review/feedback intent (allow broader synonyms)
    keywords = [
        "review", "feedback", "critique", "improve", "refine", "adjust",
        "evaluation", "assess", "assessment", "revise", "suggestions"
    ]
    mentions_intent = any(k in s for k in keywords)
    mentions_prompt_change = ("prompt" in s) and any(x in s for x in ["change", "improv", "refin", "adjust"])
    assert mentions_intent or mentions_prompt_change, f"Summary missing review signal: {summary}"
