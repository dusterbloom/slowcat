"""
LLM-backed private thoughts generator.

Providers supported via env REFLECTION_LLM_PROVIDER:
- lmstudio: OpenAI-compatible server (default http://localhost:1234/v1)
- ollama:   Ollama OpenAI-compatible API (default http://localhost:11434)

Outputs a small set (1-3) of compact private thought lines in a constrained format.
This is used outside the user-visible context (siloed).
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional
from urllib import request


def _http_post(url: str, data: dict, headers: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> dict:
    body = json.dumps(data).encode('utf-8')
    req = request.Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def generate_private_thoughts(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.5,
) -> List[Dict[str, str]]:
    """
    Generate 1-3 compact private thoughts from recent chat context.

    messages: OpenAI-style list of {role, content}
    provider: 'lmstudio' | 'ollama' (defaults via REFLECTION_LLM_PROVIDER)
    model: provider-specific model (env REFLECTION_LLM_MODEL fallback)
    
    Returns list of dicts: { 'type': str, 'content': str }
    """
    provider = (provider or os.getenv('REFLECTION_LLM_PROVIDER', os.getenv('SUMMARIZER_PROVIDER', 'lmstudio'))).lower()
    # Choose model: explicit override > REFLECTION_LLM_MODEL > LLM_MODEL > sensible local default
    if not model:
        model = os.getenv('REFLECTION_LLM_MODEL') or os.getenv('LLM_MODEL') or 'qwen/qwen3-1.7b'

    system = (
        "You are generating PRIVATE, SILOED thoughts for an assistant.\n"
        "- These are NEVER shown to the user.\n"
        "- Keep them short.\n"
        "- Output 1-3 lines. Each line MUST be in the form: \"<type>: <content>\".\n"
        "- Allowed types: observation, hypothesis, followup_seed.\n"
        "- Do not include JSON or formatting, only plain text lines.\n"
    )
    user_inst = (
        "From the following dialogue, produce 1-3 private thought lines.\n"
        "Focus on salient observations, small hypotheses, or a concrete follow-up question seed.\n"
    )
    payload_messages = (
        [{"role": "system", "content": system}] +
        [{"role": "user", "content": user_inst}] +
        messages[-16:]  # last few messages keep it cheap
    )

    if provider == 'lmstudio':
        base = os.getenv('OPENAI_BASE_URL', os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1'))
        url = f"{base.rstrip('/')}/chat/completions"
        data = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            res = _http_post(url, data)
            text = (res.get('choices', [{}])[0].get('message', {}) or {}).get('content', '')
        except Exception:
            text = ''
    elif provider == 'ollama':
        base = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        url = f"{base.rstrip('/')}/chat/completions"
        data = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            res = _http_post(url, data)
            text = (res.get('choices', [{}])[0].get('message', {}) or {}).get('content', '')
        except Exception:
            text = ''
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Parse simple lines
    out: List[Dict[str, str]] = []
    if not text:
        return out
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        # Expected format: "type: content"
        if ':' in line:
            t, c = line.split(':', 1)
            t = t.strip().lower()
            c = c.strip()
            if t in ("observation", "hypothesis", "followup_seed") and c:
                out.append({"type": t, "content": c})
    return out

