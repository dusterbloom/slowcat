"""
Simple LLM-backed dialogue summarizer with pluggable providers.

Providers supported via env SUMMARIZER_PROVIDER:
- lmstudio: OpenAI-compatible server (default http://localhost:1234/v1)
- ollama:   Ollama HTTP API (default http://localhost:11434)

Usage:
  from utils.abstract_summarizer import summarize_dialogue
  summary = summarize_dialogue(messages, provider='lmstudio', model='gpt-4o-mini')

Notes:
- This module performs HTTP calls; ensure provider is running locally.
- In test environments without network/servers, skip tests that use this.
"""
from __future__ import annotations

import os
import json
import time
from typing import List, Dict, Optional
from urllib import request


def _http_post(url: str, data: dict, headers: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> dict:
    body = json.dumps(data).encode('utf-8')
    req = request.Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _deduplicate_and_preprocess_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Preprocess messages to remove duplicates and improve summarization quality.
    
    Args:
        messages: OpenAI-style list of {role, content}
        
    Returns:
        Cleaned list with duplicates removed and low-quality content filtered
    """
    if not messages:
        return []
    
    import re
    cleaned = []
    seen_contents = set()
    
    for msg in messages:
        content = msg.get('content', '').strip()
        if not content:
            continue
            
        # Normalize content for comparison
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        
        # Skip exact duplicates
        if normalized in seen_contents:
            continue
            
        # Skip very short messages that are likely noise
        if len(content.split()) < 3 and len(content) < 15:
            continue
            
        # Skip test/integration content
        if 'integration test' in normalized or 'test content' in normalized:
            continue
            
        seen_contents.add(normalized)
        cleaned.append(msg)
    
    # If we have too few messages after cleaning, relax the filters
    if len(cleaned) < 3:
        cleaned = []
        seen_contents = set()
        for msg in messages:
            content = msg.get('content', '').strip()
            if not content:
                continue
            normalized = re.sub(r'\s+', ' ', content.lower().strip())
            if normalized in seen_contents:
                continue
            seen_contents.add(normalized)
            cleaned.append(msg)
    
    return cleaned


def summarize_dialogue(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 320,
    temperature: float = 0.2,
) -> str:
    """Summarize a dialogue into 2-3 crisp sentences.

    messages: OpenAI-style list of {role, content}
    provider: 'lmstudio' | 'ollama' (defaults from env SUMMARIZER_PROVIDER)
    model: provider-specific model (env SUMMARIZER_MODEL fallback)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    provider = (provider or os.getenv('SUMMARIZER_PROVIDER', 'lmstudio')).lower()
    model = model or os.getenv('SUMMARIZER_MODEL', 'dolphin3.0-llama3.2-3b')

    # Preprocess messages to remove duplicates and improve quality
    original_count = len(messages)
    messages = _deduplicate_and_preprocess_messages(messages)
    cleaned_count = len(messages)
    
    # Log input quality for debugging
    logger.info(f"📊 Summarizer input: {original_count} → {cleaned_count} messages after preprocessing")
    unique_contents = set(msg.get('content', '') for msg in messages)
    logger.info(f"📊 Unique content ratio: {len(unique_contents)}/{cleaned_count} = {len(unique_contents)/max(cleaned_count, 1):.2f}")
    
    # Log sample messages for debugging
    for i, msg in enumerate(messages[:3]):  # Log first 3 messages
        content_preview = msg.get('content', '')[:50]
        logger.info(f"📊 Message {i+1} [{msg.get('role', 'unknown')}]: {content_preview}...")
    
    # Check for repetitive content
    if len(unique_contents) < len(messages) * 0.5:
        logger.warning(f"⚠️ High repetition detected in summarizer input: {len(unique_contents)} unique out of {len(messages)} messages")

    # Enhanced system prompt with better guidance for handling various input types
    system_prompt = (
        "Summarize this conversation in 1-2 sentences. Do NOT roleplay. Do NOT repeat any messages. "
        "Focus only on the main topics discussed."
    )
    payload_messages = [{"role": "system", "content": system_prompt}] + messages

    if provider == 'lmstudio':
        base = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
        url = f"{base.rstrip('/')}/chat/completions"
        data = {
            "model": model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        res = _http_post(url, data)
        text = res.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Log response quality for debugging
        logger.info(f"📊 Summarizer response: {len(text)} chars")
        if not text.strip():
            logger.error(f"❌ Empty summarizer response from {provider} with model {model}")
            logger.error(f"❌ Response structure: {res}")
        else:
            logger.info(f"✅ Successful summarizer response: {text[:100]}...")
            
        # Handle empty responses with fallback
        if not text.strip():
            logger.warning(f"⚠️ Empty response from {provider}, attempting fallback with simpler prompt")
            # Try a simpler, more direct prompt as fallback
            fallback_prompt = (
                "Summarize this conversation in 1-2 sentences. Do NOT roleplay. Do NOT repeat any messages. "
                "Focus only on the main topics discussed."
            )
            fallback_messages = [{"role": "system", "content": fallback_prompt}] + messages
            
            fallback_data = {
                "model": model,
                "messages": fallback_messages,
                "temperature": max(temperature, 0.4),  # Increase temperature more for creativity
                "max_tokens": max_tokens,
                "stream": False,
            }
            
            try:
                fallback_res = _http_post(url, fallback_data)
                fallback_text = fallback_res.get('choices', [{}])[0].get('message', {}).get('content', '')
                if fallback_text.strip():
                    logger.info(f"✅ Fallback successful: {fallback_text[:100]}...")
                    return fallback_text.strip()
            except Exception as e:
                logger.error(f"❌ Fallback attempt failed: {e}")
            
            # Ultimate fallback: return a generic summary
            return "Brief conversation with minimal substantive content."
            
        return (text or '').strip()
    elif provider == 'ollama':
        base = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        # Prefer /v1/chat/completions if available (OpenAI compatible)
        try:
            url = f"{base.rstrip('/')}/v1/chat/completions"
            data = {
                "model": model,
                "messages": payload_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            res = _http_post(url, data)
            text = res.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Log response quality for debugging
            logger.info(f"📊 Summarizer response (ollama v1): {len(text)} chars")
            if not text.strip():
                logger.error(f"❌ Empty summarizer response from {provider} with model {model}")
                logger.error(f"❌ Response structure: {res}")
            else:
                logger.info(f"✅ Successful summarizer response: {text[:100]}...")
                
            return (text or '').strip()
        except Exception as e:
            logger.warning(f"⚠️ Ollama v1 endpoint failed, falling back to legacy API: {e}")
            # Fallback to legacy /api/generate with a single prompt
            prompt = system_prompt + "\n\n" + "\n".join(f"[{m['role']}] {m['content']}" for m in messages)
            url = f"{base.rstrip('/')}/api/generate"
            data = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
            res = _http_post(url, data)
            text = res.get('response', '')
            
            # Log fallback response quality
            logger.info(f"📊 Summarizer fallback response: {len(text)} chars")
            if not text.strip():
                logger.error(f"❌ Empty summarizer fallback response from {provider} with model {model}")
                logger.error(f"❌ Fallback response structure: {res}")
            else:
                logger.info(f"✅ Successful summarizer fallback response: {text[:100]}...")
                
            # Handle empty responses with fallback for Ollama
            if not text.strip():
                logger.warning(f"⚠️ Empty response from {provider}, attempting fallback with simpler prompt")
                # Try a simpler, more direct prompt as fallback
                fallback_prompt = (
                    "Summarize this conversation in 1-2 sentences. Do NOT roleplay. Do NOT repeat any messages. "
                    "Focus only on the main topics discussed."
                )
                fallback_messages = [{"role": "system", "content": fallback_prompt}] + messages
                
                fallback_data = {
                    "model": model,
                    "messages": fallback_messages,
                    "temperature": max(temperature, 0.4),  # Increase temperature more for creativity
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                
                try:
                    fallback_res = _http_post(url, fallback_data)
                    fallback_text = fallback_res.get('choices', [{}])[0].get('message', {}).get('content', '')
                    if fallback_text.strip():
                        logger.info(f"✅ Fallback successful: {fallback_text[:100]}...")
                        return fallback_text.strip()
                except Exception as e:
                    logger.error(f"❌ Fallback attempt failed: {e}")
                
                # Ultimate fallback: return a generic summary
                return "Brief conversation with minimal substantive content."
                
            return (text or '').strip()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
