"""
Reflection Daemon (SurrealDB-only)

Runs outside the live pipeline to generate siloed private thoughts after idle.

Behavior:
- Scans sessions for speakers idle longer than REFLECTION_IDLE_SECS.
- For each idle speaker, collects recent tape (speaker_id), extracts simple
  observations (keywords) and a follow-up seed, and writes private thoughts.
- Never touches user-visible context. Writes to SurrealDB 'thought' and optional
  emergent_event when ENABLE_EMERGENT_TRACKING=true.

Usage:
    cd server && python -m scripts.reflection_daemon

Env:
    USE_SURREALDB=true
    ASSISTANT_ID=slowcat
    REFLECTION_IDLE_SECS=120
    REFLECTION_COOLDOWN_SECS=300
    ENABLE_EMERGENT_TRACKING=false
    EMERGENT_LOOKBACK_TURNS=30
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import List, Dict, Any
from loguru import logger

from memory import create_smart_memory_system


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


async def _top_keywords(texts: List[str], k: int = 6) -> List[str]:
    import re
    stop = set(
        """
        a an the and or but if then else for to of in on with at by is are was were be been being i you we they he she it this that these those my your our their
        """.split()
    )
    counts: Dict[str, int] = {}
    for t in texts:
        for w in re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", (t or '').lower()):
            if w in stop:
                continue
            counts[w] = counts.get(w, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]]


async def _diff_tokens(tape_texts: List[str], thought_text: str) -> List[str]:
    import re
    def toks(texts: List[str]) -> set[str]:
        stop = set(
            """
            a an the and or but if then else for to of in on with at by is are was were be been being i you we they he she it this that these those my your our their
            """.split()
        )
        out: set[str] = set()
        for t in texts:
            for w in re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", (t or '').lower()):
                if w in stop:
                    continue
                out.add(w)
        return out
    tape = toks(tape_texts)
    th = toks([thought_text])
    return [w for w in th if w not in tape]


async def reflect_once(memory, assistant_id: str, idle_s: int, cooldown_s: int, emergent: bool, lookback_turns: int):
    # Get all sessions and pick idle ones
    sessions = []
    try:
        # SurrealMemorySystemAdapter -> SurrealMemory
        sessions = await memory.list_sessions()
    except Exception as e:
        logger.debug(f"list_sessions failed: {e}")
        return
    now = time.time()
    for row in sessions or []:
        spk = row.get('speaker_id') or ''
        last = float(row.get('last_interaction') or 0)
        if not spk or not last:
            continue
        if (now - last) < max(1, idle_s):
            continue
        # Throttle per assistant by checking last thought time
        recent_thoughts = []
        try:
            recent_thoughts = await memory.get_recent_thoughts(assistant_id, limit=1)
        except Exception:
            pass
        last_th = float(recent_thoughts[0]['ts']) if recent_thoughts else 0.0
        if last_th and (now - last_th) < max(1, cooldown_s):
            continue

        # Collect recent tape for this speaker
        items = []
        try:
            items = await memory.get_recent_for_speaker(spk, limit=max(8, lookback_turns))
        except Exception:
            items = []
        texts: List[str] = []
        for e in items or []:
            c = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
            if c:
                texts.append(c)
        if not texts:
            continue

        # Decide heuristic vs. LLM reflections
        llm_on = os.getenv('ENABLE_LLM_REFLECTIONS', 'false').lower() == 'true'
        if llm_on:
            # Build minimal chat messages from recent tape (limit size)
            msgs: List[Dict[str, str]] = []
            for e in reversed(items[-16:]):  # chronological
                role = (e.get('role') if isinstance(e, dict) else getattr(e, 'role', 'user'))
                content = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', ''))
                if content:
                    msgs.append({"role": role, "content": content})
            try:
                from utils.private_reflector import generate_private_thoughts
                thoughts = generate_private_thoughts(
                    messages=msgs,
                    provider=os.getenv('REFLECTION_LLM_PROVIDER', None),
                    model=os.getenv('REFLECTION_LLM_MODEL', None),
                    max_tokens=int(os.getenv('REFLECTION_LLM_MAX_TOKENS', '160')),
                    temperature=float(os.getenv('REFLECTION_LLM_TEMPERATURE', '0.5')),
                )
            except Exception as e:
                logger.debug(f"LLM reflections failed, falling back to heuristics: {e}")
                thoughts = []
            # Store any valid thoughts
            for t in thoughts:
                ttype = (t.get('type') or 'observation').strip()
                content = (t.get('content') or '').strip()
                if not content:
                    continue
                try:
                    await memory.add_thought(assistant_id, ttype, content)
                except Exception:
                    pass
                if emergent and ttype == 'observation':
                    try:
                        unseen = await _diff_tokens(texts, content)
                        if unseen:
                            await memory.add_emergent_event(assistant_id, 'private_topic_off_tape', content[:200], meta={'unseen_tokens': unseen[:10]}, user_id=spk)
                    except Exception:
                        pass
            # Fallback to heuristics if model returned nothing
            if not thoughts:
                logger.info("🧘 LLM reflections returned no thoughts — falling back to heuristics")
                # Heuristic reflections fallback
                kws = await _top_keywords(texts, k=6)
                if kws:
                    obs = f"observed_topics: {', '.join(kws)}"
                    try:
                        await memory.add_thought(assistant_id, 'observation', obs)
                        logger.info(f"🧘 reflection_daemon(fallback): wrote observation for {spk}: {obs}")
                    except Exception as e:
                        logger.debug(f"add_thought(observation) failed: {e}")
                    if emergent:
                        try:
                            unseen = await _diff_tokens(texts, obs)
                            if unseen:
                                await memory.add_emergent_event(assistant_id, 'private_topic_off_tape', obs[:200], meta={'unseen_tokens': unseen[:10]}, user_id=spk)
                        except Exception:
                            pass
                # Follow-up seed fallback
                last_user = ''
                for e in items:
                    if (e.get('role') if isinstance(e, dict) else getattr(e, 'role', '')) == 'user':
                        last_user = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
                        break
                if last_user:
                    hint = (last_user[:120] + '…') if len(last_user) > 120 else last_user
                    seed = f"followup_seed: consider picking up from: '{hint}'"
                    try:
                        await memory.add_thought(assistant_id, 'followup_seed', seed)
                        logger.info(f"🧘 reflection_daemon(fallback): wrote followup seed for {spk}")
                    except Exception as e:
                        logger.debug(f"add_thought(followup_seed) failed: {e}")
        else:
            # Heuristic reflections (previous behavior)
            kws = await _top_keywords(texts, k=6)
            if kws:
                obs = f"observed_topics: {', '.join(kws)}"
                try:
                    await memory.add_thought(assistant_id, 'observation', obs)
                    logger.info(f"🧘 reflection_daemon: wrote observation for {spk}: {obs}")
                except Exception as e:
                    logger.debug(f"add_thought(observation) failed: {e}")
                if emergent:
                    try:
                        unseen = await _diff_tokens(texts, obs)
                        if unseen:
                            await memory.add_emergent_event(assistant_id, 'private_topic_off_tape', obs[:200], meta={'unseen_tokens': unseen[:10]}, user_id=spk)
                    except Exception:
                        pass
            # Follow-up seed (simple)
            last_user = ''
            for e in items:
                if (e.get('role') if isinstance(e, dict) else getattr(e, 'role', '')) == 'user':
                    last_user = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
                    break
            if last_user:
                hint = (last_user[:120] + '…') if len(last_user) > 120 else last_user
                seed = f"followup_seed: consider picking up from: '{hint}'"
                try:
                    await memory.add_thought(assistant_id, 'followup_seed', seed)
                    logger.info(f"🧘 reflection_daemon: wrote followup seed for {spk}")
                except Exception as e:
                    logger.debug(f"add_thought(followup_seed) failed: {e}")


async def main():
    if os.getenv('USE_SURREALDB', 'false').lower() != 'true':
        logger.error("Reflection daemon requires USE_SURREALDB=true")
        return
    assistant_id = os.getenv('ASSISTANT_ID', 'slowcat').strip() or 'slowcat'
    idle_s = _get_env_int('REFLECTION_IDLE_SECS', 120)
    cooldown_s = _get_env_int('REFLECTION_COOLDOWN_SECS', 300)
    emergent = os.getenv('ENABLE_EMERGENT_TRACKING', 'false').lower() == 'true'
    lookback = _get_env_int('EMERGENT_LOOKBACK_TURNS', 30)

    # Create memory (adapter wraps SurrealMemory)
    memory = create_smart_memory_system()
    # Ensure we call SurrealDB-specific methods (adapter forwards)
    try:
        logger.info("🧘 Reflection daemon started")
        while True:
            try:
                await reflect_once(memory, assistant_id, idle_s, cooldown_s, emergent, lookback)
            except Exception as e:
                logger.debug(f"reflect_once error: {e}")
            await asyncio.sleep(5.0)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await memory.close()
        except Exception:
            pass


if __name__ == '__main__':
    asyncio.run(main())
