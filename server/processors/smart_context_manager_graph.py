"""
Graph Smart Context Manager (final-only storage + async concept extraction)

Goals:
- Store only FINAL user and assistant messages into SurrealDB `message` table
- Run concept/fact extraction asynchronously (non-blocking) to grow the knowledge graph
- Defer context-building to the existing SmartContextManager (optional) while this
  component focuses on reliable persistence + background enrichment.

Usage:
- Insert this processor in the pipeline so it sees STT transcription frames and
  LLM message frames. It will:
  * On final TranscriptionFrame: write a single user message
  * On final LLMMessagesFrame: write a single assistant message
  * For both: schedule an async extraction task to store facts via GraphSurrealMemory

Environment:
- SC_SCHEMA_MODE=graph (factory returns GraphSurrealMemory)
- ENABLE_ASYNC_EXTRACTION=true|false (default: true)
- EXTRACT_ASSISTANT=true|false (default: false) — whether to extract from assistant text too
"""

from __future__ import annotations

import asyncio
from typing import Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from memory import create_smart_memory_system


def _is_final_transcript(frame: TranscriptionFrame) -> bool:
    for attr in ("final", "is_final", "is_complete"):
        try:
            if bool(getattr(frame, attr)):
                return True
        except Exception:
            continue
    return False


class SmartContextManagerGraph(FrameProcessor):
    """Persist final user/assistant messages + enqueue async extraction to Graph memory."""

    def __init__(self, context, enable_async_extraction: bool = True, extract_assistant: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.context = context
        self.memory = create_smart_memory_system()
        self.enable_async_extraction = enable_async_extraction
        self.extract_assistant = extract_assistant
        # simple user id resolution; upstream pipeline can set USER_ID env for consistency
        self._user_id: Optional[str] = None

    def _speaker_key(self) -> str:
        # Prefer explicit user id if set via upstream context; else fallback to 'default_user'
        try:
            sid = getattr(self, "_user_id") or getattr(self.context, "user_id", None) or "default_user"
            return sid
        except Exception:
            return "default_user"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        try:
            # Final user messages only: treat plain TranscriptionFrame as final; skip InterimTranscriptionFrame
            if isinstance(frame, TranscriptionFrame) and not isinstance(frame, InterimTranscriptionFrame):
                txt = (getattr(frame, "text", "") or "")
                if txt:
                    user_id = getattr(frame, "speaker_id", None) or getattr(frame, "user_id", None)
                    if user_id:
                        self._user_id = user_id
                    logger.info(f"[GraphSCM] FINAL USER captured → {txt[:80]}…")
                    await self._store_message(role="user", text=txt)
                    await self._maybe_extract(txt, role="user")

            # LLMMessagesFrame carries the prompt package BEFORE the LLM runs;
            # do not treat it as an assistant message to avoid storing pre-output context.
            # LLMMessagesUpdateFrame: ignore for storage by design.
        finally:
            # Always forward frames to keep the pipeline flowing
            await self.push_frame(frame, direction)

    async def _store_message(self, role: str, text: str):
        try:
            speaker = self._speaker_key()
            # Use graph memory add_entry so we only persist a single final message
            await self.memory.add_entry(role=role, content=text, speaker_id=speaker)
            logger.info(f"[GraphSCM] PERSISTED {role.upper()} for {speaker} → {text[:80]}…")
        except Exception as e:
            logger.warning(f"[GraphSCM] store_message failed: {e}")

    async def _maybe_extract(self, text: str, role: str):
        if not self.enable_async_extraction:
            return
        async def _task():
            try:
                # Reuse GraphSurrealMemory.store_facts with speaker context
                await self.memory.store_facts(text, user_name=self._speaker_key())
                # Also store fragments for richer retrieval
                try:
                    # Attach session/turn context to fragments
                    sid = await self.memory.get_active_session_id_for_user(self._speaker_key())  # type: ignore[attr-defined]
                    turn = 0
                    if sid:
                        cnt = await self.memory.get_message_count_for_session(sid)  # type: ignore[attr-defined]
                        turn = cnt
                    await self.memory.store_fragments_from_text(self._speaker_key(), text, session_id=sid, turn=turn, role='user')  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"[GraphSCM] async extraction skipped: {e}")
        try:
            asyncio.create_task(_task())
        except Exception:
            pass

    # Sink for ResponseTap fan-out: persist final assistant response
    async def add_assistant_response(self, text: str):
        try:
            if not (text or '').strip():
                return
            await self._store_message(role='assistant', text=text)
            if self.extract_assistant:
                await self._maybe_extract(text, role='assistant')
        except Exception as e:
            logger.debug(f"[GraphSCM] add_assistant_response skipped: {e}")


def create_smart_context_manager_graph(context, **kwargs) -> SmartContextManagerGraph:
    enable_async = (str(kwargs.get("enable_async_extraction", "true")).lower() == "true")
    extract_assistant = (str(kwargs.get("extract_assistant", "false")).lower() == "true")
    return SmartContextManagerGraph(context=context,
                                    enable_async_extraction=enable_async,
                                    extract_assistant=extract_assistant)
