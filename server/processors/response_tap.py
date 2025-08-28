"""
ResponseTap

Listens to LLM response stream and commits only the final assistant message
back into SmartContextManager (and TapeStore via SCM) — avoiding streaming
chunks from polluting recent_exchanges.
"""

from typing import Optional
import asyncio
import time
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TTSTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class ResponseTap(FrameProcessor):
    def __init__(self, smart_context_manager, assistant_sinks=None, **kwargs):
        super().__init__(**kwargs)
        self.smart_context_manager = smart_context_manager
        # Optional extra sinks that can accept add_assistant_response(text)
        self._assistant_sinks = []
        try:
            if assistant_sinks:
                # Keep unique order, include smart_context_manager first
                seen = set()
                for sink in [smart_context_manager, *assistant_sinks]:
                    if sink is None:
                        continue
                    if id(sink) in seen:
                        continue
                    seen.add(id(sink))
                    # Must have add_assistant_response method
                    if hasattr(sink, 'add_assistant_response'):
                        self._assistant_sinks.append(sink)
        except Exception:
            # Fallback to single sink
            self._assistant_sinks = [smart_context_manager]
        self._in_response: bool = False
        self._buffer: str = ""
        self._announced: bool = False
        self._last_text_ts: float = 0.0
        self._last_len: int = 0
        self._committed_once: bool = False
        logger.info("🧲 ResponseTap initialized (will capture final assistant responses)")
        # Fallback commit if some providers never emit LLMFullResponseEndFrame
        self._debounce_task: Optional[asyncio.Task] = None
        try:
            import os
            self._debounce_ms = int(os.getenv('RESPONSETAP_COMMIT_DEBOUNCE_MS', '400'))
            # Extra grace to prefer sentence-complete commits when no end frame exists
            self._min_stable_ms = int(os.getenv('RESPONSETAP_MIN_STABLE_MS', '300'))
            self._enable_fallback = os.getenv('RESPONSETAP_ENABLE_FALLBACK', 'false').lower() == 'true'
        except Exception:
            self._debounce_ms = 400
            self._min_stable_ms = 300
            self._enable_fallback = False

    async def _schedule_fallback_commit(self):
        if self._debounce_task:
            self._debounce_task.cancel()
            self._debounce_task = None
        async def _debounced():
            try:
                await asyncio.sleep(self._debounce_ms / 1000.0)
                # If still in response and buffer has content, commit
                if self._in_response and (self._buffer or '').strip():
                    text = self._buffer.strip()
                    # Prefer committing at sentence boundaries or after stability
                    import time, re
                    now = time.time()
                    stable_enough = (now - self._last_text_ts) * 1000.0 >= self._min_stable_ms
                    ends_sentence = bool(re.search(r"[\.!?][\]\)\"']?$", text))
                    # If we already committed once for this response, avoid repeated fallback commits
                    if self._committed_once and not ends_sentence and not stable_enough:
                        return
                    # Commit only on sentence end or when content stopped changing for a bit
                    if not (ends_sentence or stable_enough):
                        # Re-arm a shorter fallback to check again soon
                        await asyncio.sleep(max(0.05, self._min_stable_ms/1000.0))
                        if not self._in_response:
                            return
                        text = self._buffer.strip()
                    try:
                        # Fan-out to all sinks (SCM + optional GraphWriter)
                        for sink in (self._assistant_sinks or [self.smart_context_manager]):
                            try:
                                await sink.add_assistant_response(text)
                            except Exception as e:
                                logger.debug(f"ResponseTap: sink commit failed: {e}")
                        logger.info(f"[ResponseTap] FINAL ASSISTANT (fallback) → {text[:80]}…")
                        self._committed_once = True
                    except Exception as e:
                        logger.debug(f"ResponseTap: fallback commit failed: {e}")
                    finally:
                        self._in_response = False
                        self._buffer = ""
            except asyncio.CancelledError:
                return
        self._debounce_task = asyncio.create_task(_debounced())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track LLM response lifecycle
        if isinstance(frame, LLMFullResponseStartFrame) and direction == FrameDirection.DOWNSTREAM:
            self._in_response = True
            self._buffer = ""
            self._committed_once = False
            self._last_len = 0
            if not self._announced:
                self._announced = True
                logger.info("🧲 ResponseTap engaged: detected LLM response start (downstream)")
            # Do not commit yet; wait for end
        elif direction == FrameDirection.DOWNSTREAM:
            # Prefer LLM frames; avoid capturing generic TextFrame unless fallback is enabled
            from pipecat.frames.frames import LLMTextFrame
            if isinstance(frame, LLMTextFrame):
                text = (getattr(frame, 'text', '') or '').strip()
                if text:
                    if not self._in_response:
                        self._in_response = True
                        self._buffer = ""
                        if not self._announced:
                            self._announced = True
                            logger.info("🧲 ResponseTap engaged: inferred response start from LLMTextFrame")
                    if len(text) >= len(self._buffer) and text.startswith(self._buffer):
                        self._buffer = text
                    else:
                        if self._buffer and not self._buffer.endswith(' ') and not text.startswith(' '):
                            self._buffer += ' '
                        self._buffer += text
                    self._last_text_ts = time.time()
                    self._last_len = len(self._buffer)
                    if self._enable_fallback:
                        await self._schedule_fallback_commit()
            elif isinstance(frame, TextFrame) and self._enable_fallback:
                # Fallback path for providers that don't emit LLMTextFrame
                text = (getattr(frame, 'text', '') or '').strip()
                if text:
                    if not self._in_response:
                        self._in_response = True
                        self._buffer = ""
                        if not self._announced:
                            self._announced = True
                            logger.info("🧲 ResponseTap engaged: fallback via TextFrame")
                    if len(text) >= len(self._buffer) and text.startswith(self._buffer):
                        self._buffer = text
                    else:
                        if self._buffer and not self._buffer.endswith(' ') and not text.startswith(' '):
                            self._buffer += ' '
                        self._buffer += text
                    self._last_text_ts = time.time()
                    self._last_len = len(self._buffer)
                    await self._schedule_fallback_commit()
        elif isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._in_response:
                final_text = (self._buffer or '').strip()
                if final_text:
                    try:
                        # Fan-out to all sinks (SCM + optional GraphWriter)
                        for sink in (self._assistant_sinks or [self.smart_context_manager]):
                            try:
                                await sink.add_assistant_response(final_text)
                            except Exception as e:
                                logger.debug(f"ResponseTap: sink commit failed: {e}")
                        logger.info(f"[ResponseTap] FINAL ASSISTANT → {final_text[:80]}…")
                        self._committed_once = True
                    except Exception as e:
                        logger.debug(f"ResponseTap: add_assistant_response (final) failed: {e}")
                # Reset state
                self._in_response = False
                self._buffer = ""
                # Cancel any pending debounce task
                if self._debounce_task:
                    self._debounce_task.cancel()
                    self._debounce_task = None

        await self.push_frame(frame, direction)
