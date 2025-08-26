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
    def __init__(self, smart_context_manager, **kwargs):
        super().__init__(**kwargs)
        self.smart_context_manager = smart_context_manager
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
        except Exception:
            self._debounce_ms = 400
            self._min_stable_ms = 300

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
                        await self.smart_context_manager.add_assistant_response(text)
                        logger.debug(f"🧲 ResponseTap fallback-committed assistant response ({len(text)} chars)")
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
            # Only use TextFrame for memory accumulation; ignore TTSTextFrame to avoid
            # TTS-side tokenization artifacts (e.g., letter-split names) polluting memory/summary.
            if isinstance(frame, TextFrame):
                text = (getattr(frame, 'text', '') or '').strip()
                if not text:
                    pass
                else:
                    # Treat downstream TextFrames as part of an assistant response stream.
                    # Some providers may not emit LLMFullResponseStartFrame; begin buffering on first chunk.
                    if not self._in_response:
                        self._in_response = True
                        self._buffer = ""
                        if not self._announced:
                            self._announced = True
                            logger.info("🧲 ResponseTap engaged: inferred response start from TextFrame (no start frame)")
                    # Accumulate robustly: prefer overwrite if cumulative, else append
                    # Accumulate robustly: prefer overwrite if cumulative, else append
                    if len(text) >= len(self._buffer) and text.startswith(self._buffer):
                        self._buffer = text
                    else:
                        if self._buffer and not self._buffer.endswith(' ') and not text.startswith(' '):
                            self._buffer += ' '
                        self._buffer += text
                    self._last_text_ts = time.time()
                    self._last_len = len(self._buffer)
                    # Schedule fallback commit in case end frame never arrives
                    await self._schedule_fallback_commit()
            # For TTSTextFrame, do nothing here (only pass through below)
        elif isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._in_response:
                final_text = (self._buffer or '').strip()
                if final_text:
                    try:
                        await self.smart_context_manager.add_assistant_response(final_text)
                        logger.debug(f"🧲 ResponseTap committed assistant response ({len(final_text)} chars)")
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
