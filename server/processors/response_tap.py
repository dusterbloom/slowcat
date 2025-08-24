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
        logger.info("🧲 ResponseTap initialized (will capture final assistant responses)")
        # Fallback commit if some providers never emit LLMFullResponseEndFrame
        self._debounce_task: Optional[asyncio.Task] = None
        self._last_text_ts: float = 0.0
        try:
            import os
            self._debounce_ms = int(os.getenv('RESPONSETAP_COMMIT_DEBOUNCE_MS', '400'))
        except Exception:
            self._debounce_ms = 400

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
                    try:
                        await self.smart_context_manager.add_assistant_response(text)
                        logger.debug(f"🧲 ResponseTap fallback-committed assistant response ({len(text)} chars)")
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
