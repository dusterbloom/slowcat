"""
ResponseTap

Listens to LLM response stream and commits only the final assistant message
back into SmartContextManager (and TapeStore via SCM) — avoiding streaming
chunks from polluting recent_exchanges.
"""

from typing import Optional
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track LLM response lifecycle
        if isinstance(frame, LLMFullResponseStartFrame) and direction == FrameDirection.DOWNSTREAM:
            self._in_response = True
            self._buffer = ""
            # Do not commit yet; wait for end
        elif isinstance(frame, (TextFrame, TTSTextFrame)) and direction == FrameDirection.DOWNSTREAM:
            text = (getattr(frame, 'text', '') or '').strip()
            if not text:
                pass
            elif self._in_response:
                # Accumulate robustly: prefer overwrite if cumulative, else append
                if len(text) >= len(self._buffer) and text.startswith(self._buffer):
                    self._buffer = text
                else:
                    # Some adapters stream deltas — append with space if needed
                    if self._buffer and not self._buffer.endswith(' ') and not text.startswith(' '):
                        self._buffer += ' '
                    self._buffer += text
            else:
                # Not within a declared response; treat this as a final one-off
                try:
                    if text:
                        self.smart_context_manager.add_assistant_response(text)
                except Exception as e:
                    logger.debug(f"ResponseTap: add_assistant_response (immediate) failed: {e}")
        elif isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._in_response:
                final_text = (self._buffer or '').strip()
                if final_text:
                    try:
                        self.smart_context_manager.add_assistant_response(final_text)
                    except Exception as e:
                        logger.debug(f"ResponseTap: add_assistant_response (final) failed: {e}")
                # Reset state
                self._in_response = False
                self._buffer = ""

        await self.push_frame(frame, direction)
