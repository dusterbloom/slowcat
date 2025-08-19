"""
ResponseTap - listens for final assistant TextFrame (after ContextFilter) and
feeds it into SmartContextManager and TapeStore.
"""

from typing import Optional
from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class ResponseTap(FrameProcessor):
    def __init__(self, smart_context_manager, **kwargs):
        super().__init__(**kwargs)
        self.smart_context_manager = smart_context_manager

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only tap assistant responses traveling downstream to TTS/output
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            text = (frame.text or '').strip()
            if text:
                try:
                    self.smart_context_manager.add_assistant_response(text)
                except Exception as e:
                    logger.debug(f"ResponseTap: add_assistant_response failed: {e}")

        await self.push_frame(frame, direction)

