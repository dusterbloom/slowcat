"""
Context Filter - Blocks streaming TextFrames from reaching context aggregator
Only allows complete message frames to reach context for clean memory
"""

from pipecat.frames.frames import Frame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class ContextFilter(FrameProcessor):
    """
    CRITICAL FIX: Blocks streaming TextFrames from polluting context aggregator
    Only allows complete response messages through to context/memory
    """
    
    def __init__(self):
        super().__init__()
        self._in_response = False
        self._accumulated_response = ""
        logger.info("🚫 Context filter initialized - will block streaming frames from context")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Filter frames going to context aggregator"""
        
        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._accumulated_response = ""
            logger.debug("🚫 Context filter: LLM response started - blocking streaming chunks")
            # Allow start frame to pass through
            await self.push_frame(frame, direction)
            return
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            
            # Send ONLY the complete response as single TextFrame to context
            if self._accumulated_response.strip():
                logger.info(f"✅ Context filter: Sending clean complete response: '{self._accumulated_response.strip()}'")
                complete_frame = TextFrame(self._accumulated_response.strip())
                await self.push_frame(complete_frame, direction)
            
            # Send the end frame too
            await self.push_frame(frame, direction)
            self._accumulated_response = ""
            return
            
        elif isinstance(frame, TextFrame) and self._in_response:
            # This is a streaming chunk - accumulate but DON'T send to context
            self._accumulated_response += frame.text
            logger.debug(f"🚫 BLOCKED streaming chunk from context: '{frame.text}'")
            # Block streaming text from context aggregator - don't send to context
            return
        
        # Allow all other frames through to context aggregator
        await self.push_frame(frame, direction)