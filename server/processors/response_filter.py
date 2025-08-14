"""
Response Debug Logger - Traces exactly what LLM generates to identify duplication source
"""

from pipecat.frames.frames import Frame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class ResponseFilter(FrameProcessor):
    """
    DEBUG PROCESSOR: Traces LLM output step by step to identify where duplication occurs
    """
    
    def __init__(self):
        super().__init__()
        self._in_response = False
        self._accumulated_response = ""
        self._frame_count = 0
        logger.info("🔍 DEBUG: Response tracer initialized - monitoring LLM output")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Debug trace every frame from LLM"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._accumulated_response = ""
            self._frame_count = 0
            logger.info("🔍 DEBUG: LLM Response Started")
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            logger.info(f"🔍 DEBUG: LLM Response Complete - Total frames: {self._frame_count}")
            logger.info(f"🔍 DEBUG: Final accumulated text: '{self._accumulated_response}'")
            
            # Check for duplication patterns
            if "Hello.Hello." in self._accumulated_response or "What'sWhat's" in self._accumulated_response:
                logger.error(f"🚨 DEBUG: DUPLICATION DETECTED IN LLM OUTPUT: '{self._accumulated_response}'")
            else:
                logger.info(f"✅ DEBUG: LLM output looks clean: '{self._accumulated_response[:100]}...'")
            
            self._accumulated_response = ""
            
        elif isinstance(frame, TextFrame) and self._in_response:
            self._frame_count += 1
            self._accumulated_response += frame.text
            
            logger.info(f"🔍 DEBUG: Text frame #{self._frame_count}: '{frame.text}'")
            logger.info(f"🔍 DEBUG: Accumulated so far: '{self._accumulated_response}'")
            
            # Check if individual frames are duplicated
            if frame.text.count('.') > 1 or "Hello" in frame.text and frame.text.count("Hello") > 1:
                logger.warning(f"⚠️  DEBUG: Suspicious duplication in single frame: '{frame.text}'")
        
        elif isinstance(frame, LLMFullResponseEndFrame):
            # When response is complete, send accumulated text as single frame
            if self._accumulated_response.strip():
                logger.info(f"🎯 T-JUNCTION: Sending complete response: '{self._accumulated_response.strip()}'")
                complete_frame = TextFrame(self._accumulated_response.strip())
                await self.push_frame(complete_frame, direction)
            
            # Reset for next response
            self._accumulated_response = ""
            self._in_response = False
            logger.info(f"🔍 DEBUG: LLM Response Complete - Total frames: {self._frame_count}")
            
            # Don't push the EndFrame - we've replaced streaming with complete response
            return
        
        # Pass through all other frames (but NOT streaming TextFrames)
        await self.push_frame(frame, direction)