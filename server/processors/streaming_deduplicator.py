"""
Streaming Deduplicator - Prevents cumulative token duplication in streaming responses
"""

import re
from typing import AsyncGenerator
from pipecat.frames.frames import Frame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger


class StreamingDeduplicator(FrameProcessor):
    """
    Prevents cumulative token duplication in streaming LLM responses.
    Detects patterns like "Hello.Hello.What'sWhat's up?" and fixes them.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_response = False
        self._accumulated_text = ""
        self._previous_segments = []
        logger.info("🔄 Streaming deduplicator initialized")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and deduplicate streaming text"""
        
        # Always call parent first
        await super().process_frame(frame, direction)
        
        # Handle StartFrame specially
        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._accumulated_text = ""
            self._previous_segments = []
            await self.push_frame(frame, direction)
            return
        
        # Handle EndFrame
        if isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            await self.push_frame(frame, direction)
            return
        
        # Process text frames for deduplication
        if isinstance(frame, TextFrame) and self._in_response:
            deduplicated_text = self._detect_cumulative_duplication(frame.text)
            if deduplicated_text != frame.text:
                logger.info(f"🔄 Fixed streaming duplication: '{frame.text}' -> '{deduplicated_text}'")
                deduplicated_frame = TextFrame(deduplicated_text)
                await self.push_frame(deduplicated_frame, direction)
            else:
                await self.push_frame(frame, direction)
            return
        
        # Forward all other frames
        await self.push_frame(frame, direction)
    
    def _detect_cumulative_duplication(self, text: str) -> str:
        """
        Detect and fix cumulative duplication patterns.
        
        Example: "Hello.Hello.What'sWhat's up?" -> "Hello. What's up?"
        """
        if not text:
            return text
        
        # Look for patterns where text is duplicated progressively
        # This regex catches cases where words repeat with potential punctuation
        pattern = r'(\b\w+[\'.,!?]*)\1+'
        
        def replace_duplication(match):
            original = match.group()
            deduplicated = match.group(1)
            
            # Count how many times it was duplicated
            duplication_count = len(original) // len(deduplicated)
            
            if duplication_count > 1:
                logger.debug(f"🔄 Fixed duplication: '{original}' -> '{deduplicated}' (was {duplication_count}x)")
            return deduplicated
        
        # Apply the fix
        cleaned_text = re.sub(pattern, replace_duplication, text, flags=re.IGNORECASE)
        
        # Additional pattern for more complex cumulative duplication
        # Handle cases like "I'mI'm Slowcat,I'm Slowcat, your"
        complex_pattern = r'(\b\w+\'?\w*)\1+([^\w]|\s|$)'
        
        def replace_complex_duplication(match):
            word = match.group(1)
            suffix = match.group(2)
            return word + suffix
        
        cleaned_text = re.sub(complex_pattern, replace_complex_duplication, cleaned_text)
        
        return cleaned_text