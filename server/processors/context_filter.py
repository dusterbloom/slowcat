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
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_response = False
        self._accumulated_response = ""
        logger.info("🚫 Context filter initialized - will block streaming frames from context")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Filter frames going to context aggregator"""
        
        # Always call parent first
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._accumulated_response = ""
            logger.debug("🚫 Context filter: LLM response started - blocking streaming chunks")
            # Allow start frame to pass through
            await self.push_frame(frame, direction)
            return
        
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            
            # Send ONLY our clean accumulated response as TextFrame to context
            # NEVER send the original end frame - it might trigger another corrupted response
            if self._accumulated_response.strip():
                logger.info(f"✅ Context filter: Sending ONLY our clean response: '{self._accumulated_response.strip()[:50]}...'")
                complete_frame = TextFrame(self._accumulated_response.strip())
                await self.push_frame(complete_frame, direction)
            else:
                logger.warning("🚫 Context filter: No accumulated response to send!")
            
            # CRITICAL: Do NOT send the original LLMFullResponseEndFrame 
            # The LLM's "complete" response might also be corrupted
            logger.debug("🚫 Context filter: Blocking original LLMFullResponseEndFrame")
            self._accumulated_response = ""
            return
        
        elif isinstance(frame, TextFrame) and self._in_response:
            # CRITICAL FIX: LLM sends REPLACEMENT frames, not delta frames!
            # Each TextFrame contains the complete response so far, not just new content
            if frame.text:
                # Simply replace our accumulated response with the latest complete version
                self._accumulated_response = frame.text
                logger.debug(f"🚫 Context filter: Updated complete response: '{frame.text[:50]}...'")
            
            # CRITICAL: Block ALL TextFrames during LLM response (streaming OR complete)
            # We'll send our own clean version at the end
            logger.debug(f"🚫 Context filter: Blocking TextFrame during response: '{frame.text[:30] if frame.text else 'None'}...'")
            return
        
        # For all other frames, pass through normally
        await self.push_frame(frame, direction)
    
    def _extract_delta_content(self, frame_text: str) -> str:
        """
        Extract only the NEW content from a potentially cumulative frame.
        
        Handles cases where LLM sends cumulative text like:
        - Frame 1: "I'm"
        - Frame 2: "I'mI'm Slowcat" <- Extract only " Slowcat"
        - Frame 3: "I'mI'm Slowcat,I'm Slowcat, your" <- Extract only ", your"
        """
        if not self._accumulated_response:
            # First frame - take it as is but check for internal duplication
            return self._deduplicate_internal(frame_text)
        
        # Try to find where the new content starts
        current_clean = self._accumulated_response
        
        # Simple case: frame_text ends with new content after our current response
        if frame_text.startswith(current_clean):
            # Extract the suffix
            new_content = frame_text[len(current_clean):]
            return self._deduplicate_internal(new_content)
        
        # Complex case: frame_text contains duplicated version of our current response
        # Try to find the pattern and extract the new part
        if current_clean in frame_text:
            # Find the last occurrence of our clean response
            last_pos = frame_text.rfind(current_clean)
            if last_pos >= 0:
                # Extract content after the last occurrence
                new_content = frame_text[last_pos + len(current_clean):]
                return self._deduplicate_internal(new_content)
        
        # Fallback: Try to find the longest common prefix and extract the rest
        common_len = 0
        min_len = min(len(current_clean), len(frame_text))
        
        for i in range(min_len):
            if current_clean[i] == frame_text[i]:
                common_len += 1
            else:
                break
        
        if common_len > 0:
            new_content = frame_text[common_len:]
            return self._deduplicate_internal(new_content)
        
        # Last resort: the frame might be completely new content
        logger.warning(f"🚫 Context filter: Could not extract delta, taking full frame: '{frame_text[:50]}...'")
        return self._deduplicate_internal(frame_text)
    
    def _deduplicate_internal(self, text: str) -> str:
        """Remove internal duplications within a single text chunk"""
        if not text:
            return text
        
        # Handle simple word-level duplications like "yourI'm" -> "your I'm"
        # This is a simple heuristic - can be improved
        import re
        
        # Pattern to catch repeated words/phrases
        # This will catch patterns like "yourI'm" -> extract the boundary
        
        # For now, return as-is since the main delta extraction should handle most cases
        return text