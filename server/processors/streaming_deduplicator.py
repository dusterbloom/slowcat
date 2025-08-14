"""
Streaming Deduplicator - Prevents cumulative token duplication in streaming responses
"""

import re
from typing import AsyncGenerator
from pipecat.frames.frames import Frame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger


class StreamingDeduplicator(FrameProcessor):
    """
    Prevents cumulative token duplication in streaming LLM responses.
    Detects patterns like "Hello.Hello.What'sWhat's up?" and fixes them.
    """
    
    def __init__(self):
        super().__init__()
        self._in_response = False
        self._accumulated_text = ""
        self._previous_segments = []
        logger.info("🔄 Streaming deduplicator initialized")
    
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
            return original
        
        # Apply the fix
        cleaned_text = re.sub(pattern, replace_duplication, text, flags=re.IGNORECASE)
        
        # Also handle space-separated duplications like "What's What's up?"
        words = cleaned_text.split()
        deduplicated_words = []
        
        for i, word in enumerate(words):
            # Skip if this word is identical to the previous word
            if i > 0 and words[i-1].lower().rstrip('.,!?') == word.lower().rstrip('.,!?'):
                logger.debug(f"🔄 Skipped duplicate word: '{word}'")
                continue
            deduplicated_words.append(word)
        
        return ' '.join(deduplicated_words)
    
    def _is_progressive_duplication(self, new_text: str) -> bool:
        """
        Check if the new text shows signs of progressive duplication.
        
        Progressive duplication means each new chunk contains all previous chunks.
        """
        if not self._accumulated_text or not new_text:
            return False
            
        # Check if new text starts with accumulated text (progressive pattern)
        if new_text.startswith(self._accumulated_text):
            logger.debug(f"🚨 Detected progressive duplication: new text starts with accumulated text")
            return True
            
        return False
    
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process frames and fix streaming duplication"""
        
        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._accumulated_text = ""
            self._previous_segments = []
            logger.debug("🔄 Started tracking LLM response for deduplication")
            yield frame
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            if self._accumulated_text:
                logger.debug(f"🔄 Final accumulated text: '{self._accumulated_text}'")
            self._accumulated_text = ""
            self._previous_segments = []
            yield frame
            
        elif isinstance(frame, TextFrame) and self._in_response:
            original_text = frame.text
            
            # Check for progressive duplication
            if self._is_progressive_duplication(original_text):
                # Extract only the new part
                new_part = original_text[len(self._accumulated_text):]
                logger.info(f"🔄 Fixed progressive duplication: kept only new part: '{new_part}'")
                cleaned_text = self._detect_cumulative_duplication(new_part)
            else:
                # Normal deduplication
                cleaned_text = self._detect_cumulative_duplication(original_text)
            
            # Update accumulated text
            self._accumulated_text += cleaned_text
            
            # Only yield if we have actual content
            if cleaned_text.strip():
                if cleaned_text != original_text:
                    logger.info(f"🔄 Deduplicated streaming text: '{original_text}' -> '{cleaned_text}'")
                yield TextFrame(cleaned_text)
            else:
                logger.debug("🔄 Skipped empty text frame after deduplication")
                
        else:
            # Pass through all other frames unchanged
            yield frame