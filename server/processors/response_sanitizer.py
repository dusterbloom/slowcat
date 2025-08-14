"""
Response Sanitizer - Filters out malformed LLM responses to prevent word multiplication
"""
import re
import logging
from typing import List, Dict, Any
from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

logger = logging.getLogger(__name__)


class ResponseSanitizer(FrameProcessor):
    """
    Sanitizes LLM responses to prevent word multiplication issues.
    Removes assistant messages with repetitive word patterns.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("🧼 Response sanitizer initialized")
        
        # Pattern to detect word multiplication like "I I found I found a"
        self.multiplication_pattern = re.compile(
            r'\b(\w+)\s+\1(?:\s+\1)*\b',  # Matches repeated words
            re.IGNORECASE
        )
        
        # Threshold for repeated pattern detection (% of message that's repetitive)
        self.repetition_threshold = 0.3
    
    def _is_malformed_response(self, content: str) -> bool:
        """Check if response contains word multiplication patterns"""
        if not content or len(content.strip()) < 10:
            return False
            
        # Find all repetitive patterns
        matches = list(self.multiplication_pattern.finditer(content))
        
        if not matches:
            return False
        
        # Calculate total length of repetitive content
        repetitive_length = sum(len(match.group()) for match in matches)
        repetition_ratio = repetitive_length / len(content)
        
        if repetition_ratio > self.repetition_threshold:
            logger.warning(f"🚨 Detected malformed response ({repetition_ratio:.1%} repetitive): {content[:100]}...")
            return True
            
        return False
    
    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove malformed assistant messages"""
        sanitized = []
        removed_count = 0
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Only check assistant messages
            if role == "assistant" and self._is_malformed_response(content):
                logger.info(f"🧼 Removing malformed assistant message: {content[:100]}...")
                removed_count += 1
                continue
            
            sanitized.append(msg)
        
        if removed_count > 0:
            logger.info(f"🧼 Sanitized {removed_count} malformed responses from context")
            
        return sanitized
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and sanitize malformed responses"""
        await super().process_frame(frame, direction)
        
        # Only process LLMMessagesFrame going upstream (to the LLM)
        if isinstance(frame, LLMMessagesFrame) and direction == FrameDirection.UPSTREAM:
            original_count = len(frame.messages)
            
            # Sanitize messages
            frame.messages = self._sanitize_messages(frame.messages)
            
            # Log if we made changes
            removed_count = original_count - len(frame.messages)
            if removed_count > 0:
                logger.info(f"🧼 Response sanitizer: removed {removed_count} malformed messages")
        
        await self.push_frame(frame, direction)