"""
Message Deduplicator - Prevents consecutive messages from the same role
"""
import logging
from typing import List, Dict, Any
from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

logger = logging.getLogger(__name__)


class MessageDeduplicator(FrameProcessor):
    """
    Ensures messages alternate between user and assistant roles.
    Merges consecutive user messages to prevent jinja template errors.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("📝 Message deduplicator initialized")
    
    def _merge_consecutive_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consecutive messages from the same role"""
        if not messages:
            return messages
        
        merged = []
        current_message = None
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Skip empty messages
            if not content.strip():
                continue
            
            # Handle system messages - always keep them separate
            if role == "system":
                if current_message:
                    merged.append(current_message)
                    current_message = None
                merged.append(msg)
                continue
            
            # Handle tool messages - always keep them separate
            if role == "tool" or msg.get("tool_calls") or msg.get("tool_call_id"):
                if current_message:
                    merged.append(current_message)
                    current_message = None
                merged.append(msg)
                continue
            
            # For user/assistant messages
            if current_message and current_message["role"] == role:
                # Merge with previous message of same role
                current_message["content"] = current_message["content"].rstrip() + " " + content.lstrip()
                logger.debug(f"Merged {role} messages")
            else:
                # Different role or first message
                if current_message:
                    merged.append(current_message)
                current_message = {"role": role, "content": content}
        
        # Don't forget the last message
        if current_message:
            merged.append(current_message)
        
        return merged
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and fix message alternation"""
        await super().process_frame(frame, direction)
        
        # Only process LLMMessagesFrame going upstream (to the LLM)
        if isinstance(frame, LLMMessagesFrame) and direction == FrameDirection.UPSTREAM:
            original_count = len(frame.messages)
            
            # Merge consecutive messages
            frame.messages = self._merge_consecutive_messages(frame.messages)
            
            # Log if we made changes
            if len(frame.messages) != original_count:
                logger.info(f"🔄 Deduplicated messages: {original_count} → {len(frame.messages)}")
                
                # Debug: Show the role sequence
                roles = [msg.get("role", "unknown") for msg in frame.messages]
                logger.debug(f"Message roles after deduplication: {roles}")
                
                # Verify alternation (excluding system/tool messages)
                conversation_roles = [msg["role"] for msg in frame.messages 
                                    if msg.get("role") in ["user", "assistant"]]
                
                for i in range(1, len(conversation_roles)):
                    if conversation_roles[i] == conversation_roles[i-1]:
                        logger.warning(f"⚠️ Still have consecutive {conversation_roles[i]} messages!")
        
        await self.push_frame(frame, direction)