from typing import List, Dict, Optional
import asyncio

from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    TextFrame,
    StartFrame,
    EndFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger

from .local_memory import LocalMemoryProcessor


class MemoryContextInjector(FrameProcessor):
    """
    Enhanced memory context injector that intelligently injects conversation
    history into LLM context with error handling and performance optimizations.
    
    Features:
    - Token-aware context management
    - Graceful error handling
    - Watchdog timer support
    - Configurable memory injection strategies
    """
    
    def __init__(
        self,
        memory_processor: LocalMemoryProcessor,
        system_prompt: str = "Here is relevant conversation history from previous sessions:",
        inject_as_system: bool = True,
        max_memory_tokens: int = 1000,
        enable_watchdog: bool = True
    ):
        super().__init__()
        self.memory = memory_processor
        self.system_prompt = system_prompt
        self.inject_as_system = inject_as_system
        self.max_memory_tokens = max_memory_tokens
        self.enable_watchdog = enable_watchdog
        
        logger.info(f"🧠 Memory context injector initialized (max_tokens: {max_memory_tokens})")
        
    def _estimate_token_count(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for English)."""
        return len(text) // 4
    
    def _truncate_memory_for_tokens(self, memory_messages: List[Dict], max_tokens: int) -> List[Dict]:
        """Intelligently truncate memory messages to fit within token budget."""
        if not memory_messages:
            return []
            
        # Estimate tokens for each message
        token_count = 0
        truncated_messages = []
        
        # Add messages from most recent to oldest until we hit token limit
        for msg in reversed(memory_messages):
            msg_tokens = self._estimate_token_count(msg['content'])
            if token_count + msg_tokens <= max_tokens:
                truncated_messages.insert(0, msg)  # Insert at beginning to maintain order
                token_count += msg_tokens
            else:
                break
                
        if len(truncated_messages) < len(memory_messages):
            logger.debug(f"Truncated memory from {len(memory_messages)} to {len(truncated_messages)} messages ({token_count} tokens)")
            
        return truncated_messages

    async def _inject_memory_with_watchdog(self, frame: LLMMessagesFrame) -> Optional[LLMMessagesFrame]:
        """Inject memory with optional watchdog timer for performance monitoring."""
        async def _memory_injection_task():
            # Get conversation history (FIXED: now properly awaited)
            memory_messages = await self.memory.get_context_messages()
            
            if not memory_messages:
                return None
                
            # Apply token-aware truncation
            truncated_messages = self._truncate_memory_for_tokens(memory_messages, self.max_memory_tokens)
            
            if not truncated_messages:
                return None
                
            logger.debug(f"Injecting {len(truncated_messages)} memory messages into context")
            
            # Create enhanced messages list
            enhanced_messages = []
            
            # Add system message about memory if configured
            if self.inject_as_system:
                memory_summary = "\n".join([
                    f"- {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}"
                    for msg in truncated_messages[:3]  # Show preview of first 3 messages
                ])
                enhanced_messages.append({
                    "role": "system",
                    "content": f"{self.system_prompt}\n\nRecent conversation context:\n{memory_summary}"
                })
            
            # Add original messages
            enhanced_messages.extend(frame.messages)
            
            # If not injecting as system, add memory as conversation history
            if not self.inject_as_system:
                for msg in truncated_messages:
                    enhanced_messages.insert(-len(frame.messages), msg)
            
            return LLMMessagesFrame(messages=enhanced_messages)
        
        # Execute with or without watchdog timer
        if self.enable_watchdog:
            return await self.create_task(_memory_injection_task())
        else:
            return await _memory_injection_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Intercept LLM messages going upstream and inject memory
        if isinstance(frame, LLMMessagesFrame) and direction == FrameDirection.UPSTREAM:
            try:
                enhanced_frame = await self._inject_memory_with_watchdog(frame)
                
                if enhanced_frame:
                    await self.push_frame(enhanced_frame, direction)
                    return
                else:
                    logger.debug("No memory to inject, using original frame")
                    
            except Exception as e:
                logger.error(f"Memory injection failed: {e}")
                logger.debug("Continuing with original frame without memory context")
                # Graceful fallback - continue with original frame
                
        await self.push_frame(frame, direction)
    
    async def cleanup(self):
        """Clean up memory context injector resources"""
        logger.debug("MemoryContextInjectorProcessor cleanup completed")