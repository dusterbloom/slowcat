from typing import List, Dict, Optional

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
    Injects conversation memory into LLM context.
    Works between the context aggregator and LLM.
    """
    
    def __init__(
        self,
        memory_processor: LocalMemoryProcessor,
        system_prompt: str = "Here is relevant conversation history from previous sessions:",
        inject_as_system: bool = True
    ):
        super().__init__()
        self.memory = memory_processor
        self.system_prompt = system_prompt
        self.inject_as_system = inject_as_system
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Intercept LLM messages going upstream and inject memory
        if isinstance(frame, LLMMessagesFrame) and direction == FrameDirection.UPSTREAM:
            # Get conversation history
            memory_messages = self.memory.get_context_messages()
            
            if memory_messages:
                logger.debug(f"Injecting {len(memory_messages)} memory messages into context")
                
                # Create enhanced messages list
                enhanced_messages = []
                
                # Add system message about memory if configured
                if self.inject_as_system and memory_messages:
                    enhanced_messages.append({
                        "role": "system",
                        "content": f"{self.system_prompt}\n\nPrevious conversations:\n" + 
                                 "\n".join([f"- {msg['role']}: {msg['content']}" for msg in memory_messages[:5]])
                    })
                
                # Add original messages
                enhanced_messages.extend(frame.messages)
                
                # If not injecting as system, add memory as user messages
                if not self.inject_as_system:
                    for msg in memory_messages:
                        enhanced_messages.insert(0, msg)
                
                # Create new frame with enhanced context
                enhanced_frame = LLMMessagesFrame(messages=enhanced_messages)
                await self.push_frame(enhanced_frame, direction)
                return
                
        await self.push_frame(frame, direction)