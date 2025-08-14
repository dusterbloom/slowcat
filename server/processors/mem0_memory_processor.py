"""
Mem0 Memory Processor for Slowcat Voice Agent
Integrates Mem0's advanced memory system with Pipecat pipeline
"""

import asyncio
from typing import Optional
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    LLMMessagesFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger


class Mem0MemoryProcessor(FrameProcessor):
    """
    Pipecat processor that integrates Mem0 for advanced memory management.
    
    Features:
    - Semantic memory search with LM Studio embeddings
    - Intelligent memory extraction using local LLM
    - Context injection for voice conversations 
    - Speaker-aware memory storage
    """

    def __init__(
        self,
        user_id: str = "default_user",
        lm_studio_url: str = "http://localhost:1234/v1",
        chat_model: str = "qwen2.5-14b-instruct",
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5",
        max_context_memories: int = 3,
        enabled: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.user_id = user_id
        self.max_context_memories = max_context_memories
        self.enabled = enabled
        self.memory = None
        self.last_user_input = None
        
        if self.enabled:
            try:
                # Configure Mem0 with LM Studio
                config = MemoryConfig(
                    llm=LlmConfig(
                        provider="lmstudio",
                        config={
                            "model": chat_model,
                            "lmstudio_base_url": lm_studio_url,
                            "api_key": "lm-studio",  # LM Studio specific API key
                            "temperature": 0.1,
                            "max_tokens": 500,
                            "lmstudio_response_format": {"type": "text"}
                        }
                    ),
                    embedder=EmbedderConfig(
                        provider="lmstudio", 
                        config={
                            "model": embedding_model,
                            "lmstudio_base_url": lm_studio_url,
                            "api_key": "lm-studio",  # LM Studio specific API key  
                            "embedding_dims": 768
                        }
                    ),
                    vector_store=VectorStoreConfig(
                        provider="qdrant",
                        config={
                            "embedding_model_dims": 768,
                            "collection_name": f"slowcat_{user_id}"
                        }
                    )
                )
                
                self.memory = Memory(config=config)
                logger.info(f"🧠 Mem0 memory processor initialized for user: {user_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Mem0: {e}")
                self.enabled = False
        
        if not self.enabled:
            logger.warning("⚠️ Mem0 memory processor disabled - running without memory")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle memory operations"""
        
        # DEBUG: Log relevant frames
        if isinstance(frame, (TranscriptionFrame, LLMMessagesFrame, TextFrame)):
            logger.info(f"🔍 Mem0Processor: {type(frame).__name__} {direction} - {getattr(frame, 'text', getattr(frame, 'messages', 'no text'))[:100]}")
        
        if not self.enabled:
            await super().process_frame(frame, direction)
            return

        try:
            # Handle user speech (transcription going downstream)
            if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
                if frame.text and frame.text.strip():
                    self.last_user_input = frame.text.strip()
                    logger.info(f"🎤 User said: {self.last_user_input[:100]}...")
                # Pass transcription through unchanged
                await super().process_frame(frame, direction)

            # Handle LLM messages - inject memory context 
            elif isinstance(frame, LLMMessagesFrame) and direction == FrameDirection.DOWNSTREAM:
                logger.info(f"🧠 LLMMessagesFrame detected! User input: {self.last_user_input[:50] if self.last_user_input else 'None'}...")
                if self.last_user_input:
                    # Get relevant memories and create enhanced frame
                    enhanced_frame = await self._inject_memory_context(frame, self.last_user_input)
                    logger.info(f"🚀 Sending enhanced frame with memory context")
                    # Send enhanced frame instead of original
                    await super().process_frame(enhanced_frame, direction)
                else:
                    # No user input captured, pass through unchanged
                    logger.info(f"⚠️ No user input captured, passing LLMMessagesFrame unchanged")
                    await super().process_frame(frame, direction)
            
            # Handle assistant responses (text going upstream)
            elif isinstance(frame, TextFrame) and direction == FrameDirection.UPSTREAM:
                if frame.text and self.last_user_input:
                    # Store the conversation pair asynchronously
                    asyncio.create_task(self._store_conversation(
                        self.last_user_input, 
                        frame.text.strip()
                    ))
                    self.last_user_input = None  # Reset after storing
                # Pass response through unchanged
                await super().process_frame(frame, direction)
                
            else:
                # For all other frame types, just pass through
                await super().process_frame(frame, direction)

        except Exception as e:
            logger.error(f"❌ Memory processing error: {e}")
            # Continue without memory if there's an error
            await super().process_frame(frame, direction)

    async def _inject_memory_context(self, llm_frame: LLMMessagesFrame, user_input: str) -> LLMMessagesFrame:
        """Inject relevant memories into LLM context"""
        try:
            # Search for relevant memories
            memories = await self._search_memories(user_input)
            
            if not memories:
                return llm_frame  # No memories found, return original
            
            # Create context from memories
            memory_context = self._format_memory_context(memories)
            
            # Add memory context as system message
            enhanced_messages = [
                {
                    "role": "system", 
                    "content": f"Relevant conversation history:\n{memory_context}\n\nUse this context to provide personalized responses."
                }
            ]
            enhanced_messages.extend(llm_frame.messages)
            
            if memories:
                logger.info(f"🧠 Injected {len(memories)} memories into context:")
                for i, memory in enumerate(memories[:2], 1):
                    logger.info(f"   {i}. {memory[:80]}...")
            else:
                logger.info("🧠 No relevant memories found for context")
            return LLMMessagesFrame(messages=enhanced_messages)
            
        except Exception as e:
            logger.error(f"❌ Memory context injection failed: {e}")
            return llm_frame

    async def _search_memories(self, query: str) -> list:
        """Search for relevant memories"""
        if not self.memory:
            return []
        
        try:
            # Run memory search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            memories = await loop.run_in_executor(
                None, 
                lambda: self.memory.search(query, user_id=self.user_id)
            )
            
            # Extract actual memory content
            memory_texts = []
            if memories:
                for memory in memories[:self.max_context_memories]:
                    if hasattr(memory, 'memory'):
                        memory_texts.append(memory.memory)
                    elif isinstance(memory, dict) and 'memory' in memory:
                        memory_texts.append(memory['memory'])
            
            return memory_texts
            
        except Exception as e:
            logger.error(f"❌ Memory search failed: {e}")
            return []

    def _format_memory_context(self, memories: list) -> str:
        """Format memories for LLM context"""
        if not memories:
            return ""
        
        context_parts = []
        for i, memory in enumerate(memories, 1):
            context_parts.append(f"• {memory}")
        
        return "\n".join(context_parts)

    async def _store_conversation(self, user_input: str, assistant_response: str):
        """Store conversation pair in memory (async)"""
        if not self.memory:
            return
            
        try:
            # Store the full conversation exchange
            conversation_text = f"User said: '{user_input}' and I responded: '{assistant_response}'"
            
            # Run memory storage in thread pool 
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.memory.add(conversation_text, user_id=self.user_id)
            )
            
            logger.info(f"💾 Stored in Mem0: User: '{user_input[:50]}...' → Assistant: '{assistant_response[:50]}...'")
            
        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")

    def update_user_id(self, user_id: str):
        """Update user ID for speaker recognition integration"""
        old_user_id = self.user_id
        self.user_id = user_id
        logger.info(f"🔄 Memory user ID updated: {old_user_id} → {user_id}")

    def get_memory_stats(self) -> dict:
        """Get memory statistics for monitoring"""
        if not self.enabled or not self.memory:
            return {"enabled": False}
        
        try:
            # Get user memories
            all_memories = self.memory.get_all(user_id=self.user_id)
            count = len(all_memories) if all_memories else 0
            
            return {
                "enabled": True,
                "user_id": self.user_id,
                "total_memories": count,
                "max_context_memories": self.max_context_memories
            }
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {"enabled": True, "error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        if self.memory:
            # Mem0 handles cleanup automatically
            pass
        await super().cleanup()

    # ================================
    # DROP-IN COMPATIBILITY METHODS
    # ================================
    # These methods match LocalMemoryProcessor interface for seamless replacement
    
    async def search_conversations(self, query: str, limit: int = 10, user_id: Optional[str] = None) -> list:
        """
        Drop-in replacement for LocalMemoryProcessor.search_conversations
        Returns list of memory objects for tool handlers
        """
        user_id = user_id or self.user_id
        memories = await self._search_memories(query)
        
        # Format for compatibility with tool handlers
        results = []
        for memory_text in memories[:limit]:
            results.append({
                'memory': memory_text,
                'user_id': user_id,
                'timestamp': 'recent'  # Mem0 doesn't expose timestamps in search
            })
        
        return results
    
    async def update_user_id(self, user_id: str):
        """Drop-in replacement for LocalMemoryProcessor.update_user_id"""
        self.update_user_id(user_id)  # Use existing method
    
    async def add_conversation(self, user_message: str, assistant_message: str, user_id: Optional[str] = None):
        """
        Drop-in replacement for LocalMemoryProcessor.add_conversation
        Add a conversation exchange to memory
        """
        user_id = user_id or self.user_id
        conversation_text = f"User: '{user_message}' Assistant: '{assistant_message}'"
        
        if self.memory:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.memory.add(conversation_text, user_id=user_id)
                )
                logger.debug(f"💾 Added conversation to Mem0: {user_message[:30]}...")
            except Exception as e:
                logger.error(f"❌ Failed to add conversation to Mem0: {e}")
    
    def get_context_memories(self, user_input: str, limit: int = 5) -> str:
        """
        Drop-in replacement for getting context memories as formatted string
        This is used by the memory context injector
        """
        try:
            # Run search synchronously (needed for context injector compatibility)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to be careful
                memories = []
                try:
                    # Try to get memories from cache or run in thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: self.memory.search(user_input, user_id=self.user_id))
                        raw_memories = future.result(timeout=1.0)  # 1 second timeout
                        
                    # Extract memory texts
                    if raw_memories:
                        for memory in raw_memories[:limit]:
                            if hasattr(memory, 'memory'):
                                memories.append(memory.memory)
                            elif isinstance(memory, dict) and 'memory' in memory:
                                memories.append(memory['memory'])
                except Exception as e:
                    logger.warning(f"⚠️ Could not get memories for context: {e}")
                    memories = []
            else:
                # Not in async context, can run normally
                raw_memories = self.memory.search(user_input, user_id=self.user_id) if self.memory else []
                memories = []
                if raw_memories:
                    for memory in raw_memories[:limit]:
                        if hasattr(memory, 'memory'):
                            memories.append(memory.memory)
                        elif isinstance(memory, dict) and 'memory' in memory:
                            memories.append(memory['memory'])
            
            # Format as string for context injection
            if memories:
                formatted = "Relevant conversation history:\n"
                for i, memory in enumerate(memories, 1):
                    formatted += f"• {memory}\n"
                return formatted.strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error getting context memories: {e}")
            return ""