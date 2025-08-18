"""
Memory-aware context aggregator that integrates memory injection with Pipecat context system
This replaces the current approach of injecting memory during frame processing
"""

from typing import List, Dict, Any, Optional
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator
from pipecat.frames.frames import Frame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection
from loguru import logger

from .token_counter import get_token_counter
from .stateless_memory import StatelessMemoryProcessor

class MemoryAwareOpenAILLMContext(OpenAILLMContext):
    """
    Extended OpenAI context that integrates memory injection
    
    Key improvements:
    1. Memory injection happens BEFORE LLM calls (not during frame processing)
    2. Token budget management with accurate counting
    3. Graceful fallback when context is too large
    4. Performance monitoring and logging
    """
    
    def __init__(self, 
                 messages: List[Dict[str, str]], 
                 memory_processor: Optional[StatelessMemoryProcessor] = None,
                 max_context_tokens: int = 4096,
                 memory_budget_ratio: float = 0.3,
                 **kwargs):
        super().__init__(messages, **kwargs)
        
        self.memory_processor = memory_processor
        self.max_context_tokens = max_context_tokens
        self.memory_budget_ratio = memory_budget_ratio  # % of context for memory
        self.token_counter = get_token_counter()
        
        # Performance metrics
        self.injection_count = 0
        self.total_injection_time_ms = 0.0
        self.context_overflows = 0
        self.memory_hits = 0
        
        logger.info(f"🧠 Memory-aware context initialized:")
        logger.info(f"   Max tokens: {max_context_tokens}")
        logger.info(f"   Memory budget: {memory_budget_ratio:.0%} ({int(max_context_tokens * memory_budget_ratio)} tokens)")
    
    async def _inject_memory_if_needed(self, current_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Inject memory context if memory processor is available and budget allows
        
        Returns modified message list with memory context injected
        """
        if not self.memory_processor:
            return current_messages
        
        import time
        start_time = time.perf_counter()
        
        try:
            # Count tokens in current context
            current_tokens = self.token_counter.count_message_tokens(current_messages)
            
            # Calculate available budget for memory
            memory_budget = int(self.max_context_tokens * self.memory_budget_ratio)
            available_tokens = self.max_context_tokens - current_tokens - 200  # Reserve for response
            memory_tokens_allowed = min(memory_budget, available_tokens)
            
            logger.debug(f"🧠 Context analysis:")
            logger.debug(f"   Current: {current_tokens} tokens")
            logger.debug(f"   Available for memory: {memory_tokens_allowed} tokens")
            
            if memory_tokens_allowed < 50:
                logger.debug("⚠️  Insufficient tokens for memory injection")
                return current_messages
            
            # Extract user query for memory search
            user_query = ""
            for msg in reversed(current_messages):
                if msg.get('role') == 'user':
                    user_query = msg.get('content', '')
                    break
            
            if not user_query:
                logger.debug("⚠️  No user query found for memory search")
                return current_messages
            
            # Get relevant memories with token budget
            memories = await self.memory_processor._get_relevant_memories(
                user_query, 
                self.memory_processor.current_speaker,
                max_tokens=memory_tokens_allowed
            )
            
            if not memories:
                logger.debug("📭 No relevant memories found")
                return current_messages
            
            # Build memory context string
            context_parts = []
            token_count = 0
            
            for memory in memories:
                memory_text = f"[{memory.speaker_id}]: {memory.content}"
                memory_tokens = self.token_counter.count_tokens(memory_text)
                
                if token_count + memory_tokens <= memory_tokens_allowed:
                    context_parts.append(memory_text)
                    token_count += memory_tokens
                else:
                    break
            
            if not context_parts:
                logger.debug("📭 No memories fit in token budget")
                return current_messages
            
            # Create memory context message
            memory_context = "\n".join(context_parts)
            memory_message = {
                'role': 'system',
                'content': f"[Memory Context - {len(context_parts)} items]:\n{memory_context}"
            }
            
            # Find injection point (after system message if present)
            injection_point = 1 if current_messages and current_messages[0].get('role') == 'system' else 0
            
            # Inject memory context
            enhanced_messages = current_messages.copy()
            enhanced_messages.insert(injection_point, memory_message)
            
            # Verify total context size
            total_tokens = self.token_counter.count_message_tokens(enhanced_messages)
            
            if total_tokens > self.max_context_tokens - 200:
                logger.warning(f"⚠️  Context overflow after memory injection: {total_tokens} tokens")
                self.context_overflows += 1
                return current_messages  # Return original if too large
            
            # Success metrics
            self.memory_hits += 1
            logger.info(f"✅ Memory injected: {len(context_parts)} items, {token_count} tokens")
            logger.debug(f"   Final context: {total_tokens} tokens")
            
            return enhanced_messages
            
        except Exception as e:
            logger.error(f"❌ Memory injection failed: {e}")
            return current_messages  # Return original on error
            
        finally:
            # Track performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.total_injection_time_ms += elapsed_ms
            self.injection_count += 1
            
            if elapsed_ms > 20:
                logger.warning(f"⏰ Slow memory injection: {elapsed_ms:.2f}ms")
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Override to inject memory before returning messages"""
        # This is called by the LLM service before making the API call
        # Perfect place to inject memory context!
        
        current_messages = super().get_messages()
        
        # For now, return as-is since we need async support
        # TODO: This needs to be refactored for async memory injection
        return current_messages
    
    async def get_messages_with_memory(self) -> List[Dict[str, str]]:
        """Async version that properly injects memory"""
        current_messages = super().get_messages()
        return await self._inject_memory_if_needed(current_messages)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get memory injection performance statistics"""
        avg_injection_time = (
            self.total_injection_time_ms / self.injection_count
            if self.injection_count > 0 else 0
        )
        
        return {
            'injection_count': self.injection_count,
            'memory_hits': self.memory_hits,
            'context_overflows': self.context_overflows,
            'avg_injection_time_ms': avg_injection_time,
            'memory_hit_rate': self.memory_hits / max(self.injection_count, 1),
        }

class MemoryUserContextAggregator(LLMUserContextAggregator):
    """
    User context aggregator that triggers memory injection
    This ensures memory is available before LLM processing
    """
    
    def __init__(self, context: MemoryAwareOpenAILLMContext, **kwargs):
        super().__init__(context, **kwargs)
        self.memory_context = context
        logger.debug("🧠 Memory-aware user aggregator initialized")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frame and ensure memory injection happens at right time"""
        
        # Call parent to handle normal aggregation
        await super().process_frame(frame, direction)
        
        # If this is an LLMMessagesFrame going upstream, it means
        # we're about to send context to LLM - perfect time for memory injection
        if isinstance(frame, LLMMessagesFrame) and direction == FrameDirection.UPSTREAM:
            try:
                # Inject memory into the frame's messages
                enhanced_messages = await self.memory_context.get_messages_with_memory()
                frame.messages = enhanced_messages
                
                logger.debug(f"🧠 Memory injection completed for upstream LLMMessagesFrame")
                
            except Exception as e:
                logger.error(f"❌ Memory injection failed in user aggregator: {e}")

def create_memory_context(
    initial_messages: List[Dict[str, str]],
    memory_processor: Optional[StatelessMemoryProcessor] = None,
    max_context_tokens: int = 4096,
    **kwargs
) -> MemoryAwareOpenAILLMContext:
    """
    Factory function to create memory-aware context
    
    Args:
        initial_messages: Starting messages (usually system prompt)
        memory_processor: Memory processor instance
        max_context_tokens: Maximum tokens allowed in context
        **kwargs: Additional context parameters
    
    Returns:
        Memory-aware context that will inject memories before LLM calls
    """
    return MemoryAwareOpenAILLMContext(
        messages=initial_messages,
        memory_processor=memory_processor,
        max_context_tokens=max_context_tokens,
        **kwargs
    )

# Self-test function
if __name__ == "__main__":
    import asyncio
    import tempfile
    import shutil
    
    async def test_memory_context():
        """Test memory-aware context aggregator"""
        
        print("🧠 Memory Context Aggregator Test")
        print("=" * 40)
        
        # Create temporary memory processor
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize memory processor
            from processors.stateless_memory import StatelessMemoryProcessor
            
            memory_processor = StatelessMemoryProcessor(
                db_path=temp_dir,
                max_context_tokens=512,
                perfect_recall_window=5
            )
            
            # Add some test memories
            await memory_processor._store_exchange(
                "What's the capital of France?",
                "The capital of France is Paris."
            )
            
            await memory_processor._store_exchange(
                "Tell me about the Eiffel Tower",
                "The Eiffel Tower is a famous landmark in Paris, France."
            )
            
            # Create memory-aware context
            context = create_memory_context(
                initial_messages=[
                    {"role": "system", "content": "You are a helpful assistant."}
                ],
                memory_processor=memory_processor,
                max_context_tokens=1024
            )
            
            # Test memory injection
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What do you know about Paris?"}
            ]
            
            enhanced_messages = await context._inject_memory_if_needed(test_messages)
            
            print(f"Original messages: {len(test_messages)}")
            print(f"Enhanced messages: {len(enhanced_messages)}")
            
            for i, msg in enumerate(enhanced_messages):
                print(f"Message {i}: {msg['role']} - {msg['content'][:100]}...")
            
            # Print performance stats
            stats = context.get_performance_stats()
            print(f"\nPerformance stats: {stats}")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    asyncio.run(test_memory_context())