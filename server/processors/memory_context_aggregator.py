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
                 max_context_tokens: int = 1500,  # Reduced from 4096
                 memory_token_budget: int = 1000,  # Fixed budget for memory
                 **kwargs):
        super().__init__(messages, **kwargs)
        
        self.memory_processor = memory_processor
        self.max_context_tokens = max_context_tokens
        self.memory_token_budget = memory_token_budget  # Fixed 1000 tokens for memory
        self.token_counter = get_token_counter()
        
        # Query cache for performance (last 1000 queries)
        self.query_cache = {}
        self.cache_max_size = 1000
        
        # Performance metrics
        self.injection_count = 0
        self.total_injection_time_ms = 0.0
        self.context_overflows = 0
        self.memory_hits = 0
        self.cache_hits = 0
        
        logger.info(f"🧠 Smart memory-aware context initialized:")
        logger.info(f"   Max total tokens: {max_context_tokens}")
        logger.info(f"   Memory budget: {memory_token_budget} tokens")
        logger.info(f"   Query cache size: {self.cache_max_size}")
    
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
        
        logger.info("🚨 MEMORY-AWARE get_messages() CALLED! 🚨")
        current_messages = super().get_messages()
        
        # Build context with memory synchronously
        enhanced_messages = self._build_context_with_memory_sync(current_messages)
        
        logger.info(f"🚨 RETURNING {len(enhanced_messages)} messages to LLM (vs {len(current_messages)} original)")
        return enhanced_messages
    
    def _build_context_with_memory_sync(self, current_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Smart context building with controlled memory injection"""
        if not self.memory_processor:
            return current_messages
        
        try:
            # STEP 1: Get exactly last 5 exchanges (10 messages max)
            base_context = self._get_last_5_exchanges(current_messages)
            
            # STEP 2: Extract meaningful user query
            user_query = self._extract_user_query(current_messages)
            if not user_query:
                logger.debug("⚠️ No meaningful query found")
                return base_context
            
            # STEP 3: Check if this query needs memory retrieval
            if not self._needs_memory_search(user_query):
                logger.debug(f"🚫 Query doesn't need memory: '{user_query[:30]}...'")
                return base_context
            
            # STEP 4: Get memories (with caching)
            memories = self._get_cached_memories(user_query)
            if not memories:
                logger.debug("📭 No relevant memories found")
                return base_context
            
            # STEP 5: Add memories within token budget
            enhanced_context = self._add_memories_to_context(base_context, memories)
            
            logger.info(f"🧠 Smart context: {len(base_context)} base + {len(enhanced_context) - len(base_context)} memory messages")
            return enhanced_context
            
        except Exception as e:
            logger.error(f"❌ Smart context building failed: {e}")
            return current_messages
    
    def _get_last_5_exchanges(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Get exactly the last 5 exchanges (max 10 messages)"""
        # Find system message
        system_msg = None
        conversation_msgs = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_msg = msg
            else:
                conversation_msgs.append(msg)
        
        # Take only last 10 conversation messages (5 exchanges)
        recent_conversation = conversation_msgs[-10:] if len(conversation_msgs) > 10 else conversation_msgs
        
        # Build final context
        base_context = []
        if system_msg:
            base_context.append(system_msg)
        base_context.extend(recent_conversation)
        
        logger.debug(f"📝 Base context: {len(base_context)} messages (last 5 exchanges)")
        return base_context
    
    def _extract_user_query(self, messages: List[Dict[str, str]]) -> str:
        """Extract the most recent meaningful user query"""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '').strip()
                # Skip very short or meaningless queries
                if len(content) > 2 and content not in ['?', '.', 'ok', 'yes', 'no', '!']:
                    return content
        return ""
    
    def _needs_memory_search(self, query: str) -> bool:
        """Determine if a query needs memory retrieval"""
        query_lower = query.lower()
        
        # Memory trigger patterns
        memory_triggers = [
            'remember', 'recall', 'what is my', 'what was my', 'my name',
            'my dog', 'my location', 'where do i', 'do you know',
            'tell me about', 'what do you know about me', 'location',
            'dog', 'favorite number'
        ]
        
        # Question words that might need context
        question_words = ['what', 'where', 'who', 'when', 'how', 'why', 'which']
        
        # Check for memory triggers
        if any(trigger in query_lower for trigger in memory_triggers):
            return True
        
        # Check for questions that might benefit from memory
        if any(word in query_lower for word in question_words) and len(query) > 10:
            return True
        
        # Don't search for simple conversational responses
        if len(query) < 5 or query_lower in ['thanks', 'thank you', 'bye', 'goodbye', 'hi', 'hello']:
            return False
        
        return False
    
    def _get_cached_memories(self, query: str) -> List:
        """Get memories with caching for performance"""
        # Check cache first
        if query in self.query_cache:
            self.cache_hits += 1
            logger.debug(f"💾 Cache hit for query: '{query[:30]}...'")
            return self.query_cache[query]
        
        # Search and cache result
        memories = self._get_memories_sync(query, max_memories=10)  # Get more candidates
        
        # Cache management - keep only last 1000 queries
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.query_cache.keys())[:100]  # Remove 100 oldest
            for key in oldest_keys:
                del self.query_cache[key]
        
        self.query_cache[query] = memories
        logger.debug(f"🔍 Cache miss - searched and cached for: '{query[:30]}...'")
        return memories
    
    def _add_memories_to_context(self, base_context: List[Dict[str, str]], memories: List) -> List[Dict[str, str]]:
        """Add memories to context within token budget"""
        enhanced_context = base_context.copy()
        current_tokens = 0
        memories_added = 0
        
        # Count base context tokens
        base_tokens = sum(self.token_counter.count_tokens(msg['content']) for msg in base_context)
        available_tokens = self.memory_token_budget
        
        logger.debug(f"🧮 Token budget: {available_tokens} available for memories")
        
        # Add memories within budget
        for memory in memories:
            memory_tokens = self.token_counter.count_tokens(memory.content)
            
            if current_tokens + memory_tokens <= available_tokens:
                # Add as conversation pair
                if memory.speaker_id == 'assistant':
                    enhanced_context.append({
                        'role': 'assistant',
                        'content': memory.content
                    })
                else:
                    enhanced_context.append({
                        'role': 'user',
                        'content': memory.content
                    })
                
                current_tokens += memory_tokens
                memories_added += 1
                logger.debug(f"   Added memory {memories_added}: {memory.content[:50]}... ({memory_tokens} tokens)")
            else:
                logger.debug(f"   Skipping memory - would exceed budget ({memory_tokens} tokens)")
                break
        
        # Add current user query at the end
        if base_context and base_context[-1].get('role') == 'user':
            current_query = base_context[-1]
            enhanced_context.append(current_query)
        
        logger.info(f"💰 Token usage: {base_tokens} base + {current_tokens} memory = {base_tokens + current_tokens} total")
        logger.info(f"🧠 Added {memories_added} memories to context")
        
        return enhanced_context
    
    def _get_memories_sync(self, query: str, max_memories: int = 5) -> List:
        """Memory retrieval using persistent BM25 index"""
        if not self.memory_processor:
            return []
        
        try:
            # Try persistent BM25 search first
            if hasattr(self.memory_processor, 'search_memories_bm25'):
                bm25_memories = self.memory_processor.search_memories_bm25(query, max_memories)
                if bm25_memories:
                    logger.debug(f"🎯 Persistent BM25 found {len(bm25_memories)} memories for '{query[:30]}...'")
                    return bm25_memories
            
            # Fallback to tier-based search if BM25 not available
            candidate_memories = []
            
            # 1. HOT TIER: Get from in-memory deque (fast)
            hot_count = 0
            for item in list(self.memory_processor.hot_tier):
                if hasattr(item, 'content') and hasattr(item, 'speaker_id'):
                    if (item.speaker_id == self.memory_processor.current_speaker or 
                        item.speaker_id == 'assistant'):
                        candidate_memories.append(item)
                        hot_count += 1
            
            # 2. WARM TIER: Get from LMDB if available (sync read)
            warm_count = self._load_warm_memories_sync(candidate_memories)
            
            # 3. COLD TIER: Get from LMDB if available (sync read)  
            cold_count = self._load_cold_memories_sync(candidate_memories)
            
            if not candidate_memories:
                logger.debug(f"🔍 No candidate memories found across all tiers")
                return []
            
            # Use fallback BM25 to rank memories by relevance
            relevant_memories = self._is_relevant_bm25(candidate_memories, query, top_k=max_memories)
            
            logger.debug(f"🔍 FALLBACK search: {hot_count}🔥 + {warm_count}🌡️ + {cold_count}🧊 = {len(candidate_memories)} candidates → {len(relevant_memories)} relevant for '{query[:30]}...'")
            return relevant_memories
                
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    def _load_warm_memories_sync(self, candidate_memories: List, max_items: int = 50) -> int:
        """Load warm tier memories synchronously (quick LMDB read)"""
        try:
            if not hasattr(self.memory_processor, 'env') or not hasattr(self.memory_processor, 'warm_db'):
                logger.debug("No LMDB env or warm_db available")
                return 0  # No LMDB available
            
            # Quick synchronous read from warm tier
            with self.memory_processor.env.begin() as txn:
                cursor = txn.cursor(db=self.memory_processor.warm_db)
                prefix = f"{self.memory_processor.current_speaker}:".encode()
                
                count = 0
                successful_loads = 0
                
                if cursor.set_range(prefix):
                    for key, value in cursor:
                        if not key.startswith(prefix) or count >= max_items:
                            break
                        
                        count += 1
                        # Quick deserialize (sync version)
                        memory = self._quick_deserialize_memory(value)
                        if memory:
                            candidate_memories.append(memory)
                            successful_loads += 1
                        
                        # Stop if we have enough good memories
                        if successful_loads >= max_items // 2:
                            break
                
                logger.debug(f"🌡️ Warm tier: tried {count}, loaded {successful_loads} memories")
                return successful_loads
                
        except Exception as e:
            logger.debug(f"Warm tier sync load failed: {e}")
            return 0
    
    def _load_cold_memories_sync(self, candidate_memories: List, max_items: int = 20) -> int:
        """Load cold tier memories synchronously (quick LMDB read)"""
        try:
            if not hasattr(self.memory_processor, 'env') or not hasattr(self.memory_processor, 'cold_db'):
                logger.debug("No LMDB env or cold_db available")
                return 0  # No LMDB available
            
            # Quick synchronous read from cold tier  
            with self.memory_processor.env.begin() as txn:
                cursor = txn.cursor(db=self.memory_processor.cold_db)
                prefix = f"{self.memory_processor.current_speaker}:".encode()
                
                count = 0
                successful_loads = 0
                
                if cursor.set_range(prefix):
                    for key, value in cursor:
                        if not key.startswith(prefix) or count >= max_items:
                            break
                        
                        count += 1
                        # Quick deserialize (sync version)
                        memory = self._quick_deserialize_memory(value)
                        if memory:
                            candidate_memories.append(memory)
                            successful_loads += 1
                        
                        # Stop if we have enough good memories
                        if successful_loads >= max_items // 2:
                            break
                
                logger.debug(f"🧊 Cold tier: tried {count}, loaded {successful_loads} memories")
                return successful_loads
                
        except Exception as e:
            logger.debug(f"Cold tier sync load failed: {e}")
            return 0
    
    def _quick_deserialize_memory(self, compressed_data: bytes):
        """Quick synchronous memory deserialization with robust error handling"""
        try:
            import json
            
            # Try multiple decompression methods
            data = None
            
            # Method 1: Try LZ4 decompression
            try:
                import lz4.frame
                data = lz4.frame.decompress(compressed_data)
                # logger.debug("Decompressed with LZ4")  # Reduce logging spam
            except Exception as e1:
                # logger.debug(f"LZ4 decompression failed: {e1}")  # Reduce logging spam
                
                # Method 2: Try gzip decompression
                try:
                    import gzip
                    data = gzip.decompress(compressed_data)
                    # logger.debug("Decompressed with gzip")  # Reduce logging spam
                except Exception as e2:
                    # logger.debug(f"Gzip decompression failed: {e2}")  # Reduce logging spam
                    
                    # Method 3: Assume uncompressed
                    data = compressed_data
                    # logger.debug("Using raw data (no compression)")  # Reduce logging spam
            
            # Try to decode and parse JSON
            try:
                # Try UTF-8 first
                json_str = data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # Try latin-1 as fallback
                    json_str = data.decode('latin-1')
                    # logger.debug("Used latin-1 encoding")  # Reduce logging spam
                except UnicodeDecodeError:
                    # Skip if can't decode
                    # logger.debug("Could not decode data to string")  # Reduce logging spam
                    return None
            
            # Parse JSON
            memory_dict = json.loads(json_str)
            
            # Create memory item
            from processors.enhanced_stateless_memory import MemoryItem, MemoryTier
            return MemoryItem(
                content=memory_dict['content'],
                timestamp=memory_dict['timestamp'],
                speaker_id=memory_dict['speaker_id'],
                tier=MemoryTier(memory_dict.get('tier', 'warm')),
                access_count=memory_dict.get('access_count', 0),
                last_accessed=memory_dict.get('last_accessed', 0),
                importance_score=memory_dict.get('importance_score', 1.0)
            )
            
        except Exception as e:
            # logger.debug(f"Quick deserialize failed completely: {e}")  # Reduce logging spam
            return None
    
    def _is_relevant_bm25(self, memories: List, query: str, top_k: int = 10) -> List:
        """BM25-based relevance ranking for memories with enhanced query expansion"""
        if not memories:
            return []
        
        try:
            import bm25s
            
            # Extract content from memories for indexing
            corpus = [memory.content for memory in memories]
            
            # Create and index BM25
            retriever = bm25s.BM25()
            retriever.index(bm25s.tokenize(corpus))
            
            # Expand query with related terms for better semantic matching
            expanded_query = self._expand_query_terms(query)
            logger.debug(f"🔍 Expanded query: '{query}' → '{expanded_query}'")
            
            # Search for relevant memories with expanded query
            query_tokens = bm25s.tokenize([expanded_query])
            results, scores = retriever.retrieve(query_tokens, k=min(top_k, len(corpus)))
            
            # Return memories sorted by relevance score with lower threshold for expanded search
            relevant_memories = []
            for idx, score in zip(results[0], scores[0]):
                memory = memories[idx]
                # Include more results since we expanded the query
                if score > 0.05:  # Lower threshold for expanded search
                    relevant_memories.append(memory)
                    logger.debug(f"   Found: {memory.content[:60]}... (score: {score:.3f})")
            
            logger.debug(f"🎯 BM25 found {len(relevant_memories)} relevant memories for expanded query")
            return relevant_memories
            
        except ImportError:
            logger.warning("⚠️ BM25s not available, falling back to simple matching")
            return self._fallback_simple_search(memories, query, top_k)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}, falling back to simple matching")
            return self._fallback_simple_search(memories, query, top_k)
    
    def _expand_query_terms(self, query: str) -> str:
        """Expand query with related terms for better semantic matching"""
        query_lower = query.lower()
        expanded_terms = [query]  # Start with original query
        
        # Add related terms based on query content
        if "dog" in query_lower:
            expanded_terms.extend(["pet", "animal", "puppy", "canine"])
        
        if "name" in query_lower:
            expanded_terms.extend(["called", "named", "known as"])
        
        if "recall" in query_lower or "remember" in query_lower:
            expanded_terms.extend(["told", "said", "mentioned", "is", "was"])
        
        if "favorite" in query_lower:
            expanded_terms.extend(["like", "love", "prefer"])
        
        if "number" in query_lower:
            expanded_terms.extend(["digit", "count", "numerical"])
        
        if "color" in query_lower:
            expanded_terms.extend(["colour", "hue", "shade"])
        
        # Join all terms
        return " ".join(expanded_terms)
    
    def _are_memories_repetitive(self, memories: List, threshold: float = 0.6) -> bool:
        """Check if memories are mostly repetitive questions rather than answers"""
        if len(memories) < 2:
            return False
        
        # Check if memories are all questions asking for the same thing
        question_patterns = ['recall', 'remember', 'what', 'can you', 'do you', 'tell me']
        question_count = 0
        
        for memory in memories:
            content_lower = memory.content.lower()
            # Check if this is a question pattern
            if any(pattern in content_lower for pattern in question_patterns) and '?' in memory.content:
                question_count += 1
        
        # If more than 60% are questions, it's repetitive
        question_ratio = question_count / len(memories)
        is_repetitive = question_ratio >= threshold
        
        if is_repetitive:
            logger.debug(f"🔄 Memories are repetitive questions: {question_count}/{len(memories)} questions ({question_ratio:.2%})")
        
        return is_repetitive
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for secondary searches"""
        import re
        
        # Remove common question words and focus on content words
        stop_words = {'can', 'you', 'recall', 'remember', 'what', 'is', 'the', 'my', 'a', 'an', 'do', 'does'}
        
        # Extract words, keeping phrases like "dog's name" together
        words = re.findall(r'\b\w+(?:\'s)?\b', query.lower())
        
        # Filter out stop words and short words
        key_terms = []
        for word in words:
            if word not in stop_words and len(word) > 2:
                key_terms.append(word)
        
        # Also try common combinations
        query_lower = query.lower()
        if "dog" in query_lower and "name" in query_lower:
            key_terms.extend(["dog name", "dogs name", "dog's name"])
        
        logger.debug(f"🔑 Key terms extracted: {key_terms}")
        return key_terms
    
    def _fallback_simple_search(self, memories: List, query: str, top_k: int) -> List:
        """Fallback simple search when BM25 fails"""
        relevant_memories = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for memory in memories:
            content_lower = memory.content.lower()
            # Check if any query words appear in content
            for word in query_words:
                if len(word) > 2 and word in content_lower:
                    relevant_memories.append(memory)
                    break
                    
            if len(relevant_memories) >= top_k:
                break
        
        return relevant_memories
    
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
    max_context_tokens: int = 1500,
    memory_token_budget: int = 1000,
    **kwargs
) -> MemoryAwareOpenAILLMContext:
    """
    Factory function to create smart memory-aware context
    
    Args:
        initial_messages: Starting messages (usually system prompt)
        memory_processor: Memory processor instance
        max_context_tokens: Maximum total tokens allowed in context
        memory_token_budget: Fixed budget for memory retrieval (1000 tokens)
        **kwargs: Additional context parameters
    
    Returns:
        Smart memory-aware context with controlled size and caching
    """
    return MemoryAwareOpenAILLMContext(
        messages=initial_messages,
        memory_processor=memory_processor,
        max_context_tokens=max_context_tokens,
        memory_token_budget=memory_token_budget,
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