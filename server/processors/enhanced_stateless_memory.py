"""
Enhanced StatelessMemoryProcessor with three-tier architecture

This implements the natural degradation memory system:
- Hot Tier: Last 10 conversations in memory (perfect recall)
- Warm Tier: Last 100 conversations with LZ4 compression
- Cold Tier: Everything older with Zstd compression and gradual summarization
"""

import asyncio
import time
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum

from pipecat.frames.frames import (
    Frame, StartFrame, EndFrame, TextFrame, LLMMessagesFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger

# Define MemoryItem and tier enums
class MemoryTier(Enum):
    HOT = "hot"      # In-memory, no compression
    WARM = "warm"    # LMDB with LZ4 compression
    COLD = "cold"    # LMDB with Zstd compression + summarization

@dataclass
class MemoryItem:
    """Enhanced memory item with tier tracking"""
    content: str
    timestamp: float
    speaker_id: str
    tier: MemoryTier = MemoryTier.HOT
    access_count: int = 0
    last_accessed: float = 0
    importance_score: float = 1.0
    compressed_size: int = 0
    original_size: int = 0

@dataclass  
class MemoryStats:
    """Memory system performance statistics"""
    hot_tier_items: int = 0
    warm_tier_items: int = 0
    cold_tier_items: int = 0
    total_storage_bytes: int = 0
    compression_ratio: float = 1.0
    avg_retrieval_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    degradation_events: int = 0

try:
    import lmdb
    import lz4.frame
    STORAGE_AVAILABLE = True
    
    # Try to import zstandard for cold tier compression
    try:
        import zstandard as zstd
        ZSTD_AVAILABLE = True
    except ImportError:
        ZSTD_AVAILABLE = False
        logger.warning("Zstandard not available, using gzip for cold tier compression")
        
except ImportError:
    logger.warning("LMDB or LZ4 not available, using in-memory storage only")
    STORAGE_AVAILABLE = False
    ZSTD_AVAILABLE = False


class EnhancedStatelessMemoryProcessor(FrameProcessor):
    """
    Three-tier memory system with natural degradation
    
    Architecture:
    - Hot Tier: deque(maxlen=10) - Perfect recall, no compression
    - Warm Tier: LMDB with LZ4 - Recent but compressed
    - Cold Tier: LMDB with Zstd - Heavily compressed + summarized
    
    Natural Degradation:
    - Items age from hot -> warm -> cold automatically
    - Compression increases with age
    - Access patterns influence retention
    - Constant resource usage through lifecycle management
    """
    
    def __init__(self, 
                 db_path: str = "data/enhanced_memory",
                 max_context_tokens: int = 1024,
                 hot_tier_size: int = 200,  # Increased from 10 to 200 (100 conversation turns)
                 warm_tier_size: int = 500,  # Increased proportionally
                 cold_tier_size: int = 2000,  # Increased proportionally
                 degradation_interval: int = 300,  # 5 minutes
                 **kwargs):
        # CRITICAL: Always call parent init first
        super().__init__(**kwargs)
        
        # Configuration
        self.max_context_tokens = max_context_tokens
        self.hot_tier_size = hot_tier_size
        self.warm_tier_size = warm_tier_size  
        self.cold_tier_size = cold_tier_size
        self.degradation_interval = degradation_interval
        
        # Hot tier: Perfect recall in memory
        self.hot_tier = deque(maxlen=hot_tier_size)
        
        # Warm/Cold tier storage (LMDB or fallback)
        self.persistent_memory = {}  # Fallback storage
        self.db_path = Path(db_path)
        
        # Current state
        self.current_speaker = "default_user"
        self.current_user_message = ""
        self.last_injected_message = ""
        
        # Performance tracking
        self.stats = MemoryStats()
        self.retrieval_times = deque(maxlen=100)
        self.last_degradation = time.time()
        
        # Background task tracking
        self._degradation_task = None
        self._startup_logged = False
        
        # Initialize storage
        self._init_storage()
        
        # Initialize BM25 search index
        self._init_bm25_index()
        
        logger.info(f"🧠 ENHANCED MEMORY INITIALIZED")
        logger.info(f"   Hot tier: {hot_tier_size} items (perfect recall)")
        logger.info(f"   Warm tier: {warm_tier_size} items (LZ4 compressed)")
        logger.info(f"   Cold tier: {cold_tier_size} items (Zstd + summarization)")
        logger.info(f"   Storage: {'LMDB' if STORAGE_AVAILABLE else 'in-memory'}")
        logger.info(f"   Degradation interval: {degradation_interval}s")
    
    def _init_storage(self):
        """Initialize LMDB storage with separate databases for tiers"""
        if not STORAGE_AVAILABLE:
            logger.info("Using in-memory fallback storage")
            return
        
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Larger map size for three-tier system
            self.env = lmdb.open(
                str(self.db_path),
                map_size=1024*1024*1024,  # 1GB
                max_dbs=3,  # hot, warm, cold
                writemap=True,
                metasync=False,
                sync=False
            )
            
            with self.env.begin(write=True) as txn:
                self.warm_db = self.env.open_db(b'warm', txn=txn)
                self.cold_db = self.env.open_db(b'cold', txn=txn) 
                self.meta_db = self.env.open_db(b'meta', txn=txn)
            
            logger.info(f"✅ LMDB three-tier storage initialized at {self.db_path}")
            
        except Exception as e:
            logger.warning(f"LMDB initialization failed, using in-memory: {e}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with proper Pipecat patterns"""
        
        # STEP 1: Always call parent first (required for initialization)
        await super().process_frame(frame, direction)
        
        # STEP 2: Handle StartFrame specially
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            
            # Start background tasks now that TaskManager is available
            if self._degradation_task is None:
                self._degradation_task = self.get_event_loop().create_task(self._degradation_loop())
                logger.info("🔄 Enhanced memory degradation task started")
            
            # Load existing memories from LMDB on startup
            if not self._startup_logged:
                # Load memories from persistent storage
                self.get_event_loop().create_task(self._load_memories_on_startup())
                
                # Log startup stats
                self.get_event_loop().create_task(self._log_startup_stats())
                self._startup_logged = True
            
            logger.debug("EnhancedMemory: StartFrame processed and forwarded")
            return
        
        # STEP 3: Handle EndFrame
        if isinstance(frame, EndFrame):
            # Cancel background tasks
            if self._degradation_task and not self._degradation_task.done():
                self._degradation_task.cancel()
                logger.info("🔄 Enhanced memory degradation task cancelled")
            
            await self.push_frame(frame, direction)
            logger.debug("EnhancedMemory: EndFrame processed and forwarded")
            return
        
        # STEP 4: Process other frames with error isolation
        try:
            await self._safe_process_frame(frame, direction)
        except Exception as e:
            # CRITICAL: Never let memory errors break the pipeline
            logger.error(f"Enhanced memory processing error (isolated): {e}")
        
        # STEP 5: Always forward frames
        await self.push_frame(frame, direction)
    
    async def _safe_process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with enhanced three-tier logic"""
        
        frame_name = type(frame).__name__
        
        # Skip audio/video frames
        if frame_name in ['AudioRawFrame', 'TTSAudioRawFrame', 'InputAudioRawFrame', 'VADParamsFrame', 'VideoFrame']:
            return
        
        # logger.debug(f"🧠 ENHANCED: Processing {frame_name} direction={direction}")
        
        # Track speaker changes
        if isinstance(frame, UserStartedSpeakingFrame):
            self.current_speaker = getattr(frame, 'speaker_id', 'default_user')
            logger.debug(f"Speaker started: {self.current_speaker}")
            return
        
        # Capture user input and inject memory
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.text and frame.text.strip():
                self.current_user_message = frame.text.strip()
                logger.info(f"🧠 ENHANCED: Captured user message: {self.current_user_message[:50]}...")
                
                # Store user message immediately (don't wait for assistant response)
                await self._store_user_message_immediately(self.current_user_message)
            return
        
        # Handle text frames for response storage
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.text and frame.text.strip() and self.current_user_message:
                await self._store_exchange_three_tier(self.current_user_message, frame.text.strip())
                self.current_user_message = ""
                self.last_injected_message = ""
            return
    
    async def _store_user_message_immediately(self, user_message: str):
        """Store user message immediately without waiting for assistant response"""
        try:
            timestamp = time.time()
            
            # Clean up fragmented transcriptions
            cleaned_message = user_message.strip()
            # Remove leading punctuation from fragmented transcriptions
            if cleaned_message and cleaned_message[0] in ',.;:':
                cleaned_message = cleaned_message[1:].strip()
            
            # Skip very short fragments
            if len(cleaned_message) < 3:
                logger.debug(f"Skipping short fragment: '{cleaned_message}'")
                return
            
            # Create user memory item
            user_memory = MemoryItem(
                content=cleaned_message,
                timestamp=timestamp,
                speaker_id=self.current_speaker,
                tier=MemoryTier.HOT
            )
            
            # Add to hot tier immediately
            self.hot_tier.append(user_memory)
            
            # PERSIST to LMDB for survival across restarts
            if hasattr(self, 'env'):
                try:
                    memory_dict = {
                        'content': user_memory.content,
                        'timestamp': user_memory.timestamp,
                        'speaker_id': user_memory.speaker_id,
                        'tier': user_memory.tier.value,
                        'access_count': user_memory.access_count,
                        'last_accessed': user_memory.last_accessed,
                        'importance_score': user_memory.importance_score
                    }
                    
                    # Serialize and compress
                    memory_json = json.dumps(memory_dict).encode()
                    if STORAGE_AVAILABLE and lz4:
                        compressed_data = lz4.frame.compress(memory_json)
                    else:
                        compressed_data = memory_json
                    
                    # Store in LMDB warm tier
                    key = f"{user_memory.speaker_id}:{user_memory.timestamp}".encode()
                    
                    with self.env.begin(write=True) as txn:
                        txn.put(key, compressed_data, db=self.warm_db)
                    
                    logger.info(f"🧠 ENHANCED: 💾 Persisted to LMDB warm tier")
                    
                except Exception as e:
                    logger.error(f"Failed to persist to LMDB: {e}")
            
            logger.info(f"🧠 ENHANCED: ✅ Stored user message immediately: '{user_message[:30]}...'")
            logger.info(f"🧠 ENHANCED: Hot tier now has {len(self.hot_tier)} items")
            
        except Exception as e:
            logger.error(f"Failed to store user message immediately: {e}")
    
    async def _get_relevant_memories_three_tier(self, query: str, speaker_id: str, max_tokens: int = None) -> List[MemoryItem]:
        """Enhanced memory retrieval across all three tiers"""
        
        # Import token counter
        try:
            from .token_counter import get_token_counter
            token_counter = get_token_counter()
        except ImportError:
            token_counter = None
        
        memories = []
        token_count = 0
        
        if max_tokens is None:
            max_tokens = self.max_context_tokens // 2
        
        logger.debug(f"🔍 Three-tier search for '{query[:50]}...' (budget: {max_tokens} tokens)")
        
        try:
            # 1. HOT TIER: Perfect recall from recent conversations
            hot_memories = await self._search_hot_tier(query, speaker_id, max_tokens // 2, token_counter)
            memories.extend(hot_memories)
            
            if token_counter:
                token_count = sum(token_counter.count_tokens(m.content) for m in hot_memories)
            else:
                token_count = sum(len(m.content.split()) * 1.3 for m in hot_memories)
            
            logger.debug(f"🔥 HOT: {len(hot_memories)} memories ({token_count} tokens)")
            
            # 2. WARM TIER: Recent compressed memories
            remaining_tokens = max_tokens - token_count
            if remaining_tokens > 50:
                warm_memories = await self._search_warm_tier(query, speaker_id, remaining_tokens // 2, token_counter)
                memories.extend(warm_memories)
                
                if token_counter:
                    warm_tokens = sum(token_counter.count_tokens(m.content) for m in warm_memories)
                else:
                    warm_tokens = sum(len(m.content.split()) * 1.3 for m in warm_memories)
                
                token_count += warm_tokens
                logger.debug(f"🌡️ WARM: {len(warm_memories)} memories ({warm_tokens} tokens)")
            
            # 3. COLD TIER: Archived memories with high importance scores
            remaining_tokens = max_tokens - token_count
            if remaining_tokens > 50:
                cold_memories = await self._search_cold_tier(query, speaker_id, remaining_tokens, token_counter)
                memories.extend(cold_memories)
                
                if token_counter:
                    cold_tokens = sum(token_counter.count_tokens(m.content) for m in cold_memories)
                else:
                    cold_tokens = sum(len(m.content.split()) * 1.3 for m in cold_memories)
                
                token_count += cold_tokens
                logger.debug(f"🧊 COLD: {len(cold_memories)} memories ({cold_tokens} tokens)")
        
        except Exception as e:
            logger.error(f"Three-tier memory search failed: {e}")
        
        logger.info(f"🧠 ENHANCED: Retrieved {len(memories)} memories across all tiers ({token_count} tokens)")
        return memories
    
    async def _search_hot_tier(self, query: str, speaker_id: str, max_tokens: int, token_counter=None) -> List[MemoryItem]:
        """Search hot tier (in-memory perfect recall)"""
        
        memories = []
        token_count = 0
        
        # Search most recent items in hot tier
        for item in reversed(self.hot_tier):
            if isinstance(item, MemoryItem):
                # Match speaker or assistant responses to user queries
                if item.speaker_id == speaker_id or item.speaker_id == "assistant":
                    if self._is_relevant_enhanced(item.content, query):
                        if token_counter:
                            item_tokens = token_counter.count_tokens(item.content)
                        else:
                            item_tokens = len(item.content.split()) * 1.3
                        
                        if token_count + item_tokens <= max_tokens:
                            memories.append(item)
                            token_count += item_tokens
                        else:
                            break
        
        return memories
    
    async def _search_warm_tier(self, query: str, speaker_id: str, max_tokens: int, token_counter=None) -> List[MemoryItem]:
        """Search warm tier (LMDB with LZ4 compression)"""
        
        memories = []
        
        if not hasattr(self, 'env'):
            # Fallback to in-memory
            speaker_memories = self.persistent_memory.get(f"{speaker_id}:warm", [])
            for memory in reversed(speaker_memories[-20:]):
                if self._is_relevant_enhanced(memory.content, query):
                    memories.append(memory)
                    if len(memories) >= 5:
                        break
            return memories
        
        try:
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self.warm_db)
                prefix = f"{speaker_id}:".encode()
                
                if cursor.set_range(prefix):
                    items_checked = 0
                    while items_checked < 30:  # Check more items in warm tier
                        key, value = cursor.item()
                        
                        if not key.startswith(prefix):
                            break
                        
                        memory = await self._deserialize_memory_enhanced(value, MemoryTier.WARM)
                        if memory and self._is_relevant_enhanced(memory.content, query):
                            memories.append(memory)
                            if len(memories) >= 8:  # More items from warm tier
                                break
                        
                        if not cursor.next():
                            break
                        items_checked += 1
        
        except Exception as e:
            logger.error(f"Warm tier search failed: {e}")
        
        return memories
    
    async def _search_cold_tier(self, query: str, speaker_id: str, max_tokens: int, token_counter=None) -> List[MemoryItem]:
        """Search cold tier (LMDB with Zstd compression, importance-based)"""
        
        memories = []
        
        if not hasattr(self, 'env'):
            # Fallback to in-memory
            speaker_memories = self.persistent_memory.get(f"{speaker_id}:cold", [])
            # Sort by importance score for cold tier
            sorted_memories = sorted(speaker_memories, key=lambda m: m.importance_score, reverse=True)
            for memory in sorted_memories[:10]:
                if self._is_relevant_enhanced(memory.content, query):
                    memories.append(memory)
                    if len(memories) >= 3:  # Fewer items from cold tier
                        break
            return memories
        
        try:
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self.cold_db)
                prefix = f"{speaker_id}:".encode()
                
                # Collect candidates with importance scores
                candidates = []
                if cursor.set_range(prefix):
                    items_checked = 0
                    while items_checked < 50:  # Check more items but filter by importance
                        key, value = cursor.item()
                        
                        if not key.startswith(prefix):
                            break
                        
                        memory = await self._deserialize_memory_enhanced(value, MemoryTier.COLD)
                        if memory and self._is_relevant_enhanced(memory.content, query):
                            candidates.append(memory)
                        
                        if not cursor.next():
                            break
                        items_checked += 1
                
                # Sort by importance score and take top items
                candidates.sort(key=lambda m: m.importance_score, reverse=True)
                memories = candidates[:3]  # Only top 3 from cold tier
        
        except Exception as e:
            logger.error(f"Cold tier search failed: {e}")
        
        return memories
    
    def _is_relevant_enhanced(self, memory_content: str, query: str) -> bool:
        """Enhanced relevance check with better keyword matching"""
        
        import re
        
        # Clean text by removing punctuation and normalizing
        def clean_text(text):
            # Remove punctuation and normalize whitespace
            cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
            # Split and filter out empty strings
            return [word.strip() for word in cleaned.split() if word.strip()]
        
        memory_words = set(clean_text(memory_content))
        query_words = set(clean_text(query))
        
        # Enhanced stop words list
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Remove stop words but keep important short words like "AI", "ML", "UI", etc.
        important_short_words = {'ai', 'ml', 'ui', 'ux', 'io', 'os', 'db', 'api', 'css', 'js', 'ts', 'go', 'r'}
        
        def should_keep_word(word):
            return (len(word) > 2 and word not in stop_words) or (word.lower() in important_short_words)
        
        query_keywords = {w for w in query_words if should_keep_word(w)}
        memory_keywords = {w for w in memory_words if should_keep_word(w)}
        
        if not query_keywords:
            return False
        
        # Check for exact matches, partial matches, and semantic similarity
        exact_matches = query_keywords.intersection(memory_keywords)
        
        # Partial word matching for better recall
        partial_matches = 0
        for query_word in query_keywords:
            for memory_word in memory_keywords:
                if len(query_word) > 3 and len(memory_word) > 3:
                    if query_word in memory_word or memory_word in query_word:
                        partial_matches += 1
                        break
        
        # Relevance threshold - more lenient for better recall
        relevance_score = len(exact_matches) + (partial_matches * 0.5)
        
        # If no query keywords, return False
        if not query_keywords:
            return False
        
        # Lower threshold for better recall during testing
        threshold = max(0.5, len(query_keywords) * 0.2)  # At least 20% keyword overlap
        
        # Also return true if we have any exact matches for short queries
        if len(query_keywords) <= 2 and exact_matches:
            return True
            
        return relevance_score >= threshold
    
    async def _store_exchange_three_tier(self, user_message: str, assistant_response: str):
        """Store conversation exchange in three-tier system"""
        
        try:
            timestamp = time.time()
            
            # Create enhanced memory items starting in hot tier
            user_memory = MemoryItem(
                content=user_message,
                timestamp=timestamp,
                speaker_id=self.current_speaker,
                tier=MemoryTier.HOT,
                access_count=1,
                last_accessed=timestamp,
                importance_score=self._calculate_importance_score(user_message, is_user=True),
                original_size=len(user_message.encode()),
                compressed_size=len(user_message.encode())
            )
            
            assistant_memory = MemoryItem(
                content=assistant_response,
                timestamp=timestamp + 0.001,
                speaker_id="assistant",
                tier=MemoryTier.HOT,
                access_count=1,
                last_accessed=timestamp,
                importance_score=self._calculate_importance_score(assistant_response, is_user=False),
                original_size=len(assistant_response.encode()),
                compressed_size=len(assistant_response.encode())
            )
            
            # Add to hot tier (will trigger degradation if full)
            self.hot_tier.extend([user_memory, assistant_memory])
            
            # Update BM25 index with new memories
            self._update_bm25_index(user_memory)
            self._update_bm25_index(assistant_memory)
            
            # Update statistics
            self.stats.hot_tier_items = len(self.hot_tier)
            
            logger.debug(f"🔥 Stored exchange in hot tier: {user_message[:30]}... -> {assistant_response[:30]}...")
            logger.debug(f"   User importance: {user_memory.importance_score:.2f}")
            logger.debug(f"   Assistant importance: {assistant_memory.importance_score:.2f}")
        
        except Exception as e:
            logger.error(f"Failed to store three-tier exchange: {e}")
    
    def _calculate_importance_score(self, content: str, is_user: bool = True) -> float:
        """Calculate importance score for memory item"""
        
        base_score = 1.0
        
        # Content length factor
        length_factor = min(2.0, len(content) / 100)
        
        # Question/command indicators (higher importance)
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which']
        command_indicators = ['please', 'can you', 'could you', 'help', 'explain', 'tell me']
        
        content_lower = content.lower()
        importance_boost = 0.0
        
        if any(indicator in content_lower for indicator in question_indicators):
            importance_boost += 0.3
        
        if any(indicator in content_lower for indicator in command_indicators):
            importance_boost += 0.2
        
        # Technical terms or specific topics (higher importance)
        technical_terms = ['code', 'function', 'class', 'algorithm', 'database', 'api', 'error', 'bug', 'fix']
        if any(term in content_lower for term in technical_terms):
            importance_boost += 0.4
        
        # User messages generally more important than assistant
        role_factor = 1.2 if is_user else 1.0
        
        final_score = (base_score + length_factor + importance_boost) * role_factor
        return min(5.0, final_score)  # Cap at 5.0
    
    async def _update_access_patterns(self, memories: List[MemoryItem]):
        """Update access patterns for retrieved memories"""
        
        current_time = time.time()
        
        for memory in memories:
            memory.access_count += 1
            memory.last_accessed = current_time
            
            # Boost importance score based on access frequency
            access_boost = min(0.5, memory.access_count * 0.1)
            memory.importance_score = min(5.0, memory.importance_score + access_boost)
        
        logger.debug(f"Updated access patterns for {len(memories)} memories")
    
    async def _degradation_loop(self):
        """Background loop for natural memory degradation"""
        
        while True:
            try:
                await asyncio.sleep(self.degradation_interval)
                await self._perform_degradation()
                
            except Exception as e:
                logger.error(f"Degradation loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_degradation(self):
        """Perform natural memory degradation across tiers"""
        
        logger.debug("🔄 Starting memory degradation cycle...")
        
        try:
            # 1. Move items from hot to warm tier if hot tier is full
            if len(self.hot_tier) >= self.hot_tier_size:
                await self._degrade_hot_to_warm()
            
            # 2. Move items from warm to cold tier
            await self._degrade_warm_to_cold()
            
            # 3. Compress and summarize cold tier items
            await self._compress_cold_tier()
            
            # 4. Remove least important cold tier items if over limit
            await self._cleanup_cold_tier()
            
            # 5. Update statistics
            await self._update_degradation_stats()
            
            self.last_degradation = time.time()
            self.stats.degradation_events += 1
            
            logger.info("✅ Memory degradation cycle completed")
            
        except Exception as e:
            logger.error(f"Degradation cycle failed: {e}")
    
    async def _degrade_hot_to_warm(self):
        """Move oldest items from hot tier to warm tier with LZ4 compression"""
        
        if not hasattr(self, 'env'):
            # In-memory fallback
            items_to_move = list(self.hot_tier)[:self.hot_tier_size // 2]
            
            for item in items_to_move:
                item.tier = MemoryTier.WARM
                key = f"{item.speaker_id}:warm"
                if key not in self.persistent_memory:
                    self.persistent_memory[key] = []
                self.persistent_memory[key].append(item)
            
            # Remove from hot tier
            for _ in range(len(items_to_move)):
                if self.hot_tier:
                    self.hot_tier.popleft()
            
            logger.debug(f"🌡️ Degraded {len(items_to_move)} items to warm tier (in-memory)")
            return
        
        try:
            # Move oldest half of hot tier to warm tier
            items_to_move = []
            move_count = len(self.hot_tier) // 2
            
            for _ in range(move_count):
                if self.hot_tier:
                    item = self.hot_tier.popleft()
                    item.tier = MemoryTier.WARM
                    items_to_move.append(item)
            
            # Store in warm tier with LZ4 compression
            with self.env.begin(write=True) as txn:
                for item in items_to_move:
                    key = f"{item.speaker_id}:{item.timestamp}".encode()
                    # Convert MemoryTier enum to string for JSON serialization
                    item_dict = asdict(item)
                    item_dict['tier'] = item_dict['tier'].value  # Convert enum to string
                    data = json.dumps(item_dict).encode()
                    
                    if STORAGE_AVAILABLE:
                        compressed_data = lz4.frame.compress(data)
                        item.compressed_size = len(compressed_data)
                    else:
                        compressed_data = data
                    
                    txn.put(key, compressed_data, db=self.warm_db)
            
            self.stats.warm_tier_items += len(items_to_move)
            logger.debug(f"🌡️ Degraded {len(items_to_move)} items to warm tier (LZ4 compressed)")
            
        except Exception as e:
            logger.error(f"Hot to warm degradation failed: {e}")
    
    async def _degrade_warm_to_cold(self):
        """Move old items from warm tier to cold tier with Zstd compression"""
        
        if not hasattr(self, 'env'):
            # In-memory fallback - move items older than 24 hours
            cutoff_time = time.time() - (24 * 3600)
            
            for speaker_key in list(self.persistent_memory.keys()):
                if ':warm' in speaker_key:
                    warm_items = self.persistent_memory[speaker_key]
                    items_to_move = [item for item in warm_items if item.timestamp < cutoff_time]
                    
                    if items_to_move:
                        # Move to cold tier
                        cold_key = speaker_key.replace(':warm', ':cold')
                        if cold_key not in self.persistent_memory:
                            self.persistent_memory[cold_key] = []
                        
                        for item in items_to_move:
                            item.tier = MemoryTier.COLD
                            self.persistent_memory[cold_key].append(item)
                        
                        # Remove from warm tier
                        self.persistent_memory[speaker_key] = [
                            item for item in warm_items if item not in items_to_move
                        ]
                        
                        logger.debug(f"🧊 Degraded {len(items_to_move)} items to cold tier (in-memory)")
            return
        
        try:
            # Move items older than 1 hour from warm to cold
            cutoff_time = time.time() - 3600  # 1 hour
            items_moved = 0
            
            with self.env.begin(write=True) as txn:
                cursor = txn.cursor(db=self.warm_db)
                keys_to_delete = []
                
                for key, value in cursor:
                    try:
                        memory = await self._deserialize_memory_enhanced(value, MemoryTier.WARM)
                        if memory and memory.timestamp < cutoff_time:
                            # Move to cold tier with higher compression
                            memory.tier = MemoryTier.COLD
                            memory_dict = asdict(memory)
                            memory_dict['tier'] = memory_dict['tier'].value  # Convert enum to string
                            cold_data = json.dumps(memory_dict).encode()
                            
                            if ZSTD_AVAILABLE:
                                cctx = zstd.ZstdCompressor(level=6)
                                compressed_data = cctx.compress(cold_data)
                            else:
                                compressed_data = zlib.compress(cold_data, level=6)
                            
                            memory.compressed_size = len(compressed_data)
                            
                            # Store in cold tier
                            txn.put(key, compressed_data, db=self.cold_db)
                            keys_to_delete.append(key)
                            items_moved += 1
                    
                    except Exception as e:
                        logger.warning(f"Failed to process warm tier item: {e}")
                        continue
                
                # Delete from warm tier
                for key in keys_to_delete:
                    txn.delete(key, db=self.warm_db)
            
            self.stats.cold_tier_items += items_moved
            logger.debug(f"🧊 Degraded {items_moved} items to cold tier (Zstd compressed)")
            
        except Exception as e:
            logger.error(f"Warm to cold degradation failed: {e}")
    
    async def _compress_cold_tier(self):
        """Apply additional compression and summarization to cold tier"""
        
        # This is a placeholder for future summarization logic
        # Could integrate with LLM to create summaries of old conversations
        logger.debug("🗜️ Cold tier compression (placeholder for future summarization)")
    
    async def _cleanup_cold_tier(self):
        """Remove least important items from cold tier if over limit"""
        
        if not hasattr(self, 'env'):
            # In-memory cleanup
            for speaker_key in list(self.persistent_memory.keys()):
                if ':cold' in speaker_key:
                    cold_items = self.persistent_memory[speaker_key]
                    if len(cold_items) > self.cold_tier_size:
                        # Sort by importance and keep top items
                        sorted_items = sorted(cold_items, key=lambda x: x.importance_score, reverse=True)
                        self.persistent_memory[speaker_key] = sorted_items[:self.cold_tier_size]
                        removed = len(cold_items) - self.cold_tier_size
                        logger.debug(f"🗑️ Cleaned {removed} least important items from cold tier (in-memory)")
            return
        
        try:
            # Count and clean up cold tier items per speaker
            speaker_counts = {}
            
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self.cold_db)
                for key, value in cursor:
                    try:
                        speaker_id = key.decode().split(':')[0]
                        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
                    except:
                        continue
            
            # Clean up speakers with too many cold items
            for speaker_id, count in speaker_counts.items():
                if count > self.cold_tier_size:
                    await self._cleanup_speaker_cold_items(speaker_id, count)
        
        except Exception as e:
            logger.error(f"Cold tier cleanup failed: {e}")
    
    async def _cleanup_speaker_cold_items(self, speaker_id: str, current_count: int):
        """Clean up cold tier items for specific speaker"""
        
        items_to_remove = current_count - self.cold_tier_size
        
        try:
            with self.env.begin(write=True) as txn:
                cursor = txn.cursor(db=self.cold_db)
                prefix = f"{speaker_id}:".encode()
                
                # Collect items with importance scores
                items = []
                if cursor.set_range(prefix):
                    for key, value in cursor:
                        if not key.startswith(prefix):
                            break
                        
                        try:
                            memory = await self._deserialize_memory_enhanced(value, MemoryTier.COLD)
                            if memory:
                                items.append((key, memory))
                        except:
                            continue
                
                # Sort by importance (lowest first) and remove least important
                items.sort(key=lambda x: x[1].importance_score)
                
                removed = 0
                for key, memory in items[:items_to_remove]:
                    txn.delete(key, db=self.cold_db)
                    removed += 1
                
                logger.debug(f"🗑️ Cleaned {removed} least important cold tier items for {speaker_id}")
        
        except Exception as e:
            logger.error(f"Speaker cold cleanup failed for {speaker_id}: {e}")
    
    async def _update_degradation_stats(self):
        """Update memory statistics after degradation"""
        
        try:
            # Count items in each tier
            self.stats.hot_tier_items = len(self.hot_tier)
            
            if hasattr(self, 'env'):
                with self.env.begin() as txn:
                    # Count warm tier
                    warm_cursor = txn.cursor(db=self.warm_db)
                    self.stats.warm_tier_items = sum(1 for _ in warm_cursor)
                    
                    # Count cold tier
                    cold_cursor = txn.cursor(db=self.cold_db)
                    self.stats.cold_tier_items = sum(1 for _ in cold_cursor)
            else:
                # Count in-memory items
                self.stats.warm_tier_items = sum(
                    len(items) for key, items in self.persistent_memory.items() 
                    if ':warm' in key
                )
                self.stats.cold_tier_items = sum(
                    len(items) for key, items in self.persistent_memory.items() 
                    if ':cold' in key
                )
            
            # Calculate compression ratio
            total_original = sum(item.original_size for item in self.hot_tier)
            total_compressed = sum(item.compressed_size for item in self.hot_tier)
            
            if total_original > 0:
                self.stats.compression_ratio = total_compressed / total_original
            
            # Update retrieval time average
            if self.retrieval_times:
                self.stats.avg_retrieval_time_ms = sum(self.retrieval_times) / len(self.retrieval_times)
        
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    async def _deserialize_memory_enhanced(self, data: bytes, expected_tier: MemoryTier) -> Optional[MemoryItem]:
        """Deserialize memory item with tier-specific decompression"""
        
        try:
            if expected_tier == MemoryTier.WARM and STORAGE_AVAILABLE:
                # LZ4 decompression
                decompressed = lz4.frame.decompress(data)
                memory_dict = json.loads(decompressed)
            elif expected_tier == MemoryTier.COLD:
                # Zstd or gzip decompression
                if ZSTD_AVAILABLE:
                    dctx = zstd.ZstdDecompressor()
                    decompressed = dctx.decompress(data)
                else:
                    decompressed = zlib.decompress(data)
                memory_dict = json.loads(decompressed)
            else:
                # No compression or fallback
                try:
                    memory_dict = json.loads(data.decode())
                except:
                    # Try decompression anyway
                    if STORAGE_AVAILABLE:
                        decompressed = lz4.frame.decompress(data)
                        memory_dict = json.loads(decompressed)
                    else:
                        raise
            
            # Create MemoryItem with proper enum conversion
            if 'tier' in memory_dict and isinstance(memory_dict['tier'], str):
                memory_dict['tier'] = MemoryTier(memory_dict['tier'])
            
            return MemoryItem(**memory_dict)
        
        except Exception as e:
            logger.warning(f"Failed to deserialize enhanced memory: {e}")
            return None
    
    def _build_enhanced_context_string(self, memories: List[MemoryItem]) -> str:
        """Build clean, simple context string without emojis"""
        
        context_parts = []
        seen_content = set()  # Prevent duplicates
        
        for memory in memories:
            # Skip duplicates
            content_key = f"{memory.speaker_id}:{memory.content}"
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            
            # Simple timestamp
            timestamp_str = time.strftime('%H:%M:%S', time.localtime(memory.timestamp))
            
            # Clean format: no emojis, clear speaker labels
            if memory.speaker_id == "assistant":
                context_parts.append(f"[{timestamp_str}] Assistant: {memory.content}")
            elif memory.speaker_id == "default_user":
                context_parts.append(f"[{timestamp_str}] User: {memory.content}")
            else:
                # Named speaker
                context_parts.append(f"[{timestamp_str}] {memory.speaker_id}: {memory.content}")
        
        return "\n".join(context_parts)
    
    async def _load_memories_on_startup(self):
        """Load existing memories from LMDB into hot tier on startup"""
        try:
            if not hasattr(self, 'env'):
                logger.info("🧠 No LMDB environment, skipping memory load")
                return
            
            loaded_count = 0
            with self.env.begin() as txn:
                # Load from warm tier (most recent memories)
                cursor = txn.cursor(db=self.warm_db)
                
                # Get last N items to fill hot tier
                memories_to_load = []
                for key, value in cursor:
                    try:
                        # Decompress if needed
                        if STORAGE_AVAILABLE and lz4:
                            try:
                                memory_data = lz4.frame.decompress(value)
                            except:
                                memory_data = value  # Not compressed
                        else:
                            memory_data = value
                        
                        memory_dict = json.loads(memory_data.decode() if isinstance(memory_data, bytes) else memory_data)
                        
                        # Recreate MemoryItem
                        memory = MemoryItem(
                            content=memory_dict['content'],
                            timestamp=memory_dict['timestamp'],
                            speaker_id=memory_dict['speaker_id'],
                            tier=MemoryTier.HOT,  # Move to hot tier on load
                            access_count=memory_dict.get('access_count', 0),
                            last_accessed=memory_dict.get('last_accessed', 0),
                            importance_score=memory_dict.get('importance_score', 1.0)
                        )
                        memories_to_load.append(memory)
                        
                    except Exception as e:
                        logger.debug(f"Skipping corrupted memory: {e}")
                        continue
                
                # Load most recent memories into hot tier (up to hot_tier_size)
                for memory in memories_to_load[-self.hot_tier_size:]:
                    self.hot_tier.append(memory)
                    loaded_count += 1
            
            if loaded_count > 0:
                logger.info(f"🧠 ENHANCED: Loaded {loaded_count} memories from LMDB into hot tier")
                logger.info(f"🧠 ENHANCED: Hot tier now has {len(self.hot_tier)} items")
            else:
                logger.info("🧠 ENHANCED: No existing memories found in LMDB")
                
        except Exception as e:
            logger.error(f"Failed to load memories on startup: {e}")
    
    async def _log_startup_stats(self):
        """Log enhanced memory statistics on startup"""
        
        try:
            await asyncio.sleep(2)  # Wait for initialization
            
            # Count actual items in LMDB
            warm_count = 0
            cold_count = 0
            
            if hasattr(self, 'env'):
                with self.env.begin() as txn:
                    # Count warm tier
                    cursor = txn.cursor(db=self.warm_db)
                    warm_count = cursor.count()
                    
                    # Count cold tier
                    cursor = txn.cursor(db=self.cold_db)
                    cold_count = cursor.count()
            
            logger.info("🧠 ENHANCED MEMORY STARTUP STATS:")
            logger.info(f"   🔥 Hot tier: {len(self.hot_tier)} items (in memory)")
            logger.info(f"   🌡️ Warm tier: {warm_count} items (in LMDB)") 
            logger.info(f"   🧊 Cold tier: {cold_count} items (in LMDB)")
            logger.info(f"   📊 Total: {len(self.hot_tier) + warm_count + cold_count} items")
            
            if hasattr(self, 'env'):
                logger.info("   💾 Storage: LMDB with three-tier compression")
            else:
                logger.info("   💾 Storage: In-memory fallback")
        
        except Exception as e:
            logger.error(f"Failed to log enhanced startup stats: {e}")
    
    async def update_user_id(self, user_id: str):
        """Update current user ID for memory tracking"""
        self.current_speaker = user_id
        logger.info(f"🧠 Enhanced memory switched to user: {user_id}")
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        return {
            'tier_distribution': {
                'hot': self.stats.hot_tier_items,
                'warm': self.stats.warm_tier_items,
                'cold': self.stats.cold_tier_items,
            },
            'performance': {
                'avg_retrieval_time_ms': self.stats.avg_retrieval_time_ms,
                'degradation_events': self.stats.degradation_events,
                'compression_ratio': self.stats.compression_ratio,
            },
            'storage': {
                'total_items': self.stats.hot_tier_items + self.stats.warm_tier_items + self.stats.cold_tier_items,
                'storage_type': 'LMDB' if hasattr(self, 'env') else 'in-memory',
                'compression_available': STORAGE_AVAILABLE and ZSTD_AVAILABLE,
            }
        }
    
    def _init_bm25_index(self):
        """Initialize persistent BM25 search index"""
        try:
            import bm25s
            
            # BM25 configuration
            self.bm25_index_path = self.db_path / "bm25_index"
            self.bm25_corpus_path = self.db_path / "bm25_corpus.json"
            self.bm25_retriever = None
            self.bm25_corpus = []
            self.bm25_memory_ids = []  # Maps corpus index to memory
            
            # Create index directory
            self.bm25_index_path.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing index
            if self._load_bm25_index():
                logger.info(f"📚 Loaded existing BM25 index with {len(self.bm25_corpus)} documents")
            else:
                # Build fresh index from existing memories
                self._rebuild_bm25_index()
                logger.info(f"🏗️ Built new BM25 index with {len(self.bm25_corpus)} documents")
                
        except ImportError:
            logger.warning("⚠️ BM25s not available - search will use fallback methods")
            self.bm25_retriever = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize BM25 index: {e}")
            self.bm25_retriever = None
    
    def _load_bm25_index(self) -> bool:
        """Load existing BM25 index from disk following BM25s pattern"""
        try:
            import bm25s
            
            # Check if index exists
            if not self.bm25_index_path.exists():
                logger.debug("BM25 index directory doesn't exist")
                return False
            
            # Load BM25 index with corpus (following BM25s pattern)
            self.bm25_retriever = bm25s.BM25.load(str(self.bm25_index_path), load_corpus=True)
            
            # Load memory IDs mapping
            if self.bm25_corpus_path.exists():
                import json
                with open(self.bm25_corpus_path, 'r', encoding='utf-8') as f:
                    corpus_data = json.load(f)
                    self.bm25_memory_ids = corpus_data['memory_ids']
                    logger.debug(f"Loaded {len(self.bm25_memory_ids)} memory ID mappings")
            else:
                logger.warning("BM25 memory ID mapping not found")
                return False
            
            logger.info(f"✅ BM25 index loaded successfully from {self.bm25_index_path}")
            return True
            
        except Exception as e:
            logger.debug(f"Could not load BM25 index: {e}")
            return False
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index following BM25s quickstart pattern"""
        try:
            import bm25s
            import json
            
            # Collect all memories from all tiers
            all_memories = []
            
            # Hot tier (in memory)
            for memory in self.hot_tier:
                all_memories.append(memory)
            
            # Warm and cold tiers (LMDB)
            if hasattr(self, 'env'):
                for tier_name in ['warm', 'cold']:
                    tier_db = getattr(self, f'{tier_name}_db', None)
                    if tier_db:
                        with self.env.begin(db=tier_db) as txn:
                            cursor = txn.cursor()
                            for key, value in cursor:
                                try:
                                    memory = self._quick_deserialize_memory(value)
                                    if memory:
                                        all_memories.append(memory)
                                except:
                                    continue
            
            if not all_memories:
                logger.info("No memories found to index")
                return
            
            # Build corpus and memory mapping following BM25s pattern
            corpus = []
            self.bm25_memory_ids = []
            
            for memory in all_memories:
                corpus.append(memory.content)
                # Store memory identifier (timestamp + speaker)  
                memory_id = f"{memory.speaker_id}:{memory.timestamp}"
                self.bm25_memory_ids.append(memory_id)
            
            # Create stemmer for better search quality (optional but recommended)
            stemmer = None
            try:
                import Stemmer
                stemmer = Stemmer.Stemmer("english")
                logger.debug("Using English stemmer for BM25")
            except ImportError:
                logger.debug("Stemmer not available, using basic tokenization")
            
            # Tokenize corpus following BM25s pattern
            corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer, show_progress=False)
            
            # Create and index BM25 following BM25s pattern
            self.bm25_retriever = bm25s.BM25()
            self.bm25_retriever.index(corpus_tokens)
            
            # Save index with corpus following BM25s pattern
            self.bm25_retriever.save(str(self.bm25_index_path), corpus=corpus)
            
            # Save memory ID mapping separately
            corpus_data = {
                'memory_ids': self.bm25_memory_ids
            }
            
            with open(self.bm25_corpus_path, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ BM25 index built and saved with {len(corpus)} documents using BM25s pattern")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild BM25 index: {e}")
            logger.error(f"   Error details: {str(e)}")
            self.bm25_retriever = None
    
    def _update_bm25_index(self, new_memory: MemoryItem):
        """Incrementally update BM25 index with new memory"""
        try:
            if not self.bm25_retriever:
                return
            
            # Add to corpus
            self.bm25_corpus.append(new_memory.content)
            memory_id = f"{new_memory.speaker_id}:{new_memory.timestamp}"
            self.bm25_memory_ids.append(memory_id)
            
            # For now, rebuild index (TODO: implement incremental update)
            # BM25s doesn't support true incremental updates yet
            self._rebuild_bm25_index()
            
        except Exception as e:
            logger.debug(f"Failed to update BM25 index: {e}")
    
    def search_memories_bm25(self, query: str, max_results: int = 5) -> List[MemoryItem]:
        """Search memories using persistent BM25 index following BM25s pattern"""
        if not self.bm25_retriever:
            logger.debug("BM25 index not available, falling back to tier search")
            return []
        
        try:
            import bm25s
            
            # Create stemmer (same as used for indexing)
            stemmer = None
            try:
                import Stemmer
                stemmer = Stemmer.Stemmer("english")
            except ImportError:
                pass
            
            # Tokenize query following BM25s pattern
            query_tokens = bm25s.tokenize([query], stemmer=stemmer, show_progress=False)
            
            # Search with BM25 following BM25s pattern
            results, scores = self.bm25_retriever.retrieve(
                query_tokens, 
                k=min(max_results, len(self.bm25_memory_ids)),
                corpus=self.bm25_retriever.corpus  # Use corpus from loaded index
            )
            
            # Convert results back to MemoryItem objects
            found_memories = []
            
            for idx, score in zip(results[0], scores[0]):
                if score > 0.01:  # Lower threshold for better recall
                    try:
                        memory_id = self.bm25_memory_ids[idx]
                        content = self.bm25_retriever.corpus[idx]  # Get content from loaded corpus
                        
                        # Parse memory_id to get speaker and timestamp
                        speaker_id, timestamp_str = memory_id.split(':', 1)
                        timestamp = float(timestamp_str)
                        
                        # Create memory item
                        memory = MemoryItem(
                            content=content,
                            timestamp=timestamp,
                            speaker_id=speaker_id,
                            tier=MemoryTier.WARM,
                            importance_score=float(score)
                        )
                        
                        found_memories.append(memory)
                        logger.debug(f"BM25 found: {content[:60]}... (score: {score:.3f})")
                        
                    except (IndexError, ValueError) as e:
                        logger.debug(f"Error processing result {idx}: {e}")
                        continue
            
            logger.info(f"🎯 BM25 search found {len(found_memories)} results for '{query[:30]}...'")
            return found_memories
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _quick_deserialize_memory(self, compressed_data: bytes) -> Optional[MemoryItem]:
        """Quick memory deserialization for BM25 indexing"""
        try:
            import json
            
            # Try LZ4 decompression first
            try:
                import lz4.frame
                data = lz4.frame.decompress(compressed_data)
            except:
                try:
                    import gzip
                    data = gzip.decompress(compressed_data)
                except:
                    data = compressed_data
            
            # Decode to string
            try:
                json_str = data.decode('utf-8')
            except UnicodeDecodeError:
                json_str = data.decode('latin-1', errors='ignore')
            
            # Parse JSON
            memory_dict = json.loads(json_str)
            
            return MemoryItem(
                content=memory_dict['content'],
                timestamp=memory_dict['timestamp'],
                speaker_id=memory_dict['speaker_id'],
                tier=MemoryTier(memory_dict.get('tier', 'warm')),
                access_count=memory_dict.get('access_count', 0),
                last_accessed=memory_dict.get('last_accessed', 0),
                importance_score=memory_dict.get('importance_score', 1.0)
            )
            
        except Exception:
            return None