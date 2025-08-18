"""
Fixed StatelessMemoryProcessor that follows Pipecat patterns exactly

This version addresses the common frame processing issues:
1. Proper StartFrame handling
2. Correct parent method calls
3. Frame forwarding for all frame types
4. Error handling that doesn't break the pipeline
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque

from pipecat.frames.frames import (
    Frame, StartFrame, EndFrame, TextFrame, LLMMessagesFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger

# Import the original classes we need
from processors.stateless_memory import MemoryItem, MemoryDegradation, SemanticValidator

try:
    import lmdb
    import lz4.frame
    STORAGE_AVAILABLE = True
except ImportError:
    logger.warning("LMDB or LZ4 not available, using in-memory storage only")
    STORAGE_AVAILABLE = False


class FixedStatelessMemoryProcessor(FrameProcessor):
    """
    Fixed StatelessMemoryProcessor that follows Pipecat patterns exactly
    
    Key fixes:
    1. Proper StartFrame handling with immediate forwarding
    2. All parent method calls in correct order
    3. Frame forwarding for every frame type
    4. Error isolation that doesn't break pipeline
    5. Simplified processing logic
    """
    
    def __init__(self, 
                 db_path: str = "data/stateless_memory",
                 max_context_tokens: int = 1024,
                 perfect_recall_window: int = 10,
                 **kwargs):
        # CRITICAL: Always call parent init first
        super().__init__(**kwargs)
        
        # Configuration
        self.max_context_tokens = max_context_tokens
        self.perfect_recall_window = perfect_recall_window
        
        # Simple in-memory storage (fallback if LMDB fails)
        self.perfect_recall_cache = deque(maxlen=perfect_recall_window)
        self.persistent_memory = {}  # speaker_id -> list of memories
        
        # Current state
        self.current_speaker = "unknown"
        self.current_user_message = ""
        
        # Performance metrics
        self.total_conversations = 0
        self.injection_times = deque(maxlen=100)
        
        # Initialize storage (don't fail if LMDB unavailable)
        self._init_storage(db_path)
        
        logger.info(f"Fixed stateless memory initialized")
        logger.info(f"Storage: {'LMDB' if STORAGE_AVAILABLE and hasattr(self, 'env') else 'in-memory'}")
    
    def _init_storage(self, db_path: str):
        """Initialize storage with graceful fallback"""
        if not STORAGE_AVAILABLE:
            logger.info("Using in-memory storage (LMDB not available)")
            return
        
        try:
            Path(db_path).mkdir(parents=True, exist_ok=True)
            
            self.env = lmdb.open(
                db_path,
                map_size=512*1024*1024,  # 512MB (smaller for testing)
                max_dbs=2,
                writemap=True,
                metasync=False,
                sync=False
            )
            
            with self.env.begin(write=True) as txn:
                self.memory_db = self.env.open_db(b'memory', txn=txn)
            
            logger.info(f"LMDB storage initialized at {db_path}")
            
        except Exception as e:
            logger.warning(f"LMDB initialization failed, using in-memory: {e}")
            # Don't fail - just use in-memory storage
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process frames with proper Pipecat patterns
        
        CRITICAL PATTERN:
        1. Always call parent first
        2. Handle StartFrame specially (forward immediately)
        3. Process other frames
        4. Always forward frames at the end
        """
        
        # STEP 1: Always call parent first (required for initialization)
        await super().process_frame(frame, direction)
        
        # STEP 2: Handle StartFrame specially
        if isinstance(frame, StartFrame):
            # Push StartFrame downstream IMMEDIATELY
            await self.push_frame(frame, direction)
            # Initialize processor-specific logic here if needed
            logger.debug("StatelessMemory: StartFrame processed and forwarded")
            return
        
        # STEP 3: Handle EndFrame
        if isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)
            logger.debug("StatelessMemory: EndFrame processed and forwarded")
            return
        
        # STEP 4: Process other frames with error isolation
        try:
            await self._safe_process_frame(frame, direction)
        except Exception as e:
            # CRITICAL: Never let memory errors break the pipeline
            logger.error(f"Memory processing error (isolated): {e}")
            # Continue processing - don't re-raise
        
        # STEP 5: Always forward frames (most important!)
        await self.push_frame(frame, direction)
    
    async def _safe_process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with error isolation"""
        
        # Track speaker changes
        if isinstance(frame, UserStartedSpeakingFrame):
            self.current_speaker = getattr(frame, 'speaker_id', 'unknown')
            logger.debug(f"Speaker started: {self.current_speaker}")
            return
        
        # Capture user input from transcription
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.UPSTREAM:
            if frame.text and frame.text.strip():
                self.current_user_message = frame.text.strip()
                logger.debug(f"Captured user message: {self.current_user_message[:50]}...")
            return
        
        # Handle LLM messages for context injection
        if isinstance(frame, LLMMessagesFrame):
            if direction == FrameDirection.UPSTREAM:
                # Inject memory context before LLM
                await self._inject_memory_context(frame)
                
            elif direction == FrameDirection.DOWNSTREAM:
                # Extract and store assistant response
                await self._extract_and_store_response(frame)
            return
        
        # Handle text frames (alternative path for responses)
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.text and frame.text.strip() and self.current_user_message:
                await self._store_exchange(self.current_user_message, frame.text.strip())
                self.current_user_message = ""
            return
    
    async def _inject_memory_context(self, frame: LLMMessagesFrame):
        """Inject memory context into LLM messages"""
        
        start_time = time.perf_counter()
        
        try:
            # Extract user message if not already captured
            if not self.current_user_message:
                for msg in reversed(frame.messages):
                    if msg.get('role') == 'user':
                        self.current_user_message = msg.get('content', '').strip()
                        break
            
            # Get relevant memories
            memories = await self._get_relevant_memories(
                self.current_user_message,
                self.current_speaker
            )
            
            # Inject context if we have memories
            if memories:
                context = self._build_context_string(memories)
                
                # Find injection point (after system message if present)
                injection_point = 1 if frame.messages and frame.messages[0].get('role') == 'system' else 0
                
                memory_message = {
                    'role': 'system',
                    'content': f"[Memory Context - {len(memories)} items]:\n{context}"
                }
                
                frame.messages.insert(injection_point, memory_message)
                
                logger.debug(f"Injected {len(memories)} memories")
        
        except Exception as e:
            logger.error(f"Memory injection failed: {e}")
        
        finally:
            # Track performance
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.injection_times.append(elapsed_ms)
            
            if elapsed_ms > 10:
                logger.warning(f"Slow memory injection: {elapsed_ms:.2f}ms")
    
    async def _extract_and_store_response(self, frame: LLMMessagesFrame):
        """Extract assistant response and store exchange"""
        
        try:
            # Find assistant message
            for msg in reversed(frame.messages):
                if msg.get('role') == 'assistant':
                    assistant_response = msg.get('content', '').strip()
                    if assistant_response and self.current_user_message:
                        # Store exchange
                        await self._store_exchange(
                            self.current_user_message,
                            assistant_response
                        )
                        # Clear for next exchange
                        self.current_user_message = ""
                    break
        
        except Exception as e:
            logger.error(f"Failed to extract response: {e}")
    
    async def _get_relevant_memories(self, query: str, speaker_id: str) -> List[MemoryItem]:
        """Get relevant memories (simplified version)"""
        
        memories = []
        token_count = 0
        max_tokens = self.max_context_tokens // 2  # Use half for memory
        
        try:
            # 1. Get from perfect recall cache (most recent)
            for item in reversed(self.perfect_recall_cache):
                if token_count >= max_tokens:
                    break
                
                if isinstance(item, MemoryItem) and item.speaker_id == speaker_id:
                    item_tokens = len(item.content.split()) * 1.3
                    if token_count + item_tokens <= max_tokens:
                        memories.append(item)
                        token_count += item_tokens
            
            # 2. Get from persistent storage if needed
            remaining_tokens = max_tokens - token_count
            if remaining_tokens > 100:
                stored_memories = await self._search_persistent_memory(
                    query, speaker_id, remaining_tokens
                )
                memories.extend(stored_memories)
        
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
        
        return memories
    
    async def _search_persistent_memory(self, query: str, speaker_id: str, max_tokens: int) -> List[MemoryItem]:
        """Search persistent memory storage"""
        
        memories = []
        
        try:
            # Use in-memory storage if LMDB not available
            if not hasattr(self, 'env'):
                speaker_memories = self.persistent_memory.get(speaker_id, [])
                # Simple keyword matching
                query_words = set(query.lower().split())
                
                for memory in reversed(speaker_memories[-20:]):  # Check last 20
                    memory_words = set(memory.content.lower().split())
                    if query_words.intersection(memory_words):
                        memories.append(memory)
                        if len(memories) >= 5:
                            break
                
                return memories
            
            # Use LMDB storage
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self.memory_db)
                
                # Look for speaker's memories
                prefix = f"{speaker_id}:".encode()
                if cursor.set_range(prefix):
                    items_checked = 0
                    while items_checked < 20:  # Limit search scope
                        key, value = cursor.item()
                        
                        if not key.startswith(prefix):
                            break
                        
                        memory = self._deserialize_memory(value)
                        if memory and self._is_relevant(memory.content, query):
                            memories.append(memory)
                            if len(memories) >= 5:
                                break
                        
                        if not cursor.next():
                            break
                        items_checked += 1
        
        except Exception as e:
            logger.error(f"Persistent memory search failed: {e}")
        
        return memories
    
    def _is_relevant(self, memory_content: str, query: str) -> bool:
        """Simple relevance check using keyword matching"""
        memory_words = set(memory_content.lower().split())
        query_words = set(query.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_keywords = query_words - stop_words
        
        # Check for intersection
        return bool(query_keywords.intersection(memory_words))
    
    async def _store_exchange(self, user_message: str, assistant_response: str):
        """Store conversation exchange"""
        
        try:
            timestamp = time.time()
            
            # Create memory items
            user_memory = MemoryItem(
                content=user_message,
                timestamp=timestamp,
                speaker_id=self.current_speaker
            )
            
            assistant_memory = MemoryItem(
                content=assistant_response,
                timestamp=timestamp + 0.001,
                speaker_id="assistant"
            )
            
            # Add to cache
            self.perfect_recall_cache.extend([user_memory, assistant_memory])
            
            # Store persistently
            await self._store_persistent([user_memory, assistant_memory])
            
            self.total_conversations += 1
            
            logger.debug(f"Stored exchange: {user_message[:30]}... -> {assistant_response[:30]}...")
        
        except Exception as e:
            logger.error(f"Failed to store exchange: {e}")
    
    async def _store_persistent(self, memories: List[MemoryItem]):
        """Store memories persistently"""
        
        try:
            # Use in-memory storage if LMDB not available
            if not hasattr(self, 'env'):
                for memory in memories:
                    if memory.speaker_id not in self.persistent_memory:
                        self.persistent_memory[memory.speaker_id] = []
                    self.persistent_memory[memory.speaker_id].append(memory)
                    
                    # Keep only last 100 per speaker
                    if len(self.persistent_memory[memory.speaker_id]) > 100:
                        self.persistent_memory[memory.speaker_id] = \
                            self.persistent_memory[memory.speaker_id][-100:]
                return
            
            # Use LMDB storage
            with self.env.begin(write=True) as txn:
                for memory in memories:
                    key = f"{memory.speaker_id}:{memory.timestamp}".encode()
                    data = json.dumps(asdict(memory)).encode()
                    
                    # Compress if large
                    if STORAGE_AVAILABLE and len(data) > 200:
                        data = lz4.frame.compress(data)
                    
                    txn.put(key, data, db=self.memory_db)
        
        except Exception as e:
            logger.error(f"Persistent storage failed: {e}")
    
    def _build_context_string(self, memories: List[MemoryItem]) -> str:
        """Build context string from memories"""
        
        context_parts = []
        
        for memory in memories:
            timestamp_str = time.strftime(
                '%H:%M:%S',
                time.localtime(memory.timestamp)
            )
            context_parts.append(f"[{timestamp_str}] {memory.content}")
        
        return "\n".join(context_parts)
    
    def _deserialize_memory(self, data: bytes) -> Optional[MemoryItem]:
        """Deserialize memory from storage"""
        
        try:
            # Try decompressing first
            try:
                if STORAGE_AVAILABLE:
                    decompressed = lz4.frame.decompress(data)
                    memory_dict = json.loads(decompressed)
                else:
                    memory_dict = json.loads(data.decode())
            except:
                # Not compressed
                memory_dict = json.loads(data.decode())
            
            return MemoryItem(**memory_dict)
        
        except Exception as e:
            logger.warning(f"Failed to deserialize memory: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        avg_injection_time = (
            sum(self.injection_times) / len(self.injection_times)
            if self.injection_times else 0
        )
        
        return {
            'total_conversations': self.total_conversations,
            'cache_size': len(self.perfect_recall_cache),
            'avg_injection_time_ms': avg_injection_time,
            'max_injection_time_ms': max(self.injection_times) if self.injection_times else 0,
            'storage_type': 'LMDB' if hasattr(self, 'env') else 'in-memory'
        }