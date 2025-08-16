import asyncio
import os
import sys
import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import re

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    LLMMessagesFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger

try:
    from memobase import MemoBaseClient
    from memobase.patch.openai import openai_memory
    from openai import OpenAI
    MEMOBASE_AVAILABLE = True
except ImportError:
    logger.warning("MemoBase not available. Install with: pip install memobase")
    MEMOBASE_AVAILABLE = False


class MemobaseMemoryProcessor(FrameProcessor):
    """
    MemoBase memory processor with hybrid sync/async approach:
    - Uses sync OpenAI client with MemoBase patching for memory storage/retrieval
    - Integrates with async Pipecat pipeline without blocking
    - Supports user identification and session management
    - Falls back gracefully if MemoBase unavailable
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        max_context_size: Optional[int] = None,
        flush_on_session_end: bool = True,
        fallback_to_local: bool = True
    ):
        super().__init__()
        
        # Configuration from config or parameters
        self.user_id = user_id or config.memory.default_user_id
        self.max_context_size = max_context_size or config.memobase.max_context_size
        self.max_token_limit = config.memobase.max_token_limit
        self.enable_compression = config.memobase.enable_compression
        self.compression_ratio = config.memobase.compression_ratio
        self.auto_flush_threshold = config.memobase.auto_flush_threshold
        self.enable_relevance_filtering = config.memobase.enable_relevance_filtering
        self.relevance_threshold = config.memobase.relevance_threshold
        self.flush_on_session_end = flush_on_session_end or config.memobase.flush_on_session_end
        self.fallback_to_local = fallback_to_local or config.memobase.fallback_to_local
        
        # Session tracking
        self._session_active = False
        self._conversation_buffer: List[Dict[str, str]] = []
        self._buffer_token_count = 0
        self._lock = asyncio.Lock()
        
        # Redis-based memory caching and injection tracking
        self._redis_client = None
        self._cache_ttl = 300  # 5 minutes TTL for memory cache
        self._injection_cache = {}  # Fallback for when Redis unavailable
        
        # Initialize Redis connection if available
        if REDIS_AVAILABLE:
            try:
                self._redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                logger.info("🔗 Redis connection initialized for MemoBase caching")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed, using in-memory cache: {e}")
                self._redis_client = None
        else:
            logger.warning("⚠️ Redis not available, using in-memory cache")
        
        # MemoBase clients - hybrid approach
        self.mb_client: Optional[Any] = None
        self.sync_openai_client: Optional[Any] = None
        self.patched_sync_client: Optional[Any] = None
        self.is_enabled = False
        
        # Initialize MemoBase if available and enabled
        if MEMOBASE_AVAILABLE and config.memobase.enabled:
            self._initialize_memobase_hybrid()
        else:
            logger.info("🧠 MemoBase memory processor disabled or unavailable")

    def _initialize_memobase_hybrid(self):
        """Initialize MemoBase with flexible provider architecture."""
        try:
            # Create MemoBase client (using keyword arguments for local connection)
            self.mb_client = MemoBaseClient(
                project_url=config.memobase.project_url,
                api_key=config.memobase.api_key
            )
            
            # Test connection
            if self.mb_client.ping():
                # Determine which client configuration to use
                use_separate_memory_llm = (
                    config.memobase.memory_llm.base_url != config.network.llm_base_url or
                    config.memobase.memory_llm.model != config.models.default_llm_model
                )
                
                if use_separate_memory_llm:
                    # Use separate Memory LLM provider (Option 1: Different provider for memory)
                    logger.info(f"🔄 Using separate Memory LLM: {config.memobase.memory_llm.provider_name}")
                    self.sync_openai_client = OpenAI(
                        api_key=config.memobase.memory_llm.api_key,
                        base_url=config.memobase.memory_llm.base_url
                    )
                    # Set the model for memory operations
                    self.memory_model = config.memobase.memory_llm.model
                    logger.info(f"   📍 Memory LLM: {config.memobase.memory_llm.base_url}")
                    logger.info(f"   🤖 Memory Model: {self.memory_model}")
                else:
                    # Use same LLM as main conversation (Option 2: Simple setup)
                    logger.info(f"🔄 Using main LLM for memory: {config.network.llm_base_url}")
                    self.sync_openai_client = OpenAI(
                        api_key=config.memobase.main_llm.api_key,
                        base_url=config.network.llm_base_url
                    )
                    self.memory_model = config.models.default_llm_model
                
                # Log embedding provider info
                logger.info(f"🔍 Embedding provider: {config.memobase.embedding.provider_name}")
                logger.info(f"   📍 Embedding URL: {config.memobase.embedding.base_url}")
                logger.info(f"   🧲 Embedding Model: {config.memobase.embedding.model}")
                
                # Apply MemoBase patching to the sync client with smart limits
                self.patched_sync_client = openai_memory(
                    self.sync_openai_client, 
                    self.mb_client,
                    max_context_size=min(self.max_context_size, self.max_token_limit)
                )
                
                self.is_enabled = True
                logger.info(f"🧠 MemoBase multi-provider architecture initialized")
                logger.info(f"🔧 Main LLM: {config.network.llm_base_url} | Memory LLM: {config.memobase.memory_llm.base_url}")
                logger.info(f"🎛️ Context limits: max_size={self.max_context_size}, token_limit={self.max_token_limit}, compression={self.enable_compression}")
                
                # Debug: Check actual UUID mapping
                from memobase.utils import string_to_uuid
                uuid_for_user = string_to_uuid(self.user_id)
                logger.info(f"🔍 MemoBase user_id '{self.user_id}' maps to UUID: {uuid_for_user}")
            else:
                logger.error("❌ MemoBase connection failed")
                self._handle_fallback()
                
        except Exception as e:
            logger.error(f"❌ MemoBase initialization failed: {e}")
            self._handle_fallback()

    def _handle_fallback(self):
        """Handle fallback when MemoBase is unavailable."""
        if self.fallback_to_local:
            logger.warning("⚠️ Falling back to local memory mode")
            self.is_enabled = False
        else:
            logger.error("❌ MemoBase required but unavailable")
            raise RuntimeError("MemoBase memory service unavailable")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)"""
        return len(text) // 4
    
    def _generate_cache_key(self, user_message: str, cache_type: str = "memory") -> str:
        """Generate cache key for Redis storage"""
        # Skip caching for very short or unclear queries
        if len(user_message.strip()) <= 3 or user_message.strip() in ['?', '??', '...', 'hmm', 'ok']:
            # Return a unique key that won't match anything to force fresh retrieval
            return f"memobase:{cache_type}:{self.user_id}:no_cache_{int(time.time())}"
        
        message_hash = hashlib.md5(user_message.encode()).hexdigest()[:12]
        return f"memobase:{cache_type}:{self.user_id}:{message_hash}"
    
    def _generate_injection_key(self) -> str:
        """Generate injection tracking key for Redis"""
        return f"memobase:injection:{self.user_id}"
    
    async def _get_cached_memory(self, user_message: str) -> Optional[str]:
        """Get cached memory context from Redis"""
        if not self._redis_client:
            # Fallback to in-memory cache
            cache_key = self._generate_cache_key(user_message)
            return self._injection_cache.get(cache_key)
        
        try:
            cache_key = self._generate_cache_key(user_message)
            cached_data = await self._redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                logger.debug(f"🎯 Cache hit for memory context: {cache_key}")
                return data.get('memory_context')
        except Exception as e:
            logger.warning(f"⚠️ Redis cache read failed: {e}")
        
        return None
    
    async def _cache_memory(self, user_message: str, memory_context: str):
        """Cache memory context in Redis with TTL"""
        if not self._redis_client:
            # Fallback to in-memory cache (limited size)
            cache_key = self._generate_cache_key(user_message)
            self._injection_cache[cache_key] = memory_context
            # Keep only last 10 entries to prevent memory bloat
            if len(self._injection_cache) > 10:
                oldest_key = next(iter(self._injection_cache))
                del self._injection_cache[oldest_key]
            return
        
        try:
            cache_key = self._generate_cache_key(user_message)
            cache_data = {
                'memory_context': memory_context,
                'timestamp': time.time(),
                'user_id': self.user_id
            }
            await self._redis_client.setex(cache_key, self._cache_ttl, json.dumps(cache_data))
            logger.debug(f"💾 Cached memory context: {cache_key}")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache write failed: {e}")
    
    async def _should_inject_memory(self, user_message: str) -> bool:
        """Check if memory should be injected (prevent duplicates)"""
        if not self._redis_client:
            # Simple in-memory duplicate prevention
            return user_message != getattr(self, '_last_user_message', None)
        
        try:
            injection_key = self._generate_injection_key()
            last_injection_data = await self._redis_client.get(injection_key)
            
            if last_injection_data:
                data = json.loads(last_injection_data)
                last_message = data.get('last_message')
                last_time = data.get('timestamp', 0)
                
                # Don't inject if same message within 30 seconds
                if last_message == user_message and (time.time() - last_time) < 30:
                    logger.debug(f"🚫 Skipping duplicate memory injection for: {user_message[:30]}...")
                    return False
            
            # Update injection tracking
            injection_data = {
                'last_message': user_message,
                'timestamp': time.time(),
                'user_id': self.user_id
            }
            await self._redis_client.setex(injection_key, 60, json.dumps(injection_data))  # 1 minute TTL
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Redis injection tracking failed: {e}")
            return True  # Default to allowing injection if Redis fails
    
    async def _check_and_flush_buffer(self):
        """Check if buffer needs flushing based on token count"""
        if self._buffer_token_count >= self.auto_flush_threshold:
            logger.info(f"🔄 Auto-flushing MemoBase buffer ({self._buffer_token_count} tokens)")
            await self.flush_memory()
    
    async def _add_to_conversation_buffer(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add conversation to buffer and store in MemoBase with smart token management."""
        if not self.is_enabled or not self.patched_sync_client:
            logger.debug(f"🔄 MemoBase disabled, skipping: {content[:50]}...")
            return
            
        # Estimate token cost
        content_tokens = self._estimate_tokens(content)
        
        async with self._lock:
            self._conversation_buffer.append({
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
                "tokens": content_tokens
            })
            self._buffer_token_count += content_tokens
            
        # Auto-flush if buffer is getting large (async, non-blocking)
        asyncio.create_task(self._check_and_flush_buffer())
            
        # Store individual messages in MemoBase immediately (BLOCKING to ensure storage)
        try:
            await self._store_message_in_memobase(role, content)
        except Exception as e:
            logger.error(f"❌ Critical: Failed to store message in MemoBase: {e}")
            # Continue execution even if storage fails
            
        logger.info(f"🧠 Added to MemoBase: {role} - {content[:50]}... ({content_tokens} tokens, buffer: {self._buffer_token_count})")

    async def _store_message_in_memobase(self, role: str, content: str):
        """Store a single message in MemoBase using the patched client - BLOCKING to ensure storage."""
        if not self.patched_sync_client:
            logger.warning(f"⚠️ No patched sync client available for storing {role} message")
            return
            
        try:
            # Create a minimal conversation with just the new message
            # The patched client expects a conversation format
            if role == "user":
                # For user messages, create a minimal conversation that triggers storage
                messages = [
                    {"role": "user", "content": content}
                ]
                
                # Use the patched client to make a minimal call that triggers MemoBase storage
                logger.info(f"🔍 Storing {role} message with user_id: {self.user_id}")
                
                try:
                    # CRITICAL FIX: Make this blocking to ensure storage happens
                    await asyncio.to_thread(
                        self.patched_sync_client.chat.completions.create,
                        model=getattr(self, 'memory_model', config.models.default_llm_model),
                        messages=messages,
                        user_id=self.user_id,
                        max_tokens=1,  # Minimal response 
                        temperature=0.0  # Deterministic
                    )
                    logger.info(f"✅ Successfully stored {role} message in MemoBase: {content[:30]}...")
                except Exception as e:
                    logger.error(f"❌ Failed to store {role} message in MemoBase: {e}")
                    logger.error(f"   Model: {getattr(self, 'memory_model', 'unknown')}")
                    logger.error(f"   User ID: {self.user_id}")
                    logger.error(f"   Error details: {str(e)}")
                    raise
            
            # For assistant messages, we'll use a different approach 
            # since MemoBase expects user+assistant pairs
            elif role == "assistant":
                # Only store if we have a recent user message to pair with
                recent_user_messages = [msg for msg in self._conversation_buffer[-5:] if msg["role"] == "user"]
                if recent_user_messages:
                    user_content = recent_user_messages[-1]["content"]
                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": content}
                    ]
                    
                    logger.info(f"🔍 Storing conversation pair with user_id: {self.user_id}")
                    try:
                        # CRITICAL FIX: Make this blocking to ensure storage happens
                        await asyncio.to_thread(
                            self.patched_sync_client.chat.completions.create,
                            model=getattr(self, 'memory_model', config.models.default_llm_model),
                            messages=messages,
                            user_id=self.user_id,
                            max_tokens=1,
                            temperature=0.0
                        )
                        logger.info(f"✅ Successfully stored conversation pair in MemoBase")
                    except Exception as e:
                        logger.error(f"❌ Failed to store conversation pair in MemoBase: {e}")
                        logger.error(f"   Model: {getattr(self, 'memory_model', 'unknown')}")
                        logger.error(f"   User ID: {self.user_id}")
                        logger.error(f"   Error details: {str(e)}")
                        raise
                else:
                    logger.warning(f"⚠️ No recent user message found to pair with assistant message")
                    
        except Exception as e:
            logger.error(f"❌ Failed to store {role} message in MemoBase: {e}")
            # Don't re-raise here to avoid breaking the pipeline

    def _deduplicate_memory_content(self, memory_prompt: str) -> str:
        """Remove duplicate content blocks from memory while preserving user aliases"""
        if not memory_prompt or not memory_prompt.strip():
            return memory_prompt
            
        lines = memory_prompt.split('\n')
        seen_content = set()
        deduplicated_lines = []
        
        # Track different types of content
        user_alias_patterns = [r'User\'s alias:', r'User alias:', r'Speaker:', r'Name:']
        important_patterns = [r'important', r'preference', r'name', r'alias']
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                deduplicated_lines.append(line)
                continue
                
            # Always preserve user alias information (critical for voice recognition)
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in user_alias_patterns):
                if line not in deduplicated_lines:  # Avoid exact duplicates
                    deduplicated_lines.append(line)
                    logger.debug(f"🏷️ Preserving user alias: {line[:50]}...")
                continue
                
            # For other content, check for semantic duplicates
            content_hash = hashlib.md5(line_stripped.lower().encode()).hexdigest()
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated_lines.append(line)
            else:
                logger.debug(f"🗑️ Removing duplicate content: {line[:50]}...")
        
        deduplicated = '\n'.join(deduplicated_lines)
        original_tokens = self._estimate_tokens(memory_prompt)
        deduplicated_tokens = self._estimate_tokens(deduplicated)
        
        if deduplicated_tokens < original_tokens:
            logger.info(f"🔄 Deduplicated memory: {original_tokens} → {deduplicated_tokens} tokens ({original_tokens - deduplicated_tokens} saved)")
        
        return deduplicated

    def _compress_memory_content(self, memory_prompt: str) -> str:
        """Compress memory content when it exceeds token limits"""
        # First apply deduplication
        memory_prompt = self._deduplicate_memory_content(memory_prompt)
        
        if not self.enable_compression:
            return memory_prompt
            
        current_tokens = self._estimate_tokens(memory_prompt)
        if current_tokens <= self.max_token_limit:
            return memory_prompt
            
        target_tokens = int(self.max_token_limit * self.compression_ratio)
        target_chars = target_tokens * 4  # Rough conversion
        
        # Simple compression: keep most recent and most relevant parts
        lines = memory_prompt.split('\n')
        
        # Keep header and summary lines
        compressed_lines = []
        char_count = 0
        
        # Prioritize lines with recent dates, important keywords, and user aliases
        priority_patterns = [r'2025', r'today', r'recently', r'important', r'name', r'preference', r'alias', r'speaker']
        
        # First pass: high priority lines (including user aliases)
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in priority_patterns):
                if char_count + len(line) <= target_chars:
                    compressed_lines.append(line)
                    char_count += len(line)
        
        # Second pass: fill remaining space with other lines
        for line in lines:
            if line not in compressed_lines and char_count + len(line) <= target_chars:
                compressed_lines.append(line)
                char_count += len(line)
                
        compressed = '\n'.join(compressed_lines)
        logger.info(f"🗜️ Compressed memory from {current_tokens} to {self._estimate_tokens(compressed)} tokens")
        return compressed
    
    def _fix_memory_instructions(self, memory_content: str) -> str:
        """Fix memory instructions to be helpful instead of restrictive"""
        return memory_content.replace(
            "Unless the user has relevant queries, do not actively mention those memories in the conversation.",
            "When users ask about information in your memory, provide helpful answers using that information."
        )
    
    async def _get_contextual_memory(self, user_message: str) -> Optional[str]:
        """Get contextual memory using MemoBase's User.context() API with proper temporal filtering"""
        try:
            # Import MemoBase utilities
            from memobase.utils import string_to_uuid
            from memobase import MemoBaseClient
            
            # Get MemoBase client and user
            mb_client = MemoBaseClient(
                project_url=config.memobase.project_url,
                api_key=config.memobase.api_key
            )
            
            uuid_for_user = string_to_uuid(self.user_id)
            user = mb_client.get_user(uuid_for_user, no_get=True)
            
            # CRITICAL DEBUG: Log the actual user_id being used
            logger.error(f"🔍 CRITICAL DEBUG - Using user_id: '{self.user_id}' -> UUID: {uuid_for_user}")
            logger.error(f"🔍 Expected UUID for 'default_user': {string_to_uuid('default_user')}")
            
            # Use the proper context() API with temporal-friendly parameters
            memory_context = await asyncio.to_thread(
                user.context,
                max_token_size=min(self.max_context_size, self.max_token_limit),
                chats=[{"role": "user", "content": user_message}],  # Context-aware retrieval
                event_similarity_threshold=0.1,  # Very low threshold for maximum recall
                fill_window_with_events=True,  # Fill remaining space with events
                profile_event_ratio=0.9,  # 90% profile, 10% events - prioritize profile data
                require_event_summary=False  # Don't require event summaries to save space
            )
            
            logger.error(f"🧠 Retrieved memory length: {len(memory_context)} chars")
            logger.error(f"🐕 Contains dog info: {'Bobby' in memory_context.lower() or 'dog' in memory_context.lower()}")
            return memory_context
            
        except Exception as e:
            logger.warning(f"⚠️ User.context() API failed, falling back to patched client: {e}")
            
            # Fallback to patched client method
            if not self.patched_sync_client:
                return None
                
            return await asyncio.to_thread(
                self.patched_sync_client.get_memory_prompt,
                self.user_id
            )
    
    async def _inject_memory_context(self, context, user_message: str):
        """Retrieve relevant memories from MemoBase and inject into context with smart compression and Redis caching."""
        if not self.is_enabled or not self.patched_sync_client:
            return
        
        # Check if we should inject memory (prevent duplicates)
        if not await self._should_inject_memory(user_message):
            return
            
        try:
            # Try to get cached memory first
            cached_memory = await self._get_cached_memory(user_message)
            
            if cached_memory:
                logger.debug(f"🎯 Using cached memory for: {user_message[:30]}...")
                compressed_memory = cached_memory
            else:
                # Get fresh memory from MemoBase
                logger.debug(f"🔍 Retrieving fresh memory for user_id: {self.user_id}")
                memory_prompt = await self._get_contextual_memory(user_message)
                logger.debug(f"🔍 Raw memory prompt: {repr(memory_prompt)[:200]}...")
                
                if not memory_prompt or not memory_prompt.strip():
                    return
                
                # Apply compression if needed
                compressed_memory = self._compress_memory_content(memory_prompt)
                
                # Fix the memory instruction to be more helpful
                compressed_memory = self._fix_memory_instructions(compressed_memory)
                
                # Cache the compressed memory for future use
                await self._cache_memory(user_message, compressed_memory)
            
            # Insert memory context before the last user message
            memory_msg = {
                "role": "system",
                "content": compressed_memory
            }
            
            # Insert into context messages before the last user message
            if len(context._messages) >= 1:
                context._messages.insert(-1, memory_msg)
                token_count = self._estimate_tokens(compressed_memory)
                cache_status = "🎯 cached" if cached_memory else "🔍 fresh"
                logger.info(f"🧠 Injected MemoBase memories ({token_count} tokens, {cache_status}) for: {user_message[:50]}...")
                
                # Debug: Log if dog info is present in injected memory
                has_dog_info = any(word in compressed_memory.lower() for word in ['Bobby', 'dog named', 'pet'])
                logger.info(f"🐕 Dog info in injected memory: {has_dog_info}")
                
                # Store for fallback tracking
                self._last_user_message = user_message
                    
        except Exception as e:
            logger.error(f"❌ Failed to retrieve memories from MemoBase: {e}")

    async def start_session(self, user_id: Optional[str] = None):
        """Start a new conversation session."""
        if user_id:
            await self.update_user_id(user_id)
        
        self._session_active = True
        logger.info(f"🧠 MemoBase session started for user: {self.user_id}")

    async def end_session(self):
        """End conversation session and flush if configured."""
        if self._session_active and self.flush_on_session_end and self.is_enabled:
            await self.flush_memory()
        
        self._session_active = False
        async with self._lock:
            self._conversation_buffer.clear()
        
        logger.info(f"🧠 MemoBase session ended for user: {self.user_id}")

    async def flush_memory(self):
        """Flush conversation buffer to MemoBase."""
        if not self.is_enabled or not self.patched_sync_client:
            return
        
        try:
            async with self._lock:
                if self.patched_sync_client and hasattr(self.patched_sync_client, 'flush'):
                    await asyncio.to_thread(self.patched_sync_client.flush, self.user_id)
                    logger.info(f"🧠 Flushed MemoBase memory for user: {self.user_id} ({self._buffer_token_count} tokens)")
                
                self._conversation_buffer.clear()
                self._buffer_token_count = 0
                    
        except Exception as e:
            logger.error(f"❌ Error flushing MemoBase memory: {e}")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.is_enabled:
            return {"enabled": False, "error": "MemoBase not available"}
        
        stats = {
            "enabled": True,
            "user_id": self.user_id,
            "session_active": self._session_active,
            "buffer_size": len(self._conversation_buffer),
            "buffer_tokens": self._buffer_token_count,
            "max_context_size": self.max_context_size,
            "max_token_limit": self.max_token_limit,
            "compression_enabled": self.enable_compression,
            "relevance_filtering": self.enable_relevance_filtering,
            "auto_flush_threshold": self.auto_flush_threshold,
            "integration_method": "hybrid sync/async with smart compression"
        }
        
        # Get user profile if available
        if self.patched_sync_client and hasattr(self.patched_sync_client, 'get_profile'):
            try:
                profile = await asyncio.to_thread(self.patched_sync_client.get_profile, self.user_id)
                stats["profile"] = [p.describe for p in profile] if profile else []
            except Exception as e:
                stats["profile_error"] = str(e)
        
        return stats

    async def update_user_id(self, user_id: str):
        """Update the user ID for this processor."""
        old_user_id = self.user_id
        self.user_id = user_id
        
        # If session was active, end it and start new one
        if self._session_active:
            await self.end_session()
            await self.start_session()
        
        logger.info(f"🧠 Updated MemoBase user ID from {old_user_id} to {user_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and add to memory asynchronously."""
        await super().process_frame(frame, direction)

        # Debug: Log important frame types only
        frame_name = type(frame).__name__
        if frame_name in ['OpenAILLMContextFrame', 'TextFrame', 'TranscriptionFrame', 'LLMFullResponseEndFrame']:
            content = getattr(frame, 'text', getattr(frame, 'content', 'no text'))
            if hasattr(frame, 'text') or hasattr(frame, 'content'):
                logger.info(f"🔍 MemoBase received frame: {frame_name} - content: {str(content)[:50]}...")
            else:
                logger.info(f"🔍 MemoBase received frame: {frame_name}")
        elif not frame_name.endswith('AudioFrame') and 'Audio' not in frame_name and frame_name not in ['BotSpeakingFrame', 'TransportMessageUrgentFrame']:
            logger.debug(f"🔍 MemoBase received frame: {frame_name}")

        # Only process if MemoBase is enabled
        if not self.is_enabled:
            await self.push_frame(frame, direction)
            return

        # Handle OpenAI LLM Context frames (contains the full conversation)
        if hasattr(frame, 'context') and hasattr(frame.context, '_messages'):
            messages = frame.context._messages
            if messages and len(messages) >= 2:
                # Get the last user message (most recent)
                user_messages = [msg for msg in messages if msg.get('role') == 'user']
                if user_messages:
                    latest_user_msg = user_messages[-1].get('content', '')
                    if latest_user_msg and latest_user_msg.strip():
                        # FIRST: Retrieve and inject relevant memories from MemoBase (MUST be synchronous)
                        await self._inject_memory_context(frame.context, latest_user_msg)
                        
                        # THEN: Store the user message in MemoBase (async, non-blocking)
                        asyncio.create_task(
                            self._add_to_conversation_buffer(
                                'user', 
                                latest_user_msg, 
                                {'user_id': self.user_id, 'frame_type': 'llm_context'}
                            )
                        )
                        logger.info(f"🧠 Added user message to MemoBase: {latest_user_msg[:50]}...")

        # Handle transcriptions (user input) - fallback
        elif isinstance(frame, TranscriptionFrame) and frame.text:
            # Fire and forget - don't block on memory operations
            asyncio.create_task(
                self._add_to_conversation_buffer(
                    'user', 
                    frame.text, 
                    {'user_id': getattr(frame, 'user_id', self.user_id), 'frame_type': 'transcription'}
                )
            )
            logger.info(f"🧠 Added user transcription to MemoBase buffer: {frame.text[:50]}...")

        # Handle text frames (assistant output) 
        elif isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame) and frame.text:
            # Skip if this looks like a tool call
            if frame.text.strip().startswith('[') and ']' in frame.text:
                logger.debug(f"🔧 Skipping tool call from MemoBase memory: {frame.text[:50]}...")
            else:
                # Fire and forget - don't block on memory operations
                asyncio.create_task(
                    self._add_to_conversation_buffer(
                        'assistant', 
                        frame.text,
                        {'frame_type': 'text'}
                    )
                )
                logger.info(f"🧠 Added assistant text to MemoBase buffer: {frame.text[:50]}...")

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up MemoBase connection and flush any remaining memory."""
        if self._session_active:
            await self.end_session()
        
        if self.is_enabled:
            try:
                # Final flush
                await self.flush_memory()
                logger.info("🧠 MemoBase processor cleanup completed")
            except Exception as e:
                logger.error(f"❌ Error during MemoBase cleanup: {e}")
        
        await super().cleanup()

    async def __aenter__(self):
        """Async context manager entry."""
        if self.is_enabled:
            await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()