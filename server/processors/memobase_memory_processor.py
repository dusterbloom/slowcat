import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

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
        max_context_size: int = 500,
        flush_on_session_end: bool = True,
        fallback_to_local: bool = True
    ):
        super().__init__()
        
        # Configuration from config or parameters
        self.user_id = user_id or config.memory.default_user_id
        self.max_context_size = max_context_size or config.memobase.max_context_size
        self.flush_on_session_end = flush_on_session_end or config.memobase.flush_on_session_end
        self.fallback_to_local = fallback_to_local or config.memobase.fallback_to_local
        
        # Session tracking
        self._session_active = False
        self._conversation_buffer: List[Dict[str, str]] = []
        self._lock = asyncio.Lock()
        
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
        """Initialize MemoBase with hybrid sync/async approach."""
        try:
            # Create MemoBase client (using keyword arguments for local connection)
            self.mb_client = MemoBaseClient(
                project_url=config.memobase.project_url,
                api_key=config.memobase.api_key
            )
            
            # Test connection
            if self.mb_client.ping():
                # Create a sync OpenAI client for MemoBase patching
                self.sync_openai_client = OpenAI(
                    api_key=None,  # No API key needed for LM Studio
                    base_url=config.network.llm_base_url
                )
                
                # Apply MemoBase patching to the sync client
                self.patched_sync_client = openai_memory(
                    self.sync_openai_client, 
                    self.mb_client,
                    max_context_size=self.max_context_size
                )
                
                self.is_enabled = True
                logger.info(f"🧠 MemoBase hybrid integration initialized - connected to {config.memobase.project_url}")
                logger.info(f"🔧 Using sync OpenAI client with MemoBase patching for user: {self.user_id}")
                
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

    async def _add_to_conversation_buffer(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add conversation to buffer and store in MemoBase."""
        if not self.is_enabled or not self.patched_sync_client:
            logger.debug(f"🔄 MemoBase disabled, skipping: {content[:50]}...")
            return
            
        async with self._lock:
            self._conversation_buffer.append({
                "role": role,
                "content": content,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            })
            
        # Store individual messages in MemoBase immediately
        try:
            await self._store_message_in_memobase(role, content)
        except Exception as e:
            logger.error(f"❌ Failed to store message in MemoBase: {e}")
            
        logger.info(f"🧠 Added to MemoBase: {role} - {content[:50]}...")

    async def _store_message_in_memobase(self, role: str, content: str):
        """Store a single message in MemoBase using the patched client."""
        if not self.patched_sync_client:
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
                logger.debug(f"🔍 Storing {role} message with user_id: {self.user_id}")
                await asyncio.to_thread(
                    self.patched_sync_client.chat.completions.create,
                    model=config.models.default_llm_model,
                    messages=messages,
                    user_id=self.user_id,
                    max_tokens=1,  # Minimal response 
                    temperature=0.0  # Deterministic
                )
                logger.debug(f"🧠 Stored {role} message in MemoBase: {content[:30]}...")
            
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
                    
                    # Store the conversation pair
                    logger.debug(f"🔍 Storing conversation pair with user_id: {self.user_id}")
                    await asyncio.to_thread(
                        self.patched_sync_client.chat.completions.create,
                        model=config.models.default_llm_model,
                        messages=messages,
                        user_id=self.user_id,
                        max_tokens=1,
                        temperature=0.0
                    )
                    logger.debug(f"🧠 Stored conversation pair in MemoBase: user+assistant")
                    
        except Exception as e:
            logger.error(f"❌ Failed to store {role} message in MemoBase: {e}")
            logger.error(f"❌ Error details: {str(e)}")

    async def _inject_memory_context(self, context, user_message: str):
        """Retrieve relevant memories from MemoBase and inject into context."""
        if not self.is_enabled or not self.patched_sync_client:
            return
            
        try:
            # Get memory prompt from the patched client
            logger.debug(f"🔍 Retrieving memory for user_id: {self.user_id}")
            memory_prompt = await asyncio.to_thread(
                self.patched_sync_client.get_memory_prompt,
                self.user_id
            )
            logger.debug(f"🔍 Raw memory prompt: {repr(memory_prompt)[:200]}...")
            
            if memory_prompt and memory_prompt.strip():
                # Insert memory context before the last user message
                memory_msg = {
                    "role": "system",
                    "content": memory_prompt
                }
                
                # Insert into context messages before the last user message
                if len(context._messages) >= 1:
                    context._messages.insert(-1, memory_msg)
                    logger.info(f"🧠 Injected MemoBase memories into context for: {user_message[:50]}...")
                    
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
                    logger.info(f"🧠 Flushed MemoBase memory for user: {self.user_id}")
                
                self._conversation_buffer.clear()
                    
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
            "integration_method": "hybrid sync/async with patched client"
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