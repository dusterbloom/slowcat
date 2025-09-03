"""
SurrealDB Message Store Processor

Captures and stores final STT transcriptions and LLM responses
in SurrealDB for conversation persistence and retrieval.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# Try to import SurrealDB connection
try:
    from memory.surreal_connection import get_surreal_connection, Message
    from memory.session_manager import SessionManager
    SURREAL_AVAILABLE = True
except ImportError:
    SURREAL_AVAILABLE = False
    SessionManager = None
    logger.warning("SurrealDB connection not available - install surrealdb package")

# GLOBAL session management - ensures ALL instances across the entire app use same session
_GLOBAL_CONVERSATION_SESSION = None


class SurrealMessageStore(FrameProcessor):
    """
    Processor that captures user transcriptions and assistant responses
    and stores them in SurrealDB for conversation persistence.
    """
    
    # Class-level session cache to share sessions across instances
    _shared_sessions = {}
    
    def __init__(self,
                 speaker_id: str = 'default_user',
                 session_id: str = None,
                 auto_create_session: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.speaker_id = speaker_id
        self.session_id = session_id
        self.auto_create_session = auto_create_session
        
        # Initialize SurrealDB connection
        self.surreal = None
        if SURREAL_AVAILABLE:
            self.surreal = get_surreal_connection()
        else:
            logger.warning("SurrealDB not available - message storage disabled")
        
        # Message tracking
        self._current_user_message = None
        self._in_assistant_response = False
        self._assistant_buffer = ""
        self._message_queue = []
        
        # Session management
        self._session_created = False
        self._persistent_session_id = None  # Cache session_id to prevent regeneration
        
        logger.info(f"📝 SurrealMessageStore initialized (speaker: {speaker_id})")
    
    async def _ensure_session(self):
        """Ensure we have an active session - use cached session_id to prevent regeneration"""
        if not self.surreal or self._session_created:
            return
        
        try:
            # Use persistent session_id to avoid creating new sessions every time
            if not self.session_id:
                # Check if we have a cached persistent session
                if self._persistent_session_id:
                    self.session_id = self._persistent_session_id
                    logger.info(f"🔄 Using cached session: {self.session_id}")
                else:
                    # Try to get existing session from SessionManager
                    existing_session = SessionManager.get_current_session() if SessionManager else None
                    
                    if existing_session:
                        self.session_id = existing_session
                        self._persistent_session_id = existing_session  # Cache it
                        logger.info(f"🔄 Using existing global session: {self.session_id}")
                    elif self.auto_create_session:
                        # Create new session only as last resort
                        if SessionManager:
                            self.session_id = SessionManager.create_new_session(
                                speaker_id=self.speaker_id,
                                metadata={
                                    'created_by': 'SurrealMessageStore',
                                    'auto_created': True,
                                    'global_conversation': True
                                }
                            )
                        else:
                            # Generate simple session_id if SessionManager unavailable
                            import uuid
                            self.session_id = f"session_{uuid.uuid4().hex[:12]}"
                        
                        # Cache the session_id to prevent regeneration
                        self._persistent_session_id = self.session_id
                        
                        # Create in SurrealDB using the same session_id from SessionManager
                        if self.surreal:
                            db_session = await self.surreal.create_session(
                                speaker_id=self.speaker_id,
                                metadata={
                                    'created_by': 'SurrealMessageStore',
                                    'auto_created': True,
                                    'global_conversation': True
                                },
                                session_id=self.session_id  # Pass SessionManager's session_id
                            )
                            logger.info(f"🎬 Created new session: {self.session_id} (SurrealDB: {db_session})")
                        else:
                            logger.info(f"🎬 Created new session: {self.session_id}")
                    else:
                        logger.warning("Cannot create session - auto_create_session disabled")
                        return
            else:
                # Session_id provided during init - cache it and set in SessionManager  
                self._persistent_session_id = self.session_id
                if SessionManager:
                    SessionManager.set_session(self.session_id, self.speaker_id)
                    logger.info(f"🔄 Set global session: {self.session_id}")
            
            self._session_created = True
        
        except Exception as e:
            logger.error(f"Session setup failed: {e}")
    
    async def _store_message_async(self, message: Message):
        """Store message asynchronously"""
        if not self.surreal:
            return
        
        try:
            message_id = await self.surreal.store_message(message)
            if message_id:
                logger.debug(f"📝 Stored {message.role} message: {message_id}")
                # Kick off post-persist fact extraction for user messages so we can attach edges
                try:
                    if message.role == 'user':
                        asyncio.create_task(self._extract_and_store_for_message(message_id, message.content))
                except Exception as e:
                    logger.debug(f"Post-persist extraction not scheduled: {e}")
            else:
                logger.warning(f"Failed to store {message.role} message")
        
        except Exception as e:
            logger.error(f"Error storing {message.role} message: {e}")
    
    def _enqueue_message_store(self, message: Message):
        """Queue message for async storage"""
        if self.surreal:
            asyncio.create_task(self._store_message_async(message))

    async def _extract_and_store_for_message(self, message_id: str, text: str):
        """Extract relations from the just-stored message and persist knowledge with edges.

        Creates knowledge with source_message and RELATE edges via SurrealConnectionManager.
        """
        try:
            t = (text or '').strip()
            if not t or len(t.split()) < 3:
                return
            # Prefer DSPy extractor if available; fallback to hybrid
            try:
                from memory.dspy_integration import extract_facts_from_text_dspy as _extract
            except Exception:
                from memory.hybrid_fact_extractor import extract_facts_from_text as _extract  # type: ignore

            facts = _extract(t) or []
            if not facts:
                return
            stored = 0
            for f in facts:
                subj = f.get('subject') or 'user'
                pred = f.get('predicate') or 'related_to'
                obj = f.get('value') or f.get('object') or ''
                if not obj:
                    continue
                ok = await self.surreal.store_knowledge_relation(
                    subject_name=subj,
                    predicate=pred,
                    object_name=obj,
                    subject_type='user' if subj == 'user' else 'concept',
                    object_type='concept',
                    confidence=float(f.get('confidence', 0.7)),
                    source_message_id=message_id,
                    session_id=self.session_id,
                )
                if ok:
                    stored += 1
            if stored:
                logger.info(f"🔗 Post-persist: stored {stored} knowledge relation(s) for {message_id}")
        except Exception as e:
            logger.debug(f"Post-persist extract/store failed: {e}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # COMPLETELY DISABLE FRAME PROCESSING - this is now storage-only helper
        # Just forward frames without any processing to prevent duplicate storage
        await self.push_frame(frame, direction)
    
    async def _handle_user_message(self, text: str):
        """Handle user transcription"""
        try:
            # Store current user message for context
            self._current_user_message = text
            
            # Ensure session is created first
            await self._ensure_session()
            
            # Ensure we have a session_id (required for Message validation)
            if not self.session_id:
                logger.warning("No session_id available for user message - skipping storage")
                return
            
            # Create message object
            message = Message(
                role='user',
                content=text,
                speaker_id=self.speaker_id,
                session_id=self.session_id,
                timestamp=datetime.now(timezone.utc),
                tokens=self._estimate_tokens(text)
            )
            
            # Store asynchronously
            self._enqueue_message_store(message)
            
            logger.info(f"🎤 INTEGRATION: Storing user message speaker={self.speaker_id} session={self.session_id} instance={id(self)}")
        
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
    
    async def _handle_assistant_message(self, text: str):
        """Handle assistant response"""
        try:
            # Ensure session exists and is the SAME as user's session
            await self._ensure_session()
            
            # Ensure we have a session_id (required for Message validation)
            if not self.session_id:
                logger.warning("No session_id available for assistant message - skipping storage")
                return
            
            # Create message object with proper assistant speaker_id
            import os
            assistant_id = os.getenv('ASSISTANT_ID', 'slowcat')
            message = Message(
                role='assistant',
                content=text,
                speaker_id=assistant_id,  # Use ASSISTANT_ID from env, not user's speaker_id
                session_id=self.session_id,
                timestamp=datetime.now(timezone.utc),
                tokens=self._estimate_tokens(text)
            )
            
            # Store asynchronously
            self._enqueue_message_store(message)
            
            logger.info(f"🤖 INTEGRATION: Storing assistant message speaker={assistant_id} session={self.session_id} instance={id(self)}")
            
            # Clear current user message for next turn
            self._current_user_message = None
        
        except Exception as e:
            logger.error(f"Error handling assistant message: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return max(1, int(len(text.split()) * 1.3))  # Approximation, cast to int
    
    def update_speaker(self, speaker_id: str):
        """Update the current speaker"""
        if speaker_id != self.speaker_id:
            self.speaker_id = speaker_id
            logger.info(f"🎭 Speaker updated: {speaker_id}")
    
    def update_session(self, session_id: str):
        """Update the current session"""
        if session_id != self.session_id:
            self.session_id = session_id
            self._session_created = False  # Will need to verify new session
            logger.info(f"🎬 Session updated: {session_id}")
    
    async def finalize_session(self, summary: str = None):
        """End the current session (best-effort)."""
        if self.surreal and self.session_id:
            try:
                await self.surreal.end_session(self.session_id, summary)
                logger.info(f"🏁 Session finalized: {self.session_id}")
                
                # Remove conversation session from shared cache
                CONVERSATION_KEY = "current_conversation"
                if CONVERSATION_KEY in self._shared_sessions:
                    del self._shared_sessions[CONVERSATION_KEY]
                    logger.debug(f"🗑️ Removed conversation session from cache")
                    
            except Exception as e:
                logger.error(f"Failed to finalize session: {e}")
    
    @classmethod
    def clear_session_cache(cls, speaker_id: str = None):
        """Clear session cache including global session"""
        global _GLOBAL_CONVERSATION_SESSION
        
        _GLOBAL_CONVERSATION_SESSION = None
        cls._shared_sessions.clear()
        logger.info("🗑️ Cleared ALL session cache including global conversation session")
    
    @classmethod
    def get_shared_session(cls, speaker_id: str = None) -> str:
        """Get the global conversation session ID"""
        global _GLOBAL_CONVERSATION_SESSION
        return _GLOBAL_CONVERSATION_SESSION


def create_surreal_message_store(speaker_id: str = 'default_user',
                                session_id: str = None,
                                auto_create_session: bool = True) -> Optional[SurrealMessageStore]:
    """
    Factory function to create a SurrealMessageStore
    
    Args:
        speaker_id: Default speaker identifier
        session_id: Optional existing session ID
        auto_create_session: Whether to auto-create sessions
        
    Returns:
        SurrealMessageStore instance or None if SurrealDB not available
    """
    if not SURREAL_AVAILABLE:
        logger.warning("SurrealDB not available - cannot create message store")
        return None
    
    return SurrealMessageStore(
        speaker_id=speaker_id,
        session_id=session_id,
        auto_create_session=auto_create_session
    )


# Integration with SmartContextManager
class SurrealIntegrationMixin:
    """Mixin to add SurrealDB message storage to SmartContextManager"""
    
    def __init__(self, *args, **kwargs):
        # Extract SurrealDB settings
        self.enable_surreal = kwargs.pop('enable_surreal', True)
        self.surreal_auto_session = kwargs.pop('surreal_auto_session', True)
        
        super().__init__(*args, **kwargs)
        
        # Initialize SurrealDB integration
        self.message_store = None
        if self.enable_surreal and SURREAL_AVAILABLE:
            try:
                self.message_store = create_surreal_message_store(
                    speaker_id=getattr(self, '_user_id', 'default_user'),
                    auto_create_session=self.surreal_auto_session
                )
                
                if self.message_store:
                    logger.info("📝 SurrealDB message storage enabled")
            
            except Exception as e:
                logger.warning(f"SurrealDB integration failed: {e}")
    
    async def _store_user_message_surreal(self, text: str):
        """Store user message in SurrealDB"""
        if self.message_store:
            await self.message_store._handle_user_message(text)
    
    async def _store_assistant_message_surreal(self, text: str):
        """Store assistant message in SurrealDB"""
        if self.message_store:
            await self.message_store._handle_assistant_message(text)
    
    def update_surreal_speaker(self, speaker_id: str):
        """Update SurrealDB speaker"""
        if self.message_store:
            self.message_store.update_speaker(speaker_id)
    
    async def finalize_surreal_session(self, summary: str = None):
        """Finalize SurrealDB session"""
        if self.message_store:
            await self.message_store.finalize_session(summary)
