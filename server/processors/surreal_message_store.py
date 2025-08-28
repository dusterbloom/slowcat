"""
SurrealDB Message Store Processor

Captures and stores final STT transcriptions and LLM responses
in SurrealDB for conversation persistence and retrieval.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# Try to import SurrealDB connection
try:
    from memory.surreal_connection import get_surreal_connection, Message
    SURREAL_AVAILABLE = True
except ImportError:
    SURREAL_AVAILABLE = False
    logger.warning("SurrealDB connection not available - install surrealdb package")


class SurrealMessageStore(FrameProcessor):
    """
    Processor that captures user transcriptions and assistant responses
    and stores them in SurrealDB for conversation persistence.
    """
    
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
        
        logger.info(f"📝 SurrealMessageStore initialized (speaker: {speaker_id})")
    
    async def _ensure_session(self):
        """Ensure we have an active session"""
        if not self.surreal or self._session_created:
            return
        
        try:
            if not self.session_id and self.auto_create_session:
                # Create new session
                self.session_id = await self.surreal.create_session(
                    speaker_id=self.speaker_id,
                    metadata={
                        'created_by': 'SurrealMessageStore',
                        'auto_created': True
                    }
                )
                
                if self.session_id:
                    logger.info(f"🎬 Created session: {self.session_id}")
                    self._session_created = True
                else:
                    logger.warning("Failed to create session")
            
            elif self.session_id:
                # Mark existing session as active
                self._session_created = True
                logger.info(f"🎬 Using session: {self.session_id}")
        
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
            else:
                logger.warning(f"Failed to store {message.role} message")
        
        except Exception as e:
            logger.error(f"Error storing {message.role} message: {e}")
    
    def _enqueue_message_store(self, message: Message):
        """Queue message for async storage"""
        if self.surreal:
            asyncio.create_task(self._store_message_async(message))
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Ensure we have a session
        if not self._session_created:
            await self._ensure_session()
        
        # Capture user transcriptions (final STT output)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            text = (frame.text or '').strip()
            if text:
                await self._handle_user_message(text)
        
        # Track assistant response lifecycle  
        elif isinstance(frame, LLMFullResponseStartFrame) and direction == FrameDirection.DOWNSTREAM:
            self._in_assistant_response = True
            self._assistant_buffer = ""
        
        elif isinstance(frame, LLMFullResponseEndFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._in_assistant_response and self._assistant_buffer.strip():
                await self._handle_assistant_message(self._assistant_buffer.strip())
            
            self._in_assistant_response = False
            self._assistant_buffer = ""
        
        # Accumulate assistant response text
        elif isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM and self._in_assistant_response:
            text = getattr(frame, 'text', '') or ''
            if text:
                # LLM sends REPLACEMENT frames - each frame contains complete response so far
                # We want the final, complete response, so just replace (don't concatenate)
                self._assistant_buffer = text
                logger.debug(f"📝 Assistant response update: {len(text)} chars")
        
        # Forward all frames
        await self.push_frame(frame, direction)
    
    async def _handle_user_message(self, text: str):
        """Handle user transcription"""
        try:
            # Store current user message for context
            self._current_user_message = text
            
            # Create message object
            message = Message(
                role='user',
                content=text,
                speaker_id=self.speaker_id,
                session_id=self.session_id,
                timestamp=datetime.utcnow(),
                tokens=self._estimate_tokens(text)
            )
            
            # Store asynchronously
            self._enqueue_message_store(message)
            
            logger.info(f"🎤 Storing user message: '{text}'")
        
        except Exception as e:
            logger.error(f"Error handling user message: {e}")
    
    async def _handle_assistant_message(self, text: str):
        """Handle assistant response"""
        try:
            # Create message object
            message = Message(
                role='assistant',
                content=text,
                speaker_id='assistant',
                session_id=self.session_id,
                timestamp=datetime.utcnow(),
                tokens=self._estimate_tokens(text)
            )
            
            # Store asynchronously
            self._enqueue_message_store(message)
            
            logger.info(f"🤖 Storing assistant response: '{text}'")
            
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
        """End the current session"""
        if self.surreal and self.session_id and self._session_created:
            try:
                await self.surreal.end_session(self.session_id, summary)
                logger.info(f"🏁 Session finalized: {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to finalize session: {e}")


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