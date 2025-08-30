"""
Global Session Manager for Slowcat Consciousness Engine

Provides centralized session management to ensure consistent session_ids across
all processors and prevent the race condition where different components 
create different sessions for the same conversation.
"""

import os
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from loguru import logger

# Global session state
_CURRENT_SESSION_ID: Optional[str] = None
_SESSION_METADATA: Dict[str, Any] = {}


class SessionManager:
    """
    Centralized session management to prevent session ID race conditions.
    
    This ensures that all processors (SurrealMessageStore, SmartContextManager, etc.)
    use the same session_id for a conversation.
    """
    
    @classmethod
    def get_current_session(cls) -> Optional[str]:
        """Get the current global session ID"""
        return _CURRENT_SESSION_ID
    
    @classmethod
    def create_new_session(cls, 
                          speaker_id: str = 'default_user',
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new global session that all processors will use
        
        Args:
            speaker_id: Primary speaker for this session
            metadata: Optional session metadata
            
        Returns:
            New session ID
        """
        global _CURRENT_SESSION_ID, _SESSION_METADATA
        
        # Generate new session ID
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        _CURRENT_SESSION_ID = session_id
        
        # Store metadata
        _SESSION_METADATA = {
            'session_id': session_id,
            'speaker_id': speaker_id,
            'created_at': datetime.now(timezone.utc),
            'metadata': metadata or {}
        }
        
        logger.info(f"🎬 Created GLOBAL session: {session_id} for speaker: {speaker_id}")
        return session_id
    
    @classmethod
    def set_session(cls, session_id: str, 
                   speaker_id: str = 'default_user',
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set an existing session as the current global session
        
        Args:
            session_id: Existing session ID to use
            speaker_id: Primary speaker for this session
            metadata: Optional session metadata
        """
        global _CURRENT_SESSION_ID, _SESSION_METADATA
        
        _CURRENT_SESSION_ID = session_id
        _SESSION_METADATA = {
            'session_id': session_id,
            'speaker_id': speaker_id,
            'set_at': datetime.now(timezone.utc),
            'metadata': metadata or {}
        }
        
        logger.info(f"🔄 Set GLOBAL session: {session_id} for speaker: {speaker_id}")
    
    @classmethod
    def ensure_session(cls, speaker_id: str = 'default_user') -> str:
        """
        Ensure there's an active session, creating one if needed
        
        Args:
            speaker_id: Speaker ID if creating new session
            
        Returns:
            Current or new session ID
        """
        if _CURRENT_SESSION_ID:
            return _CURRENT_SESSION_ID
        
        return cls.create_new_session(speaker_id)
    
    @classmethod
    def get_session_metadata(cls) -> Dict[str, Any]:
        """Get metadata for the current session"""
        return _SESSION_METADATA.copy()
    
    @classmethod
    def clear_session(cls) -> None:
        """Clear the current global session"""
        global _CURRENT_SESSION_ID, _SESSION_METADATA
        
        old_session = _CURRENT_SESSION_ID
        _CURRENT_SESSION_ID = None
        _SESSION_METADATA = {}
        
        if old_session:
            logger.info(f"🗑️ Cleared GLOBAL session: {old_session}")
    
    @classmethod
    def is_session_active(cls) -> bool:
        """Check if there's an active session"""
        return _CURRENT_SESSION_ID is not None


def get_global_session() -> Optional[str]:
    """Convenience function to get current global session ID"""
    return SessionManager.get_current_session()


def ensure_global_session(speaker_id: str = 'default_user') -> str:
    """Convenience function to ensure there's an active global session"""
    return SessionManager.ensure_session(speaker_id)


def create_global_session(speaker_id: str = 'default_user', 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to create a new global session"""
    return SessionManager.create_new_session(speaker_id, metadata)