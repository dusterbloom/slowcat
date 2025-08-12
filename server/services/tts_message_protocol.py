"""
TTS Message Protocol for reliable text streaming with proper concurrency handling.

This module provides a bulletproof protocol for streaming TTS text with:
- Unique message IDs (UUIDs)
- Clear message state (partial/incremental/complete)
- Concurrency safety
- Duplicate prevention
"""

import json
import uuid
import threading
from typing import Dict, Optional, Literal, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from loguru import logger


MessageType = Literal["incremental", "cumulative", "complete"]
MessageState = Literal["started", "streaming", "completed", "error"]


@dataclass
class TTSMessage:
    """Represents a single TTS message with tracking metadata."""
    message_id: str
    original_text: str
    sanitized_text: str
    total_chunks: int
    created_at: float
    
    def __post_init__(self):
        self.chunks_sent = 0
        self.last_sent_position = 0
        self.state: MessageState = "started"
        self.lock = threading.Lock()


@dataclass 
class TTSTextChunk:
    """Represents a single chunk of TTS text with metadata."""
    message_id: str
    chunk_index: int
    total_chunks: int
    text: str
    message_type: MessageType
    is_final: bool
    timestamp: float
    
    def to_json(self) -> str:
        """Serialize to JSON for transmission."""
        return json.dumps({
            "protocol": "tts_v2",
            "message_id": self.message_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "text": self.text,
            "message_type": self.message_type,
            "is_final": self.is_final,
            "timestamp": self.timestamp
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> Optional['TTSTextChunk']:
        """Deserialize from JSON."""
        try:
            data = json.loads(json_str)
            if data.get("protocol") != "tts_v2":
                return None
            return cls(
                message_id=data["message_id"],
                chunk_index=data["chunk_index"],
                total_chunks=data["total_chunks"],
                text=data["text"],
                message_type=data["message_type"],
                is_final=data["is_final"],
                timestamp=data["timestamp"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse TTS chunk: {e}")
            return None


class TTSMessageTracker:
    """
    Thread-safe tracker for TTS messages.
    Ensures each message is properly tracked and prevents duplicates.
    """
    
    def __init__(self, max_messages: int = 100):
        self._messages: Dict[str, TTSMessage] = {}
        self._lock = threading.RLock()
        self._max_messages = max_messages
        self._message_order = []  # Track insertion order for cleanup
    
    def create_message(self, original_text: str, sanitized_text: str) -> TTSMessage:
        """Create and track a new TTS message."""
        with self._lock:
            message_id = str(uuid.uuid4())
            
            # Calculate total chunks based on word count (estimate)
            words = sanitized_text.split()
            # Estimate chunks: roughly 5-10 words per chunk for natural pacing
            estimated_chunks = max(1, len(words) // 7)
            
            message = TTSMessage(
                message_id=message_id,
                original_text=original_text,
                sanitized_text=sanitized_text,
                total_chunks=estimated_chunks,
                created_at=datetime.now().timestamp()
            )
            
            # Store message
            self._messages[message_id] = message
            self._message_order.append(message_id)
            
            # Cleanup old messages if needed
            if len(self._messages) > self._max_messages:
                oldest_id = self._message_order.pop(0)
                del self._messages[oldest_id]
                logger.debug(f"Cleaned up old message: {oldest_id}")
            
            logger.info(f"Created TTS message: {message_id} ({len(words)} words, ~{estimated_chunks} chunks)")
            return message
    
    def get_message(self, message_id: str) -> Optional[TTSMessage]:
        """Get a message by ID."""
        with self._lock:
            return self._messages.get(message_id)
    
    def update_message_state(self, message_id: str, state: MessageState) -> bool:
        """Update the state of a message."""
        with self._lock:
            message = self._messages.get(message_id)
            if message:
                message.state = state
                logger.debug(f"Updated message {message_id} state to: {state}")
                return True
            return False
    
    def mark_chunk_sent(self, message_id: str, chunk_index: int, position: int) -> bool:
        """Mark a chunk as sent for a message."""
        with self._lock:
            message = self._messages.get(message_id)
            if message:
                with message.lock:
                    message.chunks_sent = max(message.chunks_sent, chunk_index + 1)
                    message.last_sent_position = max(message.last_sent_position, position)
                    return True
            return False
    
    def is_duplicate_chunk(self, message_id: str, chunk_index: int) -> bool:
        """Check if a chunk has already been sent."""
        with self._lock:
            message = self._messages.get(message_id)
            if message:
                with message.lock:
                    return chunk_index < message.chunks_sent
            return False
    
    def cleanup_completed_messages(self, age_seconds: float = 60):
        """Clean up completed messages older than specified age."""
        with self._lock:
            current_time = datetime.now().timestamp()
            to_remove = []
            
            for msg_id, message in self._messages.items():
                if (message.state == "completed" and 
                    current_time - message.created_at > age_seconds):
                    to_remove.append(msg_id)
            
            for msg_id in to_remove:
                del self._messages[msg_id]
                self._message_order.remove(msg_id)
                logger.debug(f"Cleaned up completed message: {msg_id}")
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} completed messages")


class TTSProtocolHelper:
    """Helper class for using the TTS protocol in the Kokoro service."""
    
    def __init__(self):
        self.tracker = TTSMessageTracker()
        # Start cleanup task
        self._cleanup_task = None
    
    async def start_cleanup_task(self):
        """Start the background cleanup task."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background task to clean up old messages."""
        while True:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                self.tracker.cleanup_completed_messages(age_seconds=60)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def create_message_chunks(
        self, 
        original_text: str, 
        sanitized_text: str,
        incremental_mode: bool = True
    ) -> tuple[TTSMessage, list[TTSTextChunk]]:
        """
        Create a message and prepare chunks for streaming.
        
        Args:
            original_text: The original text before sanitization
            sanitized_text: The sanitized text for TTS
            incremental_mode: If True, send incremental text. If False, send cumulative.
            
        Returns:
            The message and list of prepared chunks
        """
        message = self.tracker.create_message(original_text, sanitized_text)
        
        # Split text into words for chunking
        words = sanitized_text.split()
        chunks = []
        
        if not words:
            return message, chunks
        
        # Create chunks with proper sizing
        words_per_chunk = max(1, len(words) // message.total_chunks) if message.total_chunks > 0 else len(words)
        
        position = 0
        for chunk_idx in range(message.total_chunks):
            start = chunk_idx * words_per_chunk
            end = min(start + words_per_chunk, len(words))
            
            if start >= len(words):
                break
                
            chunk_words = words[start:end]
            
            if incremental_mode:
                # Send only the new words
                chunk_text = " ".join(chunk_words)
            else:
                # Send cumulative text up to this point
                chunk_text = " ".join(words[:end])
            
            is_final = (end >= len(words))
            
            chunk = TTSTextChunk(
                message_id=message.message_id,
                chunk_index=chunk_idx,
                total_chunks=message.total_chunks,
                text=chunk_text,
                message_type="incremental" if incremental_mode else "cumulative",
                is_final=is_final,
                timestamp=datetime.now().timestamp()
            )
            
            chunks.append(chunk)
            
            if is_final:
                break
        
        # Update actual total chunks
        message.total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return message, chunks
    
    def create_streaming_chunk(
        self,
        message: TTSMessage,
        chunk_index: int,
        text: str,
        is_incremental: bool = True,
        is_final: bool = False
    ) -> Optional[TTSTextChunk]:
        """
        Create a single streaming chunk with duplicate prevention.
        
        Args:
            message: The message this chunk belongs to
            chunk_index: The index of this chunk
            text: The text content
            is_incremental: Whether this is incremental or cumulative
            is_final: Whether this is the final chunk
            
        Returns:
            The chunk or None if it's a duplicate
        """
        # Check for duplicate
        if self.tracker.is_duplicate_chunk(message.message_id, chunk_index):
            logger.debug(f"Skipping duplicate chunk {chunk_index} for message {message.message_id}")
            return None
        
        chunk = TTSTextChunk(
            message_id=message.message_id,
            chunk_index=chunk_index,
            total_chunks=message.total_chunks,
            text=text,
            message_type="incremental" if is_incremental else "cumulative",
            is_final=is_final,
            timestamp=datetime.now().timestamp()
        )
        
        # Mark as sent
        self.tracker.mark_chunk_sent(message.message_id, chunk_index, len(text))
        
        return chunk
    
    def mark_message_complete(self, message_id: str):
        """Mark a message as completed."""
        self.tracker.update_message_state(message_id, "completed")
    
    def mark_message_error(self, message_id: str):
        """Mark a message as having an error."""
        self.tracker.update_message_state(message_id, "error")