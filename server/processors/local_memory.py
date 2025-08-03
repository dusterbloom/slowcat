import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    StartInterruptionFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger


class LocalMemoryProcessor(FrameProcessor):
    """
    A local memory processor that stores conversation history without cloud APIs.
    Stores conversations in JSON files organized by user ID.
    """
    
    def __init__(
        self,
        data_dir: str = "data/memory",
        user_id: Optional[str] = None,
        max_history_items: int = 100,
        include_in_context: int = 10
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id or "default_user"
        self.max_history_items = max_history_items
        self.include_in_context = include_in_context
        self.current_session: List[Dict] = []
        self.memory_file = self.data_dir / f"{self.user_id}_memory{config.memory.file_extension if hasattr(config.memory, 'file_extension') else '.json'}"
        self._load_memory()
        
    def _load_memory(self):
        """Load existing memory from file"""
        self.memory: List[Dict] = []
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memory = data.get('conversations', [])
                    logger.info(f"Loaded {len(self.memory)} memory items for user {self.user_id}")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                
    def _save_memory(self):
        """Save memory to file"""
        try:
            # Keep only the most recent items
            if len(self.memory) > self.max_history_items:
                self.memory = self.memory[-self.max_history_items:]
                
            data = {
                'user_id': self.user_id,
                'last_updated': datetime.now().isoformat(),
                'conversations': self.memory
            }
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            
    def _add_to_memory(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a conversation item to memory"""
        item = {
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content,
            'session_id': id(self.current_session),
            'metadata': metadata or {}
        }
        
        self.current_session.append(item)
        self.memory.append(item)
        
        # Save periodically
        if len(self.current_session) % 5 == 0:
            self._save_memory()
            
    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get recent conversation history for context"""
        if not self.memory:
            return []
            
        # Get the most recent conversations
        recent_items = self.memory[-self.include_in_context:]
        
        # Format as messages for LLM context
        messages = []
        for item in recent_items:
            messages.append({
                'role': 'user' if item['role'] == 'user' else 'assistant',
                'content': item['content']
            })
            
        return messages
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Debug logging
        # logger.debug(f"Memory processor received {type(frame).__name__} going {direction.name}")
        
        # Capture user messages from STT (TranscriptionFrame flows downstream)
        if isinstance(frame, TranscriptionFrame):
            # STT output with user transcription
            if frame.text:
                metadata = {
                    'user_id': frame.user_id,
                    'timestamp': frame.timestamp,
                    'language': str(frame.language) if frame.language else None
                }
                self._add_to_memory('user', frame.text, metadata)
                logger.info(f"📝 Stored user message from {frame.user_id}: {frame.text[:50]}...")
                
        # Capture assistant responses (TextFrame from TTS going downstream)
        elif isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame):
            # TTS output or LLM generated text
            if hasattr(frame, 'text') and frame.text:
                self._add_to_memory('assistant', frame.text)
                logger.info(f"💬 Stored assistant message: {frame.text[:50]}...")
                
        # Save on conversation end
        elif isinstance(frame, (UserStoppedSpeakingFrame, StartInterruptionFrame)):
            self._save_memory()
            
        await self.push_frame(frame, direction)
        
    def set_user_id(self, user_id: str):
        """Change the current user and reload their memory"""
        if user_id != self.user_id:
            # Save current user's memory
            self._save_memory()
            
            # Switch to new user
            self.user_id = user_id
            self.memory_file = self.data_dir / f"{user_id}_memory{config.memory.file_extension if hasattr(config.memory, 'file_extension') else '.json'}"
            self.current_session = []
            self._load_memory()