import sqlite3
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import threading

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger

class LocalMemoryProcessor(FrameProcessor):
    """
    A local memory processor that stores conversation history in a reliable
    SQLite database, ensuring data integrity and preventing corruption.
    """

    def __init__(
        self,
        data_dir: str = "data/memory",
        user_id: Optional[str] = None,
        max_history_items: int = 200,
        include_in_context: int = 10,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "memory.sqlite"
        self.user_id = user_id or config.memory.default_user_id
        self.max_history_items = max_history_items
        self.include_in_context = include_in_context
        
        # Use thread-local storage for DB connection to ensure thread safety.
        self._local = threading.local()
        self._setup_database()
        logger.info(f"📚 Memory database initialized for user: {self.user_id}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get a thread-safe database connection."""
        if not hasattr(self._local, "con"):
            self._local.con = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.con.row_factory = sqlite3.Row
        return self._local.con

    def _setup_database(self):
        """Create the database table if it doesn't exist."""
        con = self._get_db_connection()
        try:
            with con:
                cur = con.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id_timestamp ON conversations (user_id, timestamp);")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")

    def _add_to_memory(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a conversation item to the database transactionally."""
        con = self._get_db_connection()
        try:
            with con:
                cur = con.cursor()
                cur.execute(
                    "INSERT INTO conversations (user_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        self.user_id,
                        role,
                        content,
                        datetime.now().isoformat(),
                        json.dumps(metadata or {}),
                    ),
                )
            self._prune_history()
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")

    def _prune_history(self):
        """Keep the conversation history for the current user within max_history_items."""
        con = self._get_db_connection()
        try:
            with con:
                cur = con.cursor()
                cur.execute("SELECT COUNT(*) FROM conversations WHERE user_id = ?", (self.user_id,))
                count = cur.fetchone()[0]

                if count > self.max_history_items:
                    limit = count - self.max_history_items
                    cur.execute(
                        "DELETE FROM conversations WHERE id IN (SELECT id FROM conversations WHERE user_id = ? ORDER BY timestamp ASC LIMIT ?)",
                        (self.user_id, limit)
                    )
                    logger.debug(f"Pruned {limit} old memory items for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error pruning memory history: {e}")

    def get_context_messages(self) -> List[Dict[str, str]]:
        """Get recent conversation history from the database for LLM context."""
        con = self._get_db_connection()
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT role, content FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (self.user_id, self.include_in_context),
            )
            rows = cur.fetchall()
            # Reverse to get chronological order for the LLM context.
            return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        except Exception as e:
            logger.error(f"Error getting context messages: {e}")
            return []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text and direction == FrameDirection.DOWNSTREAM:
            metadata = {
                'user_id': frame.user_id,
                'timestamp': frame.timestamp,
                'language': str(frame.language) if frame.language else None
            }
            self._add_to_memory('user', frame.text, metadata)
            logger.info(f"📝 Stored user message from {frame.user_id}: {frame.text[:50]}...")
        elif isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame) and frame.text and direction == FrameDirection.DOWNSTREAM:
            self._add_to_memory('assistant', frame.text)
            logger.info(f"💬 Stored assistant message: {frame.text[:50]}...")

        await self.push_frame(frame, direction)

    def set_user_id(self, user_id: str):
        """Change the current user for memory operations."""
        if user_id != self.user_id:
            logger.info(f"Memory context switching from '{self.user_id}' to '{user_id}'")
            self.user_id = user_id