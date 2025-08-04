import aiosqlite
import asyncio
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
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger

class LocalMemoryProcessor(FrameProcessor):
    """
    A local memory processor that stores conversation history in a reliable
    SQLite database using async operations to prevent blocking.
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
        self.db_path = str(self.data_dir / "memory.sqlite")
        self.user_id = user_id or config.memory.default_user_id
        self.max_history_items = max_history_items
        self.include_in_context = include_in_context
        
        # Async connection will be initialized when first needed
        self._db_connection: Optional[aiosqlite.Connection] = None
        self._db_lock = asyncio.Lock()
        logger.info(f"📚 Memory processor initialized for user: {self.user_id}")

    async def _get_db_connection(self) -> aiosqlite.Connection:
        """Get or create the async database connection."""
        async with self._db_lock:
            if self._db_connection is None:
                self._db_connection = await aiosqlite.connect(self.db_path)
                self._db_connection.row_factory = aiosqlite.Row
                await self._setup_database()
            return self._db_connection

    async def _setup_database(self):
        """Create the database table if it doesn't exist."""
        try:
            await self._db_connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            await self._db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_id_timestamp ON conversations (user_id, timestamp);"
            )
            await self._db_connection.commit()
            logger.info(f"📚 Async memory database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")

    async def _add_to_memory(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a conversation item to the database asynchronously."""
        con = await self._get_db_connection()
        try:
            await con.execute(
                "INSERT INTO conversations (user_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    self.user_id,
                    role,
                    content,
                    datetime.now().isoformat(),
                    json.dumps(metadata or {}),
                ),
            )
            await con.commit()
            # Prune history in background to not block
            asyncio.create_task(self._prune_history())
        except Exception as e:
            logger.error(f"Error adding to memory: {e}")
            await con.rollback()

    async def _prune_history(self):
        """Keep the conversation history for the current user within max_history_items."""
        con = await self._get_db_connection()
        try:
            # Get the current count
            async with con.execute(
                "SELECT COUNT(*) FROM conversations WHERE user_id = ?", 
                (self.user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                count = row[0] if row else 0

            # Delete oldest entries if we exceed the limit
            if count > self.max_history_items:
                limit = count - self.max_history_items
                await con.execute(
                    """DELETE FROM conversations 
                    WHERE id IN (
                        SELECT id FROM conversations 
                        WHERE user_id = ? 
                        ORDER BY timestamp ASC 
                        LIMIT ?
                    )""",
                    (self.user_id, limit)
                )
                await con.commit()
                logger.debug(f"Pruned {limit} old memory entries for user {self.user_id}")
        except Exception as e:
            logger.error(f"Error pruning memory history: {e}")

    async def get_context_messages(self) -> List[Dict[str, str]]:
        """Get the most recent messages for context asynchronously."""
        con = await self._get_db_connection()
        try:
            async with con.execute(
                """SELECT role, content 
                FROM conversations 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?""",
                (self.user_id, self.include_in_context),
            ) as cursor:
                rows = await cursor.fetchall()
            
            # Reverse to get chronological order
            return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        except Exception as e:
            logger.error(f"Error getting context messages: {e}")
            return []

    async def update_user_id(self, user_id: str):
        """Update the user ID for this processor."""
        self.user_id = user_id
        logger.info(f"Updated memory processor user ID to: {user_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and add to memory asynchronously."""
        await super().process_frame(frame, direction)

        # Handle transcriptions (user input)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            # Fire and forget - don't block on DB write
            asyncio.create_task(
                self._add_to_memory('user', frame.text, {'user_id': frame.user_id})
            )
            logger.debug(f"💭 Added user message to memory: {frame.text[:50]}...")

        # Handle text frames (assistant output) but exclude transcriptions
        elif isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame) and frame.text:
            # Fire and forget - don't block on DB write
            asyncio.create_task(
                self._add_to_memory('assistant', frame.text)
            )
            logger.debug(f"🤖 Added assistant message to memory: {frame.text[:50]}...")

        await self.push_frame(frame, direction)

    async def cleanup(self):
        """Clean up database connection."""
        if self._db_connection:
            await self._db_connection.close()
            self._db_connection = None
        await super().cleanup()

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        # Ensure connection is established
        await self._get_db_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        await super().__aexit__(exc_type, exc_val, exc_tb)