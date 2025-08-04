import aiosqlite
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
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
            # Create indexes for better performance
            await self._db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_id_timestamp ON conversations (user_id, timestamp);"
            )
            await self._db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_id_content ON conversations (user_id, content);"
            )
            await self._db_connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp);"
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

    async def search_conversations(self, query: str, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search through conversation history for specific text."""
        con = await self._get_db_connection()
        try:
            # Use the provided user_id or default to current user
            search_user_id = user_id or self.user_id
            
            # Simple LIKE search (case-insensitive)
            search_pattern = f"%{query}%"
            
            async with con.execute(
                """SELECT id, role, content, timestamp, user_id 
                FROM conversations 
                WHERE user_id = ? AND content LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?""",
                (search_user_id, search_pattern, limit),
            ) as cursor:
                rows = await cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "user_id": row["user_id"]
                })
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    async def get_conversation_summary(self, days_back: int = 7, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of conversations within a date range."""
        con = await self._get_db_connection()
        try:
            # Use the provided user_id or default to current user
            summary_user_id = user_id or self.user_id
            
            # Calculate date threshold
            if days_back > 0:
                date_threshold = (datetime.now() - timedelta(days=days_back)).isoformat()
                date_condition = "AND timestamp >= ?"
                params = (summary_user_id, date_threshold)
            else:
                date_condition = ""
                params = (summary_user_id,)
            
            # Get total count
            async with con.execute(
                f"""SELECT COUNT(*) as total 
                FROM conversations 
                WHERE user_id = ? {date_condition}""",
                params
            ) as cursor:
                row = await cursor.fetchone()
                total_messages = row["total"] if row else 0
            
            # Get breakdown by role
            async with con.execute(
                f"""SELECT role, COUNT(*) as count 
                FROM conversations 
                WHERE user_id = ? {date_condition}
                GROUP BY role""",
                params
            ) as cursor:
                rows = await cursor.fetchall()
            
            role_breakdown = {row["role"]: row["count"] for row in rows}
            
            # Get recent topics (most recent 5 user messages)
            async with con.execute(
                f"""SELECT content 
                FROM conversations 
                WHERE user_id = ? AND role = 'user' {date_condition}
                ORDER BY timestamp DESC 
                LIMIT 5""",
                params
            ) as cursor:
                rows = await cursor.fetchall()
            
            recent_topics = [row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"] 
                           for row in rows]
            
            summary = {
                "total_messages": total_messages,
                "user_messages": role_breakdown.get("user", 0),
                "assistant_messages": role_breakdown.get("assistant", 0),
                "days_back": days_back,
                "recent_topics": recent_topics,
                "user_id": summary_user_id
            }
            
            logger.info(f"Generated conversation summary: {total_messages} messages in {days_back} days")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {
                "error": str(e),
                "total_messages": 0,
                "user_messages": 0,
                "assistant_messages": 0
            }

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