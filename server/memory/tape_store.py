"""
SQLite-backed TapeStore for verbatim conversation history.

Schema:
- entries(ts REAL, speaker_id TEXT, role TEXT, content TEXT)

APIs:
- add_entry(role, content, speaker_id, ts)
- search(query, limit)
- get_recent(limit, since)
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class TapeEntry:
    ts: float
    speaker_id: str
    role: str
    content: str


class TapeStore:
    def __init__(self, db_path: str = "data/tape.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info(f"🧾 TapeStore initialized: {self.db_path}")

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                ts REAL NOT NULL,
                speaker_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entries_ts ON entries(ts DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entries_role ON entries(role)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entries_content ON entries(content)")
        # Session summaries table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                turns INTEGER DEFAULT 0,
                duration_s INTEGER DEFAULT 0,
                summary TEXT NOT NULL,
                keywords TEXT DEFAULT '[]'
            )
            """
        )
        self.conn.commit()

    def add_entry(self, role: str, content: str, speaker_id: str = "default_user", ts: Optional[float] = None):
        try:
            ts = ts or time.time()
            self.conn.execute(
                "INSERT INTO entries(ts, speaker_id, role, content) VALUES(?,?,?,?)",
                (ts, speaker_id, role, content)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"TapeStore add_entry failed: {e}")

    def search(self, query: str, limit: int = 10) -> List[TapeEntry]:
        try:
            like = f"%{query}%"
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT ts, speaker_id, role, content
                FROM entries
                WHERE content LIKE ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (like, limit)
            )
            rows = cur.fetchall()
            return [TapeEntry(ts=row["ts"], speaker_id=row["speaker_id"], role=row["role"], content=row["content"]) for row in rows]
        except Exception as e:
            logger.error(f"TapeStore search failed: {e}")
            return []

    def get_recent(self, limit: int = 10, since: Optional[float] = None) -> List[TapeEntry]:
        try:
            cur = self.conn.cursor()
            if since is None:
                cur.execute(
                    "SELECT ts, speaker_id, role, content FROM entries ORDER BY ts DESC LIMIT ?",
                    (limit,)
                )
            else:
                cur.execute(
                    "SELECT ts, speaker_id, role, content FROM entries WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                    (since, limit)
                )
            rows = cur.fetchall()
            return [TapeEntry(ts=row["ts"], speaker_id=row["speaker_id"], role=row["role"], content=row["content"]) for row in rows]
        except Exception as e:
            logger.error(f"TapeStore get_recent failed: {e}")
            return []

    # Convenience helpers for session windows
    def get_entries_since(self, since_ts: float) -> List[TapeEntry]:
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT ts, speaker_id, role, content FROM entries WHERE ts >= ? ORDER BY ts ASC",
                (since_ts,)
            )
            rows = cur.fetchall()
            return [TapeEntry(ts=row["ts"], speaker_id=row["speaker_id"], role=row["role"], content=row["content"]) for row in rows]
        except Exception as e:
            logger.error(f"TapeStore get_entries_since failed: {e}")
            return []

    # Session summaries API
    def add_summary(self, session_id: str, summary_text: str, keywords_json: str = "[]", turns: int = 0, duration_s: int = 0):
        try:
            ts = time.time()
            self.conn.execute(
                "REPLACE INTO summaries(session_id, ts, turns, duration_s, summary, keywords) VALUES(?,?,?,?,?,?)",
                (session_id, ts, turns, duration_s, summary_text, keywords_json),
            )
            self.conn.commit()
            logger.info(f"🧾 TapeStore: stored session summary (turns={turns}, duration={duration_s}s)")
        except Exception as e:
            logger.error(f"TapeStore add_summary failed: {e}")

    def get_last_summary(self) -> Optional[sqlite3.Row]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM summaries ORDER BY ts DESC LIMIT 1")
            return cur.fetchone()
        except Exception as e:
            logger.error(f"TapeStore get_last_summary failed: {e}")
            return None

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
