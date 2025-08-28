"""
Graph-Native SurrealDB Memory System

This module provides a complete graph-based memory implementation
leveraging SurrealDB's native graph capabilities with relationships.
"""

import os
import time
import asyncio
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    logger.warning("SurrealDB client not available")
    SURREALDB_AVAILABLE = False
    AsyncSurreal = None

@dataclass
class GraphFact:
    """Fact represented as a graph relationship"""
    subject: str
    predicate: str
    value: Optional[str] = None
    fidelity: int = 3
    strength: float = 0.6
    source_text: str = ""
    learned_from_session: Optional[str] = None

@dataclass 
class GraphMessage:
    """Message in conversation flow"""
    content: str
    speaker_type: str  # 'user' or 'assistant'
    session_id: str
    timestamp: float = 0
    sequence_num: int = 0
    embedding: Optional[List[float]] = None
    sender_id: Optional[str] = None

@dataclass
class GraphSession:
    """Conversation session with user relationship"""
    user_name: str
    agent_id: str = "slowcat"
    summary: str = ""
    keywords: List[str] = None
    status: str = "active"
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class GraphSurrealMemory:
    """
    Graph-native SurrealDB memory system using relationships
    
    This class provides:
    - User -> knows -> Concept relationships for facts
    - Session -> contains -> Message relationships for conversations  
    - Session -> reflects -> Thought relationships for reflections
    - Message -> mentions -> Concept relationships for content links
    """
    
    def __init__(self):
        self.db = None
        self.connected = False
        
        # Performance counters
        self.new_facts = 0
        self.reinforcements = 0
        # Track a lightweight active session per user for compatibility APIs
        self._active_sessions: Dict[str, str] = {}
        # Expose legacy-compatible adapters used by SmartContextManager + QueryRouter
        # - tape_store: for add_entry/search_tape/get_recent
        # - facts_graph: for search_facts/get_facts
        self.tape_store = self
        self.facts_graph = self
        
    # ----------------------------------------
    # Result normalization helper (client-safe)
    # ----------------------------------------
    def _rows_from_query(self, res: Any) -> List[Dict[str, Any]]:
        try:
            if not res:
                return []
            if isinstance(res, list):
                first = res[0]
                if isinstance(first, dict) and 'result' in first and isinstance(first['result'], list):
                    return first['result']
                if isinstance(first, dict) and 'status' not in first and 'result' not in first:
                    return res  # type: ignore[return-value]
                rows: List[Dict[str, Any]] = []
                for item in res:
                    if isinstance(item, dict) and 'result' in item and isinstance(item['result'], list):
                        rows.extend(item['result'])
                return rows
            if isinstance(res, dict) and 'result' in res:
                r = res.get('result')
                return r if isinstance(r, list) else []
        except Exception:
            pass
        return []
        
    async def connect(self):
        """Connect to SurrealDB"""
        if not SURREALDB_AVAILABLE:
            raise RuntimeError("SurrealDB not available")
        
        url = os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc')
        user = os.getenv('SURREALDB_USER', 'root')
        password = os.getenv('SURREALDB_PASS', 'slowcat_secure_2024')
        namespace = os.getenv('SURREALDB_NAMESPACE', 'slowcat')
        database = os.getenv('SURREALDB_DATABASE', 'memory')
        
        logger.info(f"Connecting to SurrealDB at {url} ns={namespace} db={database}")
        self.db = AsyncSurreal(url)
        await self.db.connect()
        # Try multiple auth payloads for client compatibility
        try:
            await self.db.signin({"user": user, "pass": password})
        except Exception:
            try:
                await self.db.signin({"username": user, "password": password})
            except Exception:
                # Allow unauthenticated if server permits
                pass
        await self.db.use(namespace, database)

        self.connected = True
        logger.info(f"Connected to SurrealDB: {namespace}.{database}")
        # Remember ns/db for explicit reuse before critical queries
        self._ns = namespace
        self._db = database
        # Ensure important indexes/constraints exist (idempotent)
        try:
            await self._ensure_indexes()
        except Exception as e:
            logger.debug(f"[GraphMemory] ensure_indexes skipped: {e}")

    async def _ensure_indexes(self):
        """Define minimal indexes/constraints for graph schema (idempotent)."""
        stmts = [
            # Message: fast lookup by session and time; uniqueness on (session_id, sequence_num)
            "DEFINE INDEX IF NOT EXISTS message_by_session ON TABLE message COLUMNS session_id;",
            "DEFINE INDEX IF NOT EXISTS message_by_time ON TABLE message COLUMNS timestamp;",
            "DEFINE INDEX IF NOT EXISTS message_unique_seq ON TABLE message COLUMNS session_id, sequence_num UNIQUE;",
            # Session: fast lookup by user
            "DEFINE INDEX IF NOT EXISTS session_by_user ON TABLE session COLUMNS user_id;",
            # Concept: speed up name lookups
            "DEFINE INDEX IF NOT EXISTS concept_by_name ON TABLE concept COLUMNS name;",
            # Fragments table (semantic/episodic memory)
            "DEFINE TABLE IF NOT EXISTS fragments SCHEMAFULL;",
            "DEFINE FIELD fragment_id ON fragments TYPE string;",
            "DEFINE FIELD user_id ON fragments TYPE string;",
            "DEFINE FIELD type ON fragments TYPE string;",
            "DEFINE FIELD content ON fragments TYPE object;",
            "DEFINE FIELD subject ON fragments TYPE option<string>;",
            "DEFINE FIELD predicate ON fragments TYPE option<string>;",
            "DEFINE FIELD value ON fragments TYPE option<string>;",
            "DEFINE FIELD context_tags ON fragments TYPE array<string>;",
            "DEFINE FIELD strength ON fragments TYPE number;",
            "DEFINE FIELD last_accessed ON fragments TYPE option<datetime>;",
            "DEFINE FIELD access_count ON fragments TYPE number DEFAULT 0;",
            "DEFINE FIELD source_interactions ON fragments TYPE array<string> DEFAULT [];",
            "DEFINE FIELD created_at ON fragments TYPE datetime DEFAULT time::now();",
            "DEFINE FIELD session_id ON fragments TYPE option<string>;",
            "DEFINE FIELD turn ON fragments TYPE option<number>;",
            "DEFINE FIELD message_id ON fragments TYPE option<string>;",
            "DEFINE FIELD role ON fragments TYPE option<string>;",
            "DEFINE INDEX IF NOT EXISTS fragments_by_user ON TABLE fragments COLUMNS user_id;",
            "DEFINE INDEX IF NOT EXISTS fragments_by_strength ON TABLE fragments COLUMNS strength;",
            "DEFINE INDEX IF NOT EXISTS fragments_by_last ON TABLE fragments COLUMNS last_accessed;",
            "DEFINE INDEX IF NOT EXISTS fragments_by_session_turn ON TABLE fragments COLUMNS user_id, session_id, turn;",
        ]
        for q in stmts:
            try:
                await self.db.query(q)
            except Exception as e:
                # Continue attempting remaining indexes
                logger.debug(f"[GraphMemory] index define failed: {e}")
        # Best-effort: ensure doc facts table exists
        try:
            await self.db.query("DEFINE TABLE fact_plain SCHEMALESS;")
        except Exception:
            pass

    # Public helpers for session context (used by SCM for fragment annotations)
    async def get_active_session_id_for_user(self, user_name: str) -> Optional[str]:
        try:
            sid = await self._ensure_active_session(user_name)
            return sid
        except Exception:
            return None

    async def get_message_count_for_session(self, session_id: str) -> int:
        try:
            key = session_id.split(':', 1)[1] if ':' in session_id else session_id
            res = await self.db.query(
                "SELECT count() FROM message WHERE session_id = type::thing('session', $k) GROUP ALL",
                {'k': key},
            )
            rows = self._rows_from_query(res)
            return int(rows[0].get('count', 0)) if rows else 0
        except Exception:
            return 0
    
    # ========================================
    # User Management (Graph Nodes)
    # ========================================
    
    async def ensure_user(self, user_name: str) -> str:
        """Ensure user exists, create if needed. Returns user ID."""
        if not self.connected:
            await self.connect()
        
        user_id = f"user:{self._safe_id(user_name)}"
        
        try:
            # Prefer client-native select/create on things
            try:
                existing = await self.db.select(user_id)
                if existing:
                    return user_id
            except Exception:
                # Fallback to query path
                result = await self.db.query("SELECT * FROM $user_id", {"user_id": user_id})
                rows = self._rows_from_query(result)
                if rows:
                    return user_id

            # Create new user
            try:
                await self.db.create(user_id, {
                    'name': user_name,
                    'first_seen': None,  # server default time::now()
                    'last_seen': None,
                    'total_interactions': 0,
                    'metadata': {},
                })
            except Exception:
                await self.db.query(
                    """
                    CREATE $user_id SET
                        name = $name,
                        first_seen = time::now(),
                        last_seen = time::now(),
                        total_interactions = 0,
                        metadata = {}
                    """,
                    {"user_id": user_id, "name": user_name},
                )
            logger.debug(f"Created user: {user_id}")
            return user_id

        except Exception as e:
            logger.error(f"Failed to ensure user {user_name}: {e}")
            raise
    
    # ========================================
    # Session Management (Graph Relationships)
    # ========================================
    
    async def create_session(self, session: GraphSession) -> str:
        """Create a new session with user relationship"""
        if not self.connected:
            await self.connect()
        
        user_id = await self.ensure_user(session.user_name)
        session_key = str(int(time.time()))
        session_id = f"session:{session_key}"
        
        try:
            # Store user_id as a record id string (e.g., "user:alice"). This works across
            # driver versions and matches Surrealist results in your DB.
            result = await self.db.query(
                """
                CREATE type::thing('session', $session_key) SET
                    user_id = $user_id_str,
                    agent_id = $agent_id,
                    started_at = time::now(),
                    turn_count = 0,
                    summary = $summary,
                    keywords = $keywords,
                    status = $status
                """,
                {
                    "session_key": session_key,
                    "user_id_str": user_id,
                    "agent_id": session.agent_id,
                    "summary": session.summary,
                    "keywords": session.keywords,
                    "status": session.status,
                },
            )
            
            logger.info(f"[GraphMemory] SESSION CREATED id={session_id} user={session.user_name}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_user_sessions(self, user_name: str, limit: int = 10) -> List[Dict]:
        """Get sessions for a user via client-side filtering (most robust)."""
        if not self.connected:
            await self.connect()
        # Ensure context is set for engines/clients sensitive to implicit state
        try:
            await self.db.use(getattr(self, '_ns', os.getenv('SURREALDB_NAMESPACE', 'slowcat')), getattr(self, '_db', os.getenv('SURREALDB_DATABASE', 'memory_graph')))
        except Exception:
            pass

        user_key = self._safe_id(user_name)
        target = f"user:{user_key}"

        try:
            # Fetch and filter client-side using Python, matching db_probe behavior.
            rows_all = []
            try:
                rows_all = await self.db.select('session')  # type: ignore[attr-defined]
            except Exception:
                # Fallback: plain query without WHERE and filter locally
                res = await self.db.query("SELECT id, started_at, turn_count, summary, status, user_id FROM session ORDER BY started_at DESC LIMIT 200")
                rows_all = self._rows_from_query(res)

            filtered = []
            for r in rows_all or []:
                try:
                    uid = r.get('user_id') if isinstance(r, dict) else getattr(r, 'user_id', None)
                    if str(uid) == target:
                        filtered.append(r)
                except Exception:
                    continue
            # Already ordered by started_at desc in query; if using select(), best-effort sort
            def _ts(row):
                v = row.get('started_at') if isinstance(row, dict) else getattr(row, 'started_at', None)
                try:
                    return v.timestamp() if hasattr(v, 'timestamp') else float(v)
                except Exception:
                    return 0.0
            if filtered and not isinstance(filtered[0].get('started_at', None), (float, int)):
                filtered.sort(key=_ts, reverse=True)
            return filtered[: max(1, int(limit))]

        except Exception as e:
            logger.error(f"Failed to get sessions for {user_name}: {e}")
            return []

    # --- Compatibility: FactsGraph-style session APIs used by SCM ---
    async def start_session(self, speaker_id: str):
        """Start a NEW session for the speaker and set it active.

        This increments the user's session_count by creating a fresh session row.
        """
        if not self.connected:
            await self.connect()
        try:
            # Always create a new session
            new_sid = await self.create_session(GraphSession(user_name=speaker_id))
            # Mark as active for compatibility APIs
            self._active_sessions[speaker_id] = new_sid
            return new_sid
        except Exception as e:
            logger.debug(f"start_session failed: {e}")
            return None

    async def update_session(self, speaker_id: str):
        """Update last interaction and increment turn count for active session."""
        if not self.connected:
            await self.connect()
        try:
            sid = await self._ensure_active_session(speaker_id)
            # Increment turn_count on session
            try:
                sess_key = sid.split(':', 1)[1] if ':' in sid else sid
                await self.db.query(
                    """
                    UPDATE type::thing('session', $session_key) SET turn_count = (turn_count ?? 0) + 1
                    """,
                    {"session_key": sess_key},
                )
            except Exception:
                await self.db.query(
                    """
                    UPDATE $sid SET turn_count = (turn_count ?? 0) + 1
                    """,
                    {"sid": sid},
                )
        except Exception as e:
            logger.debug(f"update_session failed: {e}")
    
    # ========================================
    # Message Management (Graph Relationships)
    # ========================================
    
    async def add_message(self, message: GraphMessage) -> str:
        """Add message with session relationship"""
        if not self.connected:
            await self.connect()
        
        try:
            # Use SurrealQL with RETURN AFTER to reliably get the id
            emb = message.embedding if isinstance(message.embedding, list) else []
            # Bind session as a record id via type::thing to satisfy SCHEMAFULL record<session>
            try:
                sess_key = message.session_id.split(':', 1)[1] if isinstance(message.session_id, str) and ':' in message.session_id else str(message.session_id)
            except Exception:
                sess_key = str(message.session_id)
            res = await self.db.query(
                """
                CREATE message SET
                    session_id = type::thing('session', $session_key),
                    speaker_type = $speaker_type,
                    role = $speaker_type,
                    content = $content,
                    raw_content = $content,
                    timestamp = time::now(),
                    sequence_num = $sequence_num,
                    embedding = $embedding,
                    sender_id = $sender_id,
                    message_type = 'conversation',
                    metadata = {}
                RETURN id
                """,
                {
                    "session_key": sess_key,
                    "speaker_type": message.speaker_type,
                    "content": message.content,
                    "sequence_num": message.sequence_num,
                    "embedding": emb,
                    "sender_id": message.sender_id or 'unknown',
                },
            )
            rows = self._rows_from_query(res)
            message_id: Optional[str] = None
            if rows:
                first = rows[0]
                try:
                    if isinstance(first, dict) and first.get('id') is not None:
                        rid = first.get('id')
                        message_id = rid if isinstance(rid, str) else str(rid)
                    elif isinstance(first, str):
                        message_id = first
                    else:
                        # Some drivers return a list of ids under a field
                        for k in ('id', 'result', 'value'):
                            v = first.get(k) if isinstance(first, dict) else None
                            if v:
                                message_id = v if isinstance(v, str) else str(v)
                                break
                except Exception:
                    message_id = None

            # Fallback: query by (session_id, sequence_num)
            if not message_id:
                try:
                    # Bind session as a typed record to match SCHEMAFULL
                    try:
                        sess_key = message.session_id.split(':', 1)[1] if isinstance(message.session_id, str) and ':' in message.session_id else str(message.session_id)
                    except Exception:
                        sess_key = str(message.session_id)
                    # Rely on unique(session_id, sequence_num); no ORDER BY needed.
                    res2 = await self.db.query(
                        """
                        SELECT id FROM message
                        WHERE session_id = type::thing('session', $session_key)
                          AND sequence_num = $sequence_num
                        LIMIT 1
                        """,
                        {"session_key": sess_key, "sequence_num": message.sequence_num},
                    )
                    rows2 = self._rows_from_query(res2)
                    if rows2:
                        r0 = rows2[0]
                        if isinstance(r0, dict) and r0.get('id') is not None:
                            rid = r0.get('id')
                            message_id = rid if isinstance(rid, str) else str(rid)
                        elif isinstance(r0, str):
                            message_id = r0
                except Exception as _e:
                    logger.debug(f"id fallback failed: {_e}")

            if not message_id:
                raise RuntimeError("Failed to determine created message id")
            
            # Create relationship: session -> contains -> message
            await self.db.query("""
                RELATE $session_id->contains->$message_id SET
                    sequence_num = $sequence_num,
                    created_at = time::now()
            """, {
                "session_id": message.session_id,
                "message_id": message_id,
                "sequence_num": message.sequence_num
            })
            
            # Update session turn count
            await self.db.query("""
                UPDATE $session_id SET
                    turn_count = (
                        SELECT count() FROM message WHERE session_id = $session_id
                    )[0]
            """, {"session_id": message.session_id})
            
            logger.debug(f"Added message: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise

    # ========================================
    # Compatibility APIs (TapeStore-style)
    # ========================================

    async def _ensure_active_session(self, speaker_id: str) -> str:
        """Get or create a currently active session for a speaker."""
        if not self.connected:
            await self.connect()

        # Return cached active session if present
        sid = self._active_sessions.get(speaker_id)
        if sid:
            return sid

        # Try to find the most recent session for this user
        sessions = await self.get_user_sessions(speaker_id, limit=1)
        if sessions:
            sid = sessions[0].get('id')
            if sid:
                self._active_sessions[speaker_id] = sid
                return sid

        # Otherwise create a new session
        new_sid = await self.create_session(GraphSession(user_name=speaker_id))
        self._active_sessions[speaker_id] = new_sid
        return new_sid

    async def add_entry(self, role: str, content: str, speaker_id: str = "default_user", ts: float | None = None, agent_id: str | None = None) -> str:
        """Compatibility wrapper: write conversation entries to message table.

        Maps legacy TapeStore.add_entry(...) to graph message/session records.
        """
        if not self.connected:
            await self.connect()

        try:
            session_id = await self._ensure_active_session(speaker_id)

            # Determine next sequence number
            seq_res = await self.db.query(
                "SELECT count() FROM message WHERE session_id = type::thing('session', $session_key) GROUP ALL",
                {"session_key": session_id.split(':', 1)[1] if ':' in session_id else session_id},
            )
            seq = 1
            try:
                rows = self._rows_from_query(seq_res)
                cnt = rows[0].get('count', 0) if rows else 0
                seq = int(cnt) + 1
            except Exception:
                seq = 1

            msg = GraphMessage(
                content=content,
                speaker_type='assistant' if role == 'assistant' else 'user',
                session_id=session_id,
                timestamp=ts or time.time(),
                sequence_num=seq,
                embedding=None,
                sender_id=speaker_id,
            )
            mid = await self.add_message(msg)
            logger.debug(f"add_entry: wrote message {mid} in session {session_id} for {speaker_id}")
            return mid
        except Exception as e:
            logger.error(f"Failed to add entry (compat): {e}")
            raise

    async def get_recent(self, limit: int = 10, since: float | None = None, agent_id: str | None = None) -> List[Dict]:
        """Compatibility wrapper: read recent messages like tape entries."""
        if not self.connected:
            await self.connect()

        try:
            if since is None:
                q = """
                    SELECT content, raw_content, role, speaker_type, sender_id, timestamp, session_id
                    FROM message
                    ORDER BY timestamp DESC
                    LIMIT $limit
                """
                params = {"limit": limit}
            else:
                q = """
                    SELECT content, raw_content, role, speaker_type, sender_id, timestamp, session_id
                    FROM message
                    WHERE timestamp >= time::from::secs($since)
                    ORDER BY timestamp DESC
                    LIMIT $limit
                """
                params = {"limit": limit, "since": since}

            res = await self.db.query(q, params)
            rows = self._rows_from_query(res)
            out: List[Dict] = []
            for row in rows:
                ts_val = row.get('timestamp')
                if hasattr(ts_val, 'timestamp'):
                    ts_float = ts_val.timestamp()
                elif hasattr(ts_val, 'timetuple'):
                    ts_float = time.mktime(ts_val.timetuple())
                elif isinstance(ts_val, (int, float)):
                    ts_float = float(ts_val)
                else:
                    ts_float = time.time()
                r = row.get('role') or row.get('speaker_type') or 'user'
                text = row.get('content') or row.get('raw_content') or ''
                out.append({
                    'ts': ts_float,
                    'speaker_id': row.get('sender_id') or 'unknown',
                    'role': 'assistant' if r == 'assistant' else 'user',
                    'content': text,
                    'session_id': row.get('session_id'),
                })
            return out
        except Exception as e:
            logger.error(f"Failed to get recent (compat): {e}")
            return []
    
    async def get_conversation(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation messages using graph traversal"""
        if not self.connected:
            await self.connect()
        
        try:
            result = await self.db.query("""
                SELECT 
                    content,
                    speaker_type,
                    timestamp,
                    sequence_num,
                    (->mentions->concept.*) as mentioned_concepts
                FROM $session_id->contains->message
                ORDER BY sequence_num ASC
                LIMIT $limit
            """, {"session_id": session_id, "limit": limit})
            return self._rows_from_query(result)
            
        except Exception as e:
            logger.error(f"Failed to get conversation for {session_id}: {e}")
            return []

    # ========================================
    # Facts Compatibility (for QueryRouter)
    # ========================================

    async def search_facts(self, query: str, limit: int = 10) -> List[GraphFact]:
        """Provide a FactsGraph-like search over knowledge relations."""
        if not self.connected:
            await self.connect()
        try:
            user_name = os.getenv('USER_ID', 'user')
            user_key = self._safe_id(user_name)
            # Attempt 1: direct traversal from user -> knows
            res = await self.db.query(
                """
                SELECT relationship, fidelity, strength,
                       out.name as concept_name,
                       out.kind as concept_kind,
                       source_text
                FROM type::thing('user', $user_key)->knows
                WHERE string::contains(string::lowercase(out.name), string::lowercase($q))
                   OR string::contains(string::lowercase(relationship), string::lowercase($q))
                   OR string::contains(string::lowercase(source_text), string::lowercase($q))
                ORDER BY strength DESC
                LIMIT $limit
                """,
                {"user_key": user_key, "q": query, "limit": limit},
            )
            rows = self._rows_from_query(res)
            if not rows:
                # Attempt 2: query relation table explicitly (knows), filter by IN edge
                # First try record-typed match on IN
                res_rel = await self.db.query(
                    """
                    SELECT relationship, fidelity, strength,
                           out.name AS concept_name,
                           source_text
                    FROM knows
                    WHERE in = type::thing('user', $user_key)
                    ORDER BY strength DESC
                    LIMIT $limit
                    """,
                    {"user_key": user_key, "limit": limit},
                )
                rows = self._rows_from_query(res_rel)
            if not rows:
                # Attempt 2b: string-based IN match for broader compatibility
                res_rel2 = await self.db.query(
                    """
                    SELECT relationship, fidelity, strength,
                           out.name AS concept_name,
                           source_text
                    FROM knows
                    WHERE string::concat('', in) = $user_id_str
                    ORDER BY strength DESC
                    LIMIT $limit
                    """,
                    {"user_id_str": f"user:{user_key}", "limit": limit},
                )
                rows = self._rows_from_query(res_rel2)
            if not rows:
                # Attempt 3: broader pull + client-side match on concept name
                res2 = await self.db.query(
                    """
                    SELECT relationship, fidelity, strength,
                           out.name as concept_name,
                           out.kind as concept_kind,
                           source_text
                    FROM knows
                    WHERE in = type::thing('user', $user_key)
                    ORDER BY strength DESC
                    LIMIT 100
                    """,
                    {"user_key": user_key},
                )
                rows = [r for r in self._rows_from_query(res2) if r.get('concept_name') and (str(query).lower() in str(r.get('concept_name','')).lower())]
                rows = rows[: max(1, int(limit))]

            if not rows:
                # Attempt 4: fallback to doc facts (fact_plain)
                res_plain = await self.db.query(
                    """
                    SELECT subject, predicate, value, fidelity, strength, source_text
                    FROM fact_plain
                    WHERE user_id = $user_id_str
                      AND (
                        string::contains(string::lowercase(value), string::lowercase($q)) OR
                        string::contains(string::lowercase(predicate), string::lowercase($q)) OR
                        string::contains(string::lowercase(source_text), string::lowercase($q))
                      )
                    ORDER BY strength DESC
                    LIMIT $limit
                    """,
                    {"user_id_str": f"user:{user_key}", "q": query, "limit": limit},
                )
                rows = self._rows_from_query(res_plain)

            facts: List[GraphFact] = []
            for r in rows:
                # Prefer concept_name (from graph queries); fallback to value (from fact_plain)
                val = r.get('concept_name') or r.get('value')
                facts.append(GraphFact(
                    subject=user_name,
                    predicate=r.get('relationship') or 'knows',
                    value=val,
                    fidelity=r.get('fidelity', 3),
                    strength=r.get('strength', 0.6),
                    source_text=r.get('source_text') or 'graph_relationship',
                ))
            return facts
        except Exception as e:
            logger.error(f"search_facts failed: {e}")
            return []

    # ========================================
    # Fragments (Semantic/Episodic) API
    # ========================================

    async def store_fragments_from_text(self, user_name: str, text: str, *, session_id: Optional[str] = None, turn: Optional[int] = None, role: str = 'user') -> int:
        """Extract simple semantic fragments from text and store them.

        Uses the existing spaCy fact extractor when available.
        """
        try:
            from memory.facts_graph import extract_facts_from_text as _extract
        except Exception:
            return 0
        facts = _extract(text) or []
        count = 0
        for f in facts:
            try:
                frag = {
                    'type': 'semantic',
                    'content': {
                        'subject': f.get('subject', user_name),
                        'predicate': f.get('predicate', ''),
                        'value': f.get('value', ''),
                    },
                    'subject': f.get('subject', user_name),
                    'predicate': f.get('predicate', ''),
                    'value': f.get('value', ''),
                    'context_tags': [
                        str(f.get('predicate', '')).lower(),
                        str(f.get('value', '')).lower(),
                        user_name.lower(),
                    ],
                    'strength': float(f.get('strength', 0.6)),
                    'session_id': session_id,
                    'turn': turn,
                    'role': role,
                }
                await self._insert_fragment(user_name, frag)
                count += 1
            except Exception:
                continue
        return count

    async def _insert_fragment(self, user_name: str, fragment: Dict[str, Any]) -> str:
        if not self.connected:
            await self.connect()
        import time as _t
        frag_id = f"frag:{int(_t.time()*1000)}"
        try:
            await self.db.query(
                """
                CREATE fragments SET
                    fragment_id = $fid,
                    user_id = $uid,
                    type = $type,
                    content = $content,
                    subject = $subject,
                    predicate = $predicate,
                    value = $value,
                    context_tags = $tags,
                    strength = $strength,
                    created_at = time::now(),
                    access_count = 0,
                    session_id = $session_id,
                    turn = $turn,
                    message_id = $message_id,
                    role = $role
                """,
                {
                    'fid': frag_id,
                    'uid': f"user:{self._safe_id(user_name)}",
                    'type': fragment.get('type', 'semantic'),
                    'content': fragment.get('content', {}),
                    'subject': fragment.get('subject'),
                    'predicate': fragment.get('predicate'),
                    'value': fragment.get('value'),
                    'tags': fragment.get('context_tags', []),
                    'strength': float(fragment.get('strength', 0.6)),
                    'session_id': fragment.get('session_id'),
                    'turn': fragment.get('turn'),
                    'message_id': fragment.get('message_id'),
                    'role': fragment.get('role'),
                },
            )
        except Exception:
            pass
        return frag_id

    async def search_fragments(self, query: str, limit: int = 10, user_name: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.connected:
            await self.connect()
        u = user_name or os.getenv('USER_ID', 'user')
        ukey = self._safe_id(u)
        try:
            res = await self.db.query(
                """
                SELECT fragment_id, type, content, subject, predicate, value, strength, context_tags, session_id, turn, role
                FROM fragments
                WHERE user_id = $uid
                  AND (
                    string::contains(string::lowercase(value ?? ''), string::lowercase($q)) OR
                    string::contains(string::lowercase(content.value ?? ''), string::lowercase($q)) OR
                    string::contains(string::lowercase(array::join(context_tags, ' ')), string::lowercase($q))
                  )
                ORDER BY strength DESC, created_at DESC
                LIMIT $limit
                """,
                {'uid': f"user:{ukey}", 'q': query, 'limit': limit},
            )
            rows = self._rows_from_query(res)
            out: List[Dict[str, Any]] = []
            for r in rows:
                c = r.get('content', {}) if isinstance(r, dict) else {}
                out.append({
                    'subject': r.get('subject') or c.get('subject', u),
                    'predicate': r.get('predicate') or c.get('predicate', 'related_to'),
                    'value': r.get('value') or c.get('value', ''),
                    'fidelity': 3,
                    'strength': float(r.get('strength', 0.6)),
                    'session_id': r.get('session_id'),
                    'turn': r.get('turn'),
                })
            return out
        except Exception:
            return []

    async def upsert_fragment(self, user_name: str, fragment: Dict[str, Any]) -> Optional[str]:
        """Idempotently insert or update a fragment for a user.

        Uniqueness is determined by (user_id, content.subject, content.predicate, content.value).
        If an existing fragment is found, this will update strength = max(old, new)
        and bump last_accessed/access_count.
        """
        if not self.connected:
            await self.connect()
        uid = f"user:{self._safe_id(user_name)}"
        subj = fragment.get('content', {}).get('subject', user_name)
        pred = fragment.get('content', {}).get('predicate', '')
        val = fragment.get('content', {}).get('value', '')
        try:
            sel = await self.db.query(
                """
                SELECT fragment_id, strength FROM fragments
                WHERE user_id = $uid
                  AND content.subject = $subj
                  AND content.predicate = $pred
                  AND content.value = $val
                LIMIT 1
                """,
                {'uid': uid, 'subj': subj, 'pred': pred, 'val': val},
            )
            rows = self._rows_from_query(sel)
            if rows:
                # Update existing
                fid = rows[0].get('fragment_id')
                try:
                    await self.db.query(
                        """
                        UPDATE fragments SET
                            strength = math::max(strength, $new_strength),
                            last_accessed = time::now(),
                            access_count = (access_count ?? 0) + 1
                        WHERE fragment_id = $fid AND user_id = $uid
                        """,
                        {'fid': fid, 'uid': uid, 'new_strength': float(fragment.get('strength', 0.6))},
                    )
                except Exception:
                    pass
                return fid
            # Insert new
            return await self._insert_fragment(user_name, fragment)
        except Exception:
            return None

    async def get_facts(self, subject: str = None, predicate: str = None, min_fidelity: int = 0, limit: int = 50) -> List[GraphFact]:
        """Return a simple list of facts; used by semantic fallback."""
        if not self.connected:
            await self.connect()
        try:
            user_name = os.getenv('USER_ID', 'user')
            user_key = self._safe_id(user_name)
            where = []
            params: Dict[str, Any] = {"user_id": user_id, "limit": limit, "min_fid": min_fidelity}
            if predicate:
                where.append("relationship = $pred")
                params["pred"] = predicate
            if min_fidelity:
                where.append("fidelity >= $min_fid")
            where_clause = (" WHERE " + " AND ".join(where)) if where else ""
            q = f"""
                SELECT relationship, fidelity, strength, out.name AS concept_name, source_text
                FROM type::thing('user', $user_key)->knows{where_clause}
                ORDER BY strength DESC
                LIMIT $limit
            """
            params = {"user_key": user_key, "limit": limit, "min_fid": min_fidelity, **({"pred": predicate} if predicate else {})}
            res = await self.db.query(q, params)
            rows = self._rows_from_query(res)
            return [GraphFact(
                subject=user_name,
                predicate=r.get('relationship') or 'knows',
                value=r.get('concept_name'),
                fidelity=r.get('fidelity', 3),
                strength=r.get('strength', 0.6),
                source_text=r.get('source_text') or 'graph_relationship',
            ) for r in rows]
        except Exception:
            return []
    
    # ========================================
    # Knowledge Management (Graph Relationships)
    # ========================================
    
    async def store_fact(self, user_name: str, fact: GraphFact) -> bool:
        """Store fact as user->knows->concept relationship"""
        if not self.connected:
            await self.connect()
        
        try:
            user_id = await self.ensure_user(user_name)
            concept_id = await self._ensure_concept(fact.value, self._infer_concept_kind(fact))
            
            # Check if relationship already exists
            existing = await self.db.query(
                "SELECT * FROM $user_id->knows WHERE out = $concept_id",
                {"user_id": user_id, "concept_id": concept_id},
            )
            rows = self._rows_from_query(existing)
            if rows:
                # Reinforce existing relationship
                await self.db.query("""
                    UPDATE $user_id->knows SET
                        strength = math::max(strength * 0.7 + $new_strength * 0.3, strength),
                        fidelity = math::max(fidelity, $fidelity),
                        reinforced_at = time::now(),
                        access_count = access_count + 1
                    WHERE out = $concept_id
                """, {
                    "user_id": user_id,
                    "concept_id": concept_id,
                    "new_strength": fact.strength,
                    "fidelity": fact.fidelity
                })
                
                self.reinforcements += 1
                logger.debug(f"Reinforced: {user_name} knows {fact.value}")
                updated = True
            else:
                # Create new knowledge relationship
                await self.db.query(
                    """
                    RELATE $user_id->knows->$concept_id SET
                        relationship = $predicate,
                        fidelity = $fidelity,
                        strength = $strength,
                        learned_at = time::now(),
                        reinforced_at = time::now(),
                        source_text = $source_text,
                        learned_from_session = $learned_sid,
                        access_count = 0,
                        decay_rate = 1.0
                    """,
                    {
                        "user_id": user_id,
                        "concept_id": concept_id,
                        "predicate": fact.predicate,
                        "fidelity": fact.fidelity,
                        "strength": fact.strength,
                        "source_text": fact.source_text,
                        "learned_sid": fact.learned_from_session,
                    },
                )
                
                self.new_facts += 1
                logger.debug(f"New knowledge: {user_name} {fact.predicate} {fact.value}")
                updated = False
                
        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            updated = False
        # Also persist a doc fact row for compatibility/fallback
        try:
            await self.db.query(
                """
                CREATE fact_plain SET
                    user_id = $user_id_str,
                    subject = $subject,
                    predicate = $predicate,
                    value = $value,
                    fidelity = $fidelity,
                    strength = $strength,
                    source_text = $source_text,
                    created = time::now()
                """,
                {
                    "user_id_str": f"user:{self._safe_id(user_name)}",
                    "subject": user_name,
                    "predicate": fact.predicate,
                    "value": fact.value,
                    "fidelity": fact.fidelity,
                    "strength": fact.strength,
                    "source_text": fact.source_text,
                },
            )
        except Exception:
            pass
        return updated
    
    async def get_user_knowledge(self, user_name: str, limit: int = 20) -> List[Dict]:
        """Get user's knowledge using graph traversal"""
        if not self.connected:
            await self.connect()
        
        user_id = f"user:{self._safe_id(user_name)}"
        
        try:
            result = await self.db.query("""
                SELECT 
                    relationship,
                    fidelity,
                    strength,
                    learned_at,
                    reinforced_at,
                    access_count,
                    out.name as concept_name,
                    out.kind as concept_kind,
                    source_text
                FROM $user_id->knows
                ORDER BY strength DESC, reinforced_at DESC
                LIMIT $limit
            """, {"user_id": user_id, "limit": limit})
            return self._rows_from_query(result)
            
        except Exception as e:
            logger.error(f"Failed to get knowledge for {user_name}: {e}")
            return []
    
    async def search_knowledge(self, user_name: str, query: str) -> List[Dict]:
        """Search user's knowledge graph"""
        if not self.connected:
            await self.connect()
        
        user_id = f"user:{self._safe_id(user_name)}"
        
        try:
            result = await self.db.query("""
                SELECT 
                    relationship,
                    fidelity,
                    strength,
                    out.name as concept_name,
                    out.kind as concept_kind
                FROM $user_id->knows
                WHERE 
                    string::contains(string::lowercase(out.name), string::lowercase($query))
                    OR string::contains(string::lowercase(relationship), string::lowercase($query))
                    OR string::contains(string::lowercase(source_text), string::lowercase($query))
                ORDER BY strength DESC
                LIMIT 10
            """, {"user_id": user_id, "query": query})
            return self._rows_from_query(result)
            
        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []
    
    # ========================================
    # Search Operations
    # ========================================
    
    async def search_conversations(self, user_name: str, query: str, limit: int = 10) -> List[Dict]:
        """Search conversations using full-text search"""
        if not self.connected:
            await self.connect()
        
        user_id = f"user:{self._safe_id(user_name)}"
        
        try:
            # Try FTS first
            result = await self.db.query("""
                SELECT 
                    session_id,
                    content,
                    raw_content,
                    speaker_type,
                    role,
                    timestamp,
                    (SELECT summary FROM session WHERE id = $parent.session_id)[0] as session_summary
                FROM message 
                WHERE content @@ $query OR raw_content @@ $query
                    AND session_id IN (
                        SELECT id FROM session WHERE user_id = $user_id
                    )
                ORDER BY timestamp DESC
                LIMIT $limit
            """, {"query": query, "user_id": user_id, "limit": limit})
            rows = self._rows_from_query(result)
            if rows:
                return rows
                
        except Exception:
            # Fallback to contains search
            result = await self.db.query("""
                SELECT 
                    session_id,
                    content,
                    raw_content,
                    speaker_type,
                    role,
                    timestamp
                FROM message 
                WHERE string::contains(string::lowercase(content), string::lowercase($query))
                   OR string::contains(string::lowercase(raw_content), string::lowercase($query))
                    AND session_id IN (
                        SELECT id FROM session WHERE user_id = $user_id
                    )
                ORDER BY timestamp DESC
                LIMIT $limit
            """, {"query": query, "user_id": user_id, "limit": limit})
            return self._rows_from_query(result)
        
        return []
    
    # ========================================
    # Graph Analytics
    # ========================================
    
    async def get_related_concepts(self, user_name: str, concept_name: str, depth: int = 2) -> List[Dict]:
        """Find concepts related through user's knowledge graph"""
        if not self.connected:
            await self.connect()
        
        user_id = f"user:{self._safe_id(user_name)}"
        
        try:
            if depth == 1:
                # Direct relationships
                result = await self.db.query("""
                    SELECT DISTINCT
                        out.name as concept_name,
                        out.kind as concept_kind,
                        relationship,
                        strength
                    FROM $user_id->knows
                    WHERE string::contains(string::lowercase(out.name), string::lowercase($concept))
                    ORDER BY strength DESC
                """, {
                    "user_id": user_id,
                    "concept": concept_name
                })
            else:
                # Multi-hop relationships (simplified)
                result = await self.db.query("""
                    SELECT DISTINCT
                        out.name as concept_name,
                        out.kind as concept_kind,
                        relationship
                    FROM $user_id->knows
                    WHERE out.kind = (
                        SELECT out.kind FROM $user_id->knows 
                        WHERE string::contains(string::lowercase(out.name), string::lowercase($concept))
                        LIMIT 1
                    )[0]
                    ORDER BY strength DESC
                    LIMIT 10
                """, {
                    "user_id": user_id,
                    "concept": concept_name
                })
            
            return result[0].get('result', [])
            
        except Exception as e:
            logger.error(f"Failed to get related concepts: {e}")
            return []
    
    async def get_conversation_context(self, session_id: str) -> Dict:
        """Get rich context for a conversation including related knowledge"""
        if not self.connected:
            await self.connect()
        
        try:
            # Get session info with user details
            session_result = await self.db.query("""
                SELECT 
                    *,
                    user_id.name as user_name,
                    (SELECT count() FROM message WHERE session_id = $parent.id)[0] as message_count
                FROM $session_id
            """, {"session_id": session_id})
            session_rows = self._rows_from_query(session_result)
            if not session_rows:
                return {}
            session_info = session_rows[0]
            user_name = session_info.get('user_name', 'unknown')
            
            # Get recent messages
            messages = await self.get_conversation(session_id, limit=20)
            
            # Get user's relevant knowledge
            knowledge = await self.get_user_knowledge(user_name, limit=10)
            
            return {
                'session': session_info,
                'messages': messages,
                'user_knowledge': knowledge,
                'user_name': user_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return {}

    async def get_session_info(self, speaker_id: str) -> Dict:
        """Provide session metadata compatible with legacy API using graph tables.

        Returns keys: session_count, last_interaction, first_seen, total_turns
        """
        if not self.connected:
            await self.connect()
        try:
            user_key = self._safe_id(speaker_id)
            # 1) Count sessions
            res1 = await self.db.query(
                "SELECT count() AS c FROM session WHERE user_id = type::thing('user', $uk)",
                {"uk": user_key},
            )
            rows1 = self._rows_from_query(res1)
            session_count = int((rows1[0].get('c') if rows1 else 0) or 0)

            # 2) Sum total turns
            res2 = await self.db.query(
                "SELECT turn_count FROM session WHERE user_id = type::thing('user', $uk)",
                {"uk": user_key},
            )
            rows2 = self._rows_from_query(res2)
            total_turns = 0
            for r in rows2:
                try:
                    total_turns += int(r.get('turn_count') or 0)
                except Exception:
                    continue

            # 3) Last interaction: latest message timestamp across user's sessions
            res3 = await self.db.query(
                """
                SELECT timestamp FROM message
                WHERE session_id IN (SELECT id FROM session WHERE user_id = type::thing('user', $uk))
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                {"uk": user_key},
            )
            rows3 = self._rows_from_query(res3)
            last_interaction = None
            if rows3:
                v = rows3[0].get('timestamp')
                try:
                    if hasattr(v, 'timestamp'):
                        last_interaction = v.timestamp()
                    elif isinstance(v, (int, float)):
                        last_interaction = float(v)
                except Exception:
                    last_interaction = None

            # 4) First seen: earliest session start
            res4 = await self.db.query(
                """
                SELECT started_at FROM session
                WHERE user_id = type::thing('user', $uk)
                ORDER BY started_at ASC
                LIMIT 1
                """,
                {"uk": user_key},
            )
            rows4 = self._rows_from_query(res4)
            first_seen = None
            if rows4:
                v = rows4[0].get('started_at')
                try:
                    if hasattr(v, 'timestamp'):
                        first_seen = v.timestamp()
                    elif isinstance(v, (int, float)):
                        first_seen = float(v)
                except Exception:
                    first_seen = None

            info_obj = {
                'session_count': session_count,
                'total_turns': total_turns,
                'last_interaction': last_interaction,
                'first_seen': first_seen,
            }
            if session_count > 0:
                logger.info(f"[GraphMemory] SESSION INFO speaker={speaker_id} sessions={session_count} turns={total_turns}")
            return info_obj
        except Exception as e:
            logger.debug(f"get_session_info (graph) failed: {e}")
            return {
                'session_count': 0,
                'total_turns': 0,
                'last_interaction': None,
                'first_seen': None,
            }
    
    # ========================================
    # Helper Methods
    # ========================================
    
    async def _ensure_concept(self, concept_name: str, kind: str = "unknown") -> str:
        """Ensure concept exists, create if needed"""
        concept_id = f"concept:{self._safe_id(concept_name)}"
        
        try:
            # Check if exists using type::thing to bind record id safely
            res = await self.db.query(
                "SELECT * FROM type::thing('concept', $cid)",
                {"cid": concept_id.split(':',1)[1] if ':' in concept_id else concept_id},
            )
            rows = self._rows_from_query(res)
            if rows:
                return concept_id
            
            # Create new concept
            await self.db.query(
                """
                CREATE type::thing('concept', $cid) SET
                    name = $name,
                    kind = $kind,
                    properties = {},
                    mentioned_count = 0,
                    first_mentioned = time::now(),
                    last_mentioned = time::now()
                """,
                {
                    "cid": concept_id.split(':',1)[1] if ':' in concept_id else concept_id,
                    "name": concept_name,
                    "kind": kind,
                },
            )
            
            return concept_id
            
        except Exception as e:
            logger.error(f"Failed to ensure concept {concept_name}: {e}")
            raise
    
    def _safe_id(self, name: str) -> str:
        """Create a safe ID from a name"""
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(name).lower())
        return safe_name[:50]
    
    def _infer_concept_kind(self, fact: GraphFact) -> str:
        """Infer concept kind from fact predicate"""
        predicate_map = {
            'has_pet': 'pet',
            'dog_name': 'pet',
            'likes': 'preference',
            'location': 'place',
            'name': 'person',
            'lives_in': 'place'
        }
        return predicate_map.get(fact.predicate, 'unknown')
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
            self.connected = False

    # TapeStore compatibility: ignore or lightly persist session summaries
    async def add_summary(self, session_id: str, summary: str, keywords_json: str = '[]', turns: int = 0, duration_s: int = 0):
        """Compatibility shim for SmartContextManager.finalize_summary.

        Graph schema stores rolling summaries on session records; this shim
        avoids errors when SCM tries to persist a legacy summary row.
        """
        try:
            if not self.connected:
                await self.connect()
            # Best-effort: if there is an active session for the configured user, update its summary
            user_name = os.getenv('USER_ID', 'user')
            sid = self._active_sessions.get(user_name)
            if sid:
                await self.db.query(
                    "UPDATE $sid SET summary = $summary",
                    {"sid": sid, "summary": summary or ''},
                )
        except Exception as e:
            logger.debug(f"add_summary shim skipped: {e}")
    
    # ========================================
    # Legacy Compatibility Methods
    # ========================================
    
    async def reinforce_or_insert(self, fact_data: Dict) -> bool:
        """Legacy compatibility method"""
        fact = GraphFact(
            subject=fact_data.get('subject', 'user'),
            predicate=fact_data.get('predicate', 'related_to'),
            value=fact_data.get('value'),
            fidelity=fact_data.get('fidelity', 3),
            strength=fact_data.get('strength', 0.6),
            source_text=fact_data.get('source_text', '')
        )
        return await self.store_fact(fact.subject, fact)
    
    async def search_tape(self, query: str, limit: int = 10, agent_id: str = None) -> List[Dict]:
        """Legacy compatibility method - searches all users"""
        # This is simplified - in practice you'd specify user
        if not self.connected:
            await self.connect()
        
        try:
            result = await self.db.query("""
                SELECT 
                    content,
                    raw_content,
                    speaker_type as role,
                    timestamp as ts,
                    session_id
                FROM message 
                WHERE content @@ $query OR raw_content @@ $query
                ORDER BY timestamp DESC
                LIMIT $limit
            """, {"query": query, "limit": limit})
            return self._rows_from_query(result)
            
        except Exception:
            # Fallback
            result = await self.db.query("""
                SELECT 
                    content,
                    raw_content,
                    speaker_type as role,
                    timestamp as ts,
                    session_id
                FROM message 
                WHERE string::contains(string::lowercase(content), string::lowercase($query))
                   OR string::contains(string::lowercase(raw_content), string::lowercase($query))
                ORDER BY timestamp DESC
                LIMIT $limit
            """, {"query": query, "limit": limit})
            rows = self._rows_from_query(result)
            # Normalize role and content
            for row in rows:
                if not row.get('content') and row.get('raw_content'):
                    row['content'] = row.get('raw_content')
                if not row.get('role') and row.get('speaker_type'):
                    row['role'] = row.get('speaker_type')
            return rows

    async def store_facts(self, text: str, user_name: Optional[str] = None) -> int:
        """Extract facts from text and store as user->knows->concept.

        Keeps compatibility with SmartContextManager which calls store_facts(text).
        """
        try:
            if not text or not isinstance(text, str):
                return 0
            # Lazy import to avoid circulars
            from memory.facts_graph import extract_facts_from_text
            facts = extract_facts_from_text(text) or []
            count = 0
            # Determine user
            user_name = user_name or os.getenv('USER_ID', 'user')
            for f in facts:
                gf = GraphFact(
                    subject=f.get('subject', user_name),
                    predicate=f.get('predicate', 'related_to'),
                    value=f.get('value'),
                    fidelity=f.get('fidelity', 3),
                    strength=f.get('strength', 0.6),
                    source_text=f.get('source_text', text),
                    learned_from_session=self._active_sessions.get(user_name),
                )
                # Use subject if provided; otherwise env user
                subj = gf.subject or user_name
                ok = await self.store_fact(subj, gf)
                if ok is not None:  # ok True=reinforced / False=new, count both
                    count += 1
            return count
        except Exception as e:
            logger.debug(f"store_facts failed: {e}")
            return 0

# Factory function for compatibility
def create_graph_surreal_memory():
    """Create graph-native SurrealDB memory instance"""
    return GraphSurrealMemory()
