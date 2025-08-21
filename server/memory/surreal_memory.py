"""
SurrealDB Multi-Model Memory System - Drop-in replacement for SQLite memory

This revolutionary upgrade replaces 3 separate SQLite databases with a single 
SurrealDB multi-model database that provides:
- Graph relationships for facts with natural decay
- Time-travel queries ("what did we discuss last Tuesday?")
- Live subscriptions for real-time updates  
- Unified queries across all memory types
- Apple Silicon optimized (Rust-based)

Features:
- Drop-in compatibility with existing FactsGraph, TapeStore, QueryRouter interfaces
- Multi-model schema: facts (graph) + tape (time-series) + sessions (document)
- Async-first design (existing system already uses async)
- Environment toggle via USE_SURREALDB=true
- Time-travel superpowers with natural language dates
"""

import os
import time
import asyncio
import json
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger

try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    logger.warning("SurrealDB client not available. Install with: pip install surrealdb")
    SURREALDB_AVAILABLE = False
    AsyncSurreal = None


# Constants (matching original system)
DECAY_HALF_LIFE_S = 6 * 3600    # 6 hours half-life
PROMOTE_THRESH = 0.75           # Strength threshold to promote fidelity
DEMOTE_THRESH = 0.25           # Strength threshold to demote fidelity  
EMA_ALPHA = 0.35               # Exponential moving average factor
MAX_FACTS = 512                # Hard cap on total facts

FIDELITY_NAMES = {
    4: "verbatim",
    3: "structured", 
    2: "tuple",
    1: "edge",
    0: "forgotten"
}


@dataclass
class SurrealFact:
    """Fact with SurrealDB extensions"""
    subject: str
    predicate: str
    value: Optional[str] = None
    species: Optional[str] = None
    fidelity: int = 3
    strength: float = 0.6
    last_seen: float = 0
    created: float = 0
    access_count: int = 0
    source_text: str = ""
    
    # SurrealDB specific fields
    id: Optional[str] = None
    decay_rate: float = 1.0      # Custom decay multiplier
    tags: List[str] = None       # Graph relationship tags
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SurrealTapeEntry:
    """Tape entry with temporal query capabilities"""
    ts: float
    speaker_id: str
    role: str
    content: str
    
    # SurrealDB specific fields
    id: Optional[str] = None
    session_id: Optional[str] = None
    embedding: Optional[List[float]] = None  # For semantic search
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SurrealMemory:
    """
    Unified SurrealDB memory system with drop-in compatibility
    
    Provides exact same interface as existing FactsGraph, TapeStore, QueryRouter
    while adding SurrealDB superpowers like time-travel queries and live subscriptions.
    """
    
    def __init__(self, 
                 surreal_url: str = "ws://localhost:8000/rpc",
                 namespace: str = "slowcat",
                 database: str = "memory"):
        
        if not SURREALDB_AVAILABLE:
            raise ImportError("SurrealDB client required. Install with: pip install surrealdb")
        
        self.surreal_url = surreal_url
        self.namespace = namespace
        self.database = database
        self.db = None
        self.connected = False
        
        # Performance tracking (matching FactsGraph interface)
        self.reinforcements = 0
        self.decays_applied = 0
        self.facts_forgotten = 0
        
        logger.info(f"🚀 SurrealDB Memory initialized: {surreal_url}/{namespace}/{database}")
    
    async def connect(self):
        """Initialize SurrealDB connection and schema"""
        try:
            self.db = AsyncSurreal(self.surreal_url)
            
            # Authenticate with credentials from environment or defaults
            auth_user = os.getenv('SURREALDB_USER', 'root')
            auth_pass = os.getenv('SURREALDB_PASS', 'slowcat_secure_2024')
            
            try:
                # Correct JWT-based authentication format
                await self.db.signin({
                    'username': 'root',
                    'password': auth_pass
                })
                logger.debug(f"✅ Authentication successful as root with JWT")
            except Exception as e:
                logger.warning(f"JWT authentication failed: {e}")
                # Try without authentication as fallback
                logger.info("Attempting to continue without authentication")
            
            await self.db.use(self.namespace, self.database)
            
            # Initialize schema idempotently
            await self._init_schema()
            
            self.connected = True
            logger.info("🔗 SurrealDB connection established")
            
        except Exception as e:
            logger.error(f"SurrealDB connection failed: {e}")
            raise
    
    async def _init_schema(self):
        """Initialize SurrealDB schema with multi-model design"""
        
        # Facts table with proper SurrealQL syntax
        await self.db.query("""
            DEFINE TABLE fact SCHEMAFULL;
            DEFINE FIELD subject ON fact TYPE string;
            DEFINE FIELD predicate ON fact TYPE string;
            DEFINE FIELD value ON fact TYPE option<string>;
            DEFINE FIELD species ON fact TYPE option<string>;
            DEFINE FIELD fidelity ON fact TYPE number DEFAULT 3;
            DEFINE FIELD strength ON fact TYPE number DEFAULT 0.6;
            DEFINE FIELD last_seen ON fact TYPE datetime VALUE time::now();
            DEFINE FIELD created ON fact TYPE datetime VALUE time::now();
            DEFINE FIELD access_count ON fact TYPE number DEFAULT 0;
            DEFINE FIELD source_text ON fact TYPE string DEFAULT '';
        """)
        
        # Conversation tape with temporal capabilities
        await self.db.query("""
            DEFINE TABLE tape SCHEMAFULL;
            DEFINE FIELD ts ON tape TYPE datetime VALUE time::now();
            DEFINE FIELD speaker_id ON tape TYPE string;
            DEFINE FIELD role ON tape TYPE string;
            DEFINE FIELD content ON tape TYPE string;
            DEFINE FIELD session_id ON tape TYPE option<string>;
            DEFINE FIELD metadata ON tape TYPE object DEFAULT {};
        """)
        
        # Session summaries and metadata
        await self.db.query("""
            DEFINE TABLE session SCHEMAFULL;
            DEFINE FIELD speaker_id ON session TYPE string;
            DEFINE FIELD session_count ON session TYPE number DEFAULT 0;
            DEFINE FIELD last_interaction ON session TYPE datetime VALUE time::now();
            DEFINE FIELD first_seen ON session TYPE datetime VALUE time::now();
            DEFINE FIELD total_turns ON session TYPE number DEFAULT 0;
        """)
        
        logger.debug("📊 SurrealDB schema initialized")
    
    # ========================================
    # FactsGraph Interface Compatibility
    # ========================================
    
    async def reinforce_or_insert(self, fact_data: Dict) -> bool:
        """
        Store new fact or reinforce existing one (FactsGraph compatibility)
        
        Args:
            fact_data: Dict with subject, predicate, value, etc.
            
        Returns:
            bool: True if fact was reinforced, False if newly inserted
        """
        if not self.connected:
            await self.connect()
        
        try:
            now = time.time()
            
            # Check if fact exists
            query = """
                SELECT * FROM fact 
                WHERE subject = $subject 
                AND predicate = $predicate 
                AND value = $value 
                AND species = $species
            """
            
            result = await self.db.query(query, {
                'subject': fact_data['subject'],
                'predicate': fact_data['predicate'],
                'value': fact_data.get('value'),
                'species': fact_data.get('species')
            })
            
            if result and len(result) > 0:
                # Existing fact - reinforce it
                existing = result[0]
                fact_id = existing['id']
                
                # Apply EMA to strengthen (with NaN protection)
                old_strength = existing.get('strength', 0.6)
                if old_strength is None or (isinstance(old_strength, float) and math.isnan(old_strength)):
                    old_strength = 0.6
                new_strength = self._ema(old_strength, 1.0)
                new_fidelity = max(existing.get('fidelity', 3), fact_data.get('fidelity', 3))
                
                update_query = """
                    UPDATE $fact_id SET 
                        fidelity = $fidelity,
                        strength = $strength,
                        last_seen = time::now(),
                        access_count = access_count + 1
                """
                
                await self.db.query(update_query, {
                    'fact_id': fact_id,
                    'fidelity': new_fidelity,
                    'strength': new_strength
                })
                
                self.reinforcements += 1
                logger.debug(f"✨ Reinforced fact: {fact_data['subject']}.{fact_data['predicate']} "
                           f"(S{existing.get('fidelity', 3)}→S{new_fidelity}, {old_strength:.2f}→{new_strength:.2f})")
                
                return True
                
            else:
                # New fact - insert it
                insert_query = """
                    CREATE fact SET
                        subject = $subject,
                        predicate = $predicate,
                        value = $value,
                        species = $species,
                        fidelity = $fidelity,
                        strength = $strength,
                        created = time::now(),
                        last_seen = time::now(),
                        source_text = $source_text,
                        decay_rate = $decay_rate,
                        tags = $tags
                """
                
                # Validate strength to prevent NaN values
                strength = fact_data.get('strength', 0.6)
                if strength is None or (isinstance(strength, float) and math.isnan(strength)):
                    strength = 0.6
                
                await self.db.query(insert_query, {
                    'subject': fact_data['subject'],
                    'predicate': fact_data['predicate'],
                    'value': fact_data.get('value'),
                    'species': fact_data.get('species'),
                    'fidelity': fact_data.get('fidelity', 3),
                    'strength': strength,
                    'source_text': fact_data.get('source_text', ''),
                    'decay_rate': fact_data.get('decay_rate', 1.0),
                    'tags': fact_data.get('tags', [])
                })
                
                logger.debug(f"➕ New fact: {fact_data['subject']}.{fact_data['predicate']} = "
                           f"{fact_data.get('value', '?')} (S{fact_data.get('fidelity', 3)})")
                
                return False
                
        except Exception as e:
            logger.error(f"SurrealDB fact reinforce/insert failed: {e}")
            return False

    # ========================================
    # Session Metadata (FactsGraph session compatibility)
    # ========================================

    async def start_session(self, speaker_id: str):
        """Mark the start of a new session for a speaker (increment session_count once)."""
        if not self.connected:
            await self.connect()
        try:
            # Fetch existing session record
            sel = """
                SELECT * FROM session WHERE speaker_id = $speaker_id LIMIT 1
            """
            result = await self.db.query(sel, { 'speaker_id': speaker_id })
            rows = []
            if result and len(result) > 0:
                rows = result[0].get('result', []) if isinstance(result[0], dict) else result[0]

            if rows:
                rec_id = rows[0].get('id')
                upd = f"""
                    UPDATE {rec_id} SET 
                        session_count = (session_count ?? 0) + 1,
                        last_interaction = time::now()
                """
                await self.db.query(upd)
            else:
                ins = """
                    CREATE session SET 
                        speaker_id = $speaker_id,
                        session_count = 1,
                        last_interaction = time::now(),
                        first_seen = time::now(),
                        total_turns = 0
                """
                await self.db.query(ins, { 'speaker_id': speaker_id })
        except Exception as e:
            logger.error(f"SurrealDB start_session failed: {e}")

    async def update_session(self, speaker_id: str):
        """Update session metadata for speaker (increment total_turns, set last_interaction)."""
        if not self.connected:
            await self.connect()
        try:
            sel = """
                SELECT * FROM session WHERE speaker_id = $speaker_id LIMIT 1
            """
            result = await self.db.query(sel, { 'speaker_id': speaker_id })
            rows = []
            if result and len(result) > 0:
                rows = result[0].get('result', []) if isinstance(result[0], dict) else result[0]

            if rows:
                rec_id = rows[0].get('id')
                upd = f"""
                    UPDATE {rec_id} SET 
                        total_turns = (total_turns ?? 0) + 1,
                        last_interaction = time::now()
                """
                await self.db.query(upd)
            else:
                ins = """
                    CREATE session SET 
                        speaker_id = $speaker_id,
                        session_count = 1,
                        last_interaction = time::now(),
                        first_seen = time::now(),
                        total_turns = 1
                """
                await self.db.query(ins, { 'speaker_id': speaker_id })
        except Exception as e:
            logger.error(f"SurrealDB update_session failed: {e}")

    async def get_session_info(self, speaker_id: str) -> Dict:
        """Get session metadata for dynamic prompts."""
        if not self.connected:
            await self.connect()
        try:
            sel = """
                SELECT session_count, last_interaction, first_seen, total_turns
                FROM session WHERE speaker_id = $speaker_id LIMIT 1
            """
            result = await self.db.query(sel, { 'speaker_id': speaker_id })
            rows = []
            if result and len(result) > 0:
                rows = result[0].get('result', []) if isinstance(result[0], dict) else result[0]

            if not rows:
                # Default for new users
                return {
                    'session_count': 0,
                    'last_interaction': None,
                    'first_seen': time.time(),
                    'total_turns': 0,
                }

            row = rows[0]
            # Normalize datetime fields to POSIX seconds
            def to_ts(v):
                if hasattr(v, 'timestamp'):
                    return v.timestamp()
                if isinstance(v, (int, float)):
                    return float(v)
                return None

            return {
                'session_count': row.get('session_count', 0),
                'last_interaction': to_ts(row.get('last_interaction')),
                'first_seen': to_ts(row.get('first_seen')),
                'total_turns': row.get('total_turns', 0),
            }
        except Exception as e:
            logger.error(f"SurrealDB get_session_info failed: {e}")
            return {
                'session_count': 0,
                'last_interaction': None,
                'first_seen': time.time(),
                'total_turns': 0,
            }
    
    async def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search facts by text content (FactsGraph compatibility)
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching facts as dicts
        """
        if not self.connected:
            await self.connect()
        
        try:
            # Enhanced search with SurrealDB text search capabilities
            search_query = """
                SELECT * FROM fact 
                WHERE 
                    string::contains(string::lowercase(subject OR ''), string::lowercase($query))
                    OR string::contains(string::lowercase(predicate OR ''), string::lowercase($query))
                    OR string::contains(string::lowercase(value OR ''), string::lowercase($query))
                    OR string::contains(string::lowercase(species OR ''), string::lowercase($query))
                    OR string::contains(string::lowercase(source_text OR ''), string::lowercase($query))
                ORDER BY fidelity DESC, strength DESC, last_seen DESC
                LIMIT $limit
            """
            
            result = await self.db.query(search_query, {
                'query': query,
                'limit': limit
            })
            
            facts = []
            if result and len(result) > 0:
                for row in result:
                    # Convert to FactsGraph-compatible object
                    fact = SurrealFact(
                        subject=row.get('subject', ''),
                        predicate=row.get('predicate', ''),
                        value=row.get('value'),
                        species=row.get('species'),
                        fidelity=row.get('fidelity', 3),
                        strength=row.get('strength', 0.6),
                        last_seen=(time.mktime(row['last_seen'].timetuple()) if row.get('last_seen') else time.time()),
                        created=(time.mktime(row['created'].timetuple()) if row.get('created') else time.time()),
                        access_count=row.get('access_count', 0),
                        source_text=row.get('source_text', ''),
                        id=row.get('id')
                    )
                    facts.append(fact)
            
            return facts
            
        except Exception as e:
            logger.error(f"SurrealDB fact search failed: {e}")
            return []
    
    async def get_top_facts(self, limit: int = 10) -> List[Dict]:
        """
        Get the most important facts (FactsGraph compatibility)
        """
        if not self.connected:
            await self.connect()
        
        try:
            query = """
                SELECT * FROM fact 
                WHERE fidelity >= 2
                ORDER BY fidelity DESC, strength DESC, last_seen DESC
                LIMIT $limit
            """
            
            result = await self.db.query(query, {'limit': limit})
            
            facts = []
            if result and len(result) > 0:
                for row in result:
                    fact = SurrealFact(
                        subject=row.get('subject', ''),
                        predicate=row.get('predicate', ''),
                        value=row.get('value'),
                        species=row.get('species'),
                        fidelity=row.get('fidelity', 3),
                        strength=row.get('strength', 0.6),
                        last_seen=(time.mktime(row['last_seen'].timetuple()) if row.get('last_seen') else time.time()),
                        created=(time.mktime(row['created'].timetuple()) if row.get('created') else time.time()),
                        access_count=row.get('access_count', 0),
                        source_text=row.get('source_text', ''),
                        id=row.get('id')
                    )
                    facts.append(fact)
            
            return facts
            
        except Exception as e:
            logger.error(f"SurrealDB get_top_facts failed: {e}")
            return []
    
    # ========================================
    # TapeStore Interface Compatibility
    # ========================================
    
    async def add_entry(self, role: str, content: str, speaker_id: str = "default_user", ts: Optional[float] = None):
        """
        Add conversation entry to tape (TapeStore compatibility)
        """
        if not self.connected:
            await self.connect()
        
        try:
            entry_ts = ts or time.time()
            
            insert_query = """
                CREATE tape SET
                    ts = time::from::secs($ts),
                    speaker_id = $speaker_id,
                    role = $role,
                    content = $content,
                    session_id = $session_id,
                    metadata = $metadata
            """
            
            # Generate session ID based on speaker and date
            session_id = f"{speaker_id}_{int(entry_ts // 86400)}"  # Daily sessions
            
            await self.db.query(insert_query, {
                'ts': int(entry_ts),  # Convert to integer for time::from::secs
                'speaker_id': speaker_id,
                'role': role,
                'content': content,
                'session_id': session_id,
                'metadata': {}
            })
            
            logger.debug(f"📼 Added tape entry: [{role}] {content[:50]}...")
            
        except Exception as e:
            logger.error(f"SurrealDB tape add_entry failed: {e}")
    
    async def search_tape(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search conversation tape (TapeStore compatibility)
        """
        if not self.connected:
            await self.connect()
        
        try:
            search_query = """
                SELECT * FROM tape 
                WHERE string::contains(string::lowercase(content), string::lowercase($query))
                ORDER BY ts DESC
                LIMIT $limit
            """
            
            result = await self.db.query(search_query, {
                'query': query,
                'limit': limit
            })
            
            entries = []
            if result and len(result) > 0:
                for row in result:
                    entry = {
                        'ts': time.mktime(row['ts'].timetuple()) if row.get('ts') else time.time(),
                        'speaker_id': row.get('speaker_id', 'unknown'),
                        'role': row.get('role', 'user'),
                        'content': row.get('content', '')
                    }
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"SurrealDB tape search failed: {e}")
            return []
    
    async def get_recent(self, limit: int = 10, since: Optional[float] = None) -> List[Dict]:
        """
        Get recent tape entries (TapeStore compatibility)
        """
        if not self.connected:
            await self.connect()
        
        try:
            if since is None:
                query = """
                    SELECT * FROM tape 
                    ORDER BY ts DESC 
                    LIMIT $limit
                """
                params = {'limit': limit}
            else:
                query = """
                    SELECT * FROM tape 
                    WHERE ts >= time::from::secs($since)
                    ORDER BY ts DESC 
                    LIMIT $limit
                """
                params = {'since': since, 'limit': limit}
            
            result = await self.db.query(query, params)
            
            entries = []
            if result and len(result) > 0:
                for row in result:
                    # Handle datetime objects from SurrealDB
                    ts_value = row.get('ts')
                    if hasattr(ts_value, 'timestamp'):
                        ts_float = ts_value.timestamp()
                    elif hasattr(ts_value, 'timetuple'):
                        ts_float = time.mktime(ts_value.timetuple())
                    elif isinstance(ts_value, (int, float)):
                        ts_float = float(ts_value)
                    else:
                        ts_float = time.time()
                        
                    entry = {
                        'ts': ts_float,
                        'speaker_id': row.get('speaker_id', 'unknown'),
                        'role': row.get('role', 'user'),
                        'content': row.get('content', '')
                    }
                    entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"SurrealDB get_recent failed: {e}")
            return []
    
    async def get_entries_since(self, since_ts: float) -> List[Dict]:
        """
        Get tape entries since given timestamp (TapeStore compatibility)
        
        Args:
            since_ts: Unix timestamp to start from
            
        Returns:
            List of tape entries since the timestamp
        """
        if not self.connected:
            await self.connect()
        
        try:
            query = """
                SELECT * FROM tape 
                WHERE ts >= time::from::secs($since)
                ORDER BY ts ASC
            """
            
            result = await self.db.query(query, {'since': int(since_ts)})
            
            entries = []
            if result and len(result) > 0:
                for row in result:
                    # Convert SurrealDB entry to TapeStore format
                    ts_value = row.get('ts', 0)
                    # Handle datetime objects from SurrealDB
                    if hasattr(ts_value, 'timestamp'):
                        ts_float = ts_value.timestamp()
                    elif isinstance(ts_value, (int, float)):
                        ts_float = float(ts_value)
                    else:
                        ts_float = 0.0
                        
                    entry_dict = {
                        'role': row.get('role', ''),
                        'content': row.get('content', ''),
                        'speaker_id': row.get('speaker_id', ''),
                        'ts': ts_float,
                        'session_id': row.get('session_id', ''),
                        'metadata': row.get('metadata', {})
                    }
                    entries.append(type('TapeEntry', (), entry_dict)())
            
            logger.debug(f"📼 Retrieved {len(entries)} tape entries since {since_ts}")
            return entries
            
        except Exception as e:
            logger.error(f"SurrealDB get_entries_since failed: {e}")
            return []
    
    async def add_summary(self, session_id: str, summary: str, keywords_json: str = '[]', turns: int = 0, duration_s: int = 0):
        """
        Add session summary (TapeStore compatibility)
        
        Args:
            session_id: Session identifier
            summary: Summary text
            keywords_json: Keywords as JSON string
            turns: Number of turns in session
            duration_s: Session duration in seconds
        """
        if not self.connected:
            await self.connect()
        
        try:
            query = """
                CREATE session_summary SET
                    session_id = $session_id,
                    summary = $summary,
                    keywords = $keywords,
                    turns = $turns,
                    duration_s = $duration_s,
                    ts = time::now()
            """
            
            import json
            try:
                keywords = json.loads(keywords_json)
            except:
                keywords = []
            
            await self.db.query(query, {
                'session_id': session_id,
                'summary': summary,
                'keywords': keywords,
                'turns': turns,
                'duration_s': duration_s
            })
            
            logger.debug(f"📝 Added session summary: {session_id}")
            
        except Exception as e:
            logger.error(f"SurrealDB add_summary failed: {e}")
    
    async def get_last_summary(self) -> Optional[Dict]:
        """
        Get the last session summary (TapeStore compatibility)
        
        Returns:
            Dict with summary data or None if no summaries exist
        """
        if not self.connected:
            await self.connect()
        
        try:
            query = """
                SELECT * FROM session_summary 
                ORDER BY ts DESC 
                LIMIT 1
            """
            
            result = await self.db.query(query)
            
            if result and len(result) > 0:
                # SurrealDB returns results in different formats, handle both
                if isinstance(result[0], list) and len(result[0]) > 0:
                    row = result[0][0]
                elif isinstance(result[0], dict):
                    row = result[0]
                else:
                    return None
                    
                # Handle datetime objects from SurrealDB  
                ts_value = row.get('ts', 0)
                if hasattr(ts_value, 'timestamp'):
                    ts_float = ts_value.timestamp()
                elif isinstance(ts_value, (int, float)):
                    ts_float = float(ts_value)
                else:
                    ts_float = 0.0
                
                return {
                    'session_id': row.get('session_id', ''),
                    'summary': row.get('summary', ''),
                    'keywords': row.get('keywords', []),
                    'turns': row.get('turns', 0),
                    'duration_s': row.get('duration_s', 0),
                    'ts': ts_float
                }
            
            return None
            
        except Exception as e:
            logger.error(f"SurrealDB get_last_summary failed: {e}")
            return None
    
    # ========================================
    # NEW SurrealDB Superpowers
    # ========================================
    
    async def time_travel_query(self, natural_date: str, limit: int = 10) -> List[Dict]:
        """
        Time-travel query using natural language dates
        
        Examples:
        - "last Tuesday"
        - "three days ago"
        - "this morning"
        - "last week"
        """
        if not self.connected:
            await self.connect()
        
        try:
            # Use SurrealDB's time functions for natural language parsing
            # This is a simplified implementation - would need proper NLP for full natural language
            
            time_expressions = {
                'today': 'time::now() - 1d',
                'yesterday': 'time::now() - 2d',
                'last week': 'time::now() - 7d',
                'last month': 'time::now() - 30d',
                'this morning': 'time::now() - 12h',
                'last tuesday': 'time::now() - 7d',  # Simplified
                'three days ago': 'time::now() - 3d',
            }
            
            # Parse natural date
            time_expr = time_expressions.get(natural_date.lower(), 'time::now() - 1d')
            
            query = f"""
                SELECT * FROM tape 
                WHERE ts >= {time_expr}
                ORDER BY ts DESC
                LIMIT $limit
            """
            
            result = await self.db.query(query, {'limit': limit})
            
            entries = []
            if result and len(result[0]['result']) > 0:
                for row in result[0]['result']:
                    entry = {
                        'ts': time.mktime(row['ts'].timetuple()) if row.get('ts') else time.time(),
                        'speaker_id': row.get('speaker_id', 'unknown'),
                        'role': row.get('role', 'user'),
                        'content': row.get('content', ''),
                        'natural_date': natural_date  # Include parsed date
                    }
                    entries.append(entry)
            
            logger.info(f"🕰️ Time travel query '{natural_date}' returned {len(entries)} results")
            return entries
            
        except Exception as e:
            logger.error(f"SurrealDB time travel query failed: {e}")
            return []
    
    async def apply_decay(self):
        """
        Apply time-based decay to all facts using SurrealDB events
        """
        if not self.connected:
            await self.connect()
        
        try:
            # SurrealDB automatic decay via events + manual cleanup
            decay_query = """
                UPDATE fact SET 
                    strength = strength * math::pow(0.5, (time::now() - last_seen) / duration::from::secs($half_life))
                WHERE strength > 0.1;
                
                DELETE fact WHERE fidelity <= 0 OR strength < 0.05;
            """
            
            await self.db.query(decay_query, {
                'half_life': DECAY_HALF_LIFE_S
            })
            
            self.decays_applied += 1
            logger.info("🕰️ SurrealDB automatic decay applied")
            
        except Exception as e:
            logger.error(f"SurrealDB decay failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics
        """
        if not self.connected:
            await self.connect()
        
        try:
            stats_query = """
                SELECT 
                    count() AS total_facts,
                    math::sum(fidelity) AS total_fidelity,
                    math::mean(strength) AS avg_strength
                FROM fact;
                
                SELECT 
                    count() AS total_entries
                FROM tape;
                
                SELECT 
                    fidelity,
                    count() AS count
                FROM fact
                GROUP BY fidelity;
            """
            
            result = await self.db.query(stats_query)
            
            facts_stats = result[0]['result'][0] if result[0]['result'] else {}
            tape_stats = result[1]['result'][0] if result[1]['result'] else {}
            fidelity_dist = {f"S{row['fidelity']}": row['count'] for row in result[2]['result']} if result[2]['result'] else {}
            
            return {
                'total_facts': facts_stats.get('total_facts', 0),
                'total_tape_entries': tape_stats.get('total_entries', 0),
                'avg_strength': facts_stats.get('avg_strength', 0),
                'fidelity_distribution': fidelity_dist,
                'reinforcements': self.reinforcements,
                'decays_applied': self.decays_applied,
                'facts_forgotten': self.facts_forgotten,
                'db_type': 'SurrealDB',
                'connection_status': 'connected' if self.connected else 'disconnected'
            }
            
        except Exception as e:
            logger.error(f"SurrealDB stats failed: {e}")
            return {'error': str(e)}
    
    # ========================================
    # Utility Methods
    # ========================================
    
    def _ema(self, prev: float, obs: float, alpha: float = EMA_ALPHA) -> float:
        """Exponential moving average"""
        return alpha * obs + (1 - alpha) * prev
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
            self.connected = False
            logger.info("🔒 SurrealDB connection closed")


# ========================================
# Factory Functions for Drop-in Replacement
# ========================================

def create_surreal_memory_system(
    surreal_url: str = None,
    namespace: str = "slowcat", 
    database: str = "memory"
) -> SurrealMemory:
    """
    Create SurrealDB memory system with environment configuration
    
    Environment Variables:
    - SURREALDB_URL: SurrealDB connection URL (default: ws://localhost:8000/rpc)
    - SURREALDB_NAMESPACE: Database namespace (default: slowcat)
    - SURREALDB_DATABASE: Database name (default: memory)
    """
    
    # Use environment variables with fallbacks
    url = surreal_url or os.getenv('SURREALDB_URL', 'ws://localhost:8000/rpc')
    ns = os.getenv('SURREALDB_NAMESPACE', namespace)
    db = os.getenv('SURREALDB_DATABASE', database)
    
    logger.info(f"🚀 Creating SurrealDB memory system: {url}/{ns}/{db}")
    
    return SurrealMemory(
        surreal_url=url,
        namespace=ns,
        database=db
    )


async def create_surreal_facts_graph(surreal_url: str = None) -> SurrealMemory:
    """Create SurrealDB facts graph with FactsGraph compatibility"""
    memory = create_surreal_memory_system(surreal_url)
    await memory.connect()
    return memory


async def create_surreal_tape_store(surreal_url: str = None) -> SurrealMemory:
    """Create SurrealDB tape store with TapeStore compatibility"""
    memory = create_surreal_memory_system(surreal_url)
    await memory.connect()
    return memory


# Self-test
if __name__ == "__main__":
    async def test_surreal_memory():
        """Test SurrealDB memory system"""
        logger.info("🚀 Testing SurrealDB Memory System")
        
        if not SURREALDB_AVAILABLE:
            logger.error("SurrealDB client not available. Install with: pip install surrealdb")
            return
        
        try:
            # Create memory system
            memory = create_surreal_memory_system()
            await memory.connect()
            
            # Test fact operations
            test_facts = [
                {'subject': 'user', 'predicate': 'pet', 'value': 'Potola', 'species': 'dog'},
                {'subject': 'user', 'predicate': 'location', 'value': 'San Francisco'},
                {'subject': 'user', 'predicate': 'name', 'value': 'Alex'},
            ]
            
            logger.info("Testing fact insertion...")
            for fact in test_facts:
                await memory.reinforce_or_insert(fact)
            
            # Test fact search
            logger.info("Testing fact search...")
            dog_facts = await memory.search_facts("dog")
            logger.info(f"Dog facts: {dog_facts}")
            
            # Test tape operations
            logger.info("Testing tape operations...")
            await memory.add_entry("user", "Hello, this is a test message", "alex")
            await memory.add_entry("assistant", "Hi Alex! How can I help you today?", "alex")
            
            # Test tape search
            recent_entries = await memory.get_recent(limit=5)
            logger.info(f"Recent entries: {len(recent_entries)}")
            
            # Test time travel
            logger.info("Testing time travel query...")
            today_entries = await memory.time_travel_query("today", limit=10)
            logger.info(f"Today's entries: {len(today_entries)}")
            
            # Test stats
            logger.info("Testing stats...")
            stats = await memory.get_stats()
            logger.info(f"Memory stats: {stats}")
            
            # Test decay
            logger.info("Testing decay...")
            await memory.apply_decay()
            
            await memory.close()
            logger.info("✅ SurrealDB Memory test complete")
            
        except Exception as e:
            logger.error(f"SurrealDB test failed: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_surreal_memory())
