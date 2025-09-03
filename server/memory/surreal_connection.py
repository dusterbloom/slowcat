"""
SurrealDB Connection Manager for Consciousness Engine

Provides async connection pooling, query execution, and error handling
for SurrealDB interactions. Manages message storage and retrieval.
"""

import asyncio
import os
import time
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from loguru import logger

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    SURREALDB_AVAILABLE = False
    AsyncSurreal = None
    logger.warning("SurrealDB client not available - install with: pip install surrealdb")

# Try to import MLX sentence transformers first (Apple Silicon optimized), fallback to standard
try:
    from mlx_sentence_transformers import SentenceTransformer
    _use_mlx = True
    logger.debug("MLX sentence transformers available for query embedding")
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        _use_mlx = False
        logger.debug("Standard sentence transformers available for query embedding")
    except ImportError:
        logger.warning("No sentence transformer library available - advanced search will use text-only")
        SentenceTransformer = None

# Global sentence transformer for query embeddings
_query_sentence_transformer = None

def get_query_sentence_transformer():
    """Get singleton sentence transformer for query embeddings"""
    global _query_sentence_transformer
    if _query_sentence_transformer is None and SentenceTransformer is not None:
        try:
            model_name = "all-MiniLM-L6-v2"
            _query_sentence_transformer = SentenceTransformer(model_name)
            backend = "MLX" if _use_mlx else "Standard"
            logger.info(f"🔧 {backend} sentence transformer '{model_name}' loaded for query embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to load sentence transformer for queries: {e}")
            _query_sentence_transformer = None
    return _query_sentence_transformer


@dataclass
class Message:
    """Represents a conversation message"""
    role: str  # 'user' or 'assistant'
    content: str
    speaker_id: str = 'default_user'
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    tokens: int = 0
    metadata: Optional[Dict[str, Any]] = None
    parent_message: Optional[str] = None
    
    def validate(self) -> None:
        """Validate message data before storage"""
        # session_id is now REQUIRED - no auto-generation to prevent race conditions
        if not self.session_id:
            raise ValueError("session_id is required - use SessionManager to ensure consistent sessions")
            
        if self.role not in ['user', 'assistant', 'system']:
            raise ValueError(f"Invalid role: {self.role}")
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")
        if not self.speaker_id:
            self.speaker_id = 'default_user'  # Auto-fix missing speaker_id
    
    def to_surreal(self) -> Dict[str, Any]:
        """Convert to SurrealDB format"""
        self.validate()  # Validate before conversion
        data = {
            'role': self.role,
            'content': self.content,
            'speaker_id': self.speaker_id,
            'tokens': int(self.tokens) if self.tokens is not None else 0,  # Convert to int
            'embedding': self.embedding if self.embedding else [],  # Always provide array
            'metadata': self.metadata if self.metadata else {},  # Always provide object
            'session_id': self.session_id,  # NEVER fall back to 'default' - require valid session_id
            'timestamp': self.timestamp if self.timestamp else datetime.now(timezone.utc)  # Always provide timestamp as datetime object
        }
        
        if self.parent_message:
            # Convert string ID to RecordID if needed
            if isinstance(self.parent_message, str) and ':' in self.parent_message:
                from surrealdb import RecordID
                table, record_id = self.parent_message.split(':', 1)
                data['parent_message'] = RecordID(table, record_id)
            else:
                data['parent_message'] = self.parent_message
            
        return data


class SurrealConnectionManager:
    """Manages connections and queries to SurrealDB"""
    
    def __init__(self,
                 url: str = None,
                 namespace: str = None,
                 database: str = None,
                 username: str = None,
                 password: str = None,
                 max_retries: int = 3,
                 retry_delay: float = 0.5):
        """
        Initialize SurrealDB connection manager
        
        Args:
            url: SurrealDB URL (defaults to env SURREAL_URL or ws://localhost:8000)
            namespace: Database namespace
            database: Database name
            username: Optional username for auth
            password: Optional password for auth
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        
        if not SURREALDB_AVAILABLE:
            raise ImportError("SurrealDB client not available. Install with: pip install surrealdb")
        
        # Connection parameters  
        self.url = url or os.getenv('SURREALDB_URL', 'ws://localhost:8000/rpc')
        self.namespace = namespace or os.getenv('SURREALDB_NAMESPACE', 'slowcat')
        self.database = database or os.getenv('SURREALDB_DATABASE', 'memory_graph')
        self.username = username or os.getenv('SURREALDB_USER')
        self.password = password or os.getenv('SURREALDB_PASS')
        
        # Connection management
        self.db: Optional[AsyncSurreal] = None
        self.connected = False
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._lock = asyncio.Lock()
        
        # Cache for active sessions
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"SurrealDB manager initialized for {self.url} (ns: {self.namespace}, db: {self.database})")
    
    async def connect(self) -> bool:
        """Establish connection to SurrealDB"""
        async with self._lock:
            if self.connected and self.db:
                return True
            
            for attempt in range(self.max_retries):
                try:
                    self.db = AsyncSurreal(self.url)
                    await self.db.connect()
                    
                    # Authenticate as ROOT user first (BEFORE use())
                    if self.username and self.password:
                        await self.db.signin({
                            'username': self.username,
                            'password': self.password
                        })
                    
                    # Select namespace and database after authentication  
                    await self.db.use(self.namespace, self.database)
                    
                    self.connected = True
                    logger.info(f"✅ Connected to SurrealDB at {self.url}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(f"❌ Failed to connect to SurrealDB after {self.max_retries} attempts")
                        raise
            
            return False
    
    async def disconnect(self):
        """Close connection to SurrealDB"""
        if self.db:
            try:
                await self.db.close()
                self.connected = False
                logger.info("Disconnected from SurrealDB")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
    
    async def ensure_connected(self):
        """Ensure database connection is active"""
        if not self.connected:
            await self.connect()
    
    async def store_message(self, message: Message) -> Optional[str]:
        """
        Store a conversation message in SurrealDB
        
        Args:
            message: Message object to store
            
        Returns:
            Message ID if successful, None otherwise
        """
        await self.ensure_connected()
        
        try:
            # INTEGRATION DEBUG: Log every store_message call with stack trace
            import traceback
            stack = ''.join(traceback.format_stack()[-3:-1])  # Get calling context
            logger.info(f"🔍 INTEGRATION: store_message called role={message.role} speaker={message.speaker_id} session={message.session_id} tokens={getattr(message, 'tokens', 0)} caller={stack.strip()}")
            # Convert message to SurrealDB format
            data = message.to_surreal()
            
            # Create the message record using query method for better compatibility
            result = await self.db.query(
                "CREATE messages CONTENT $data;",
                {'data': data}
            )
            
            if result and len(result) > 0:
                created_record = result[0]  # First record
                message_id = created_record.get('id')
                logger.debug(f"📝 Stored {message.role} message: {message_id}")

                # Canonical relation: messages -> message_belongs_to -> sessions
                try:
                    if message.session_id:
                        sess_res = await self.db.query(
                            "SELECT * FROM sessions WHERE session_id = $sid LIMIT 1;",
                            {"sid": message.session_id},
                        )
                        if isinstance(sess_res, list) and sess_res:
                            session_record_id = sess_res[0].get('id')
                            if session_record_id and message_id:
                                await self.db.query(
                                    """
                                    RELATE $mid -> message_belongs_to -> $sid SET
                                        timestamp = $ts,
                                        speaker_role = $role,
                                        message_order = 0
                                    """,
                                    {
                                        "mid": message_id,
                                        "sid": session_record_id,
                                        "ts": data.get('timestamp'),
                                        "role": data.get('role', 'user'),
                                    },
                                )
                except Exception as e:
                    logger.debug(f"message_belongs_to creation skipped: {e}")

                # Update session if provided
                if message.session_id and message.session_id in self._active_sessions:
                    await self._update_session_stats(message.session_id, message.tokens)

                return str(message_id) if message_id else None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            return None
    
    async def store_messages_batch(self, messages: List[Message]) -> List[str]:
        """
        Store multiple messages in batch
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of message IDs
        """
        await self.ensure_connected()
        
        try:
            # Convert all messages to SurrealDB format
            data = [msg.to_surreal() for msg in messages]
            
            # Batch create
            query = """
                LET $messages = $data;
                FOR $msg IN $messages {
                    CREATE messages CONTENT $msg;
                };
            """
            
            result = await self.db.query(query, {'data': data})
            
            # Extract IDs from result
            ids = []
            if result and isinstance(result, list):
                for item in result:
                    if isinstance(item, dict) and 'id' in item:
                        ids.append(item['id'])
            
            logger.debug(f"📝 Batch stored {len(ids)} messages")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to batch store messages: {e}")
            return []
    
    async def get_conversation(self, 
                              session_id: str = None,
                              speaker_id: str = None,
                              limit: int = 50,
                              offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve conversation messages
        
        Args:
            session_id: Optional session ID filter
            speaker_id: Optional speaker ID filter  
            limit: Maximum messages to return
            offset: Number of messages to skip
            
        Returns:
            List of message dictionaries
        """
        await self.ensure_connected()
        
        try:
            # Build query based on filters
            conditions = []
            params = {'limit': limit, 'offset': offset}
            
            if session_id:
                conditions.append("session_id = $session_id")
                params['session_id'] = session_id
            
            if speaker_id:
                conditions.append("speaker_id = $speaker_id")
                params['speaker_id'] = speaker_id
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            query = f"""
                SELECT * FROM messages
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT $limit
                START $offset;
            """
            
            result = await self.db.query(query, params)
            
            if result and isinstance(result, list) and len(result) > 0:
                messages = result[0].get('result', [])
                logger.debug(f"Retrieved {len(messages)} messages")
                return messages
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation: {e}")
            return []
    
    async def get_recent_messages(self, 
                                 minutes: int = 5,
                                 speaker_id: str = None) -> List[Dict[str, Any]]:
        """
        Get messages from the last N minutes
        
        Args:
            minutes: Time window in minutes
            speaker_id: Optional speaker filter
            
        Returns:
            List of recent messages
        """
        await self.ensure_connected()
        
        try:
            params = {'minutes': minutes}
            conditions = ["timestamp > time::now() - $minutes * 1m"]
            
            if speaker_id:
                conditions.append("speaker_id = $speaker_id")
                params['speaker_id'] = speaker_id
            
            query = f"""
                SELECT * FROM messages
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp ASC;
            """
            
            result = await self.db.query(query, params)
            
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].get('result', [])
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            return []
    
    async def create_session(self, 
                           speaker_id: str = 'default_user',
                           metadata: Dict[str, Any] = None,
                           session_id: str = None) -> Optional[str]:
        """
        Create a new conversation session
        
        Args:
            speaker_id: Speaker identifier
            metadata: Optional session metadata
            session_id: Optional existing session ID to use (will generate if None)
            
        Returns:
            Session ID if successful
        """
        await self.ensure_connected()
        
        try:
            # Use provided session_id or generate a new one
            if not session_id:
                import uuid
                session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            # Check if session already exists to prevent duplicates
            existing = await self.db.query(
                "SELECT * FROM sessions WHERE session_id = $session_id;",
                {'session_id': session_id}
            )
            
            if existing and len(existing) > 0:
                logger.info(f"🔄 Session already exists: {session_id} - using existing")
                # Cache existing session for consistency
                existing_data = existing[0]
                self._active_sessions[session_id] = existing_data
                return session_id
            
            data = {
                'session_id': session_id,
                'speaker_id': speaker_id,
                'start_time': datetime.now(timezone.utc),
                'turn_count': 0,
                'total_tokens': 0,
                'is_active': True
            }
            
            if metadata:
                data['metadata'] = metadata
            
            # Use query method with simple CREATE statement
            result = await self.db.query(
                "CREATE sessions CONTENT $data;",
                {'data': data}
            )
            logger.debug(f"Session create result: {result}")
            
            if result and len(result) > 0:
                # Extract the created record from query result  
                created_record = result[0]  # First record
                record_id = created_record.get('id', None)
                
                # Cache active session
                self._active_sessions[session_id] = data
                logger.info(f"🎬 Created session: {session_id} for speaker: {speaker_id} (record: {record_id})")
                return session_id
            else:
                logger.warning(f"Session creation failed - no result returned: {result}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    async def start_session(self, speaker_id: str = 'default_user') -> Optional[str]:
        """
        Compatibility method for SmartContextManager
        Alias for create_session()
        """
        return await self.create_session(speaker_id)
    
    async def get_recent(self, limit: int = 10, since: float = None, agent_id: str = None):
        """TapeStore compatibility - get recent entries"""
        try:
            # Use get_recent_messages for compatibility
            minutes = 60 if since is None else max(1, int((time.time() - since) / 60))
            messages = await self.get_recent_messages(
                minutes=minutes,
                speaker_id=agent_id
            )
            
            # Convert to tape format
            results = []
            for msg in messages[:limit]:
                results.append({
                    'content': msg.get('content', ''),
                    'role': msg.get('role', ''),
                    'speaker_id': msg.get('speaker_id', ''),
                    'ts': msg.get('timestamp', ''),
                    'metadata': msg.get('metadata', {})
                })
            
            return results
            
        except Exception as e:
            logger.debug(f"get_recent failed: {e}")
            return []
    
    async def add_entry(self, role: str, content: str, speaker_id: str = "default_user", ts: Optional[float] = None, session_id: Optional[str] = None, **kwargs) -> bool:
        """
        TapeStore compatibility method - add a tape entry
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            speaker_id: Speaker identifier
            ts: Optional timestamp (Unix timestamp)
            session_id: Optional session ID (will auto-generate if None)
            **kwargs: Additional metadata
        """
        try:
            # Convert timestamp if provided
            timestamp = datetime.fromtimestamp(ts, timezone.utc) if ts else datetime.now(timezone.utc)
            
            # Auto-generate session_id if not provided
            if not session_id:
                import uuid
                session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            message = Message(
                role=role,
                content=content,
                speaker_id=speaker_id,
                session_id=session_id,
                timestamp=timestamp,
                metadata=kwargs
            )
            
            message_id = await self.store_message(message)
            return message_id is not None
            
        except Exception as e:
            from loguru import logger
            logger.error(f"Failed to add tape entry: {e}")
            return False
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> list:
        """
        Search compatibility method for query router
        """
        try:
            # Use existing search_messages method
            results = await self.search_messages(query, limit=limit)
            
            # Convert to expected format for compatibility
            formatted_results = []
            for msg in results:
                # Create a simple result object that matches expected interface
                result = {
                    'content': msg.get('content', ''),
                    'speaker_id': msg.get('speaker_id', ''),
                    'timestamp': msg.get('timestamp', ''),
                    'metadata': msg.get('metadata', {}),
                    'score': 0.8  # Default relevance score
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            from loguru import logger
            logger.error(f"Search failed: {e}")
            return []
    
    async def store_facts(self, facts: list, speaker_id: str = None, session_id: str = None) -> int:
        """
        FactsGraph compatibility - convert to unified knowledge system
        """
        stored_count = 0
        
        try:
            # Use speaker_id as default subject if provided, otherwise use 'user'
            default_subject = speaker_id if speaker_id else 'user'
            
            for fact in facts:
                if isinstance(fact, dict):
                    subject = fact.get('subject', default_subject)
                    predicate = fact.get('predicate', 'mentioned')
                    obj = fact.get('value') or fact.get('object', str(fact))
                    embedding = fact.get('embedding')
                    confidence = fact.get('confidence', 0.8)
                else:
                    subject = getattr(fact, 'subject', default_subject)
                    predicate = getattr(fact, 'predicate', 'mentioned')
                    obj = getattr(fact, 'value', str(fact))
                    embedding = getattr(fact, 'embedding', None)
                    confidence = getattr(fact, 'confidence', 0.8)
                
                # DATA QUALITY GATE: Discard low-confidence facts
                if confidence < 0.75:
                    logger.debug(f"Discarding low-confidence fact: {subject} -> {predicate} -> {obj} ({confidence})")
                    continue

                # Store using unified knowledge system
                success = await self.store_knowledge_relation(
                    subject_name=subject,
                    predicate=predicate, 
                    object_name=obj,
                    confidence=confidence,
                    embedding=embedding,
                    session_id=session_id
                )
                if success:
                    stored_count += 1
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store facts: {e}")
            return 0
    
    
    def apply_decay(self):
        """
        Compatibility method for fact decay - handled by SurrealDB events
        """
        # In SurrealDB, this would be handled by scheduled events/triggers
        pass
    
    def get_stats(self) -> dict:
        """
        Get memory system statistics
        """
        return {
            'type': 'SurrealDB',
            'connected': self.connected,
            'url': self.url,
            'namespace': self.namespace,
            'database': self.database
        }
    
    async def get_session_info(self, speaker_id: str) -> dict:
        """
        Get session information for SmartContextManager compatibility
        """
        try:
            await self.ensure_connected()
            
            # Get sessions count and first_seen
            sess_res = await self.db.query(
                "SELECT count() AS session_count, math::min(start_time) AS first_seen FROM sessions WHERE speaker_id = $speaker_id GROUP ALL;",
                {'speaker_id': speaker_id}
            )
            logger.debug(f"Session info query (sessions) for {speaker_id}: {sess_res}")

            session_count = 0
            first_seen = None
            if isinstance(sess_res, list) and sess_res:
                rec = sess_res[0]
                session_count = rec.get('session_count', 0)
                first_seen = rec.get('first_seen')

            # Get last interaction from messages
            msg_res = await self.db.query(
                "SELECT math::max(timestamp) AS last_interaction FROM messages WHERE speaker_id = $speaker_id GROUP ALL;",
                {'speaker_id': speaker_id}
            )
            logger.debug(f"Session info query (messages) for {speaker_id}: {msg_res}")

            last_interaction = None
            if isinstance(msg_res, list) and msg_res:
                rec2 = msg_res[0]
                last_interaction = rec2.get('last_interaction')

            # Convert datetimes to epoch seconds if needed
            def _to_ts(dt):
                try:
                    from datetime import datetime
                    if isinstance(dt, datetime):
                        return dt.timestamp()
                    return None
                except Exception:
                    return None

            first_seen_ts = _to_ts(first_seen)
            last_interaction_ts = _to_ts(last_interaction)

            return {
                'speaker_id': speaker_id,
                'session_count': int(session_count or 0),
                'first_seen': first_seen_ts,
                'last_interaction': last_interaction_ts,
                'type': 'SurrealDB'
            }
            
        except Exception as e:
            logger.debug(f"Failed to get session info: {e}")
            return {
                'speaker_id': speaker_id,
                'session_count': 0,
                'first_seen': None,
                'last_interaction': None,
                'type': 'SurrealDB'
            }
    
    async def end_session(self, 
                         session_id: str,
                         summary: str = None) -> bool:
        """
        End a conversation session
        
        Args:
            session_id: Session to end
            summary: Optional session summary
            
        Returns:
            True if successful
        """
        await self.ensure_connected()
        
        try:
            update_data = {
                'is_active': False,
                'end_time': datetime.now(timezone.utc)
            }
            
            if summary:
                update_data['summary'] = summary
            
            query = """
                UPDATE sessions SET
                    is_active = $is_active,
                    end_time = $end_time,
                    summary = $summary
                WHERE session_id = $session_id;
            """
            
            await self.db.query(query, {
                'session_id': session_id,
                'is_active': False,
                'end_time': update_data['end_time'],
                'summary': summary
            })
            
            # Remove from cache
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
            logger.info(f"🏁 Ended session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False
    
    async def _update_session_stats(self, session_id: str, tokens: int):
        """Update session statistics"""
        try:
            query = """
                UPDATE sessions SET
                    turn_count = turn_count + 1,
                    total_tokens = total_tokens + $tokens
                WHERE session_id = $session_id;
            """
            
            await self.db.query(query, {
                'session_id': session_id,
                'tokens': tokens
            })
            
        except Exception as e:
            logger.debug(f"Failed to update session stats: {e}")
    
    async def search_messages(self,
                            query_text: str,
                            limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search messages using full-text search
        
        Args:
            query_text: Search query
            limit: Maximum results
            
        Returns:
            List of matching messages
        """
        await self.ensure_connected()
        
        try:
            # Use SurrealDB's CONTAINS for simple text search  
            query = """
                SELECT * FROM messages
                WHERE content CONTAINS $query
                ORDER BY timestamp DESC
                LIMIT $limit;
            """
            
            result = await self.db.query(query, {
                'query': query_text,
                'limit': limit
            })
            
            if result and isinstance(result, list):
                return result
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to search messages: {e}")
            return []
    
    async def get_message_graph(self,
                              message_id: str,
                              depth: int = 2) -> Dict[str, Any]:
        """
        Get message with its relationships
        
        Args:
            message_id: Message to start from
            depth: Graph traversal depth
            
        Returns:
            Message and related messages
        """
        await self.ensure_connected()
        
        try:
            query = """
                SELECT *,
                    ->responded_to->messages AS responses,
                    <-responded_to<-messages AS parent
                FROM messages
                WHERE id = $message_id
                FETCH responses, parent;
            """
            
            result = await self.db.query(query, {
                'message_id': message_id
            })
            
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].get('result', {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get message graph: {e}")
            return {}
    
    async def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
        """
        FactsGraph compatibility - search unified knowledge system
        """
        try:
            # Use unified knowledge search
            relations = await self.search_knowledge_relations(query, limit)
            
            # Convert to legacy facts format for compatibility
            facts = []
            for rel in relations:
                fact = {
                    'subject': rel.get('subject', ''),
                    'predicate': rel.get('predicate', ''),
                    'object': rel.get('object', ''),
                    'value': rel.get('object', ''),  # Compatibility alias
                    'confidence': rel.get('confidence', 0.8),
                    'strength': rel.get('confidence', 0.8),  # Compatibility alias
                    'score': int(rel.get('confidence', 0.8) * 100)
                }
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            logger.error(f"Failed to search facts: {e}")
            return []
    
    # ================================================================================
    # NEW: UNIFIED KNOWLEDGE SYSTEM METHODS
    # ================================================================================
    
    async def store_knowledge_relation(self, subject_name: str, predicate: str, object_name: str, 
                                     subject_type: str = 'user', object_type: str = 'concept',
                                     confidence: float = 0.8, source_message_id: str = None,
                                     embedding: Optional[List[float]] = None,
                                     session_id: Optional[str] = None) -> bool:
        """
        Store knowledge using direct SurrealQL with adaptive predicate normalization
        """
        await self.ensure_connected()

        # VALIDATION & NORMALIZATION (Phase 1 Fix)
        try:
            from memory.validation import normalize_entity, validate_fact
            import os

            # Normalize subject and object names first
            subject_norm = normalize_entity(subject_name)
            object_norm = normalize_entity(object_name)

            # Check if validation bypass is enabled for testing
            bypass_validation = os.getenv("BYPASS_GUARDIAN", "false").lower() == "true"
            
            if bypass_validation:
                logger.debug(f"🚧 Validation BYPASSED for testing: {subject_name} -> {predicate} -> {object_name}")
            else:
                # Validate the fact before proceeding
                if not validate_fact(subject_norm, predicate, object_norm):
                    logger.warning(f"Fact validation failed for: {subject_name} -> {predicate} -> {object_name}")
                    return False
        except ImportError:
            logger.error("Could not import validation module. Skipping fact validation.")
            subject_norm = subject_name
            object_norm = object_name
        
        # 🧬 ADAPTIVE NORMALIZATION: Let the knowledge graph learn and normalize predicates
        try:
            from memory.adaptive_knowledge_graph import normalize_predicate_adaptive
            normalized_predicate = await normalize_predicate_adaptive(predicate)
            if normalized_predicate != predicate:
                logger.debug(f"🔄 Predicate normalized: '{predicate}' → '{normalized_predicate}'")
                predicate = normalized_predicate
        except Exception as e:
            logger.warning(f"Adaptive normalization failed, using original predicate: {e}")
            # Continue with original predicate if adaptive system fails
        
        try:
            # Create safe entity IDs by removing/replacing problematic characters
            import re
            def make_safe_id(name: str) -> str:
                # Clean the name first (remove possessives, articles)
                clean_name = name.strip()
                
                # Remove possessives (fixes "Sardinia's" → "Sardinia")
                if clean_name.endswith("'s"):
                    clean_name = clean_name[:-2]
                elif clean_name.endswith("s'"):
                    clean_name = clean_name[:-1]
                
                # Remove leading articles
                import re as regex
                clean_name = regex.sub(r'^(the|a|an)\s+', '', clean_name, flags=regex.IGNORECASE).strip()
                
                # Replace problematic characters with underscores or remove them
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
                # Remove consecutive underscores and leading/trailing ones
                safe_name = re.sub(r'_+', '_', safe_name).strip('_')
                # Ensure it doesn't start with a number
                if safe_name and safe_name[0].isdigit():
                    safe_name = 'entity_' + safe_name
                return safe_name.lower() or 'unknown'  # Lowercase for consistency
            
            subject_safe = make_safe_id(subject_norm)
            object_safe = make_safe_id(object_norm)
            
            # Use cleaned names for canonical_name (prevents "Sardinia's" in display)
            subject_clean = subject_norm.strip()
            if subject_clean.endswith("'s"):
                subject_clean = subject_clean[:-2]
            elif subject_clean.endswith("s'"):
                subject_clean = subject_clean[:-1]
            
            object_clean = object_norm.strip()  
            if object_clean.endswith("'s"):
                object_clean = object_clean[:-2]
            elif object_clean.endswith("s'"):
                object_clean = object_clean[:-1]
            
            logger.debug(f"Creating entities: {subject_safe}, {object_safe}")
            
            # Create entities with parameterized queries - ignore if they exist
            try:
                await self.db.query(f"CREATE entity:{subject_safe} SET type=$subject_type, canonical_name=$subject_name, session_id=$session_id;", {
                    'subject_type': subject_type,
                    'subject_name': subject_clean,  # Use cleaned name
                    'session_id': session_id
                })
            except Exception:
                pass  # Entity might already exist
            
            try:
                await self.db.query(f"CREATE entity:{object_safe} SET type=$object_type, canonical_name=$object_name, session_id=$session_id;", {
                    'object_type': object_type,
                    'object_name': object_clean,  # Use cleaned name
                    'session_id': session_id
                })
            except Exception:
                pass  # Entity might already exist
            
            # Check for existing fact (disabled for node-based model; rely on downstream consolidation)
            existing = []
            
            if existing and len(existing) > 0:
                # Relation exists, update access stats and embedding if provided
                relation_id = existing[0].get('id')
                
                if embedding is not None:
                    # Update with embedding
                    update_result = await self.db.query(f"""
                        UPDATE {relation_id} SET
                            last_accessed = time::now(),
                            access_count = access_count + 1,
                            strength = math::min(1.0, strength + 0.1),
                            embedding = $embedding;
                    """, {'embedding': embedding})
                    logger.debug(f"Updated existing relation with embedding: {relation_id}")
                else:
                    # Update without embedding
                    update_result = await self.db.query(f"""
                        UPDATE {relation_id} SET
                            last_accessed = time::now(),
                            access_count = access_count + 1,
                            strength = math::min(1.0, strength + 0.1);
                    """)
                    logger.debug(f"Updated existing relation: {relation_id}")
                
                result = existing  # Return existing relation
            else:
                # Normalize predicate using ontology
                try:
                    normalize_result = await self.db.query("RETURN fn::normalize_predicate($pred);", {'pred': predicate})
                    canonical_predicate = normalize_result[0]['result'] if normalize_result and normalize_result[0] else predicate
                except:
                    canonical_predicate = predicate  # Fallback to original
                
                # Validate predicate usage (optional - log warnings but don't block)
                try:
                    validation_result = await self.db.query(
                        "RETURN fn::validate_predicate_usage($pred, $subj_type, $obj_type);", 
                        {'pred': canonical_predicate, 'subj_type': subject_type, 'obj_type': object_type}
                    )
                    validation = validation_result[0]['result'] if validation_result and validation_result[0] else {'valid': True}
                    if not validation.get('valid', True):
                        logger.warning(f"Predicate validation warning: {canonical_predicate} - {validation.get('reason')}")
                except:
                    pass  # Don't block on validation errors
                    
                # Create knowledge as node and link per latest schema
                result = []
                # 1) CREATE knowledge node
                k_params = {
                    'predicate': canonical_predicate,
                    'confidence': float(confidence),
                    'session_id': session_id,
                    'embedding': embedding or None,
                    'method': 'conversation'
                }
                kq = """
                    CREATE knowledge SET
                        predicate = $predicate,
                        confidence = $confidence,
                        strength = $confidence,
                        created_at = time::now(),
                        last_accessed = time::now(),
                        session_id = $session_id,
                        extraction_method = $method,
                        embedding = $embedding
                    RETURN id;
                """
                kres = await self.db.query(kq, k_params)
                knowledge_id = None
                if isinstance(kres, list) and kres:
                    knowledge_id = kres[0].get('id') if isinstance(kres[0], dict) else None

                # 2) Link subject/object via knowledge_about
                if knowledge_id:
                    try:
                        await self.db.query(
                            """
                            RELATE $kid -> knowledge_about -> $sub_eid SET
                                relationship_type = 'subject',
                                relevance_score = $confidence,
                                discovered_at = time::now();
                            """,
                            {"kid": knowledge_id, "sub_eid": f"entity:{subject_safe}", "confidence": float(confidence)},
                        )
                        await self.db.query(
                            """
                            RELATE $kid -> knowledge_about -> $obj_eid SET
                                relationship_type = 'object',
                                relevance_score = $confidence,
                                discovered_at = time::now();
                            """,
                            {"kid": knowledge_id, "obj_eid": f"entity:{object_safe}", "confidence": float(confidence)},
                        )
                    except Exception as e:
                        logger.debug(f"knowledge_about link skipped: {e}")

                # 3) Link provenance to session via knowledge_from
                if knowledge_id and session_id:
                    try:
                        sess = await self.db.query("SELECT * FROM sessions WHERE session_id = $sid LIMIT 1;", {"sid": session_id})
                        if isinstance(sess, list) and sess:
                            sess_id = sess[0].get('id')
                            if sess_id:
                                await self.db.query(
                                    """
                                    RELATE $kid -> knowledge_from -> $sid SET
                                        extraction_method = 'conversation',
                                        extraction_confidence = $confidence,
                                        learned_at = time::now();
                                    """,
                                    {"kid": knowledge_id, "sid": sess_id, "confidence": float(confidence)},
                                )
                    except Exception as e:
                        logger.debug(f"knowledge_from link skipped: {e}")

            logger.debug(f"Relation result structure: {kres}")
            
            # Consider success if we created a knowledge node id
            success = bool(knowledge_id)

            if success:
                logger.info(f"✅ Stored knowledge: {subject_norm} -{predicate}-> {object_norm}")

                # 🔗 Link message provenance and session/entity associations
                try:
                    if source_message_id and knowledge_id:
                        # Link message -> knowledge
                        await self.db.query(
                            "RELATE $mid -> message_contains -> $kid SET created_at = time::now();",
                            {"mid": source_message_id, "kid": knowledge_id},
                        )

                    if session_id:
                        sess = await self.db.query("SELECT * FROM sessions WHERE session_id = $sid LIMIT 1;", {"sid": session_id})
                        if isinstance(sess, list) and sess:
                            sess_id = sess[0].get('id')
                            # session_involves
                            if sess_id:
                                await self.db.query(
                                    "RELATE $sid -> session_involves -> $sub_eid SET message_count += 1, last_interaction = time::now();",
                                    {"sid": sess_id, "sub_eid": f"entity:{subject_safe}"},
                                )
                                await self.db.query(
                                    "RELATE $sid -> session_involves -> $obj_eid SET message_count += 1, last_interaction = time::now();",
                                    {"sid": sess_id, "obj_eid": f"entity:{object_safe}"},
                                )
                                # entity_mentioned_in (entity -> sessions)
                                await self.db.query(
                                    "RELATE $sub_eid -> entity_mentioned_in -> $sid SET last_mentioned = time::now();",
                                    {"sub_eid": f"entity:{subject_safe}", "sid": sess_id},
                                )
                                await self.db.query(
                                    "RELATE $obj_eid -> entity_mentioned_in -> $sid SET last_mentioned = time::now();",
                                    {"obj_eid": f"entity:{object_safe}", "sid": sess_id},
                                )
                except Exception as relate_error:
                    logger.warning(f"⚠️ Failed to create some relations: {relate_error}")

                # 🧬 NOTIFY EVOLUTION SERVICE: Let the adaptive system learn from this new fact
                try:
                    from services.knowledge_evolution_service import on_knowledge_stored
                    on_knowledge_stored()
                except ImportError:
                    pass  # Evolution service not available, continue normally
            else:
                logger.warning(f"Knowledge creation did not return id (predicate={predicate})")
            
            return success
            
        except Exception as e:
            logger.error(f"Knowledge storage failed: {e}")
            return False
    
    
    async def search_knowledge_relations(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Advanced search using embeddings and hybrid matching
        """
        await self.ensure_connected()
        
        try:
            # Generate query embedding for semantic search
            query_embedding = None
            transformer = get_query_sentence_transformer()
            
            if transformer:
                try:
                    query_embedding = transformer.encode(query, convert_to_numpy=True).tolist()
                    logger.debug(f"Generated query embedding ({len(query_embedding)} dims) for: '{query[:50]}...'")
                except Exception as e:
                    logger.warning(f"Failed to generate query embedding: {e}")
            
            # Use advanced search function if embedding is available, fallback to text search
            if query_embedding:
                result = await self.db.query("""
                    SELECT *, 
                           in.canonical_name as subject,
                           out.canonical_name as object
                    FROM fn::search_knowledge_advanced($query, $query_embedding, $limit);
                """, {
                    'query': query,
                    'query_embedding': query_embedding,
                    'limit': limit
                })
                logger.debug(f"🔍 Advanced search with embeddings returned {len(result)} results")
            else:
                # Fallback to text-only search
                result = await self.db.query("""
                    SELECT *, 
                           in.canonical_name as subject,
                           out.canonical_name as object
                    FROM fn::search_knowledge($query, $limit);
                """, {
                    'query': query,
                    'limit': limit
                })
                logger.debug(f"🔍 Text-only search returned {len(result)} results")
            
            # Update last_accessed timestamp for retrieved facts to track usage
            if result and isinstance(result, list) and result:
                fact_ids = []
                for fact in result:
                    if hasattr(fact, 'id') or 'id' in fact:
                        fact_id = getattr(fact, 'id', None) or fact.get('id')
                        if fact_id:
                            fact_ids.append(fact_id)
                
                if fact_ids:
                    try:
                        # Batch update access timestamps and increment access counts
                        await self.db.query("""
                            UPDATE $fact_ids SET 
                                last_accessed = time::now(),
                                access_count = access_count + 1;
                        """, {'fact_ids': fact_ids})
                        logger.debug(f"Updated access tracking for {len(fact_ids)} facts")
                    except Exception as e:
                        logger.warning(f"Failed to update access tracking: {e}")
            
            # Log result structure without embeddings (too verbose)
            if result:
                sample_result = result[0].copy() if result[0] else {}
                if 'embedding' in sample_result:
                    sample_result['embedding'] = f"[{len(sample_result['embedding'])} dims]"
                logger.debug(f"Search result structure: {sample_result}")
            else:
                logger.debug("Search result structure: None")
            
            # SurrealDB Python SDK returns the data directly as a list
            if result and isinstance(result, list):
                return result
            
            return []
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def get_entity_knowledge(self, entity_name: str, limit: int = 20) -> List[Dict]:
        """
        Get all knowledge about a specific entity
        
        Args:
            entity_name: Name of the entity
            limit: Maximum number of results
            
        Returns:
            List of knowledge relations involving this entity
        """
        await self.ensure_connected()
        
        try:
            # Direct query for entity knowledge (functions may not be working)
            result = await self.db.query(
                """
                SELECT * FROM knowledge 
                WHERE in.canonical_name = $entity_name OR out.canonical_name = $entity_name
                ORDER BY strength DESC, temporal_context.when_said DESC
                LIMIT $limit;
                """,
                {'entity_name': entity_name, 'limit': limit}
            )
            
            if result and len(result) > 0:
                # Handle different result formats
                if isinstance(result[0], dict):
                    return result[0].get('result', [])
                else:
                    return result if isinstance(result, list) else []
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get entity knowledge: {e}")
            return []
    
    async def search_tape(self, query: str, limit: int = 10, **kwargs) -> list:
        """
        Search tape/conversation messages (compatibility method)
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of message results formatted as tape entries
        """
        try:
            # Use existing search_messages method
            messages = await self.search_messages(query, limit=limit)
            
            # Format as tape-like entries for compatibility
            tape_results = []
            for msg in messages:
                tape_entry = {
                    'content': msg.get('content', ''),
                    'role': msg.get('role', ''),
                    'speaker_id': msg.get('speaker_id', ''),
                    'timestamp': msg.get('timestamp', ''),
                    'session_id': msg.get('session_id', ''),
                    'ts': msg.get('timestamp', ''),  # Compatibility alias
                    'metadata': msg.get('metadata', {})
                }
                tape_results.append(tape_entry)
            
            return tape_results
            
        except Exception as e:
            logger.error(f"Search tape failed: {e}")
            return []


# Singleton instance
_connection_manager: Optional[SurrealConnectionManager] = None


def get_surreal_connection() -> SurrealConnectionManager:
    """Get or create singleton connection manager"""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = SurrealConnectionManager()
    
    return _connection_manager


async def store_conversation_turn(user_text: str, 
                                 assistant_text: str,
                                 session_id: str = None,
                                 speaker_id: str = 'default_user') -> bool:
    """
    Convenience function to store a conversation turn
    
    Args:
        user_text: User's message
        assistant_text: Assistant's response
        session_id: Optional session ID
        speaker_id: Speaker identifier
        
    Returns:
        True if both messages stored successfully
    """
    manager = get_surreal_connection()
    
    try:
        # Create user message
        user_msg = Message(
            role='user',
            content=user_text,
            speaker_id=speaker_id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc)
        )
        
        user_id = await manager.store_message(user_msg)
        
        if user_id:
            # Create assistant message with parent reference
            assistant_msg = Message(
                role='assistant',
                content=assistant_text,
                speaker_id='assistant',
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                parent_message=user_id
            )
            
            assistant_id = await manager.store_message(assistant_msg)
            
            if assistant_id:
                # Create relationship - convert string IDs back to RecordID objects
                from surrealdb import RecordID
                user_record = RecordID('messages', user_id.split(':')[1]) if ':' in str(user_id) else user_id
                assistant_record = RecordID('messages', assistant_id.split(':')[1]) if ':' in str(assistant_id) else assistant_id
                
                await manager.db.query(
                    "RELATE $user->responded_to->$assistant",
                    {'user': user_record, 'assistant': assistant_record}
                )
                
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to store conversation turn: {e}")
        return False
