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
                           metadata: Dict[str, Any] = None) -> Optional[str]:
        """
        Create a new conversation session
        
        Args:
            speaker_id: Speaker identifier
            metadata: Optional session metadata
            
        Returns:
            Session ID if successful
        """
        await self.ensure_connected()
        
        try:
            import uuid
            session_id = f"session_{uuid.uuid4().hex[:12]}"
            
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
    
    async def store_facts(self, facts: list, speaker_id: str = None) -> int:
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
                
                # Store using unified knowledge system
                success = await self.store_knowledge_relation(
                    subject_name=subject,
                    predicate=predicate, 
                    object_name=obj,
                    confidence=confidence,
                    embedding=embedding
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
            
            # Use proper SurrealDB COUNT aggregation with GROUP ALL for single result
            result = await self.db.query(
                "SELECT count() AS session_count FROM sessions WHERE speaker_id = $speaker_id GROUP ALL;",
                {'speaker_id': speaker_id}
            )
            
            logger.debug(f"Session count query result for {speaker_id}: {result}")
            
            session_count = 0
            if result and len(result) > 0:
                # SurrealDB COUNT returns format: [{"session_count": N}] 
                count_record = result[0]  # First query result
                session_count = count_record.get('session_count', 0)
                logger.debug(f"Found {session_count} sessions for {speaker_id}")
            
            return {
                'speaker_id': speaker_id,
                'session_count': session_count,
                'type': 'SurrealDB'
            }
            
        except Exception as e:
            logger.debug(f"Failed to get session info: {e}")
            return {
                'speaker_id': speaker_id,
                'session_count': 0,
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
                                     embedding: Optional[List[float]] = None) -> bool:
        """
        Store knowledge using direct SurrealQL - simple and reliable
        """
        await self.ensure_connected()
        
        try:
            # Create safe entity IDs by removing/replacing problematic characters
            import re
            def make_safe_id(name: str) -> str:
                # Replace problematic characters with underscores or remove them
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
                # Remove consecutive underscores and leading/trailing ones
                safe_name = re.sub(r'_+', '_', safe_name).strip('_')
                # Ensure it doesn't start with a number
                if safe_name and safe_name[0].isdigit():
                    safe_name = 'entity_' + safe_name
                return safe_name or 'unknown'
            
            subject_safe = make_safe_id(subject_name)
            object_safe = make_safe_id(object_name)
            
            logger.debug(f"Creating entities: {subject_safe}, {object_safe}")
            
            # Create entities with parameterized queries - ignore if they exist
            try:
                await self.db.query(f"CREATE entity:{subject_safe} SET type=$subject_type, canonical_name=$subject_name;", {
                    'subject_type': subject_type,
                    'subject_name': subject_name
                })
            except Exception:
                pass  # Entity might already exist
            
            try:
                await self.db.query(f"CREATE entity:{object_safe} SET type=$object_type, canonical_name=$object_name;", {
                    'object_type': object_type,
                    'object_name': object_name
                })
            except Exception:
                pass  # Entity might already exist
            
            # Check if relation already exists to prevent duplicates
            existing_query = f"""
                SELECT * FROM knowledge 
                WHERE in = entity:{subject_safe} 
                  AND out = entity:{object_safe} 
                  AND predicate = $predicate
                LIMIT 1;
            """
            
            existing = await self.db.query(existing_query, {'predicate': predicate})
            
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
                # Create new relation
                query_params = {
                    'predicate': predicate,
                    'confidence': confidence
                }
                
                if embedding is not None:
                    relation_query = f"""
                        RELATE entity:{subject_safe}->knowledge->entity:{object_safe} SET
                            predicate = $predicate,
                            confidence = $confidence,
                            strength = 1.0,
                            embedding = $embedding,
                            created_at = time::now();
                    """
                    query_params['embedding'] = embedding
                else:
                    relation_query = f"""
                        RELATE entity:{subject_safe}->knowledge->entity:{object_safe} SET
                            predicate = $predicate,
                            confidence = $confidence,
                            strength = 1.0,
                            created_at = time::now();
                    """
                
                result = await self.db.query(relation_query, query_params)
            logger.debug(f"Relation result structure: {result}")
            
            # Check if relation was created successfully
            success = False
            if result and len(result) > 0:
                if hasattr(result[0], 'result'):
                    success = bool(result[0].result)
                elif isinstance(result[0], dict) and 'result' in result[0]:
                    success = bool(result[0]['result'])
                else:
                    success = bool(result[0])
            
            if success:
                logger.info(f"✅ Stored knowledge: {subject_name} -{predicate}-> {object_name}")
            else:
                logger.warning(f"Relation creation may have failed: {result}")
            
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