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

try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    SURREALDB_AVAILABLE = False
    AsyncSurreal = None
    logger.warning("SurrealDB client not available - install with: pip install surrealdb")


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
        if not self.session_id or self.session_id == 'default':
            raise ValueError(f"Message must have valid session_id, got: {self.session_id}")
        if self.role not in ['user', 'assistant', 'system']:
            raise ValueError(f"Invalid role: {self.role}")
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")
        if not self.speaker_id:
            raise ValueError("Message must have speaker_id")
    
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
    
    async def add_entry(self, role: str, content: str, speaker_id: str = "default_user", ts: Optional[float] = None, **kwargs) -> bool:
        """
        TapeStore compatibility method - add a tape entry
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            speaker_id: Speaker identifier
            ts: Optional timestamp (Unix timestamp)
            **kwargs: Additional metadata
        """
        try:
            # Convert timestamp if provided
            timestamp = datetime.fromtimestamp(ts, timezone.utc) if ts else datetime.now(timezone.utc)
            
            message = Message(
                role=role,
                content=content,
                speaker_id=speaker_id,
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
        FactsGraph compatibility method - store extracted facts
        """
        await self.ensure_connected()  # Ensure database connection
        
        try:
            stored_count = 0
            for fact in facts:
                # Convert fact to proper fact_plain table format per SURREAL_GRAPH.md schema
                import os
                user_id = f"user:{os.getenv('USER_ID', 'default_user')}"
                
                if isinstance(fact, dict):
                    fact_data = {
                        'user_id': user_id,
                        'subject': fact.get('subject', str(fact)),
                        'predicate': fact.get('predicate', 'mentioned'),
                        'value': fact.get('value', str(fact)),  # Use 'value' not 'object'
                        'fidelity': fact.get('fidelity', 4),
                        'strength': fact.get('strength', 0.8),
                        'source_text': fact.get('source_text', ''),
                        'created': datetime.now(timezone.utc),
                    }
                else:
                    fact_data = {
                        'user_id': user_id,
                        'subject': getattr(fact, 'subject', str(fact)),
                        'predicate': getattr(fact, 'predicate', 'mentioned'),
                        'value': getattr(fact, 'value', str(fact)),
                        'fidelity': getattr(fact, 'fidelity', 4),
                        'strength': getattr(fact, 'strength', 0.8),
                        'source_text': getattr(fact, 'source_text', ''),
                        'created': datetime.now(timezone.utc),
                    }
                
                try:
                    result = await self.db.query(
                        "CREATE fact_plain CONTENT $data;",
                        {'data': fact_data}
                    )
                    if result and len(result) > 0:
                        stored_count += 1
                except Exception as e:
                    logger.debug(f"Failed to store individual fact: {e}")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store facts: {e}")
            return 0
    
    async def extract_and_store_facts(self, text: str, speaker_id: str = None) -> int:
        """
        Extract facts from text and store them
        """
        await self.ensure_connected()  # Ensure database connection
        
        try:
            # Simple fact extraction - in production this would use NLP
            # For now, just store the text as a semantic fact
            if len(text.strip()) > 10:  # Only store meaningful text
                # First create a message to serve as the source
                source_message = Message(
                    role='user',
                    content=text,
                    speaker_id=speaker_id or 'default_user',
                    timestamp=datetime.now(timezone.utc)
                )
                
                source_id = await self.store_message(source_message)
                
                if source_id:
                    # Convert string ID back to RecordID for the source field
                    from surrealdb import RecordID
                    if ':' in str(source_id):
                        table, record_id = str(source_id).split(':', 1)
                        source_record = RecordID(table, record_id)
                    else:
                        source_record = source_id
                
                    fact_data = {
                        'subject': speaker_id or 'user',
                        'predicate': 'said',
                        'object': text,  # Use 'object' field name
                        'confidence': 1.0,
                        'source': source_record,  # Required field
                        'extracted_at': datetime.now(timezone.utc),
                        'last_verified': datetime.now(timezone.utc),
                        'decay_rate': 0.1,
                        'importance': 1.0,  # S4 highest fidelity = 1.0 importance
                    }
                else:
                    # If we can't create the source message, skip fact storage
                    return 0
                
                result = await self.db.create('facts', fact_data)
                return 1 if result else 0
            
            return 0
            
        except Exception as e:
            logger.debug(f"Failed to extract and store facts: {e}")
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
        Search facts by text content (FactsGraph compatibility)
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching facts as dictionaries
        """
        await self.ensure_connected()
        
        try:
            # Search facts using SurrealDB's full-text search capabilities
            search_query = """
                SELECT *,
                    math::floor((confidence * importance * 100)) AS score
                FROM facts
                WHERE 
                    subject @@ $query OR 
                    predicate @@ $query OR 
                    object @@ $query
                ORDER BY score DESC, last_verified DESC
                LIMIT $limit;
            """
            
            result = await self.db.query(search_query, {
                'query': query,
                'limit': limit
            })
            
            if result and isinstance(result, list) and len(result) > 0:
                facts = result[0].get('result', [])
                
                # Convert to expected format for compatibility
                formatted_facts = []
                for fact in facts:
                    # Handle RecordID objects
                    fact_id = fact.get('id')
                    if hasattr(fact_id, 'id'):
                        fact_id = str(fact_id)
                    
                    formatted_fact = {
                        'id': fact_id,
                        'subject': fact.get('subject', ''),
                        'predicate': fact.get('predicate', ''),
                        'object': fact.get('object', ''),  # SurrealDB uses 'object' field
                        'value': fact.get('object', ''),   # Compatibility alias
                        'confidence': fact.get('confidence', 0.8),
                        'importance': fact.get('importance', 0.5),
                        'strength': fact.get('confidence', 0.8),  # Compatibility alias
                        'score': fact.get('score', 50),
                        'extracted_at': fact.get('extracted_at'),
                        'last_verified': fact.get('last_verified')
                    }
                    formatted_facts.append(formatted_fact)
                
                logger.debug(f"🔍 Found {len(formatted_facts)} facts for query: {query}")
                return formatted_facts
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to search facts: {e}")
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