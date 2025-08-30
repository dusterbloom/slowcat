"""
Smart Memory System - Fixed context with natural decay

Components:
- FactsGraph: Structured knowledge with fidelity levels (S4→S0)
- QueryClassifier: Language-agnostic intent classification  
- QueryRouter: Multi-store retrieval with intelligent routing
- SmartContextManager: Fixed 4096-token context management

Usage:
    from memory import create_smart_memory_system
    
    memory = create_smart_memory_system()
    response = await memory.process_query("What's my dog's name?")
"""

# SurrealDB-native imports
from .surreal_connection import get_surreal_connection, Message
from .query_classifier import (
    HybridQueryClassifier, QueryIntent, ClassificationResult, 
    create_query_classifier
)
from .query_router import (
    QueryRouter, RetrievalResponse, MemoryResult, 
    create_query_router
)
from .hybrid_fact_extractor import extract_facts_from_text

__all__ = [
    # Core SurrealDB components
    'get_surreal_connection',
    'Message',
    'HybridQueryClassifier', 
    'QueryRouter',
    
    # Data classes
    'QueryIntent',
    'ClassificationResult',
    'RetrievalResponse', 
    'MemoryResult',
    
    # Factory functions
    'create_query_classifier',
    'create_query_router',
    'create_smart_memory_system',
    'create_surreal_message_store',
    
    # Utilities
    'extract_facts_from_text',
]


def create_smart_memory_system(facts_db_path: str = "data/facts.db",
                              tape_store=None,
                              embedding_store=None):
    """
    Create SurrealDB-native smart memory system
    
    Environment Variables:
        SURREAL_URL: SurrealDB connection URL (default: ws://localhost:8000)
        SURREAL_USER: SurrealDB username (default: root)  
        SURREAL_PASS: SurrealDB password (default: root)
        SURREAL_NAMESPACE: Database namespace (default: slowcat)
        SURREAL_DATABASE: Database name (default: consciousness)
    
    Args:
        facts_db_path: Ignored - SurrealDB manages persistence
        tape_store: Ignored - SurrealDB provides unified storage
        embedding_store: Ignored - SurrealDB handles embeddings
        
    Returns:
        SurrealMemorySystem instance
    """
    import os
    from loguru import logger
    
    logger.info("🚀 Creating SurrealDB-native memory system")
    
    try:
        # Get SurrealDB connection manager
        surreal_manager = get_surreal_connection()
        
        # Ensure all schema functions exist (deferred for async context)
        from .schema_init import ensure_schema_functions
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule schema initialization for later
                asyncio.create_task(ensure_schema_functions(surreal_manager))
                logger.info("📋 Schema functions scheduled for async initialization")
            else:
                # Run schema initialization in sync context
                schema_ready = loop.run_until_complete(ensure_schema_functions(surreal_manager))
                if not schema_ready:
                    logger.warning("⚠️ Schema functions not fully applied, continuing anyway")
        except Exception as e:
            logger.warning(f"⚠️ Schema initialization deferred: {e}")
        
        # Create query router that works with SurrealDB
        try:
            query_router = create_query_router(
                facts_graph=surreal_manager,  # SurrealDB acts as facts store
                tape_store=surreal_manager,   # SurrealDB acts as tape store  
                embedding_store=None          # SurrealDB handles embeddings
            )
        except Exception as e:
            logger.warning(f"Query router creation failed: {e}")
            query_router = None
        
        # Return SurrealDB memory system
        return SurrealMemorySystem(
            connection_manager=surreal_manager,
            query_router=query_router
        )
        
    except Exception as e:
        logger.error(f"SurrealDB memory system creation failed: {e}")
        raise RuntimeError(f"Cannot create memory system: {e}")


class SurrealMemorySystem:
    """
    SurrealDB-native memory system for conversation storage and retrieval
    """
    
    def __init__(self, connection_manager, query_router=None):
        self.connection_manager = connection_manager
        self.query_router = query_router
        self.current_session_id = None
        self._session_cache = {}
        
        # Provide compatibility interfaces
        self.facts_graph = connection_manager  # SurrealDB provides facts interface
        self.tape_store = connection_manager   # SurrealDB provides tape interface
        
        # Session creation is handled by SurrealMessageStore - no need to auto-create here
        # This prevents duplicate session creation
        from loguru import logger
        logger.debug("Session management delegated to SurrealMessageStore - no auto-creation")
    
    async def _init_session(self):
        """Initialize default session using configured USER_ID"""
        try:
            import os
            user_id = os.getenv('USER_ID', 'default_user')
            self.current_session_id = await self.start_session(user_id)
            from loguru import logger
            logger.info(f"🎬 Auto-created session: {self.current_session_id} for user: {user_id}")
        except Exception as e:
            from loguru import logger
            logger.warning(f"Failed to auto-create session: {e}")
    
    async def start_session(self, speaker_id: str = None):
        """Start a new session and cache it"""
        try:
            # Use configured USER_ID if no speaker_id provided
            if not speaker_id:
                import os
                speaker_id = os.getenv('USER_ID', 'default_user')
            
            session_id = await self.connection_manager.create_session(speaker_id)
            self.current_session_id = session_id
            self._session_cache[speaker_id] = session_id
            return session_id
        except Exception as e:
            # Generate fallback session ID
            import uuid
            fallback_id = f"session_{uuid.uuid4().hex[:12]}"
            self.current_session_id = fallback_id
            self._session_cache[speaker_id] = fallback_id
            return fallback_id
    
    def get_current_session(self):
        """Get current session ID, creating one if needed"""
        if not self.current_session_id:
            import uuid
            self.current_session_id = f"session_{uuid.uuid4().hex[:12]}"
        return self.current_session_id
    
    async def search_unified(self, query: str, limit: int = 20):
        """Unified search across messages and knowledge"""
        from loguru import logger
        results = []
        
        try:
            # Search messages
            messages = await self.connection_manager.search_messages(query, limit=limit//2)
            for msg in messages:
                results.append({
                    'content': msg.get('content', ''),
                    'type': 'message',
                    'score': 0.8,
                    'metadata': msg
                })
            logger.debug(f"Unified search: {len(messages)} messages")
        except Exception as e:
            logger.debug(f"Message search failed: {e}")
        
        try:
            # Search knowledge relations
            knowledge = await self.connection_manager.search_knowledge_relations(query, limit=limit//2)
            for rel in knowledge:
                content = f"{rel.get('subject', '')} {rel.get('predicate', '')} {rel.get('object', '')}"
                results.append({
                    'content': content,
                    'type': 'knowledge',
                    'score': rel.get('confidence', 0.9),
                    'metadata': rel
                })
            logger.debug(f"Unified search: {len(knowledge)} knowledge relations")
        except Exception as e:
            logger.debug(f"Knowledge search failed: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f"🔍 Unified search for '{query}' found {len(results)} total results")
        return results[:limit]
    
    async def process_query(self, query: str, context: dict = None):
        """Process query using unified search and query router"""
        try:
            if self.query_router:
                # Use query router if available
                return await self.query_router.route_query(query, context)
            else:
                # Use unified search as fallback
                results = await self.search_unified(query, limit=10)
                
                # Convert unified search results to compatible format
                class SimpleResult:
                    def __init__(self, result_dict):
                        content = result_dict.get('content', '')
                        metadata = result_dict.get('metadata', {})
                        result_type = result_dict.get('type', 'message')
                        
                        if result_type == 'knowledge':
                            self.subject = metadata.get('subject', '')
                            self.predicate = metadata.get('predicate', '')
                            self.value = metadata.get('object', '')
                            self.source_store = 'knowledge'
                        else:
                            self.subject = metadata.get('speaker_id', '')
                            self.predicate = 'said'
                            self.value = content
                            self.source_store = 'messages'
                        
                        self.species = None
                        self.fidelity = 3
                        self.strength = result_dict.get('score', 0.8)
                        self.last_seen = metadata.get('timestamp', 0)
                        self.created = metadata.get('timestamp', 0)
                        self.access_count = 0
                        self.source_text = content
                
                formatted_results = [SimpleResult(r) for r in results]
                
                class SimpleResponse:
                    def __init__(self, results):
                        self.results = results
                        self.total_results = len(results)
                        self.retrieval_time_ms = 0
                        self.strategy_used = 'surreal_direct'
                        self.stores_queried = ['surreal_messages']
                        self.classification = self._default_classification()
                    
                    def _default_classification(self):
                        class SimpleClassification:
                            def __init__(self):
                                self.intent = SimpleIntent()
                                self.confidence = 0.6
                        
                        class SimpleIntent:
                            def __init__(self):
                                self.name = 'MESSAGE_SEARCH'
                        
                        return SimpleClassification()
                
                return SimpleResponse(formatted_results)
                
        except Exception as e:
            from loguru import logger
            logger.debug(f"Query processing failed: {e}")
            # Return empty response
            class EmptyResponse:
                def __init__(self):
                    self.results = []
                    self.total_results = 0
                    self.retrieval_time_ms = 0
                    self.strategy_used = 'error_fallback'
                    self.stores_queried = []
                    self.classification = None
            
            return EmptyResponse()
    
    async def store_message(self, role: str, content: str, speaker_id: str = 'default_user'):
        """DISABLED: Message storage handled by SurrealMessageStore processor with proper token counts"""
        from loguru import logger
        logger.debug(f"📝 SurrealMemorySystem.store_message disabled - SurrealMessageStore handles storage ({role}: {content[:50]}...)")
        
        # Extract and store facts from user messages (this is still valuable)
        if role == 'user':
            try:
                await self.store_facts(content, speaker_id)
            except Exception as e:
                logger.debug(f"Fact extraction failed: {e}")
        
        return True  # Return True to not break calling code
        
        # Original method commented out to prevent duplicate storage:
        # from loguru import logger
        # try:
        #     # Ensure we have a session
        #     if not self.current_session_id:
        #         await self._init_session()
        #     
        #     # Store the message
        #     success = await self.connection_manager.add_entry(
        #         role=role,
        #         content=content,
        #         speaker_id=speaker_id,
        #         session_id=self.current_session_id
        #     )
        #     
        #     if success:
        #         logger.debug(f"📝 Stored {role} message in session {self.current_session_id}")
        #         
        #         # Also extract and store facts from user messages
        #         if role == 'user':
        #             await self.store_facts(content, speaker_id)
        #     
        #     return success
        #     
        # except Exception as e:
        #     logger.error(f"Message storage failed: {e}")
        #     return False
    
    async def store_facts(self, text: str, speaker_id: str = 'default_user') -> int:
        """Store facts using SurrealDB with SpaCy fact extraction and knowledge relations"""
        from loguru import logger
        try:
            # Extract facts using SpaCy
            logger.debug(f"🔍 About to call extract_facts_from_text function: {extract_facts_from_text}")
            logger.debug(f"🔍 Module: {extract_facts_from_text.__module__}")
            facts = extract_facts_from_text(text)
            logger.debug(f"🔍 SurrealMemorySystem extracted {len(facts)} facts from: '{text[:50]}...'")
            logger.debug(f"🔍 Facts detail: {facts}")
            
            if not facts:
                logger.debug("🔍 No facts extracted, returning 0")
                return 0
            
            stored_count = 0
            
            # Store facts as knowledge relations
            for i, fact in enumerate(facts):
                logger.debug(f"🔄 Processing fact {i+1}: {fact}")
                
                if isinstance(fact, dict):
                    subject = fact.get('subject', speaker_id)
                    predicate = fact.get('predicate', 'mentioned')
                    obj = fact.get('value') or fact.get('object', '')
                    
                    logger.debug(f"   📋 Extracted: subject={subject}, predicate={predicate}, obj={obj}")
                    
                    if obj:  # Only store if we have an object
                        logger.debug(f"   💾 Storing knowledge relation...")
                        success = await self.connection_manager.store_knowledge_relation(
                            subject_name=subject,
                            predicate=predicate,
                            object_name=obj,
                            subject_type='user' if subject == speaker_id else 'concept',
                            object_type='concept'
                        )
                        logger.debug(f"   📊 Storage success: {success}")
                        if success:
                            stored_count += 1
                            logger.debug(f"   ✅ Stored count now: {stored_count}")
                    else:
                        logger.debug(f"   ⚠️ Skipping fact with empty object")
            
            # Legacy compatibility storage removed to prevent duplicates
            # Facts are already stored as knowledge relations above
            
            logger.debug(f"🧠 Final result: extracted and stored {stored_count} facts as knowledge relations")
            return stored_count
            
        except Exception as e:
            logger.error(f"Fact storage failed: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    async def update_session(self, speaker_id: str):
        """Update session metadata"""
        # Session management handled by SurrealDB connection manager
        pass

    # Simplified methods for compatibility
    async def get_recent(self, limit: int = 10, since: float = None, agent_id: str = None):
        """Get recent messages and knowledge from SurrealDB"""
        from loguru import logger
        results = []
        
        try:
            # Get recent messages (expand time window for more data)
            minutes = 60 if since is None else max(1, int((time.time() - since) / 60))
            messages = await self.connection_manager.get_recent_messages(
                minutes=minutes, 
                speaker_id=agent_id
            )
            
            # Convert messages to DTH-compatible format
            for msg in messages[:limit//2]:  # Use half the limit for messages
                results.append({
                    'content': msg.get('content', ''),
                    'role': msg.get('role', ''),
                    'speaker_id': msg.get('speaker_id', ''),
                    'ts': msg.get('timestamp', ''),
                    'metadata': msg.get('metadata', {})
                })
            
            logger.debug(f"get_recent: Found {len(messages)} messages")
            
        except Exception as e:
            logger.debug(f"Failed to get recent messages: {e}")
        
        try:
            # Also get recent knowledge relations
            knowledge = await self.connection_manager.search_knowledge_relations(
                query='',  # Empty query to get all
                limit=limit//2  # Use other half for knowledge
            )
            
            # Convert knowledge to DTH-compatible format
            for rel in knowledge:
                content = f"{rel.get('subject', '')} {rel.get('predicate', '')} {rel.get('object', '')}"
                # Use the subject as speaker_id for user facts (e.g., 'user' -> current user)
                subject = rel.get('subject', '')
                fact_speaker_id = subject if subject in ['user'] else 'system'
                
                results.append({
                    'content': content,
                    'role': 'knowledge',
                    'speaker_id': fact_speaker_id,
                    'ts': rel.get('created_at', ''),
                    'metadata': {
                        'type': 'knowledge',
                        'confidence': rel.get('confidence', 0.8),
                        'subject': rel.get('subject', ''),
                        'predicate': rel.get('predicate', ''),
                        'object': rel.get('object', '')
                    }
                })
            
            logger.debug(f"get_recent: Found {len(knowledge)} knowledge relations")
            
        except Exception as e:
            logger.debug(f"Failed to get knowledge relations: {e}")
        
        # Get conversation from current session if we have one
        if self.current_session_id:
            try:
                session_messages = await self.connection_manager.get_conversation(
                    session_id=self.current_session_id,
                    limit=limit
                )
                
                for msg in session_messages:
                    if msg not in messages:  # Avoid duplicates
                        results.append({
                            'content': msg.get('content', ''),
                            'role': msg.get('role', ''),
                            'speaker_id': msg.get('speaker_id', ''),
                            'ts': msg.get('timestamp', ''),
                            'metadata': msg.get('metadata', {})
                        })
                
                logger.debug(f"get_recent: Found {len(session_messages)} session messages")
                
            except Exception as e:
                logger.debug(f"Failed to get session messages: {e}")
        
        logger.info(f"📦 get_recent returning {len(results)} total items (messages + knowledge)")
        return results
    
    def apply_decay(self):
        """Placeholder for fact decay - handled by SurrealDB events"""
        pass
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        return {
            'system': 'SurrealDB',
            'connection_active': self.connection_manager.connected if self.connection_manager else False,
            'query_router_available': self.query_router is not None
        }
    
    async def close(self):
        """Clean shutdown of SurrealDB connection"""
        if self.connection_manager:
            await self.connection_manager.disconnect()


def create_surreal_message_store(speaker_id: str = 'default_user', 
                                 auto_create_session: bool = True):
    """
    Create a SurrealDB message store processor for the pipeline
    
    Args:
        speaker_id: Speaker identifier for messages
        auto_create_session: Whether to create a session automatically
        
    Returns:
        SurrealMessageStore processor instance
    """
    from processors.surreal_message_store import SurrealMessageStore
    return SurrealMessageStore(
        speaker_id=speaker_id,
        auto_create_session=auto_create_session
    )


# extract_facts_from_text is imported from hybrid_fact_extractor  
# No local definition needed - using the hybrid SpaCy+Gemma implementation
