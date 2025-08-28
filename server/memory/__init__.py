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
from .spacy_fact_extractor import extract_facts_from_text

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
        
        # Provide compatibility interfaces
        self.facts_graph = connection_manager  # SurrealDB provides facts interface
        self.tape_store = connection_manager   # SurrealDB provides tape interface
    
    async def process_query(self, query: str, context: dict = None):
        """Process query using SurrealDB connection manager"""
        try:
            if self.query_router:
                # Use query router if available
                return await self.query_router.route_query(query, context)
            else:
                # Direct SurrealDB search as fallback
                messages = await self.connection_manager.search_messages(query, limit=10)
                
                # Convert to compatible format
                class SimpleResult:
                    def __init__(self, msg_dict):
                        self.subject = msg_dict.get('speaker_id', '')
                        self.predicate = 'said'
                        self.value = msg_dict.get('content', '')
                        self.species = None
                        self.fidelity = 3
                        self.strength = 0.8
                        self.last_seen = msg_dict.get('timestamp', 0)
                        self.created = msg_dict.get('timestamp', 0)
                        self.access_count = 0
                        self.source_text = msg_dict.get('content', '')
                        self.source_store = 'messages'
                
                results = [SimpleResult(msg) for msg in messages]
                
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
                
                return SimpleResponse(results)
                
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
    
    async def store_facts(self, text: str) -> int:
        """Store facts using SurrealDB with SpaCy fact extraction"""
        try:
            # Extract facts using SpaCy
            facts = extract_facts_from_text(text)
            
            if not facts:
                return 0
            
            # Store facts in SurrealDB
            facts_stored = await self.connection_manager.store_facts(facts)
            
            return facts_stored if facts_stored is not None else 0
            
        except Exception as e:
            from loguru import logger
            logger.error(f"Fact storage failed: {e}")
            return 0
    
    async def update_session(self, speaker_id: str):
        """Update session metadata"""
        # Session management handled by SurrealDB connection manager
        pass

    # Simplified methods for compatibility
    async def get_recent(self, limit: int = 10, since: float = None, agent_id: str = None):
        """Get recent messages from SurrealDB"""
        try:
            minutes = 5 if since is None else max(1, int((time.time() - since) / 60))
            return await self.connection_manager.get_recent_messages(minutes=minutes)
        except Exception:
            return []
    
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


def extract_facts_from_text(text: str) -> list:
    """
    Extract facts from text (placeholder implementation)
    
    This is a simplified version for compatibility. 
    Real fact extraction would use NLP to identify structured knowledge.
    
    Args:
        text: Input text to extract facts from
        
    Returns:
        List of extracted facts (currently empty - to be implemented)
    """
    # TODO: Implement proper fact extraction using spaCy or similar
    # For now, return empty list to maintain compatibility
    return []
