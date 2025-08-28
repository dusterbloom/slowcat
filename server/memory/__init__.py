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

from .facts_graph import FactsGraph, extract_facts_from_text
from .tape_store import TapeStore
from .query_classifier import (
    HybridQueryClassifier, QueryIntent, ClassificationResult, 
    create_query_classifier
)
from .query_router import (
    QueryRouter, RetrievalResponse, MemoryResult, 
    create_query_router
)
from loguru import logger

__all__ = [
    # Core components
    'FactsGraph',
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
    
    # Utilities
    'extract_facts_from_text',
]


def create_smart_memory_system(facts_db_path: str = "data/facts.db",
                              tape_store=None,
                              embedding_store=None):
    """
    Create complete smart memory system with all components
    
    Environment Variables:
        USE_SURREALDB: Set to 'true' to use SurrealDB instead of SQLite
        SURREALDB_URL: SurrealDB connection URL (default: ws://localhost:8000/rpc)
        SURREALDB_NAMESPACE: Database namespace (default: slowcat)
        SURREALDB_DATABASE: Database name (default: memory)
    
    Args:
        facts_db_path: Path to facts SQLite database (ignored if using SurrealDB)
        tape_store: Optional conversation tape store
        embedding_store: Optional semantic search store
        
    Returns:
        SmartMemorySystem instance
    """
    import os
    from pathlib import Path
    
    # Default to SurrealDB memory (no flag required). Legacy flags still honored.
    val = (os.getenv('USE_SURREALDB', '').strip().lower() or os.getenv('USE_SLOWCAT_MEMORY', '').strip().lower())
    # If explicitly disabled (e.g., 'false', '0', 'no'), use SQLite; otherwise prefer SurrealDB.
    use_surreal = not (val in ('false', '0', 'no'))
    if os.getenv('USE_SURREALDB') is not None:
        logger.info("🛈 USE_SURREALDB is deprecated — SurrealDB is the default now.")
    if use_surreal:
        try:
            # Prefer graph-native memory when schema mode is 'graph' (default)
            schema_mode = os.getenv('SC_SCHEMA_MODE', 'graph').strip().lower()
            if schema_mode == 'graph':
                from .graph_surreal_memory import create_graph_surreal_memory
                logger.info("🚀 Using SurrealDB Graph memory system (message/session)")
                surreal_memory = create_graph_surreal_memory()
            else:
                from .surreal_memory import create_surreal_memory_system
                logger.info("🚀 Using SurrealDB legacy memory system (compat mode)")
                surreal_memory = create_surreal_memory_system()

            # SurrealDB provides both facts and conversation functionality
            # Return the graph-native memory directly (no adapter indirection)
            return surreal_memory

        except ImportError as e:
            logger.error(f"SurrealDB not available: {e}")
            logger.info("📦 Falling back to SQLite memory system")
            # Fall through to SQLite implementation
        except Exception as e:
            logger.error(f"SurrealDB initialization failed: {e}")
            logger.info("📦 Falling back to SQLite memory system")
            # Fall through to SQLite implementation
    
    # Original SQLite implementation
    logger.info("📦 Using SQLite memory system")
    
    # Ensure data directory exists
    Path(facts_db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create components
    facts_graph = FactsGraph(facts_db_path)

    # Initialize tape store if not provided
    if tape_store is None:
        tape_store = TapeStore(str(Path(facts_db_path).with_name('tape.db')))

    query_router = create_query_router(
        facts_graph=facts_graph,
        tape_store=tape_store,
        embedding_store=embedding_store
    )
    
    return SmartMemorySystem(
        facts_graph=facts_graph,
        query_router=query_router,
        tape_store=tape_store
    )


class SurrealMemorySystemAdapter:
    """
    Adapter to make SurrealDB memory compatible with SmartMemorySystem interface
    """
    
    def __init__(self, surreal_memory):
        self.surreal_memory = surreal_memory
        self.facts_graph = surreal_memory  # SurrealDB provides facts interface
        self.tape_store = surreal_memory   # SurrealDB provides tape interface
        
        # Prefer the standard QueryRouter wired to SurrealDB stores for consistency
        try:
            from .query_router import create_query_router
            self.query_router = create_query_router(
                facts_graph=surreal_memory,
                tape_store=surreal_memory,
                embedding_store=None
            )
        except Exception as e:
            logger.warning(f"Standard QueryRouter not available for SurrealDB: {e}")
            self.query_router = None
    
    async def process_query(self, query: str, context: dict = None):
        """Process query using SurrealDB native query router"""
        if self.query_router:
            # Use native SurrealDB query router for intelligent routing
            response = await self.query_router.route_query(query, context)
            
            # Convert MemoryResult back to SimpleResult for compatibility
            class SimpleResult:
                def __init__(self, memory_result):
                    # Map MemoryResult attributes to legacy format
                    self.content = memory_result.content
                    self.source_store = memory_result.source_store
                    self.relevance_score = memory_result.relevance_score
                    self.timestamp = memory_result.timestamp
                    
                    # Legacy format compatibility - extract from metadata if available
                    metadata = getattr(memory_result, 'metadata', {})
                    self.subject = metadata.get('subject', '')
                    self.predicate = metadata.get('predicate', '')
                    self.value = metadata.get('value', memory_result.content)
                    self.species = metadata.get('species')
                    self.fidelity = metadata.get('fidelity', 3)
                    self.strength = memory_result.relevance_score
                    self.last_seen = memory_result.timestamp
                    self.created = memory_result.timestamp
                    self.access_count = metadata.get('access_count', 0)
                    self.source_text = memory_result.content
            
            results = [SimpleResult(r) for r in response.results]
            
            # Return enhanced response with SurrealDB routing metadata
            class SurrealResponse:
                def __init__(self, results, response):
                    self.results = results
                    self.total_results = response.total_results
                    self.retrieval_time_ms = response.retrieval_time_ms
                    self.strategy_used = response.strategy_used
                    self.stores_queried = response.stores_queried
                    self.classification = response.classification or self._default_classification()
                
                def _default_classification(self):
                    class SimpleClassification:
                        def __init__(self):
                            self.intent = SimpleIntent()
                            self.confidence = 0.8
                    
                    class SimpleIntent:
                        def __init__(self):
                            self.name = 'SURREAL_UNIFIED'
                    
                    return SimpleClassification()
            
            return SurrealResponse(results, response)
        
        else:
            # Fallback to simple fact search if query router not available
            raw_results = await self.surreal_memory.search_facts(query)
            
            class SimpleResult:
                def __init__(self, fact_obj):
                    self.subject = getattr(fact_obj, 'subject', '')
                    self.predicate = getattr(fact_obj, 'predicate', '')
                    self.value = getattr(fact_obj, 'value', None)
                    self.species = getattr(fact_obj, 'species', None)
                    self.fidelity = getattr(fact_obj, 'fidelity', 3)
                    self.strength = getattr(fact_obj, 'strength', 0.6)
                    self.last_seen = getattr(fact_obj, 'last_seen', 0)
                    self.created = getattr(fact_obj, 'created', 0)
                    self.access_count = getattr(fact_obj, 'access_count', 0)
                    self.source_text = getattr(fact_obj, 'source_text', '')
                    self.source_store = 'facts'
            
            results = [SimpleResult(f) for f in raw_results]
            
            class SimpleResponse:
                def __init__(self, results):
                    self.results = results
                    self.total_results = len(results)
                    self.retrieval_time_ms = 0
                    self.strategy_used = 'fallback_direct'
                    self.stores_queried = ['surreal_facts']
                    self.classification = self._default_classification()
                
                def _default_classification(self):
                    class SimpleClassification:
                        def __init__(self):
                            self.intent = SimpleIntent()
                            self.confidence = 0.6
                    
                    class SimpleIntent:
                        def __init__(self):
                            self.name = 'PERSONAL_FACTS'
                    
                    return SimpleClassification()
            
            return SimpleResponse(results)
    
    async def store_facts(self, text: str) -> int:
        """Extract and store facts from text using SurrealDB"""
        from .facts_graph import extract_facts_from_text
        facts = extract_facts_from_text(text)
        stored_count = 0
        
        for fact in facts:
            await self.surreal_memory.reinforce_or_insert(fact)
            stored_count += 1
            
        return stored_count
    
    async def update_session(self, speaker_id: str):
        """Update session metadata (async for compatibility with pipeline)."""
        try:
            await self.surreal_memory.update_session(speaker_id)
        except Exception:
            pass

    # --- Pass-throughs for DTH / retrieval helpers ---
    async def knn_tape(self, query: str, limit: int = 20, scan: int = 200, speaker_id: str | None = None, agent_id: str | None = None):
        """Expose SurrealDB-side KNN to DynamicTapeHead."""
        try:
            return await self.surreal_memory.knn_tape(query, limit=limit, scan=scan, speaker_id=speaker_id, agent_id=agent_id)
        except Exception:
            return []

    async def search_tape(self, query: str, limit: int = 10, agent_id: str | None = None):
        """Expose keyword search over tape to DynamicTapeHead."""
        try:
            return await self.surreal_memory.search_tape(query, limit=limit, agent_id=agent_id)
        except Exception:
            return []

    async def get_recent(self, limit: int = 10, since: float | None = None, agent_id: str | None = None):
        """Expose recent tape retrieval for candidates."""
        try:
            return await self.surreal_memory.get_recent(limit=limit, since=since, agent_id=agent_id)
        except Exception:
            return []

    # --- Private thoughts pass-throughs ---
    async def add_thought(self, agent_id: str, thought_type: str, content: str, links: list[str] | None = None, visibility: str = 'private'):
        try:
            return await self.surreal_memory.add_thought(agent_id=agent_id, thought_type=thought_type, content=content, links=links, visibility=visibility)
        except Exception:
            return None

    async def get_recent_thoughts(self, agent_id: str, limit: int = 20):
        try:
            return await self.surreal_memory.get_recent_thoughts(agent_id=agent_id, limit=limit)
        except Exception:
            return []

    async def search_thoughts(self, agent_id: str, query: str, limit: int = 20):
        try:
            return await self.surreal_memory.search_thoughts(agent_id=agent_id, query=query, limit=limit)
        except Exception:
            return []

    # Emergent events
    async def add_emergent_event(self, agent_id: str, kind: str, content_snippet: str,
                                 meta: dict | None = None, session_id: str | None = None,
                                 user_id: str | None = None, confidence: float | None = None):
        try:
            return await self.surreal_memory.add_emergent_event(
                agent_id=agent_id,
                kind=kind,
                content_snippet=content_snippet,
                meta=meta,
                session_id=session_id,
                user_id=user_id,
                confidence=confidence,
            )
        except Exception:
            return None

    # Session/tape helpers for daemon
    async def list_sessions(self):
        try:
            return await self.surreal_memory.list_sessions()
        except Exception:
            return []

    async def get_recent_for_speaker(self, speaker_id: str, limit: int = 20):
        try:
            return await self.surreal_memory.get_recent_for_speaker(speaker_id, limit=limit)
        except Exception:
            return []
    
    def apply_decay(self):
        """Apply natural decay to facts using SurrealDB"""
        import asyncio
        
        async def decay_async():
            await self.surreal_memory.apply_decay()
        
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(decay_async())
        except RuntimeError:
            asyncio.run(decay_async())
    
    def get_stats(self) -> dict:
        """Get comprehensive system statistics from SurrealDB"""
        import asyncio
        
        async def stats_async():
            return await self.surreal_memory.get_stats()
        
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(stats_async())
        except RuntimeError:
            return asyncio.run(stats_async())
    
    async def close(self):
        """Clean shutdown of SurrealDB connection"""
        await self.surreal_memory.close()


class SmartMemorySystem:
    """
    Complete smart memory system combining all components
    """
    
    def __init__(self, facts_graph: FactsGraph, query_router: QueryRouter, tape_store: TapeStore | None = None):
        self.facts_graph = facts_graph
        self.query_router = query_router
        self.tape_store = tape_store
        
    async def process_query(self, query: str, context: dict = None) -> RetrievalResponse:
        """
        Process a user query and return relevant memories
        
        Args:
            query: User query text
            context: Optional conversation context
            
        Returns:
            RetrievalResponse with results and metadata
        """
        return await self.query_router.route_query(query, context)
    
    async def store_facts(self, text: str) -> int:
        """
        Extract and store facts from text
        
        Args:
            text: Text to extract facts from
            
        Returns:
            Number of facts extracted and stored
        """
        facts = extract_facts_from_text(text)
        stored_count = 0
        
        for fact in facts:
            # SQLite operations are sync, but await for consistency with SurrealDB
            result = self.facts_graph.reinforce_or_insert(fact)
            if hasattr(result, '__await__'):
                await result
            stored_count += 1
            
        return stored_count
    
    def update_session(self, speaker_id: str):
        """Update session metadata"""
        self.facts_graph.update_session(speaker_id)
    
    def apply_decay(self):
        """Apply natural decay to facts"""
        self.facts_graph.decay_facts()
    
    def get_stats(self) -> dict:
        """Get comprehensive system statistics"""
        return {
            'facts': self.facts_graph.get_stats(),
            'router': self.query_router.get_performance_stats(),
            'tape_entries': None
        }
    
    async def close(self):
        """Clean shutdown"""
        result = self.facts_graph.close()
        if hasattr(result, '__await__'):
            await result
