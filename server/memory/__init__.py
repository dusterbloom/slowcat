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
    from loguru import logger
    
    # Default to SurrealDB memory (no flag required). Legacy flags still honored.
    val = (os.getenv('USE_SURREALDB', '').strip().lower() or os.getenv('USE_SLOWCAT_MEMORY', '').strip().lower())
    # If explicitly disabled (e.g., 'false', '0', 'no'), use SQLite; otherwise prefer SurrealDB.
    use_surreal = not (val in ('false', '0', 'no'))
    if os.getenv('USE_SURREALDB') is not None:
        logger.info("🛈 USE_SURREALDB is deprecated — SurrealDB is the default now.")
    if use_surreal:
        try:
            from .surreal_memory import create_surreal_memory_system
            logger.info("🚀 Using SurrealDB memory system")
            
            # Create SurrealDB unified memory system
            surreal_memory = create_surreal_memory_system()
            
            # SurrealDB provides both facts and tape functionality
            # Return adapter that maintains compatibility
            return SurrealMemorySystemAdapter(surreal_memory)
            
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
        self.query_router = None           # TODO: Create SurrealDB query router
    
    async def process_query(self, query: str, context: dict = None):
        """Process query using SurrealDB capabilities"""
        # For now, simple fact search - will enhance with proper routing
        raw_results = await self.surreal_memory.search_facts(query)
        
        # Wrap results to include source_store for compatibility with SCM filters
        class SimpleResult:
            def __init__(self, fact_obj):
                # Expect a SurrealFact-like object with attributes
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
        
        # Create a simple object with .results attribute for compatibility
        class SimpleResponse:
            def __init__(self, results, classification=None):
                self.results = results
                self.total_results = len(results)
                self.retrieval_time_ms = 0
                self.strategy_used = 'direct'
                self.stores_queried = ['surreal_facts']
                self.classification = classification or SimpleClassification()
        
        class SimpleClassification:
            def __init__(self):
                self.intent = SimpleIntent()
                self.confidence = 0.8
        
        class SimpleIntent:
            def __init__(self):
                self.name = 'PERSONAL_FACTS'
        
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
    async def knn_tape(self, query: str, limit: int = 20, scan: int = 200):
        """Expose SurrealDB-side KNN to DynamicTapeHead."""
        try:
            return await self.surreal_memory.knn_tape(query, limit=limit, scan=scan)
        except Exception:
            return []

    async def search_tape(self, query: str, limit: int = 10):
        """Expose keyword search over tape to DynamicTapeHead."""
        try:
            return await self.surreal_memory.search_tape(query, limit=limit)
        except Exception:
            return []

    async def get_recent(self, limit: int = 10, since: float | None = None):
        """Expose recent tape retrieval for candidates."""
        try:
            return await self.surreal_memory.get_recent(limit=limit, since=since)
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
