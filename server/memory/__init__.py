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
    
    Args:
        facts_db_path: Path to facts SQLite database
        tape_store: Optional conversation tape store
        embedding_store: Optional semantic search store
        
    Returns:
        SmartMemorySystem instance
    """
    from pathlib import Path
    
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
    
    def store_facts(self, text: str) -> int:
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
            self.facts_graph.reinforce_or_insert(fact)
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
    
    def close(self):
        """Clean shutdown"""
        self.facts_graph.close()
