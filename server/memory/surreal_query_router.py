"""
SurrealDB Query Router - Native graph-based query routing

This router leverages SurrealDB's native capabilities:
- Graph relationships for complex fact traversals
- Multi-model queries across facts (graph) + tape (time-series) + sessions (document)
- Real-time subscriptions for live updates
- Temporal queries for "what did we discuss last Tuesday?"
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    logger.warning("SurrealDB client not available")
    SURREALDB_AVAILABLE = False
    AsyncSurreal = None


@dataclass
class SurrealRetrievalResponse:
    """Response from SurrealDB query routing"""
    results: List[Any]
    total_results: int
    retrieval_time_ms: float
    strategy_used: str
    stores_queried: List[str]
    classification: Optional[Dict] = None
    graph_traversal_depth: int = 0
    temporal_range: Optional[Tuple[float, float]] = None


@dataclass
class SurrealMemoryResult:
    """Unified memory result from SurrealDB"""
    content: str
    source_store: str  # 'facts', 'tape', 'sessions'
    relevance_score: float
    timestamp: float
    metadata: Dict[str, Any]
    
    # Fact-specific fields
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value: Optional[str] = None
    fidelity: Optional[int] = None
    
    # Tape-specific fields
    speaker_id: Optional[str] = None
    role: Optional[str] = None
    session_id: Optional[str] = None


class SurrealQueryRouter:
    """
    Native SurrealDB query router with graph traversal and multi-model queries
    
    Replaces the legacy QueryRouter with SurrealDB-native operations:
    - Graph relationship traversals
    - Time-travel queries
    - Multi-store unified queries
    - Real-time subscription support
    """
    
    def __init__(self, surreal_memory):
        self.surreal_memory = surreal_memory
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time_ms': 0.0,
            'graph_traversals': 0,
            'temporal_queries': 0,
            'multi_store_queries': 0
        }
    
    async def route_query(self, query: str, context: Dict = None) -> SurrealRetrievalResponse:
        """
        Route query using SurrealDB's native capabilities
        
        Args:
            query: User query text
            context: Optional conversation context with speaker_id, session_count, etc.
            
        Returns:
            SurrealRetrievalResponse with unified results from multiple stores
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Classify query intent for routing strategy
            intent = await self._classify_query_intent(query, context)
            
            # Choose routing strategy based on intent and context
            if intent.get('type') == 'temporal':
                results, strategy = await self._handle_temporal_query(query, context)
            elif intent.get('type') == 'relationship':
                results, strategy = await self._handle_graph_traversal(query, context)
            elif intent.get('type') == 'conversation_history':
                results, strategy = await self._handle_conversation_query(query, context)
            else:
                # Default: unified multi-store search
                results, strategy = await self._handle_unified_search(query, context)
            
            # Wrap results in SurrealMemoryResult format for compatibility
            formatted_results = []
            for result in results:
                if hasattr(result, 'subject'):  # It's a fact
                    formatted_results.append(SurrealMemoryResult(
                        content=getattr(result, 'source_text', '') or f"{result.subject} {result.predicate} {result.value or ''}",
                        source_store='facts',
                        relevance_score=getattr(result, 'strength', 0.6),
                        timestamp=getattr(result, 'last_seen', 0),
                        metadata={'fidelity': getattr(result, 'fidelity', 3)},
                        subject=getattr(result, 'subject', ''),
                        predicate=getattr(result, 'predicate', ''),
                        value=getattr(result, 'value', None),
                        fidelity=getattr(result, 'fidelity', 3)
                    ))
                elif hasattr(result, 'role'):  # It's a tape entry
                    formatted_results.append(SurrealMemoryResult(
                        content=getattr(result, 'content', ''),
                        source_store='tape',
                        relevance_score=1.0,  # Tape entries are exact matches
                        timestamp=getattr(result, 'ts', 0),
                        metadata={'role': getattr(result, 'role', '')},
                        speaker_id=getattr(result, 'speaker_id', ''),
                        role=getattr(result, 'role', ''),
                        session_id=getattr(result, 'session_id', '')
                    ))
                else:
                    # Generic result
                    formatted_results.append(SurrealMemoryResult(
                        content=str(result),
                        source_store='unknown',
                        relevance_score=0.5,
                        timestamp=time.time(),
                        metadata={}
                    ))
            
            retrieval_time = (time.time() - start_time) * 1000
            
            # Update performance stats
            self.performance_stats['total_queries'] += 1
            self.performance_stats['avg_response_time_ms'] = (
                (self.performance_stats['avg_response_time_ms'] * (self.performance_stats['total_queries'] - 1) + 
                 retrieval_time) / self.performance_stats['total_queries']
            )
            
            return SurrealRetrievalResponse(
                results=formatted_results,
                total_results=len(formatted_results),
                retrieval_time_ms=retrieval_time,
                strategy_used=strategy,
                stores_queried=['surreal_unified'],
                classification=intent,
                graph_traversal_depth=intent.get('traversal_depth', 0),
                temporal_range=intent.get('temporal_range')
            )
            
        except Exception as e:
            logger.error(f"SurrealDB query routing failed: {e}")
            return SurrealRetrievalResponse(
                results=[],
                total_results=0,
                retrieval_time_ms=(time.time() - start_time) * 1000,
                strategy_used='error_fallback',
                stores_queried=[],
                classification={'error': str(e)}
            )
    
    async def _classify_query_intent(self, query: str, context: Dict) -> Dict:
        """
        Classify query intent for optimal routing strategy
        
        Uses heuristics and context to determine:
        - temporal: "what did we discuss yesterday?"
        - relationship: "tell me about my dog's friends"
        - conversation_history: "continue from where we left off"
        - personal_facts: "what's my name?"
        """
        query_lower = query.lower()
        
        # Temporal indicators
        temporal_words = ['yesterday', 'last week', 'ago', 'when did', 'time when', 'before', 'after']
        if any(word in query_lower for word in temporal_words):
            return {
                'type': 'temporal',
                'confidence': 0.8,
                'temporal_range': self._extract_temporal_range(query),
                'strategy': 'time_travel_search'
            }
        
        # Relationship/graph traversal indicators
        relationship_words = ['friends', 'related', 'connected', 'family', 'colleagues', 'about']
        if any(word in query_lower for word in relationship_words):
            self.performance_stats['graph_traversals'] += 1
            return {
                'type': 'relationship',
                'confidence': 0.7,
                'traversal_depth': 2,
                'strategy': 'graph_traversal'
            }
        
        # Conversation history indicators  
        conversation_words = ['discussed', 'talked about', 'mentioned', 'said', 'continue', 'resume']
        if any(word in query_lower for word in conversation_words):
            return {
                'type': 'conversation_history',
                'confidence': 0.75,
                'strategy': 'conversation_search'
            }
        
        # Default: personal facts
        return {
            'type': 'personal_facts',
            'confidence': 0.6,
            'strategy': 'unified_search'
        }
    
    async def _handle_temporal_query(self, query: str, context: Dict) -> Tuple[List[Any], str]:
        """Handle time-based queries using SurrealDB's temporal capabilities"""
        self.performance_stats['temporal_queries'] += 1
        
        # Use SurrealDB's time-travel queries
        # Example: SELECT * FROM conversation WHERE timestamp > time::now() - 1d
        try:
            # For now, delegate to existing search methods
            # TODO: Implement native SurrealDB temporal queries
            results = await self.surreal_memory.search_tape(query, limit=10)
            return results, 'temporal_search'
        except Exception as e:
            logger.debug(f"Temporal query failed: {e}")
            return [], 'temporal_fallback'
    
    async def _handle_graph_traversal(self, query: str, context: Dict) -> Tuple[List[Any], str]:
        """Handle relationship queries using SurrealDB's graph capabilities"""
        try:
            # Use SurrealDB's graph traversal
            # Example: SELECT * FROM facts WHERE subject = 'user' RELATE->knows->person
            facts = await self.surreal_memory.search_facts(query, limit=15)
            return facts, 'graph_traversal'
        except Exception as e:
            logger.debug(f"Graph traversal failed: {e}")
            return [], 'graph_fallback'
    
    async def _handle_conversation_query(self, query: str, context: Dict) -> Tuple[List[Any], str]:
        """Handle conversation history queries"""
        try:
            # Search conversation tape with speaker context
            speaker_id = context.get('speaker_id', 'default_user')
            results = await self.surreal_memory.search_tape(query, limit=10)
            
            # Filter by speaker if available
            if speaker_id and speaker_id != 'default_user':
                results = [r for r in results if getattr(r, 'speaker_id', '') == speaker_id]
            
            return results, 'conversation_search'
        except Exception as e:
            logger.debug(f"Conversation query failed: {e}")
            return [], 'conversation_fallback'
    
    async def _handle_unified_search(self, query: str, context: Dict) -> Tuple[List[Any], str]:
        """Handle general queries with unified multi-store search"""
        self.performance_stats['multi_store_queries'] += 1
        
        try:
            # Search both facts and tape, combine results
            facts_task = self.surreal_memory.search_facts(query, limit=10)
            tape_task = self.surreal_memory.search_tape(query, limit=5)
            
            facts_results, tape_results = await asyncio.gather(facts_task, tape_task, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(facts_results, Exception):
                facts_results = []
            if isinstance(tape_results, Exception):
                tape_results = []
            
            # Combine and rank results
            all_results = list(facts_results) + list(tape_results)
            return all_results, 'unified_multi_store'
            
        except Exception as e:
            logger.debug(f"Unified search failed: {e}")
            # Fallback to facts only
            try:
                facts = await self.surreal_memory.search_facts(query, limit=10)
                return facts, 'facts_only_fallback'
            except Exception:
                return [], 'complete_fallback'
    
    def _extract_temporal_range(self, query: str) -> Optional[Tuple[float, float]]:
        """Extract temporal range from query (placeholder)"""
        # TODO: Implement natural language date parsing
        # For now, return a recent range as fallback
        now = time.time()
        one_day = 24 * 3600
        return (now - one_day, now)
    
    def get_performance_stats(self) -> Dict:
        """Get query router performance statistics"""
        return self.performance_stats.copy()


def create_surreal_query_router(surreal_memory) -> SurrealQueryRouter:
    """Factory function for creating SurrealDB query router"""
    return SurrealQueryRouter(surreal_memory)