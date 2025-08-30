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
        """Handle time-based queries using SurrealDB's unified temporal capabilities"""
        self.performance_stats['temporal_queries'] += 1
        
        try:
            # NEW: Use unified temporal queries with knowledge system
            temporal_range = self._extract_temporal_range(query)
            
            if temporal_range:
                start_time, end_time = temporal_range
                # Convert to datetime objects for SurrealDB
                from datetime import datetime, timezone
                start_dt = datetime.fromtimestamp(start_time, timezone.utc)
                end_dt = datetime.fromtimestamp(end_time, timezone.utc)
                
                # Use unified temporal search function
                try:
                    temporal_knowledge = await self.surreal_memory.db.query(
                        "SELECT * FROM fn::get_memories_by_time($from, $to);",
                        {'from': start_dt, 'to': end_dt}
                    )
                    
                    if temporal_knowledge and len(temporal_knowledge) > 0:
                        results = temporal_knowledge[0].get('result', [])
                        if results:
                            return results, 'unified_temporal_search'
                except Exception as e:
                    logger.debug(f"Unified temporal search failed: {e}")
            
            # Fallback to tape search for conversation memory
            results = await self.surreal_memory.search_tape(query, limit=10)
            return results, 'tape_temporal_search'
            
        except Exception as e:
            logger.debug(f"Temporal query failed: {e}")
            return [], 'temporal_fallback'
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract key search terms from natural language queries"""
        import re
        
        # Common question patterns to remove
        query_clean = re.sub(r'\b(do you know|can you remember|what is|what was|tell me about|who is)\b', '', query, flags=re.IGNORECASE)
        query_clean = re.sub(r'\b(the name of|about)\b', '', query_clean, flags=re.IGNORECASE)
        query_clean = re.sub(r'[?!.,;]', '', query_clean)
        
        # Extract important nouns and entities
        words = query_clean.lower().split()
        
        # Key terms that are important for searching
        key_terms = []
        important_words = {'dog', 'pet', 'cat', 'animal', 'name', 'location', 'job', 'work', 'family', 'friend', 
                          'hobby', 'like', 'love', 'favorite', 'age', 'birthday', 'address', 'phone', 'email',
                          'spouse', 'partner', 'child', 'parent', 'sibling', 'car', 'house', 'apartment'}
        
        for word in words:
            # Keep important words
            if word in important_words:
                key_terms.append(word)
            # Keep possessive indicators
            elif word in {'my', 'our', 'their', 'his', 'her'}:
                key_terms.append(word)
            # Keep words longer than 3 characters that aren't common stop words
            elif len(word) > 3 and word not in {'that', 'this', 'what', 'where', 'when', 'how', 'why'}:
                key_terms.append(word)
        
        # If no key terms found, fall back to all non-stop words
        if not key_terms:
            stop_words = {'i', 'me', 'my', 'you', 'your', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms[:5]  # Limit to top 5 terms
    
    async def _handle_graph_traversal(self, query: str, context: Dict) -> Tuple[List[Any], str]:
        """Handle relationship queries using SurrealDB's unified entity-knowledge graph"""
        try:
            # NEW: Extract entity name from query for focused search
            entity_name = self._extract_entity_from_query(query, context)
            
            if entity_name:
                # Get all knowledge about the specific entity
                entity_knowledge = await self.surreal_memory.get_entity_knowledge(entity_name, limit=20)
                if entity_knowledge:
                    return entity_knowledge, 'entity_focused_graph'
            
            # Fallback to general knowledge search with graph capabilities
            search_terms = self._extract_search_terms(query)
            search_query = ' '.join(search_terms) if search_terms else query
            logger.debug(f"🔍 Extracted search terms: {search_terms} -> '{search_query}'")
            knowledge_results = await self.surreal_memory.search_knowledge_relations(search_query, limit=15)
            return knowledge_results, 'unified_graph_traversal'
            
        except Exception as e:
            logger.debug(f"Graph traversal failed: {e}")
            # Fallback to legacy facts
            try:
                facts = await self.surreal_memory.search_facts(query, limit=15)
                return facts, 'legacy_facts_fallback'
            except Exception:
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
            # NEW: Use unified knowledge system + legacy compatibility
            search_terms = self._extract_search_terms(query)
            search_query = ' '.join(search_terms) if search_terms else query
            logger.debug(f"🔍 Multi-store search terms: {search_terms} -> '{search_query}'")
            
            knowledge_task = self.surreal_memory.search_knowledge_relations(search_query, limit=8)
            facts_task = self.surreal_memory.search_facts(search_query, limit=7)
            tape_task = self.surreal_memory.search_tape(query, limit=5)  # Keep original query for tape search
            
            knowledge_results, facts_results, tape_results = await asyncio.gather(
                knowledge_task, facts_task, tape_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(knowledge_results, Exception):
                knowledge_results = []
            if isinstance(facts_results, Exception):
                facts_results = []
            if isinstance(tape_results, Exception):
                tape_results = []
            
            # Combine and prioritize: unified knowledge > legacy facts > tape
            all_results = list(knowledge_results) + list(facts_results) + list(tape_results)
            return all_results, 'unified_hybrid_search'
            
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
    
    def _extract_entity_from_query(self, query: str, context: Dict) -> Optional[str]:
        """
        Extract entity name from query for focused searches
        
        Args:
            query: User query text
            context: Query context with speaker info
            
        Returns:
            Entity name if detected, None otherwise
        """
        query_lower = query.lower()
        
        # Check for self-references
        self_indicators = ['my', 'i', 'me', 'myself']
        if any(word in query_lower for word in self_indicators):
            speaker_id = context.get('speaker_id', 'user')
            return speaker_id
        
        # Check for direct entity references (basic pattern matching)
        # This would be enhanced with NLP in production
        entity_patterns = [
            'about',
            'tell me about',
            'what do you know about',
            'information on',
            'facts about'
        ]
        
        for pattern in entity_patterns:
            if pattern in query_lower:
                # Extract the word(s) after the pattern
                pattern_index = query_lower.find(pattern)
                after_pattern = query[pattern_index + len(pattern):].strip()
                if after_pattern:
                    # Take first word/phrase as potential entity name
                    entity_candidate = after_pattern.split()[0] if after_pattern.split() else None
                    if entity_candidate and len(entity_candidate) > 1:
                        return entity_candidate.strip('?.,!').lower()
        
        return None
    
    def get_performance_stats(self) -> Dict:
        """Get query router performance statistics"""
        return self.performance_stats.copy()


def create_surreal_query_router(surreal_memory) -> SurrealQueryRouter:
    """Factory function for creating SurrealDB query router"""
    return SurrealQueryRouter(surreal_memory)