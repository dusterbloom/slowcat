"""
Query Router - Multi-store memory retrieval with intelligent routing

Routes queries to appropriate memory stores based on classified intent:
- Facts Graph: Personal facts (structured, fast)
- Tape Machine: Verbatim conversation history 
- Embedding Store: Semantic/episodic memory
- Hybrid Search: When intent unclear

Features:
- Confidence-based routing strategies
- Fallback chains for reliability  
- Performance monitoring
- Language-agnostic operation
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from memory.query_classifier import (
    HybridQueryClassifier, QueryIntent, ClassificationResult, create_query_classifier
)
from memory.facts_graph import FactsGraph


class RoutingStrategy(Enum):
    """Different routing approaches based on confidence"""
    DIRECT = "direct"                    # High confidence: route to single store
    PRIMARY_WITH_FALLBACK = "fallback"   # Medium confidence: try primary, then fallback
    HYBRID = "hybrid"                    # Low confidence: search all stores
    BYPASS = "bypass"                    # Skip memory, go direct to LLM


@dataclass
class RetrievalPlan:
    """Plan for retrieving information from memory stores"""
    strategy: RoutingStrategy
    primary_store: str
    secondary_stores: List[str]
    max_results: int
    time_filter: Optional[Tuple[float, float]] = None  # (start_time, end_time)
    entity_filter: List[str] = None
    confidence_threshold: float = 0.0
    
    def __post_init__(self):
        if self.entity_filter is None:
            self.entity_filter = []


@dataclass
class MemoryResult:
    """Single memory retrieval result"""
    content: str
    source_store: str
    relevance_score: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class RetrievalResponse:
    """Complete response from memory retrieval"""
    results: List[MemoryResult]
    total_results: int
    retrieval_time_ms: float
    strategy_used: RoutingStrategy
    stores_queried: List[str]
    classification: ClassificationResult


class MemoryStoreInterface:
    """Base interface for memory stores"""
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> List[MemoryResult]:
        """Search the store for relevant results"""
        raise NotImplementedError
    
    async def get_recent(self, limit: int = 10, since: float = None) -> List[MemoryResult]:
        """Get recent entries"""
        raise NotImplementedError
    
    def get_store_name(self) -> str:
        """Get human-readable store name"""
        raise NotImplementedError


class FactsStoreAdapter(MemoryStoreInterface):
    """Adapter for Facts Graph"""
    
    def __init__(self, facts_graph: FactsGraph):
        self.facts_graph = facts_graph
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> List[MemoryResult]:
        """Search facts by content"""
        try:
            facts = self.facts_graph.search_facts(query, limit=limit)
            # Simple semantic nudge: map location-style queries to user.location
            qlow = (query or '').lower()
            if (not facts) and any(k in qlow for k in ["where", "location", "located", "live", "from"]):
                facts = self.facts_graph.get_facts(subject='user', predicate='location', min_fidelity=1, limit=limit)
            results = []
            
            for fact in facts:
                # Format fact as readable content
                if fact.value:
                    if fact.species:
                        content = f"{fact.subject}'s {fact.predicate} is {fact.value} ({fact.species})"
                    else:
                        content = f"{fact.subject}'s {fact.predicate} is {fact.value}"
                else:
                    # S1 level - only relationship
                    content = f"{fact.subject} has {fact.predicate}"
                
                # Calculate relevance score
                relevance = (fact.fidelity / 4.0) * fact.strength
                
                result = MemoryResult(
                    content=content,
                    source_store="facts",
                    relevance_score=relevance,
                    timestamp=fact.last_seen,
                    metadata={
                        'fidelity': fact.fidelity,
                        'strength': fact.strength,
                        'subject': fact.subject,
                        'predicate': fact.predicate,
                        'value': fact.value,
                        'species': fact.species,
                        'access_count': fact.access_count
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Facts search failed: {e}")
            return []
    
    async def get_recent(self, limit: int = 10, since: float = None) -> List[MemoryResult]:
        """Get recently accessed facts"""
        try:
            facts = self.facts_graph.get_facts(limit=limit)
            # Filter by time if specified
            if since:
                facts = [f for f in facts if f.last_seen >= since]
            
            return await self._facts_to_results(facts)
        except Exception as e:
            logger.error(f"Recent facts retrieval failed: {e}")
            return []
    
    def get_store_name(self) -> str:
        return "Facts Graph"


class TapeStoreAdapter(MemoryStoreInterface):
    """Adapter for verbatim conversation storage"""
    
    def __init__(self, tape_store=None):
        self.tape_store = tape_store
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> List[MemoryResult]:
        if not self.tape_store:
            return []
        entries = self.tape_store.search(query, limit=limit)
        results: List[MemoryResult] = []
        for e in entries:
            results.append(MemoryResult(
                content=f"[{e.role}] {e.content}",
                source_store='tape',
                relevance_score=0.5,  # simple default; could be BM25 score
                timestamp=e.ts,
                metadata={'speaker_id': e.speaker_id, 'role': e.role}
            ))
        return results
    
    async def get_recent(self, limit: int = 10, since: float = None) -> List[MemoryResult]:
        if not self.tape_store:
            return []
        entries = self.tape_store.get_recent(limit=limit, since=since)
        results: List[MemoryResult] = []
        for e in entries:
            results.append(MemoryResult(
                content=f"[{e.role}] {e.content}",
                source_store='tape',
                relevance_score=0.3,
                timestamp=e.ts,
                metadata={'speaker_id': e.speaker_id, 'role': e.role}
            ))
        return results
    
    def get_store_name(self) -> str:
        return "Conversation Tape"


class EmbeddingStoreAdapter(MemoryStoreInterface):
    """Adapter for semantic/embedding-based search (placeholder)"""
    
    def __init__(self, embedding_store=None):
        self.embedding_store = embedding_store
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> List[MemoryResult]:
        """Semantic search using embeddings"""
        # Placeholder - will integrate with FAISS or similar
        return []
    
    async def get_recent(self, limit: int = 10, since: float = None) -> List[MemoryResult]:
        return []
    
    def get_store_name(self) -> str:
        return "Semantic Search"


class QueryRouter:
    """
    Intelligent router that directs queries to appropriate memory stores
    """
    
    def __init__(self, 
                 facts_graph: Optional[FactsGraph] = None,
                 tape_store=None,
                 embedding_store=None):
        
        self.classifier = create_query_classifier()
        
        # Memory store adapters
        self.stores = {}
        if facts_graph:
            self.stores['facts'] = FactsStoreAdapter(facts_graph)
        if tape_store:
            self.stores['tape'] = TapeStoreAdapter(tape_store)
        if embedding_store:
            self.stores['embeddings'] = EmbeddingStoreAdapter(embedding_store)
        
        # Confidence thresholds for routing decisions
        self.thresholds = {
            'high_confidence': 0.8,     # Direct routing
            'medium_confidence': 0.6,   # Primary + fallback
            'low_confidence': 0.4       # Hybrid search
        }
        
        # Performance tracking
        self.total_queries = 0
        self.routing_stats = {strategy: 0 for strategy in RoutingStrategy}
        self.avg_response_time_ms = 0
        
        logger.info(f"🧭 Query Router initialized with {len(self.stores)} stores: "
                   f"{list(self.stores.keys())}")
    
    async def route_query(self, 
                         query: str, 
                         context: Dict = None,
                         max_results: int = 10) -> RetrievalResponse:
        """
        Route query to appropriate memory stores
        
        Args:
            query: User query text
            context: Optional conversation context
            max_results: Maximum results to return
            
        Returns:
            RetrievalResponse with results from appropriate stores
        """
        start_time = time.time()
        self.total_queries += 1
        
        # 1. Classify the query
        classification = await self.classifier.classify(query, context)
        
        logger.debug(f"🎯 Query classified: {classification.intent.value} "
                    f"({classification.confidence:.2f}) - '{query[:50]}...'")
        
        # 2. Create retrieval plan
        plan = self._create_retrieval_plan(classification, max_results)
        
        # 3. Execute retrieval plan
        results = await self._execute_retrieval_plan(query, plan)
        
        # 4. Build response
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_stats(plan.strategy, elapsed_ms)
        
        response = RetrievalResponse(
            results=results,
            total_results=len(results),
            retrieval_time_ms=elapsed_ms,
            strategy_used=plan.strategy,
            stores_queried=self._get_queried_stores(plan),
            classification=classification
        )
        
        logger.info(f"🔍 Query routed: {len(results)} results in {elapsed_ms:.1f}ms "
                   f"using {plan.strategy.value} strategy")
        
        return response
    
    def _create_retrieval_plan(self, 
                              classification: ClassificationResult, 
                              max_results: int) -> RetrievalPlan:
        """
        Create retrieval plan based on classification results
        """
        intent = classification.intent
        confidence = classification.confidence
        
        # Determine routing strategy based on confidence
        if confidence >= self.thresholds['high_confidence']:
            strategy = RoutingStrategy.DIRECT
        elif confidence >= self.thresholds['medium_confidence']:
            strategy = RoutingStrategy.PRIMARY_WITH_FALLBACK
        elif confidence >= self.thresholds['low_confidence']:
            strategy = RoutingStrategy.HYBRID
        else:
            strategy = RoutingStrategy.BYPASS
        
        # Map intent to primary store
        store_mapping = {
            QueryIntent.PERSONAL_FACTS: 'facts',
            QueryIntent.CONVERSATION_HISTORY: 'tape',
            QueryIntent.EPISODIC_MEMORY: 'embeddings',
            QueryIntent.KNOWLEDGE_SYNTHESIS: 'embeddings',
            QueryIntent.GENERAL_KNOWLEDGE: None,  # Bypass memory
            QueryIntent.HYBRID_SEARCH: None      # Search all
        }
        
        primary_store = store_mapping.get(intent)
        
        # Define secondary stores for fallback
        secondary_stores = []
        if intent == QueryIntent.PERSONAL_FACTS:
            secondary_stores = ['tape', 'embeddings']
        elif intent == QueryIntent.CONVERSATION_HISTORY:
            secondary_stores = ['embeddings', 'facts']
        elif intent in [QueryIntent.EPISODIC_MEMORY, QueryIntent.KNOWLEDGE_SYNTHESIS]:
            secondary_stores = ['tape', 'facts']
        
        # Override for general knowledge or hybrid
        if intent == QueryIntent.GENERAL_KNOWLEDGE:
            strategy = RoutingStrategy.BYPASS
            primary_store = None
            secondary_stores = []
        elif intent == QueryIntent.HYBRID_SEARCH:
            strategy = RoutingStrategy.HYBRID
            primary_store = None
            secondary_stores = list(self.stores.keys())
        
        # Extract filters from features
        time_filter = None
        if classification.features.has_temporal_marker:
            # TODO: Parse time expressions properly
            time_filter = self._parse_temporal_filter(classification)
        
        plan = RetrievalPlan(
            strategy=strategy,
            primary_store=primary_store,
            secondary_stores=secondary_stores,
            max_results=max_results,
            time_filter=time_filter,
            entity_filter=classification.features.entities,
            confidence_threshold=confidence
        )
        
        return plan
    
    async def _execute_retrieval_plan(self, query: str, plan: RetrievalPlan) -> List[MemoryResult]:
        """
        Execute the retrieval plan across appropriate stores
        """
        all_results = []
        
        try:
            if plan.strategy == RoutingStrategy.BYPASS:
                # Skip memory entirely
                return []
            
            elif plan.strategy == RoutingStrategy.DIRECT:
                # Query only primary store
                if plan.primary_store and plan.primary_store in self.stores:
                    results = await self.stores[plan.primary_store].search(
                        query, limit=plan.max_results
                    )
                    all_results.extend(results)
            
            elif plan.strategy == RoutingStrategy.PRIMARY_WITH_FALLBACK:
                # Try primary first, then fallback if insufficient results
                if plan.primary_store and plan.primary_store in self.stores:
                    results = await self.stores[plan.primary_store].search(
                        query, limit=plan.max_results
                    )
                    all_results.extend(results)
                
                # If insufficient results, try secondary stores
                if len(all_results) < plan.max_results // 2:
                    remaining_limit = plan.max_results - len(all_results)
                    
                    for store_name in plan.secondary_stores:
                        if store_name in self.stores and len(all_results) < plan.max_results:
                            results = await self.stores[store_name].search(
                                query, limit=remaining_limit
                            )
                            all_results.extend(results)
            
            elif plan.strategy == RoutingStrategy.HYBRID:
                # Search all available stores
                stores_to_search = plan.secondary_stores if plan.secondary_stores else list(self.stores.keys())
                results_per_store = max(1, plan.max_results // len(stores_to_search))
                
                # Query stores in parallel
                search_tasks = []
                for store_name in stores_to_search:
                    if store_name in self.stores:
                        task = self.stores[store_name].search(query, limit=results_per_store)
                        search_tasks.append((store_name, task))
                
                # Wait for all searches to complete
                search_results = await asyncio.gather(
                    *[task for _, task in search_tasks], 
                    return_exceptions=True
                )
                
                # Combine results
                for (store_name, _), results in zip(search_tasks, search_results):
                    if isinstance(results, Exception):
                        logger.error(f"Search failed in {store_name}: {results}")
                    else:
                        all_results.extend(results)
            
            # Sort by relevance score and limit results
            all_results.sort(key=lambda r: r.relevance_score, reverse=True)
            return all_results[:plan.max_results]
            
        except Exception as e:
            logger.error(f"Retrieval plan execution failed: {e}")
            return []
    
    def _parse_temporal_filter(self, classification: ClassificationResult) -> Optional[Tuple[float, float]]:
        """
        Parse temporal expressions from query features
        TODO: Implement proper temporal parsing
        """
        if not classification.features.has_temporal_marker:
            return None
        
        # Placeholder - would need proper temporal parsing
        # For now, return last 24 hours if temporal marker detected
        now = time.time()
        return (now - 86400, now)  # Last 24 hours
    
    def _get_queried_stores(self, plan: RetrievalPlan) -> List[str]:
        """Get list of stores that were queried"""
        stores = []
        
        if plan.primary_store and plan.primary_store in self.stores:
            stores.append(plan.primary_store)
        
        if plan.strategy in [RoutingStrategy.PRIMARY_WITH_FALLBACK, RoutingStrategy.HYBRID]:
            for store in plan.secondary_stores:
                if store in self.stores and store not in stores:
                    stores.append(store)
        
        return stores
    
    def _update_stats(self, strategy: RoutingStrategy, elapsed_ms: float):
        """Update performance statistics"""
        self.routing_stats[strategy] += 1
        
        # Update average response time
        if self.total_queries == 1:
            self.avg_response_time_ms = elapsed_ms
        else:
            self.avg_response_time_ms = (
                (self.avg_response_time_ms * (self.total_queries - 1) + elapsed_ms) / 
                self.total_queries
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get router performance statistics"""
        return {
            'total_queries': self.total_queries,
            'routing_distribution': dict(self.routing_stats),
            'avg_response_time_ms': self.avg_response_time_ms,
            'available_stores': list(self.stores.keys()),
            'thresholds': self.thresholds
        }
    
    def update_thresholds(self, 
                         high: float = None, 
                         medium: float = None, 
                         low: float = None):
        """Update confidence thresholds for routing"""
        if high is not None:
            self.thresholds['high_confidence'] = high
        if medium is not None:
            self.thresholds['medium_confidence'] = medium  
        if low is not None:
            self.thresholds['low_confidence'] = low
            
        logger.info(f"Updated routing thresholds: {self.thresholds}")


# Factory function
def create_query_router(facts_graph: Optional[FactsGraph] = None,
                       tape_store=None,
                       embedding_store=None) -> QueryRouter:
    """Create and return a configured query router"""
    router = QueryRouter(
        facts_graph=facts_graph,
        tape_store=tape_store,
        embedding_store=embedding_store
    )
    # Optional thresholds override via environment
    import os
    try:
        hi = os.getenv('ROUTER_THRESHOLD_HIGH')
        med = os.getenv('ROUTER_THRESHOLD_MED')
        low = os.getenv('ROUTER_THRESHOLD_LOW')
        if hi or med or low:
            router.update_thresholds(
                high=float(hi) if hi else None,
                medium=float(med) if med else None,
                low=float(low) if low else None
            )
    except Exception as e:
        logger.warning(f"Router thresholds env override failed: {e}")
    return router


# Self-test
if __name__ == "__main__":
    async def test_router():
        """Test query router with mock facts"""
        logger.info("🧭 Testing Query Router")
        
        # Create test facts graph
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/test.db"
            facts_graph = FactsGraph(db_path)
            
            # Add test facts
            test_facts = [
                {'subject': 'user', 'predicate': 'pet', 'value': 'Potola', 'species': 'dog'},
                {'subject': 'user', 'predicate': 'location', 'value': 'San Francisco'},
                {'subject': 'user', 'predicate': 'name', 'value': 'Alex'},
            ]
            
            for fact in test_facts:
                facts_graph.reinforce_or_insert(fact)
            
            # Create router
            router = create_query_router(facts_graph=facts_graph)
            
            # Test queries
            test_queries = [
                "What's my dog's name?",        # Should route to facts
                "What is photosynthesis?",      # Should bypass memory  
                "What did I say yesterday?",    # Should route to tape (empty)
                "Tell me a story",              # Should route to embeddings (empty)
            ]
            
            for query in test_queries:
                response = await router.route_query(query)
                
                logger.info(f"Query: '{query}'")
                logger.info(f"  Intent: {response.classification.intent.value}")
                logger.info(f"  Strategy: {response.strategy_used.value}")
                logger.info(f"  Results: {response.total_results}")
                logger.info(f"  Time: {response.retrieval_time_ms:.1f}ms")
                
                for result in response.results:
                    logger.info(f"    - {result.content} ({result.source_store})")
                
                logger.info("")
            
            # Print stats
            stats = router.get_performance_stats()
            logger.info(f"Router stats: {stats}")
            
            facts_graph.close()
        
        logger.info("✅ Query Router test complete")
    
    asyncio.run(test_router())
