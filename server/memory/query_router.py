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
from collections import defaultdict
from loguru import logger
import os

# Optional semantic fallback for facts retrieval
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    FACTS_EMBED_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore
    FACTS_EMBED_AVAILABLE = False

from memory.query_classifier import (
    HybridQueryClassifier, QueryIntent, ClassificationResult, create_query_classifier
)
# from memory.facts_graph import FactsGraph  # Replaced with SurrealDB connection manager


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
    
    def __init__(self, facts_graph):
        self.facts_graph = facts_graph
        # Semantic fallback (optional, generic — no domain hardcoding)
        self._encoder = None
        self._fact_emb_cache = {}  # key -> np.ndarray
        self._text_cache = {}      # key -> str
        self._max_emb_facts = int(os.getenv('FACTS_EMBED_FALLBACK_MAX', '800'))
        self._min_sim = float(os.getenv('FACTS_EMBED_MIN_SIM', '0.0'))
        self._budget_ms = int(os.getenv('FACTS_EMBED_FALLBACK_BUDGET_MS', '35'))
        if FACTS_EMBED_AVAILABLE:
            try:
                self._encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("📐 FactsStoreAdapter: semantic fallback enabled (MiniLM)")
            except Exception as e:
                logger.warning(f"FactsStoreAdapter: failed to init encoder: {e}")
                self._encoder = None
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> List[MemoryResult]:
        """Search facts by content with graph traversal support"""
        try:
            # First, try graph traversal queries for relationship patterns
            graph_results = await self._try_graph_traversal_query(query, limit)
            if graph_results:
                return graph_results
            
            # Fallback to standard fact search
            if hasattr(self.facts_graph, 'search_facts'):
                facts = await self.facts_graph.search_facts(query, limit=limit)
            else:
                logger.warning(f"Facts graph {type(self.facts_graph)} has no search_facts method")
                return []
            # Simple semantic nudge: map location-style queries to user.location
            qlow = (query or '').lower()
            if (not facts) and any(k in qlow for k in ["where", "location", "located", "live", "from"]):
                if hasattr(self.facts_graph, 'get_facts'):
                    facts = await self.facts_graph.get_facts(subject='user', predicate='location', min_fidelity=1, limit=limit)
            # Heuristic: personal pet name queries
            if (not facts) and any(k in qlow for k in ["dog", "pet"]) and any(k in qlow for k in ["name", "called", "called?"]):
                try:
                    if hasattr(self.facts_graph, 'get_facts'):
                        # Try likely predicates; tolerate different schemas
                        candidate_preds = ['dog_name', 'pet_name', 'pet', 'dog']
                        agg = []
                        for p in candidate_preds:
                            try:
                                res = await self.facts_graph.get_facts(subject='user', predicate=p, min_fidelity=0, limit=limit)
                                if res:
                                    agg.extend(res)
                            except Exception:
                                continue
                        if agg:
                            facts = agg
                except Exception:
                    pass
            results = []
            
            for fact in facts:
                # Format fact as readable content - handle both dict and object formats
                value = getattr(fact, 'value', None) or fact.get('object', '') if isinstance(fact, dict) else getattr(fact, 'value', '')
                subject = getattr(fact, 'subject', None) or fact.get('subject', '') if isinstance(fact, dict) else getattr(fact, 'subject', '')
                predicate = getattr(fact, 'predicate', None) or fact.get('predicate', '') if isinstance(fact, dict) else getattr(fact, 'predicate', '')
                species = getattr(fact, 'species', None) or fact.get('species', None) if isinstance(fact, dict) else getattr(fact, 'species', None)
                
                if value:
                    if species:
                        content = f"{subject}'s {predicate} is {value} ({species})"
                    else:
                        content = f"{subject}'s {predicate} is {value}"
                else:
                    # S1 level - only relationship
                    content = f"{subject} has {predicate}"
                
                # Calculate relevance score - handle both dict and object formats
                fidelity = getattr(fact, 'fidelity', None) or fact.get('fidelity', 3) if isinstance(fact, dict) else getattr(fact, 'fidelity', 3)
                strength = getattr(fact, 'strength', None) or fact.get('strength', 1.0) if isinstance(fact, dict) else getattr(fact, 'strength', 1.0)
                last_seen = getattr(fact, 'last_seen', None) or fact.get('last_accessed', 0) if isinstance(fact, dict) else getattr(fact, 'last_seen', 0)
                
                relevance = (fidelity / 4.0) * strength
                
                result = MemoryResult(
                    content=content,
                    source_store="facts",
                    relevance_score=relevance,
                    timestamp=last_seen,
                    metadata={
                        'fidelity': fidelity,
                        'strength': strength,
                        'subject': subject,
                        'predicate': predicate,
                        'value': value,
                        'species': species,
                        'access_count': getattr(fact, 'access_count', None) or fact.get('access_count', 0) if isinstance(fact, dict) else getattr(fact, 'access_count', 0)
                    }
                )
                results.append(result)
            
            # If empty, try generic semantic fallback (no hardcoding)
            if not results:
                try:
                    sem = await self._semantic_fallback(query, limit)
                    if sem:
                        return sem
                except Exception:
                    pass
            return results
            
        except Exception as e:
            logger.error(f"Facts search failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    # --- Semantic fallback (generic, no hardcoding) ---
    def _fact_key(self, fact) -> tuple:
        return (
            getattr(fact, 'subject', '') or '',
            getattr(fact, 'predicate', '') or '',
            getattr(fact, 'value', None),
        )

    def _textualize_fact(self, fact) -> str:
        subj = (getattr(fact, 'subject', '') or '').strip()
        pred = (getattr(fact, 'predicate', '') or '').strip().replace('_', ' ')
        val = getattr(fact, 'value', None)
        if val is None or (str(val).strip() == ''):
            return f"{subj} has {pred}"
        return f"{subj}'s {pred} is {val}"

    async def _semantic_fallback(self, query: str, limit: int) -> List[MemoryResult]:
        if not self._encoder:
            return []
        try:
            start = time.time()
            import re
            # Get a bounded set of candidate facts (recency/strength ordering is up to FactsGraph)
            try:
                if hasattr(self.facts_graph, 'get_facts'):
                    candidates = await self.facts_graph.get_facts(limit=self._max_emb_facts)  # type: ignore[attr-defined]
                elif hasattr(self.facts_graph, 'get_top_facts'):
                    candidates = await self.facts_graph.get_top_facts(limit=self._max_emb_facts)  # type: ignore[attr-defined]
                else:
                    candidates = []
            except TypeError:
                # Sync variants
                try:
                    if hasattr(self.facts_graph, 'get_facts'):
                        candidates = self.facts_graph.get_facts(limit=self._max_emb_facts)  # type: ignore[assignment]
                    elif hasattr(self.facts_graph, 'get_top_facts'):
                        candidates = self.facts_graph.get_top_facts(limit=self._max_emb_facts)  # type: ignore[assignment]
                    else:
                        candidates = []
                except Exception:
                    candidates = []
            if not candidates:
                return []

            # Encode query once
            q_emb = self._encoder.encode([query])[0]
            # Normalize
            q_norm = np.linalg.norm(q_emb) or 1.0

            # Light lexical pre-filter to reduce embedding work (generic, no hardcoding)
            q_tokens = set(re.split(r"\W+", query.lower()))
            q_tokens.discard('')

            scored: List[tuple[float, Any]] = []
            # Prefer candidates whose subject/predicate tokens overlap with query tokens
            def candidate_tokens(fact) -> set:
                s = (getattr(fact, 'subject', '') or '').lower()
                p = (getattr(fact, 'predicate', '') or '').lower().replace('_', ' ')
                toks = set(re.split(r"\W+", s + ' ' + p))
                toks.discard('')
                return toks

            # Split candidates into likely (overlap) and others
            likely = []
            others = []
            for f in candidates:
                toks = candidate_tokens(f)
                if q_tokens and (toks & q_tokens):
                    likely.append(f)
                else:
                    others.append(f)

            ordered = likely + others
            for f in ordered:
                key = self._fact_key(f)
                text = self._text_cache.get(key)
                if not text:
                    text = self._textualize_fact(f)
                    self._text_cache[key] = text
                emb = self._fact_emb_cache.get(key)
                if emb is None:
                    emb = self._encoder.encode([text])[0]
                    self._fact_emb_cache[key] = emb
                # Cosine similarity
                denom = (np.linalg.norm(emb) or 1.0) * q_norm
                sim = float(np.dot(q_emb, emb) / denom)
                if sim >= self._min_sim:
                    scored.append((sim, f))
                # Time budget guard
                if (time.time() - start) * 1000.0 > self._budget_ms:
                    break

            if not scored:
                return []
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [f for _, f in scored[:limit]]

            # Map to MemoryResult
            results: List[MemoryResult] = []
            for fact in top:
                content = self._textualize_fact(fact)
                relevance = (getattr(fact, 'fidelity', 3) / 4.0) * getattr(fact, 'strength', 0.6)
                # Mix in semantic score lightly by boosting relevance (kept simple)
                # Note: the caller already sorts by relevance; this provides a stable order
                result = MemoryResult(
                    content=content,
                    source_store="facts",
                    relevance_score=relevance,
                    timestamp=getattr(fact, 'last_seen', 0.0),
                    metadata={
                        'subject': getattr(fact, 'subject', ''),
                        'predicate': getattr(fact, 'predicate', ''),
                        'value': getattr(fact, 'value', None),
                        'fidelity': getattr(fact, 'fidelity', 3),
                        'strength': getattr(fact, 'strength', 0.6),
                        'semantic_fallback': True,
                    }
                )
                results.append(result)
            return results
        except Exception as e:
            logger.debug(f"Semantic fallback failed: {e}")
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
    
    async def _try_graph_traversal_query(self, query: str, limit: int) -> List[MemoryResult]:
        """Try to answer query using graph traversal patterns"""
        
        if not hasattr(self.facts_graph, 'query_graph'):
            return []
        
        query_lower = query.lower().strip()
        results = []
        
        try:
            # Pattern 1: "When is my meeting with [person]?"
            if any(word in query_lower for word in ["when", "meeting", "appointment"]) and "with" in query_lower:
                # Extract person name after "with"
                import re
                match = re.search(r'\bwith\s+([a-zA-Z\s]+?)(?:\?|$|\.)', query_lower)
                if match:
                    person = match.group(1).strip()
                    
                    # Graph traversal query to find meetings with this person
                    graph_query = """
                        SELECT * FROM (
                            SELECT *, ->has_meeting->* AS meetings FROM entity WHERE name = $person OR lower(name) = $person_lower
                        ) WHERE meetings IS NOT NONE
                        UNION ALL
                        SELECT * FROM fact WHERE 
                            (predicate LIKE '%meeting%' AND value LIKE $person_pattern) OR
                            (predicate LIKE $meeting_with_pattern)
                    """
                    
                    params = {
                        'person': person.title(),
                        'person_lower': person.lower(),
                        'person_pattern': f'%{person}%',
                        'meeting_with_pattern': f'%meeting_with_{person.lower().replace(" ", "_")}%'
                    }
                    
                    graph_results = await self.facts_graph.query_graph(graph_query, params)
                    
                    for result in graph_results:
                        if result:
                            results.append(MemoryResult(
                                content=f"Meeting with {person}: {result}",
                                source="graph_traversal",
                                score=0.9,
                                metadata={
                                    'query_type': 'meeting_with_person',
                                    'person': person,
                                    'graph_result': result
                                }
                            ))
            
            # Pattern 2: "What's my dog's name?" or similar pet queries  
            elif any(word in query_lower for word in ["pet", "dog", "cat"]) and any(word in query_lower for word in ["name", "called"]):
                graph_query = """
                    SELECT * FROM fact WHERE 
                        subject = 'user' AND 
                        (predicate LIKE '%pet_name%' OR predicate LIKE '%dog_name%' OR predicate LIKE '%cat_name%')
                    UNION ALL
                    SELECT * FROM (
                        SELECT *, ->owns->* AS pets FROM entity WHERE lower(name) = 'user'
                    ) WHERE pets IS NOT NONE
                """
                
                graph_results = await self.facts_graph.query_graph(graph_query, {})
                
                for result in graph_results:
                    if result:
                        results.append(MemoryResult(
                            content=f"Pet information: {result}",
                            source="graph_traversal", 
                            score=0.9,
                            metadata={
                                'query_type': 'pet_name',
                                'graph_result': result
                            }
                        ))
            
            # Pattern 3: "Where do I live?" or location queries
            elif any(word in query_lower for word in ["where", "live", "location", "address"]):
                graph_query = """
                    SELECT * FROM fact WHERE 
                        subject = 'user' AND 
                        predicate = 'location'
                    UNION ALL
                    SELECT * FROM (
                        SELECT *, ->lives_at->* AS location FROM entity WHERE lower(name) = 'user'
                    ) WHERE location IS NOT NONE
                """
                
                graph_results = await self.facts_graph.query_graph(graph_query, {})
                
                for result in graph_results:
                    if result:
                        results.append(MemoryResult(
                            content=f"Location: {result}",
                            source="graph_traversal",
                            score=0.9,
                            metadata={
                                'query_type': 'location',
                                'graph_result': result
                            }
                        ))
            
            if results:
                logger.debug(f"🕸️ Graph traversal found {len(results)} results for: '{query[:50]}...'")
            
            return results[:limit]
            
        except Exception as e:
            logger.debug(f"Graph traversal query failed: {e}")
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
        
        # Use the correct method based on tape store type
        try:
            if hasattr(self.tape_store, 'search_tape'):
                # SurrealMemory interface
                entries = await self.tape_store.search_tape(query, limit=limit)
            elif hasattr(self.tape_store, 'search'):
                # Generic tape store interface
                entries = await self.tape_store.search(query, limit=limit)
            else:
                logger.warning(f"Tape store {type(self.tape_store)} has no search method")
                return []
        except Exception as e:
            logger.error(f"Tape store search failed: {e}")
            return []
            
        results: List[MemoryResult] = []
        for e in entries:
            # Handle both dict and object formats
            if isinstance(e, dict):
                role = e.get('role', 'unknown')
                content = e.get('content', '')
                ts = e.get('ts', 0)
                speaker_id = e.get('speaker_id', 'unknown')
            else:
                role = getattr(e, 'role', 'unknown')
                content = getattr(e, 'content', '')
                ts = getattr(e, 'ts', 0)
                speaker_id = getattr(e, 'speaker_id', 'unknown')
                
            results.append(MemoryResult(
                content=f"[{role}] {content}",
                source_store='tape',
                relevance_score=0.5,  # simple default; could be BM25 score
                timestamp=ts,
                metadata={'speaker_id': speaker_id, 'role': role}
            ))
        return results
    
    async def get_recent(self, limit: int = 10, since: float = None) -> List[MemoryResult]:
        if not self.tape_store:
            return []
        
        # Use the correct method based on tape store type
        try:
            if hasattr(self.tape_store, 'get_recent'):
                entries = await self.tape_store.get_recent(limit=limit, since=since)
            else:
                logger.warning(f"Tape store {type(self.tape_store)} has no get_recent method")
                return []
        except Exception as e:
            logger.error(f"Tape store get_recent failed: {e}")
            return []
            
        results: List[MemoryResult] = []
        for e in entries:
            # Handle both dict and object formats
            if isinstance(e, dict):
                role = e.get('role', 'unknown')
                content = e.get('content', '')
                ts = e.get('ts', 0)
                speaker_id = e.get('speaker_id', 'unknown')
            else:
                role = getattr(e, 'role', 'unknown')
                content = getattr(e, 'content', '')
                ts = getattr(e, 'ts', 0)
                speaker_id = getattr(e, 'speaker_id', 'unknown')
                
            results.append(MemoryResult(
                content=f"[{role}] {content}",
                source_store='tape',
                relevance_score=0.3,
                timestamp=ts,
                metadata={'speaker_id': speaker_id, 'role': role}
            ))
        return results
    
    def get_store_name(self) -> str:
        return "Conversation Tape"


class EmbeddingStoreAdapter(MemoryStoreInterface):
    """Adapter for semantic/embedding-based search using M3 system"""
    
    def __init__(self, embedding_store=None, m3_context_retriever=None):
        self.embedding_store = embedding_store  # Legacy support
        self.m3_context_retriever = m3_context_retriever
    
    async def search(self, query: str, limit: int = 10, **kwargs) -> List[MemoryResult]:
        """Semantic search using M3 context retrieval system"""
        try:
            if self.m3_context_retriever:
                # Use M3 context retriever for intelligent semantic search
                from .m3_context_retriever import ContextType, RetrievalStrategy
                
                # Determine context type based on kwargs
                context_type = kwargs.get('context_type')
                if isinstance(context_type, str):
                    context_type = getattr(ContextType, context_type.upper(), None)
                
                # Retrieve context using M3 system
                retrieval_result = await self.m3_context_retriever.retrieve_context(
                    query=query,
                    context_type=context_type,
                    max_items=limit,
                    strategy=RetrievalStrategy.SIMILARITY_FIRST,
                    entity_filter=kwargs.get('entity_filter'),
                    time_range=kwargs.get('time_range')
                )
                
                # Convert M3 context items to MemoryResult format
                results = []
                for item in retrieval_result.items:
                    result = MemoryResult(
                        content=item.content,
                        source_store='m3_semantic',
                        relevance_score=item.relevance_score,
                        timestamp=item.timestamp,
                        metadata={
                            'node_id': item.node_id,
                            'context_type': item.context_type.value,
                            'source_clip_id': item.source_clip_id,
                            'entity_refs': item.entity_refs,
                            'retrieval_strategy': retrieval_result.retrieval_strategy.value
                        }
                    )
                    results.append(result)
                
                logger.debug(f"🧠 M3 semantic search found {len(results)} results for: '{query[:50]}...'")
                return results
            
            elif self.embedding_store:
                # Fallback to legacy embedding store
                logger.debug("Using legacy embedding store (placeholder)")
                return []
            
            else:
                logger.warning("No M3 context retriever or embedding store available")
                return []
                
        except Exception as e:
            logger.error(f"M3 semantic search failed: {e}")
            return []
    
    async def get_recent(self, limit: int = 10, since: float = None) -> List[MemoryResult]:
        """Get recent semantic memories"""
        try:
            if self.m3_context_retriever:
                from .m3_context_retriever import ContextType, RetrievalStrategy
                
                # Build temporal query
                time_range = None
                if since:
                    time_range = (since, time.time())
                
                retrieval_result = await self.m3_context_retriever.retrieve_context(
                    query="recent memories",
                    context_type=ContextType.RECENT,
                    max_items=limit,
                    strategy=RetrievalStrategy.TEMPORAL_FIRST,
                    time_range=time_range
                )
                
                # Convert to MemoryResult format
                results = []
                for item in retrieval_result.items:
                    result = MemoryResult(
                        content=item.content,
                        source_store='m3_recent',
                        relevance_score=item.relevance_score,
                        timestamp=item.timestamp,
                        metadata={
                            'node_id': item.node_id,
                            'context_type': item.context_type.value,
                            'source_clip_id': item.source_clip_id
                        }
                    )
                    results.append(result)
                
                return results
            else:
                return []
                
        except Exception as e:
            logger.error(f"M3 recent search failed: {e}")
            return []
    
    def get_store_name(self) -> str:
        return "M3 Semantic Search" if self.m3_context_retriever else "Legacy Semantic Search"


class QueryRouter:
    """
    Intelligent router that directs queries to appropriate memory stores
    """
    
    def __init__(self, 
                 facts_graph = None,
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
        
        # Confidence thresholds for routing decisions - tuned for voice agent use
        self.thresholds = {
            'high_confidence': 0.7,     # Direct routing  
            'medium_confidence': 0.5,   # Primary + fallback
            'low_confidence': 0.2       # Hybrid search - be aggressive about memory search
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
        Route query to appropriate memory stores with smart keyword extraction
        
        Args:
            query: User query text
            context: Optional conversation context
            max_results: Maximum results to return
            
        Returns:
            RetrievalResponse with results from appropriate stores
        """
        start_time = time.time()
        self.total_queries += 1
        
        # 1. Smart keyword extraction for natural language queries
        from .keyword_extractor import get_keyword_extractor
        keyword_extractor = get_keyword_extractor()
        keyword_result = keyword_extractor.extract_keywords(query, max_keywords=3)
        
        # 2. Classify the original query
        classification = await self.classifier.classify(query, context)

        logger.debug(f"🎯 Query classified: {classification.intent.value} "
                    f"({classification.confidence:.2f}) - '{query[:50]}...'")
        
        # 3. Create retrieval plan (SIMPLIFIED - always HYBRID)
        plan = self._create_retrieval_plan(classification, max_results)
        
        # 4. Execute retrieval plan with keyword enhancement
        results = await self._execute_retrieval_plan_with_keywords(
            query, keyword_result, plan, classification, max_results)
        
        # 5. Build response
        elapsed_ms = (time.time() - start_time) * 1000
        self._update_stats(plan.strategy, elapsed_ms)
        
        strategy_name = f"{plan.strategy.value}"
        if not keyword_result.is_simple_query and len(keyword_result.keywords) > 1:
            strategy_name += f"_keywords"
        
        response = RetrievalResponse(
            results=results,
            total_results=len(results),
            retrieval_time_ms=elapsed_ms,
            strategy_used=plan.strategy,
            stores_queried=self._get_queried_stores(plan),
            classification=classification
        )
        
        if keyword_result.is_simple_query:
            logger.info(f"🔍 Query routed: {len(results)} results in {elapsed_ms:.1f}ms "
                       f"using {plan.strategy.value} strategy")
        else:
            logger.info(f"🔍 Query routed: {len(results)} results in {elapsed_ms:.1f}ms "
                       f"using {plan.strategy.value} strategy (keywords: {keyword_result.keywords})")
        
        return response
    
    def _create_retrieval_plan(self, 
                              classification: ClassificationResult, 
                              max_results: int) -> RetrievalPlan:
        """
        Create retrieval plan - SIMPLIFIED to always use HYBRID strategy
        This ensures consistent, reliable results across all queries
        """
        # ALWAYS use HYBRID strategy - search all stores for best results
        strategy = RoutingStrategy.HYBRID
        
        # Always search all available stores
        secondary_stores = list(self.stores.keys())
        
        # No primary store needed for HYBRID - it searches all stores equally
        primary_store = None
        
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
            confidence_threshold=classification.confidence
        )
        
        return plan
    
    async def _execute_retrieval_plan(self, query: str, plan: RetrievalPlan, classification: ClassificationResult) -> List[MemoryResult]:
        """
        Execute the retrieval plan - SIMPLIFIED to only handle HYBRID strategy
        Always searches all available stores for consistent, reliable results
        """
        all_results = []
        
        try:
            # SIMPLIFIED: Only HYBRID strategy - search all available stores
            stores_to_search = list(self.stores.keys())
            results_per_store = max(1, plan.max_results // len(stores_to_search))
            
            # Query stores in parallel
            search_tasks = []
            for store_name in stores_to_search:
                if store_name in self.stores:
                    # Use regular search for all stores (no special conversation history logic)
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
                    import traceback
                    logger.error(f"Full traceback for {store_name}: {''.join(traceback.format_exception(type(results), results, results.__traceback__))}")
                else:
                    logger.debug(f"Store {store_name} returned {len(results)} results")
                    all_results.extend(results)
            
            # Sort by relevance score and limit results
            all_results.sort(key=lambda r: r.relevance_score, reverse=True)
            return all_results[:plan.max_results]
            
        except Exception as e:
            logger.error(f"Retrieval plan execution failed: {e}")
            return []
    
    async def _execute_retrieval_plan_with_keywords(self, 
                                                   original_query: str,
                                                   keyword_result,
                                                   plan: RetrievalPlan, 
                                                   classification: ClassificationResult,
                                                   max_results: int) -> List[MemoryResult]:
        """
        Execute retrieval plan with smart keyword extraction enhancement
        
        For simple queries (single words/phrases), use direct search.
        For natural language queries, extract keywords and search each separately,
        then combine and deduplicate results.
        """
        # If it's a simple query, use the original direct search
        if keyword_result.is_simple_query:
            return await self._execute_retrieval_plan(original_query, plan, classification)
        
        # For natural language queries, search with extracted keywords
        if not keyword_result.keywords:
            # No meaningful keywords found, fall back to direct search
            return await self._execute_retrieval_plan(original_query, plan, classification)
        
        logger.debug(f"🔍 Keyword search: {keyword_result.keywords}")
        
        all_results = []
        seen_results = set()  # For deduplication
        
        try:
            # Search with each keyword
            for keyword in keyword_result.keywords:
                # Create a temporary plan for this keyword
                keyword_results = await self._execute_retrieval_plan(keyword, plan, classification)
                
                # Deduplicate results based on content hash
                for result in keyword_results:
                    # Create a hash based on first 50 chars of content to avoid exact duplicates
                    content_hash = hash(result.content[:50]) if hasattr(result, 'content') else hash(str(result))
                    
                    if content_hash not in seen_results:
                        all_results.append(result)
                        seen_results.add(content_hash)
            
            # Sort by relevance score and limit results
            all_results.sort(key=lambda r: getattr(r, 'relevance_score', 0.0), reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Keyword-enhanced retrieval failed: {e}")
            # Fall back to direct search on error
            return await self._execute_retrieval_plan(original_query, plan, classification)
    
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
def create_query_router(facts_graph = None,
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


def create_m3_query_router(m3_context_retriever=None,
                          facts_graph=None,
                          tape_store=None,
                          config: Optional[Dict] = None) -> QueryRouter:
    """
    Create M3-enabled query router with intelligent context retrieval
    
    Args:
        m3_context_retriever: M3ContextRetriever instance
        facts_graph: Facts graph (optional, can be None)
        tape_store: Tape store for conversation history
        config: Optional configuration overrides
        
    Returns:
        M3-enabled QueryRouter instance
    """
    # Create memory store adapters
    stores = {}
    
    if facts_graph:
        stores['facts'] = FactsStoreAdapter(facts_graph)
    
    if tape_store:
        stores['tape'] = TapeStoreAdapter(tape_store)
    
    # Create M3-enabled embedding store
    if m3_context_retriever:
        stores['embeddings'] = EmbeddingStoreAdapter(
            embedding_store=None,  # No legacy store
            m3_context_retriever=m3_context_retriever
        )
        logger.info("🧠 Created M3-enabled embedding store adapter")
    
    # Create router with M3 stores
    router = QueryRouter.__new__(QueryRouter)
    router.classifier = create_query_classifier()
    router.stores = stores
    
    # M3-optimized confidence thresholds (from M3 paper)
    router.thresholds = {
        'high_confidence': 0.85,   # Direct routing confidence
        'medium_confidence': 0.65, # Fallback routing confidence  
        'low_confidence': 0.45,    # Hybrid search confidence
        'bypass_threshold': 0.25   # Skip memory threshold
    }
    
    # Initialize performance tracking
    router.total_queries = 0
    router.avg_response_time_ms = 0.0
    router.routing_stats = defaultdict(int)
    
    # Apply environment overrides
    import os
    try:
        hi = os.getenv('ROUTER_THRESHOLD_HIGH')
        med = os.getenv('ROUTER_THRESHOLD_MED')
        low = os.getenv('ROUTER_THRESHOLD_LOW')
        if hi or med or low:
            if hi:
                router.thresholds['high_confidence'] = float(hi)
            if med:
                router.thresholds['medium_confidence'] = float(med)
            if low:
                router.thresholds['low_confidence'] = float(low)
    except Exception as e:
        logger.warning(f"M3 router thresholds env override failed: {e}")
    
    logger.info(f"📋 Created M3-enabled QueryRouter with {len(stores)} memory stores")
    logger.info(f"   Thresholds: {router.thresholds}")
    
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
