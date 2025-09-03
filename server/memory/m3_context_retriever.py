"""M3 Context Retriever - Intelligent memory selection with relevance ranking

Implements M3-Agent inspired context retrieval system that replaces token budgeting
with relevance ranking. Combines similarity search, equivalence resolution,
and graph traversal for optimal context selection.

Features:
- Relevance-based context ranking (replaces token budgeting)
- Multimodal memory retrieval
- Entity-aware context selection
- Temporal context filtering
- Real-time performance optimization
- Graph traversal for context expansion
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import time
import numpy as np

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context for retrieval"""
    EPISODIC = "episodic"      # Specific events and interactions
    SEMANTIC = "semantic"      # Facts and knowledge
    VOICE = "voice"           # Voice-specific memories
    RECENT = "recent"         # Recent conversation context
    ENTITY = "entity"         # Entity-related memories

class RetrievalStrategy(Enum):
    """Retrieval strategies for different query types"""
    SIMILARITY_FIRST = "similarity_first"    # Start with similarity, expand
    ENTITY_FIRST = "entity_first"           # Start with entities, expand
    TEMPORAL_FIRST = "temporal_first"       # Start with time, expand
    HYBRID = "hybrid"                       # Combine multiple strategies

@dataclass
class ContextItem:
    """Single item of retrieved context"""
    node_id: int
    content: str
    relevance_score: float
    context_type: ContextType
    source_clip_id: int
    timestamp: float
    entity_refs: List[str]
    embedding: List[float]
    metadata: Dict[str, Any]

@dataclass
class RetrievalContext:
    """Complete context retrieval result"""
    items: List[ContextItem]
    total_relevance: float
    retrieval_strategy: RetrievalStrategy
    query_type: str
    retrieval_time_ms: float
    entities_referenced: Set[str]
    clips_referenced: Set[int]
    statistics: Dict[str, Any]

class M3ContextRetriever:
    """
    M3-inspired context retriever with intelligent memory selection
    
    Replaces token budgeting with relevance ranking following M3-Agent patterns.
    Integrates similarity search, equivalence resolution, and graph traversal.
    """
    
    # M3-Agent inspired parameters
    MAX_CONTEXT_ITEMS = 20           # Maximum context items to retrieve
    MIN_RELEVANCE_THRESHOLD = 0.3    # Minimum relevance for inclusion
    ENTITY_BOOST_FACTOR = 1.2        # Boost for entity-related context
    TEMPORAL_DECAY_FACTOR = 0.95     # Decay for older memories
    GRAPH_EXPANSION_DEPTH = 2        # Maximum graph traversal depth
    
    def __init__(self, 
                 m3_integration,
                 similarity_search,
                 equivalence_resolver,
                 embedding_service=None):
        """Initialize M3 context retriever
        
        Args:
            m3_integration: M3SurrealIntegration instance
            similarity_search: M3SimilaritySearch instance
            equivalence_resolver: M3EquivalenceResolver instance
            embedding_service: Service for generating embeddings
        """
        self.m3_integration = m3_integration
        self.similarity_search = similarity_search
        self.equivalence_resolver = equivalence_resolver
        self.embedding_service = embedding_service
        
        # Context caching for performance
        self._context_cache: Dict[str, Tuple[RetrievalContext, float]] = {}
        self.cache_ttl = 60  # 1 minute cache TTL
        
        # Performance tracking
        self.stats = {
            'total_retrievals': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0.0,
            'avg_relevance_score': 0.0,
            'items_retrieved': 0
        }
        
        logger.info("🧠 M3ContextRetriever initialized")
    
    async def retrieve_context(self,
                             query: str,
                             context_type: Optional[ContextType] = None,
                             max_items: int = 10,
                             strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
                             entity_filter: Optional[List[str]] = None,
                             time_range: Optional[Tuple[float, float]] = None) -> RetrievalContext:
        """
        Main context retrieval function with relevance ranking
        
        Args:
            query: Query text or description
            context_type: Specific type of context to retrieve
            max_items: Maximum number of context items
            strategy: Retrieval strategy to use
            entity_filter: Filter for specific entities
            time_range: Time range filter (start_time, end_time)
            
        Returns:
            RetrievalContext with ranked results
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, context_type, max_items, strategy)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
            
            # Generate query embedding if embedding service available
            query_embedding = None
            if self.embedding_service:
                query_embedding = await self.embedding_service.get_embedding(query)
            
            # Execute retrieval strategy
            if strategy == RetrievalStrategy.SIMILARITY_FIRST:
                context_items = await self._similarity_first_retrieval(
                    query, query_embedding, context_type, max_items, entity_filter, time_range
                )
            elif strategy == RetrievalStrategy.ENTITY_FIRST:
                context_items = await self._entity_first_retrieval(
                    query, query_embedding, context_type, max_items, entity_filter, time_range
                )
            elif strategy == RetrievalStrategy.TEMPORAL_FIRST:
                context_items = await self._temporal_first_retrieval(
                    query, query_embedding, context_type, max_items, entity_filter, time_range
                )
            else:  # HYBRID
                context_items = await self._hybrid_retrieval(
                    query, query_embedding, context_type, max_items, entity_filter, time_range
                )
            
            # Apply relevance ranking and final filtering
            ranked_items = await self._rank_and_filter_context(context_items, query, max_items)
            
            # Build result
            retrieval_time = time.time() - start_time
            result = RetrievalContext(
                items=ranked_items,
                total_relevance=sum(item.relevance_score for item in ranked_items),
                retrieval_strategy=strategy,
                query_type=context_type.value if context_type else "mixed",
                retrieval_time_ms=retrieval_time * 1000,
                entities_referenced=set().union(*[item.entity_refs for item in ranked_items]),
                clips_referenced={item.source_clip_id for item in ranked_items},
                statistics=self._calculate_retrieval_stats(ranked_items, retrieval_time)
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update statistics
            self._update_stats(result)
            
            logger.debug(f"🧠 Retrieved {len(ranked_items)} context items using {strategy.value} (time: {retrieval_time*1000:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return RetrievalContext(
                items=[],
                total_relevance=0.0,
                retrieval_strategy=strategy,
                query_type="error",
                retrieval_time_ms=(time.time() - start_time) * 1000,
                entities_referenced=set(),
                clips_referenced=set(),
                statistics={}
            )
    
    async def _similarity_first_retrieval(self,
                                        query: str,
                                        query_embedding: Optional[List[float]],
                                        context_type: Optional[ContextType],
                                        max_items: int,
                                        entity_filter: Optional[List[str]],
                                        time_range: Optional[Tuple[float, float]]) -> List[ContextItem]:
        """Similarity-first retrieval strategy"""
        try:
            context_items = []
            
            if query_embedding:
                # Use embedding-based similarity search
                from .m3_similarity_search import ModalityType
                
                # Choose modality based on context type
                modality = ModalityType.TEXT
                if context_type == ContextType.VOICE:
                    modality = ModalityType.VOICE
                elif context_type == ContextType.SEMANTIC:
                    modality = ModalityType.SEMANTIC
                
                # Search similar nodes
                similar_results = await self.similarity_search.search_nodes(
                    query_embedding=query_embedding,
                    modality=modality,
                    max_results=max_items * 2  # Get extra for filtering
                )
                
                # Convert to context items
                for result in similar_results:
                    context_item = await self._create_context_item(
                        result.node_id,
                        result.content,
                        result.similarity_score,
                        context_type or self._infer_context_type(result.node_type),
                        result.metadata
                    )
                    
                    if context_item:
                        context_items.append(context_item)
                
                # Expand with graph traversal
                if context_items:
                    expanded_items = await self._expand_with_graph_traversal(
                        [item.node_id for item in context_items[:3]],  # Use top 3 for expansion
                        max_additional=max_items // 2
                    )
                    context_items.extend(expanded_items)
            
            else:
                # Fallback to text-based search
                context_items = await self._text_based_fallback_search(query, max_items)
            
            return context_items
            
        except Exception as e:
            logger.error(f"Similarity-first retrieval failed: {e}")
            return []
    
    async def _entity_first_retrieval(self,
                                    query: str,
                                    query_embedding: Optional[List[float]],
                                    context_type: Optional[ContextType],
                                    max_items: int,
                                    entity_filter: Optional[List[str]],
                                    time_range: Optional[Tuple[float, float]]) -> List[ContextItem]:
        """Entity-first retrieval strategy"""
        try:
            context_items = []
            
            # Extract entities from query (simple approach)
            potential_entities = self._extract_entities_from_query(query)
            
            if entity_filter:
                potential_entities.extend(entity_filter)
            
            # Find nodes related to these entities
            for entity_name in potential_entities:
                raw = await self.m3_integration.query("""
                    SELECT * FROM m3_nodes 
                    WHERE contents CONTAINS $entity OR metadata.speaker_id = $entity
                    LIMIT $limit
                """, {
                    "entity": entity_name,
                    "limit": max_items // max(len(potential_entities), 1)
                })
                entity_nodes = self._normalize_query_result(raw)
                
                for node in entity_nodes:
                    context_item = await self._create_context_item(
                        node['node_id'],
                        node.get('contents', []),
                        0.8,  # High base relevance for entity matches
                        context_type or self._infer_context_type(node.get('node_type')),
                        node.get('metadata', {})
                    )
                    
                    if context_item:
                        # Boost score for entity relevance
                        context_item.relevance_score *= self.ENTITY_BOOST_FACTOR
                        context_item.entity_refs.append(entity_name)
                        context_items.append(context_item)
            
            # If we have embedding, also do similarity search for completeness
            if query_embedding and len(context_items) < max_items:
                similar_items = await self._similarity_first_retrieval(
                    query, query_embedding, context_type, 
                    max_items - len(context_items), entity_filter, time_range
                )
                context_items.extend(similar_items)
            
            return context_items
            
        except Exception as e:
            logger.error(f"Entity-first retrieval failed: {e}")
            return []
    
    async def _temporal_first_retrieval(self,
                                      query: str,
                                      query_embedding: Optional[List[float]],
                                      context_type: Optional[ContextType],
                                      max_items: int,
                                      entity_filter: Optional[List[str]],
                                      time_range: Optional[Tuple[float, float]]) -> List[ContextItem]:
        """Temporal-first retrieval strategy"""
        try:
            context_items = []
            
            # Get recent clips first
            recent_clips_query = "SELECT * FROM m3_clips ORDER BY start_time DESC LIMIT 10"
            if time_range:
                recent_clips_query = """
                    SELECT * FROM m3_clips 
                    WHERE start_time >= $start_time AND start_time <= $end_time
                    ORDER BY start_time DESC
                    LIMIT 20
                """
            
            raw_clips = await self.m3_integration.query(
                recent_clips_query,
                {"start_time": time_range[0], "end_time": time_range[1]} if time_range else {}
            )
            clips_result = self._normalize_query_result(raw_clips)
            
            # Get nodes from recent clips
            for clip in clips_result[:5]:  # Limit to 5 most recent clips
                if not isinstance(clip, dict):
                    continue
                clip_id = clip.get('clip_id')
                if clip_id is None:
                    continue
                clip_nodes = await self.m3_integration.get_clip_nodes(clip_id)
                
                for node in clip_nodes:
                    if len(context_items) >= max_items:
                        break
                    
                    # Calculate temporal relevance
                    temporal_score = self._calculate_temporal_relevance(node, query)
                    
                    if temporal_score > self.MIN_RELEVANCE_THRESHOLD:
                        context_item = await self._create_context_item(
                            node['node_id'],
                            node.get('contents', []),
                            temporal_score,
                            context_type or self._infer_context_type(node.get('node_type')),
                            node.get('metadata', {})
                        )
                        
                        if context_item:
                            context_items.append(context_item)
            
            return context_items
            
        except Exception as e:
            logger.error(f"Temporal-first retrieval failed: {e}")
            return []
    
    async def _hybrid_retrieval(self,
                              query: str,
                              query_embedding: Optional[List[float]],
                              context_type: Optional[ContextType],
                              max_items: int,
                              entity_filter: Optional[List[str]],
                              time_range: Optional[Tuple[float, float]]) -> List[ContextItem]:
        """Hybrid retrieval combining multiple strategies"""
        try:
            # Run multiple strategies concurrently
            tasks = [
                self._similarity_first_retrieval(query, query_embedding, context_type, max_items//3, entity_filter, time_range),
                self._entity_first_retrieval(query, query_embedding, context_type, max_items//3, entity_filter, time_range),
                self._temporal_first_retrieval(query, query_embedding, context_type, max_items//3, entity_filter, time_range)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            all_items = []
            for result in results:
                if isinstance(result, list):
                    all_items.extend(result)
                else:
                    logger.error(f"Hybrid retrieval task failed: {result}")
            
            # Deduplicate by node_id
            seen_nodes = set()
            unique_items = []
            for item in all_items:
                if item.node_id not in seen_nodes:
                    seen_nodes.add(item.node_id)
                    unique_items.append(item)
            
            return unique_items
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    async def _expand_with_graph_traversal(self,
                                         seed_node_ids: List[int],
                                         max_additional: int) -> List[ContextItem]:
        """Expand context using graph traversal from seed nodes"""
        try:
            expanded_items = []
            visited_nodes = set(seed_node_ids)
            
            for seed_node_id in seed_node_ids:
                if len(expanded_items) >= max_additional:
                    break
                
                # Get graph context around this node
                graph_context = await self.m3_integration.get_graph_context(
                    seed_node_id,
                    max_depth=self.GRAPH_EXPANSION_DEPTH,
                    max_nodes=max_additional - len(expanded_items)
                )
                
                # Process connected nodes
                for connected_node in graph_context.get('nodes', []):
                    node_id = connected_node.get('node_id')
                    
                    if node_id and node_id not in visited_nodes:
                        visited_nodes.add(node_id)
                        
                        # Create context item with graph-based relevance
                        context_item = await self._create_context_item(
                            node_id,
                            connected_node.get('contents', []),
                            0.6,  # Base relevance for graph-connected nodes
                            self._infer_context_type(connected_node.get('node_type')),
                            connected_node.get('metadata', {})
                        )
                        
                        if context_item:
                            expanded_items.append(context_item)
                        
                        if len(expanded_items) >= max_additional:
                            break
            
            logger.debug(f"🔗 Expanded context with {len(expanded_items)} graph-connected items")
            return expanded_items
            
        except Exception as e:
            logger.error(f"Graph traversal expansion failed: {e}")
            return []
    
    async def _rank_and_filter_context(self,
                                     context_items: List[ContextItem],
                                     query: str,
                                     max_items: int) -> List[ContextItem]:
        """Apply final relevance ranking and filtering"""
        try:
            # Apply relevance boosting and decay
            for item in context_items:
                # Temporal decay
                age_factor = self._calculate_age_factor(item.timestamp)
                item.relevance_score *= age_factor
                
                # Content quality boost
                content_quality = self._assess_content_quality(item.content)
                item.relevance_score *= content_quality
                
                # Entity boost
                if item.entity_refs:
                    item.relevance_score *= self.ENTITY_BOOST_FACTOR
            
            # Filter by minimum relevance
            filtered_items = [
                item for item in context_items 
                if item.relevance_score >= self.MIN_RELEVANCE_THRESHOLD
            ]
            
            # Sort by relevance score
            filtered_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply diversity filter to avoid too similar items
            diverse_items = self._apply_diversity_filter(filtered_items, max_items)
            
            return diverse_items
            
        except Exception as e:
            logger.error(f"Context ranking and filtering failed: {e}")
            return context_items[:max_items]  # Fallback to simple truncation
    
    async def _create_context_item(self,
                                 node_id: int,
                                 content: List[str],
                                 relevance_score: float,
                                 context_type: ContextType,
                                 metadata: Dict[str, Any]) -> Optional[ContextItem]:
        """Create a context item from node data"""
        try:
            # Get additional node data if needed
            node_data = await self.m3_integration.get_node_by_id(node_id)
            
            if not node_data:
                return None
            
            # Extract entities from content
            entity_refs = self._extract_entities_from_content(content)
            
            # Get embedding
            embeddings = node_data.get('embeddings', [])
            embedding = embeddings[0] if embeddings else []
            
            context_item = ContextItem(
                node_id=node_id,
                content=self._format_content(content),
                relevance_score=relevance_score,
                context_type=context_type,
                source_clip_id=metadata.get('clip_id', 0),
                timestamp=time.time(),  # Could use actual timestamp from metadata
                entity_refs=entity_refs,
                embedding=embedding,
                metadata=metadata
            )
            
            return context_item
            
        except Exception as e:
            logger.error(f"Failed to create context item: {e}")
            return None
    
    def _generate_cache_key(self, query: str, context_type: Optional[ContextType], 
                          max_items: int, strategy: RetrievalStrategy) -> str:
        """Generate cache key for retrieval results"""
        import hashlib
        
        key_components = [
            query,
            context_type.value if context_type else "none",
            str(max_items),
            strategy.value
        ]
        
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _get_cached_result(self, cache_key: str) -> Optional[RetrievalContext]:
        """Get cached result if available and not expired"""
        if cache_key in self._context_cache:
            result, cached_time = self._context_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return result
            else:
                del self._context_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: RetrievalContext):
        """Cache retrieval result"""
        self._context_cache[cache_key] = (result, time.time())
        
        # Simple cache size management
        if len(self._context_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self._context_cache.keys(),
                key=lambda k: self._context_cache[k][1]
            )[:20]
            
            for key in oldest_keys:
                del self._context_cache[key]
    
    def _infer_context_type(self, node_type: str) -> ContextType:
        """Infer context type from node type"""
        if node_type == 'voice':
            return ContextType.VOICE
        elif node_type == 'semantic':
            return ContextType.SEMANTIC
        elif node_type == 'episodic':
            return ContextType.EPISODIC
        else:
            return ContextType.RECENT
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from query text (simple implementation)"""
        # Simple entity extraction - could be enhanced with NLP
        words = query.split()
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word.lower())
        
        return entities
    
    def _extract_entities_from_content(self, content: List[str]) -> List[str]:
        """Extract entities from content"""
        entities = []
        for text in content:
            entities.extend(self._extract_entities_from_query(text))
        return list(set(entities))  # Deduplicate
    
    def _format_content(self, content: List[str]) -> str:
        """Format content list into readable string"""
        if isinstance(content, list):
            return " ".join(content)
        return str(content)
    
    def _calculate_temporal_relevance(self, node: Dict, query: str) -> float:
        """Calculate temporal relevance score"""
        # Simple implementation - could be enhanced with temporal analysis
        base_score = 0.5
        
        # Check for temporal keywords in query
        temporal_keywords = ['recent', 'today', 'yesterday', 'now', 'current', 'latest']
        if any(keyword in query.lower() for keyword in temporal_keywords):
            base_score = 0.8
        
        return base_score
    
    def _calculate_age_factor(self, timestamp: float) -> float:
        """Calculate age decay factor"""
        age_hours = (time.time() - timestamp) / 3600
        return max(0.1, self.TEMPORAL_DECAY_FACTOR ** age_hours)
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality for boosting"""
        if not content:
            return 0.5
        
        # Simple quality metrics
        quality = 1.0
        
        # Length penalty for very short content
        if len(content) < 10:
            quality *= 0.7
        
        # Boost for longer, more informative content
        if len(content) > 50:
            quality *= 1.1
        
        return min(1.2, quality)
    
    def _apply_diversity_filter(self, items: List[ContextItem], max_items: int) -> List[ContextItem]:
        """Apply diversity filtering to avoid too similar items"""
        if len(items) <= max_items:
            return items
        
        # Simple diversity based on content similarity
        diverse_items = [items[0]]  # Always include top item
        
        for item in items[1:]:
            if len(diverse_items) >= max_items:
                break
            
            # Check similarity with existing items
            is_diverse = True
            for existing_item in diverse_items:
                if self._calculate_content_similarity(item.content, existing_item.content) > 0.8:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_items.append(item)
        
        return diverse_items
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _text_based_fallback_search(self, query: str, max_items: int) -> List[ContextItem]:
        """Fallback search when embeddings not available"""
        try:
            # Simple text-based search in SurrealDB
            raw = await self.m3_integration.query("""
                SELECT * FROM m3_nodes 
                WHERE contents CONTAINS $query OR metadata.speaker_id CONTAINS $query
                LIMIT $limit
            """, {"query": query, "limit": max_items})
            search_results = self._normalize_query_result(raw)
            
            context_items = []
            for node in search_results:
                context_item = await self._create_context_item(
                    node['node_id'],
                    node.get('contents', []),
                    0.6,  # Base relevance for text matches
                    self._infer_context_type(node.get('node_type')),
                    node.get('metadata', {})
                )
                
                if context_item:
                    context_items.append(context_item)
            
            return context_items
            
        except Exception as e:
            logger.error(f"Text-based fallback search failed: {e}")
            return []

    def _normalize_query_result(self, result):
        """Normalize SurrealDB query outputs into a list of dict records."""
        try:
            if result is None:
                return []
            if isinstance(result, list):
                if len(result) == 1 and isinstance(result[0], dict) and 'result' in result[0]:
                    inner = result[0]['result']
                    return inner if isinstance(inner, list) else []
                if all(isinstance(x, dict) for x in result):
                    return result
                if len(result) == 1 and isinstance(result[0], list):
                    inner = result[0]
                    return inner if all(isinstance(x, dict) for x in inner) else []
                return []
            if isinstance(result, dict) and 'result' in result and isinstance(result['result'], list):
                inner = result['result']
                return inner if all(isinstance(x, dict) for x in inner) else []
            return []
        except Exception:
            return []
    
    def _calculate_retrieval_stats(self, items: List[ContextItem], retrieval_time: float) -> Dict[str, Any]:
        """Calculate statistics for retrieval results"""
        if not items:
            return {}
        
        return {
            'items_count': len(items),
            'avg_relevance': sum(item.relevance_score for item in items) / len(items),
            'max_relevance': max(item.relevance_score for item in items),
            'min_relevance': min(item.relevance_score for item in items),
            'unique_clips': len(set(item.source_clip_id for item in items)),
            'unique_entities': len(set().union(*[item.entity_refs for item in items])),
            'retrieval_time_ms': retrieval_time * 1000
        }
    
    def _update_stats(self, result: RetrievalContext):
        """Update retrieval statistics"""
        self.stats['total_retrievals'] += 1
        self.stats['items_retrieved'] += len(result.items)
        
        if result.items:
            # Update average relevance score
            avg_relevance = result.total_relevance / len(result.items)
            if self.stats['avg_relevance_score'] == 0:
                self.stats['avg_relevance_score'] = avg_relevance
            else:
                alpha = 0.1  # Exponential moving average
                self.stats['avg_relevance_score'] = (
                    alpha * avg_relevance + (1 - alpha) * self.stats['avg_relevance_score']
                )
        
        # Update average retrieval time
        if self.stats['avg_retrieval_time'] == 0:
            self.stats['avg_retrieval_time'] = result.retrieval_time_ms
        else:
            alpha = 0.1
            self.stats['avg_retrieval_time'] = (
                alpha * result.retrieval_time_ms + (1 - alpha) * self.stats['avg_retrieval_time']
            )
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval performance statistics"""
        return {
            **self.stats,
            'cache_size': len(self._context_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_retrievals'], 1)
        }
