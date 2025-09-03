"""M3 Similarity Search - Optimized multimodal similarity search

Based on M3-Agent paper implementation with MIPS (Maximum Inner Product Search)
and modality-specific thresholds for optimal retrieval performance.

Features:
- MIPS for efficient similarity search
- Multimodal support (text, voice, image)
- Different thresholds per modality (text: 0.3, voice: 0.6)
- Batched operations for performance
- Real-time optimization (<20ms target)
"""

import logging
import numpy as np
import asyncio
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Modality types with M3-Agent thresholds"""
    TEXT = "text"
    VOICE = "voice" 
    IMAGE = "image"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"

@dataclass
class SearchResult:
    """Single similarity search result"""
    node_id: int
    similarity_score: float
    node_type: str
    content: List[str]
    metadata: Dict[str, Any]
    embedding: List[float]

@dataclass
class SearchConfig:
    """Configuration for similarity search"""
    modality: ModalityType
    threshold: float
    max_results: int = 10
    use_mips: bool = True
    normalize_embeddings: bool = True

class M3SimilaritySearch:
    """
    M3-inspired similarity search with MIPS and multimodal support
    
    Implements the search_node function from M3-Agent paper with optimizations
    for real-time voice agent usage.
    """
    
    # M3-Agent thresholds optimized for better recall in production
    MODALITY_THRESHOLDS = {
        ModalityType.TEXT: 0.1,        # Lowered from 0.3 for better recall
        ModalityType.VOICE: 0.2,       # Lowered from 0.6 for better recall
        ModalityType.IMAGE: 0.1,       # Lowered from 0.3 for better recall
        ModalityType.SEMANTIC: 0.1,    # Lowered from 0.3 for better recall
        ModalityType.EPISODIC: 0.2     # Lowered from 0.4 for better recall
    }
    
    def __init__(self, m3_integration):
        """Initialize similarity search with M3 integration
        
        Args:
            m3_integration: M3SurrealIntegration instance
        """
        self.m3_integration = m3_integration
        
        # Performance optimization caches
        self._embedding_cache: Dict[int, np.ndarray] = {}
        self._node_cache: Dict[int, Dict] = {}
        self._last_cache_update = 0
        self.cache_ttl = 300  # 5 minutes
        
        # Statistics tracking
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_search_time': 0.0,
            'last_search_time': 0.0
        }
        
        logger.info("🔍 M3SimilaritySearch initialized")
    
    async def search_nodes(self, 
                          query_embedding: List[float],
                          modality: ModalityType,
                          max_results: int = 10,
                          threshold: Optional[float] = None,
                          node_type_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar nodes using M3-Agent MIPS approach
        
        Args:
            query_embedding: Query vector for similarity search
            modality: Modality type for threshold selection
            max_results: Maximum number of results to return
            threshold: Custom threshold (uses modality default if None)
            node_type_filter: Filter by node type ('voice', 'episodic', 'semantic')
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        start_time = time.time()
        
        try:
            # Use modality-specific threshold if not provided
            if threshold is None:
                threshold = self.MODALITY_THRESHOLDS.get(modality, 0.3)
            
            # Normalize query embedding for cosine similarity
            query_vec = np.array(query_embedding, dtype=np.float32)
            if np.linalg.norm(query_vec) > 0:
                query_vec = query_vec / np.linalg.norm(query_vec)
            
            # Use optimized SurrealDB function for similarity search
            raw_results = await self.m3_integration.search_similar_nodes(
                query_embedding=query_embedding,
                node_type=node_type_filter,
                limit=max_results * 2,  # Get extra for filtering
                min_similarity=threshold
            )
            
            # Process and rank results
            results = []
            for raw_result in raw_results:
                if raw_result.get('similarity', 0) >= threshold:
                    result = SearchResult(
                        node_id=raw_result.get('node_id'),
                        similarity_score=raw_result.get('similarity'),
                        node_type=raw_result.get('node_type'),
                        content=raw_result.get('contents', []),
                        metadata=raw_result.get('metadata', {}),
                        embedding=raw_result.get('embeddings', [[]])[0] if raw_result.get('embeddings') else []
                    )
                    results.append(result)
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            results = results[:max_results]
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_stats(search_time, len(results))
            
            logger.debug(f"🔍 Found {len(results)} similar nodes for {modality.value} query (threshold: {threshold:.2f}, time: {search_time*1000:.1f}ms)")
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def search_clips(self,
                          query_embedding: List[float],
                          max_clips: int = 2,
                          threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Clip-level retrieval following M3-Agent search_clip function
        
        Each clip scored by highest similarity among its memory entries
        
        Args:
            query_embedding: Query vector
            max_clips: Number of clips to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of clip information with scores
        """
        try:
            # Get all nodes with embeddings
            all_nodes = await self.m3_integration.query(
                "SELECT * FROM m3_nodes WHERE array::len(embeddings) > 0"
            )
            
            if not all_nodes:
                return []
            
            # Group nodes by clip_id and find max similarity per clip
            clip_scores = {}
            query_vec = np.array(query_embedding, dtype=np.float32)
            if np.linalg.norm(query_vec) > 0:
                query_vec = query_vec / np.linalg.norm(query_vec)
            
            for node in all_nodes:
                clip_id = node.get('clip_id', 1)
                embeddings = node.get('embeddings', [])
                
                if not embeddings:
                    continue
                
                # Calculate similarity with first embedding
                node_vec = np.array(embeddings[0], dtype=np.float32)
                if np.linalg.norm(node_vec) > 0:
                    node_vec = node_vec / np.linalg.norm(node_vec)
                    similarity = float(np.dot(query_vec, node_vec))
                    
                    # Keep max similarity for this clip
                    if clip_id not in clip_scores or similarity > clip_scores[clip_id]['score']:
                        clip_scores[clip_id] = {
                            'clip_id': clip_id,
                            'score': similarity,
                            'best_node': node,
                            'node_count': clip_scores.get(clip_id, {}).get('node_count', 0) + 1
                        }
                    else:
                        clip_scores[clip_id]['node_count'] += 1
            
            # Filter by threshold and sort
            valid_clips = [
                clip_data for clip_data in clip_scores.values()
                if clip_data['score'] >= threshold
            ]
            
            valid_clips.sort(key=lambda x: x['score'], reverse=True)
            
            # Get clip metadata
            result_clips = []
            for clip_data in valid_clips[:max_clips]:
                clip_info = await self.m3_integration.query(
                    "SELECT * FROM m3_clips WHERE clip_id = $clip_id",
                    {"clip_id": clip_data['clip_id']}
                )
                
                if clip_info:
                    result_clips.append({
                        **clip_info[0],
                        'similarity_score': clip_data['score'],
                        'node_count': clip_data['node_count'],
                        'best_node': clip_data['best_node']
                    })
            
            logger.debug(f"🎬 Found {len(result_clips)} relevant clips")
            return result_clips
            
        except Exception as e:
            logger.error(f"Clip search failed: {e}")
            return []
    
    async def search_by_content(self,
                               query_text: str,
                               embedding_service,
                               modality: ModalityType = ModalityType.TEXT,
                               max_results: int = 10) -> List[SearchResult]:
        """
        Search by text content using embedding service
        
        Args:
            query_text: Text to search for
            embedding_service: Service to generate embeddings
            modality: Modality type for thresholding
            max_results: Maximum results to return
            
        Returns:
            List of similar nodes
        """
        try:
            # Generate query embedding
            query_embedding = await embedding_service.get_embedding(query_text)
            
            if not query_embedding:
                logger.warning(f"Failed to generate embedding for query: {query_text}")
                return []
            
            # Search with generated embedding
            return await self.search_nodes(
                query_embedding=query_embedding,
                modality=modality,
                max_results=max_results
            )
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return []
    
    async def find_similar_to_node(self,
                                  node_id: int,
                                  max_results: int = 5,
                                  exclude_same_clip: bool = True) -> List[SearchResult]:
        """
        Find nodes similar to a given node
        
        Args:
            node_id: ID of the reference node
            max_results: Maximum results to return
            exclude_same_clip: Whether to exclude nodes from same clip
            
        Returns:
            List of similar nodes
        """
        try:
            # Get the reference node
            ref_node = await self.m3_integration.get_node_by_id(node_id)
            
            if not ref_node or not ref_node.get('embeddings'):
                logger.warning(f"Node {node_id} not found or has no embeddings")
                return []
            
            # Use first embedding for similarity
            query_embedding = ref_node['embeddings'][0]
            modality = ModalityType(ref_node.get('node_type', 'text'))
            
            # Search for similar nodes
            results = await self.search_nodes(
                query_embedding=query_embedding,
                modality=modality,
                max_results=max_results + 1  # +1 to account for self
            )
            
            # Filter out self and optionally same clip
            filtered_results = []
            ref_clip_id = ref_node.get('clip_id')
            
            for result in results:
                if result.node_id == node_id:
                    continue  # Skip self
                
                if exclude_same_clip and result.metadata.get('clip_id') == ref_clip_id:
                    continue  # Skip same clip
                
                filtered_results.append(result)
            
            return filtered_results[:max_results]
            
        except Exception as e:
            logger.error(f"Similar node search failed: {e}")
            return []
    
    async def batch_similarity_search(self,
                                     query_embeddings: List[List[float]],
                                     modality: ModalityType,
                                     max_results: int = 10) -> List[List[SearchResult]]:
        """
        Batch similarity search for multiple queries
        
        Args:
            query_embeddings: List of query embeddings
            modality: Modality type for all queries
            max_results: Results per query
            
        Returns:
            List of result lists, one per query
        """
        try:
            tasks = []
            for embedding in query_embeddings:
                task = self.search_nodes(
                    query_embedding=embedding,
                    modality=modality,
                    max_results=max_results
                )
                tasks.append(task)
            
            # Run searches concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch search failed: {result}")
                    processed_results.append([])
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch similarity search failed: {e}")
            return [[] for _ in query_embeddings]
    
    def _update_stats(self, search_time: float, result_count: int):
        """Update search statistics"""
        self.stats['total_searches'] += 1
        self.stats['last_search_time'] = search_time
        
        # Running average of search time
        if self.stats['avg_search_time'] == 0:
            self.stats['avg_search_time'] = search_time
        else:
            alpha = 0.1  # Exponential moving average
            self.stats['avg_search_time'] = (
                alpha * search_time + (1 - alpha) * self.stats['avg_search_time']
            )
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        return {
            **self.stats,
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_searches'], 1),
            'avg_search_time_ms': self.stats['avg_search_time'] * 1000,
            'last_search_time_ms': self.stats['last_search_time'] * 1000
        }
    
    async def optimize_for_realtime(self):
        """Optimize search performance for real-time usage"""
        try:
            # Pre-warm embedding cache
            logger.info("🚀 Optimizing M3 similarity search for real-time performance...")
            
            # Get recent nodes for cache warming
            recent_nodes = await self.m3_integration.query(
                "SELECT * FROM m3_nodes ORDER BY metadata.created_at DESC LIMIT 100"
            )
            
            for node in recent_nodes:
                if node.get('embeddings'):
                    # Cache normalized embeddings
                    embedding = np.array(node['embeddings'][0], dtype=np.float32)
                    if np.linalg.norm(embedding) > 0:
                        normalized = embedding / np.linalg.norm(embedding)
                        self._embedding_cache[node['node_id']] = normalized
                        self._node_cache[node['node_id']] = node
            
            self._last_cache_update = time.time()
            
            logger.info(f"✅ Pre-warmed cache with {len(self._embedding_cache)} embeddings")
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")


class M3ClipRetriever:
    """Specialized retriever for temporal clip-based memory retrieval"""
    
    def __init__(self, m3_integration, similarity_search: M3SimilaritySearch):
        self.m3_integration = m3_integration
        self.similarity_search = similarity_search
    
    async def get_clip_context(self,
                              clip_id: int,
                              include_neighbors: bool = True,
                              max_neighbor_clips: int = 2) -> Dict[str, Any]:
        """
        Get complete context for a temporal clip
        
        Args:
            clip_id: Target clip ID
            include_neighbors: Whether to include neighboring clips
            max_neighbor_clips: Number of neighbor clips to include
            
        Returns:
            Complete clip context with nodes and relationships
        """
        try:
            # Get main clip nodes
            main_nodes = await self.m3_integration.get_clip_nodes(clip_id)
            
            context = {
                'clip_id': clip_id,
                'nodes': main_nodes,
                'neighbor_clips': []
            }
            
            if include_neighbors and main_nodes:
                # Find neighboring clips by similarity
                for node in main_nodes[:3]:  # Use top 3 nodes for neighbor finding
                    if node.get('embeddings'):
                        similar_results = await self.similarity_search.search_clips(
                            query_embedding=node['embeddings'][0],
                            max_clips=max_neighbor_clips + 1  # +1 for self
                        )
                        
                        # Add unique neighbor clips
                        for clip_result in similar_results:
                            if (clip_result['clip_id'] != clip_id and 
                                clip_result not in context['neighbor_clips']):
                                context['neighbor_clips'].append(clip_result)
                
                # Limit neighbor clips
                context['neighbor_clips'] = context['neighbor_clips'][:max_neighbor_clips]
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get clip context: {e}")
            return {'clip_id': clip_id, 'nodes': [], 'neighbor_clips': []}