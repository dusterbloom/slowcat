"""
AudioGraph - M3-inspired entity-centric graph for audio-first voice agents

This is a full adaptation of M3-Agent's VideoGraph for voice agents, maintaining
all sophistication including clustering, equivalence detection, edge weighting,
collision resolution, and multi-modal node management.

Based on M3-Agent's VideoGraph implementation with audio-first optimizations.
"""

import os
import json
import logging
import random
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class AudioGraph:
    """
    M3-inspired AudioGraph for entity-centric voice agent memory.
    
    Maintains all M3 VideoGraph complexity:
    - Multiple embeddings per node with max limits
    - Complex search modes (mean, sum, max, min)
    - DBSCAN clustering for semantic deduplication  
    - Edge weight dynamics with reinforcement/weakening
    - Equivalence resolution with union-find
    - Collision detection with voting mechanisms
    - Clip-based temporal organization
    """
    
    def __init__(self, 
                 max_voice_embeddings=20,  # M3 default for audio nodes
                 max_text_embeddings=10,   # M3 default for text nodes
                 voice_matching_threshold=0.6,  # M3's audio threshold
                 text_matching_threshold=0.3):  # M3's text threshold
        """
        Initialize AudioGraph with M3 parameters.
        
        Args:
            max_voice_embeddings: Maximum embeddings per voice node (M3 default: 20)
            max_text_embeddings: Maximum embeddings per text node (M3 default: 10)
            voice_matching_threshold: Threshold for voice similarity (M3: 0.6)
            text_matching_threshold: Threshold for text similarity (M3: 0.3)
        """
        
        # Core M3 structures
        self.nodes = {}  # node_id -> Node object
        self.edges = {}  # (node_id1, node_id2) -> edge weight
        
        # M3's ordered node tracking
        self.voice_nodes = []  # Voice node IDs (replacing M3's img nodes)
        self.text_nodes = []  # Text node IDs in insertion order
        self.text_nodes_by_clip = {}  # clip_id -> [node_ids]
        self.event_sequence_by_clip = {}  # clip_id -> [episodic_node_ids]
        
        # M3's equivalence tracking (speaker identity resolution)
        self.equivalences = {}  # canonical_id -> set(node_ids)
        
        # M3 parameters
        self.max_voice_embeddings = max_voice_embeddings
        self.max_text_embeddings = max_text_embeddings
        self.voice_matching_threshold = voice_matching_threshold
        self.text_matching_threshold = text_matching_threshold
        
        self.next_node_id = 0
        
        logger.info(f"🧠 AudioGraph initialized with M3 parameters:")
        logger.info(f"   Max voice embeddings: {max_voice_embeddings}")
        logger.info(f"   Max text embeddings: {max_text_embeddings}")
        logger.info(f"   Voice threshold: {voice_matching_threshold}")
        logger.info(f"   Text threshold: {text_matching_threshold}")

    class Node:
        """
        M3's Node structure adapted for audio-first voice agents.
        
        Maintains M3's design:
        - Multiple embeddings per node
        - Rich metadata structure
        - Type-specific handling
        """
        
        def __init__(self, node_id: int, node_type: str):
            """
            Initialize node with M3 structure.
            
            Args:
                node_id: Unique node identifier
                node_type: Type of node ('voice', 'episodic', 'semantic')
            """
            self.id = node_id
            self.type = node_type  # 'voice', 'episodic', 'semantic'
            self.embeddings = []  # Multiple embeddings per node (M3 key feature)
            self.metadata = {
                'contents': [],  # Raw content (transcripts, facts, etc.)
                'timestamp': None,  # Clip/conversation ID (M3's clip_id)
                'speaker_id': None,  # For voice nodes (speaker identification)
                'confidence': 1.0,  # Node confidence/importance
                'first_seen': time.time(),
                'last_updated': time.time()
            }
        
        def add_embedding(self, embedding: List[float], max_embeddings: int):
            """
            Add embedding with M3's max limit and random sampling.
            
            Args:
                embedding: New embedding to add
                max_embeddings: Maximum embeddings allowed for this node type
            """
            self.embeddings.append(embedding)
            
            # M3's random sampling when exceeding max
            if len(self.embeddings) > max_embeddings:
                self.embeddings = random.sample(self.embeddings, max_embeddings)
                logger.debug(f"Node {self.id}: Random sampled to {max_embeddings} embeddings")
        
        def add_content(self, content: str):
            """Add content with M3's accumulation pattern."""
            self.metadata['contents'].append(content)
            self.metadata['last_updated'] = time.time()
            
    # ============================================================================
    # M3's Core Node Operations
    # ============================================================================
    
    def add_voice_node(self, voice_data: Dict[str, Any]) -> int:
        """
        Add voice node with M3's full complexity.
        
        Args:
            voice_data: Dictionary containing:
                - embeddings: List of voice embeddings (from Resemblyzer)
                - contents: List of transcripts/utterances
                - speaker_id: Optional speaker identifier
                
        Returns:
            node_id: ID of created voice node
        """
        node = self.Node(self.next_node_id, 'voice')
        
        # M3's embedding management
        voice_embeddings = voice_data.get('embeddings', [])
        for embedding in voice_embeddings[:self.max_voice_embeddings]:
            node.embeddings.append(embedding)
        
        # M3's metadata storage
        node.metadata['contents'] = voice_data.get('contents', [])
        node.metadata['speaker_id'] = voice_data.get('speaker_id')
        node.metadata['confidence'] = voice_data.get('confidence', 1.0)
        
        self.nodes[self.next_node_id] = node
        self.voice_nodes.append(self.next_node_id)
        self.next_node_id += 1
        
        logger.debug(f"🎤 Voice node {node.id} created with {len(node.embeddings)} embeddings")
        
        return node.id
    
    def add_text_node(self, text_data: Dict[str, Any], clip_id: int, text_type: str = 'episodic') -> int:
        """
        Add text node with M3's episodic/semantic handling.
        
        Args:
            text_data: Dictionary containing:
                - embeddings: Text embeddings
                - contents: Text content list
            clip_id: Temporal clip identifier (M3's clip_id)
            text_type: 'episodic' or 'semantic'
            
        Returns:
            node_id: ID of created text node
        """
        if text_type not in ['episodic', 'semantic']:
            raise ValueError("text_type must be either 'episodic' or 'semantic'")
        
        node = self.Node(self.next_node_id, text_type)
        
        # M3's embedding storage
        node.embeddings = text_data.get('embeddings', [])
        
        # M3's content and metadata
        node.metadata['contents'] = text_data.get('contents', [])
        node.metadata['timestamp'] = clip_id
        
        self.nodes[self.next_node_id] = node
        self.text_nodes.append(node.id)
        
        # M3's clip-based organization
        if clip_id not in self.text_nodes_by_clip:
            self.text_nodes_by_clip[clip_id] = []
        self.text_nodes_by_clip[clip_id].append(node.id)
        
        # M3's episodic sequence tracking
        if text_type == 'episodic':
            if clip_id not in self.event_sequence_by_clip:
                self.event_sequence_by_clip[clip_id] = []
            self.event_sequence_by_clip[clip_id].append(node.id)
        
        self.next_node_id += 1
        
        logger.debug(f"📝 {text_type.capitalize()} text node {node.id} created for clip {clip_id}")
        
        return node.id
    
    def update_node(self, node_id: int, update_info: Dict[str, Any]) -> bool:
        """
        Update existing node with M3's complexity.
        
        Args:
            node_id: ID of target node
            update_info: Dictionary of update information:
                - embeddings: New embeddings to add
                - contents: New content to add
                
        Returns:
            bool: Success indicator
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        
        # M3's content accumulation
        if 'contents' in update_info:
            node.metadata['contents'].extend(update_info['contents'])
        
        # M3's embedding management with limits
        if 'embeddings' in update_info:
            new_embeddings = update_info['embeddings']
            
            # Determine max embeddings based on node type (M3 pattern)
            if node.type == 'voice':
                max_emb = self.max_voice_embeddings
            else:  # text nodes
                max_emb = self.max_text_embeddings
            
            # M3's embedding combination and sampling
            all_embeddings = node.embeddings + new_embeddings
            
            if len(all_embeddings) > max_emb:
                # M3's random sampling when exceeding limit
                node.embeddings = random.sample(all_embeddings, max_emb)
                logger.debug(f"Node {node_id}: Random sampled to {max_emb} embeddings")
            else:
                node.embeddings = all_embeddings
        
        node.metadata['last_updated'] = time.time()
        
        logger.debug(f"🔄 Node {node_id} updated with {len(update_info.get('embeddings', []))} new embeddings")
        
        return True
    
    # ============================================================================
    # M3's Sophisticated Search Operations
    # ============================================================================
    
    def search_voice_nodes(self, voice_info: Dict[str, Any]) -> List[Tuple[int, float]]:
        """
        M3's parallel numpy-based voice search with full complexity.
        
        Args:
            voice_info: Dictionary containing:
                - embeddings: Query voice embeddings
                
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        # Get all voice nodes for search
        target_nodes = [(node_id, node.embeddings) 
                       for node_id, node in self.nodes.items() 
                       if node.type == 'voice']
        
        if not target_nodes:
            logger.debug("🔍 No voice nodes found for search")
            return []
        
        # M3's query embedding preparation
        query_embeddings = np.array(voice_info.get("embeddings", []))
        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        embedding_dim = query_embeddings.shape[-1]
        node_similarities = []
        
        # M3's per-node similarity calculation
        for node_id, node_embeddings in target_nodes:
            if not node_embeddings:
                continue
                
            # M3's numpy array conversion
            node_emb_array = np.array(node_embeddings)
            if len(node_emb_array.shape) == 1:
                node_emb_array = node_emb_array.reshape(1, -1)
            
            # Verify embedding dimensions match
            if node_emb_array.shape[-1] != embedding_dim:
                logger.warning(f"Embedding dimension mismatch for node {node_id}")
                continue
            
            # M3's pairwise similarity matrix calculation
            try:
                similarities = cosine_similarity(query_embeddings, node_emb_array)
                
                # M3's max pooling across all embedding pairs
                max_similarity = np.max(similarities)
                
                # M3's threshold filtering
                if max_similarity > self.voice_matching_threshold:
                    node_similarities.append((node_id, float(max_similarity)))
                    
            except Exception as e:
                logger.error(f"Error calculating similarity for voice node {node_id}: {e}")
                continue
        
        # M3's sorted results by similarity
        result = sorted(node_similarities, key=lambda x: x[1], reverse=True)
        
        logger.debug(f"🔍 Voice search found {len(result)} matches above threshold {self.voice_matching_threshold}")
        
        return result
    
    def search_text_nodes(self, query_embeddings: List[List[float]], 
                         range_nodes: List[int] = [], 
                         mode: str = "max") -> List[Tuple[int, float]]:
        """
        M3's text search with multiple aggregation modes and full complexity.
        
        Args:
            query_embeddings: Query text embeddings
            range_nodes: Optional list of nodes to restrict search to
            mode: Similarity calculation mode ('mean', 'sum', 'max', 'min')
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        if mode not in ['mean', 'sum', 'max', 'min']:
            raise ValueError("Mode must be one of: 'mean', 'sum', 'max', 'min'")
        
        # M3's range node processing
        if range_nodes:
            # Get connected text nodes for range restriction
            text_nodes = []
            for node_id in range_nodes:
                connected = self.get_connected_nodes(node_id, type=['episodic', 'semantic'])
                text_nodes.extend(connected)
            text_nodes = list(set(text_nodes))  # Remove duplicates
        else:
            text_nodes = [nid for nid in self.text_nodes if nid in self.nodes]
        
        if not text_nodes:
            logger.debug("🔍 No text nodes found for search")
            return []
        
        # M3's query embedding preparation
        query_emb_array = np.array(query_embeddings)
        if len(query_emb_array.shape) == 1:
            query_emb_array = query_emb_array.reshape(1, -1)
        
        node_similarities = []
        
        # M3's per-node similarity with mode aggregation
        for node_id in text_nodes:
            if node_id not in self.nodes:
                continue
                
            node = self.nodes[node_id]
            if not node.embeddings:
                continue
            
            # M3's node embedding array
            node_emb_array = np.array(node.embeddings)
            if len(node_emb_array.shape) == 1:
                node_emb_array = node_emb_array.reshape(1, -1)
            
            try:
                # M3's similarity matrix calculation
                similarities = cosine_similarity(query_emb_array, node_emb_array)
                
                # M3's mode-based aggregation
                if mode == "max":
                    aggregated_score = np.max(similarities)
                elif mode == "mean":
                    aggregated_score = np.mean(similarities)
                elif mode == "sum":
                    aggregated_score = np.sum(similarities)
                elif mode == "min":
                    aggregated_score = np.min(similarities)
                
                # M3's threshold filtering
                if aggregated_score > self.text_matching_threshold:
                    node_similarities.append((node_id, float(aggregated_score)))
                    
            except Exception as e:
                logger.error(f"Error calculating similarity for text node {node_id}: {e}")
                continue
        
        # M3's sorted results
        result = sorted(node_similarities, key=lambda x: x[1], reverse=True)
        
        logger.debug(f"🔍 Text search ({mode}) found {len(result)} matches above threshold {self.text_matching_threshold}")
        
        return result
    
    # ============================================================================
    # M3's Edge Management and Graph Operations  
    # ============================================================================
    
    def add_edge(self, node_id1: int, node_id2: int, weight: float = 1.0) -> bool:
        """
        Add or update bidirectional weighted edges with M3's constraints.
        
        M3's rule: Text-to-text connections not allowed between same type nodes.
        
        Args:
            node_id1: First node ID
            node_id2: Second node ID  
            weight: Edge weight (default: 1.0)
            
        Returns:
            bool: Success indicator
        """
        # M3's validation
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            logger.warning(f"Cannot add edge: nodes {node_id1} or {node_id2} not found")
            return False
        
        node1_type = self.nodes[node_id1].type
        node2_type = self.nodes[node_id2].type
        
        # M3's constraint: no same-type text node connections
        if (node1_type == node2_type and 
            node1_type in ['episodic', 'semantic']):
            logger.debug(f"Blocked same-type text connection: {node1_type} {node_id1} <-> {node_id2}")
            return False
        
        # M3's bidirectional edge creation
        self.edges[(node_id1, node_id2)] = weight
        self.edges[(node_id2, node_id1)] = weight
        
        logger.debug(f"🔗 Edge added: {node_id1} <-> {node_id2} (weight: {weight:.3f})")
        
        return True
    
    def update_edge_weight(self, node_id1: int, node_id2: int, delta_weight: float) -> bool:
        """
        Update weight of existing bidirectional edge with M3's auto-removal.
        
        Args:
            node_id1: First node ID
            node_id2: Second node ID
            delta_weight: Weight change (can be negative)
            
        Returns:
            bool: Success indicator
        """
        edge_key = (node_id1, node_id2)
        
        if edge_key not in self.edges:
            logger.debug(f"Edge {node_id1} <-> {node_id2} not found for weight update")
            return False
        
        # M3's bidirectional weight update
        new_weight = self.edges[edge_key] + delta_weight
        self.edges[(node_id1, node_id2)] = new_weight
        self.edges[(node_id2, node_id1)] = new_weight
        
        # M3's edge removal when weight <= 0
        if new_weight <= 0:
            del self.edges[(node_id1, node_id2)]
            del self.edges[(node_id2, node_id1)]
            logger.debug(f"🗑️  Edge removed: {node_id1} <-> {node_id2} (weight dropped to {new_weight:.3f})")
        else:
            logger.debug(f"⚖️  Edge weight updated: {node_id1} <-> {node_id2} (weight: {new_weight:.3f})")
        
        return True
    
    def reinforce_node(self, node_id: int, delta_weight: float = 1.0) -> int:
        """
        Reinforce all edges connected to the given node (M3's pattern).
        
        Args:
            node_id: ID of the node to reinforce
            delta_weight: Amount to increase edge weights by (default: 1.0)
            
        Returns:
            int: Number of edges reinforced
        """
        if node_id not in self.nodes:
            logger.warning(f"Cannot reinforce node {node_id}: not found")
            return 0
        
        reinforced_count = 0
        
        # M3's pattern: iterate over copy to avoid modification during iteration
        for (n1, n2) in list(self.edges.keys()):
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, delta_weight)
                reinforced_count += 1
        
        logger.debug(f"💪 Node {node_id}: {reinforced_count} edges reinforced by {delta_weight}")
        
        return reinforced_count
    
    def weaken_node(self, node_id: int, delta_weight: float = 1.0) -> int:
        """
        Weaken all edges connected to the given node (M3's pattern).
        
        Args:
            node_id: ID of the node to weaken
            delta_weight: Amount to decrease edge weights by (default: 1.0)
            
        Returns:
            int: Number of edges weakened
        """
        if node_id not in self.nodes:
            logger.warning(f"Cannot weaken node {node_id}: not found")
            return 0
        
        weakened_count = 0
        
        # M3's pattern: use negative delta to decrease weights
        for (n1, n2) in list(self.edges.keys()):
            if n1 == node_id or n2 == node_id:
                self.update_edge_weight(n1, n2, -delta_weight)  # Negative delta
                weakened_count += 1
        
        logger.debug(f"📉 Node {node_id}: {weakened_count} edges weakened by {delta_weight}")
        
        return weakened_count
    
    def get_connected_nodes(self, node_id: int, type: List[str] = ['voice', 'episodic', 'semantic']) -> List[int]:
        """
        Get all nodes connected to given node with M3's type filtering.
        
        Args:
            node_id: Target node ID
            type: List of node types to include
            
        Returns:
            List of connected node IDs
        """
        connected = set()  # M3's set usage to avoid duplicates from bidirectional edges
        
        for (n1, n2), weight in self.edges.items():
            if n1 == node_id and n2 in self.nodes and self.nodes[n2].type in type:
                connected.add(n2)
            elif n2 == node_id and n1 in self.nodes and self.nodes[n1].type in type:
                connected.add(n1)
        
        return list(connected)
    
    # ============================================================================
    # Graph Statistics and Information  
    # ============================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        voice_count = len([n for n in self.nodes.values() if n.type == 'voice'])
        episodic_count = len([n for n in self.nodes.values() if n.type == 'episodic'])
        semantic_count = len([n for n in self.nodes.values() if n.type == 'semantic'])
        
        return {
            'total_nodes': len(self.nodes),
            'voice_nodes': voice_count,
            'episodic_nodes': episodic_count,
            'semantic_nodes': semantic_count,
            'total_edges': len(self.edges) // 2,  # Bidirectional, so divide by 2
            'clips_tracked': len(self.text_nodes_by_clip),
            'equivalence_sets': len(self.equivalences)
        }
    
    # ============================================================================
    # M3's Clustering and Collision Resolution
    # ============================================================================
    
    def _cluster_semantic_nodes(self, nodes: List[int], threshold: float = 0.9) -> List[int]:
        """
        M3's DBSCAN clustering for semantic node deduplication.
        
        Args:
            nodes: List of semantic node IDs to cluster
            threshold: Similarity threshold for clustering (default: 0.9)
            
        Returns:
            List of cluster labels for each node
        """
        if not nodes or len(nodes) == 1:
            return [0] * len(nodes)
        
        # M3's embedding collection
        embeddings = []
        valid_nodes = []
        
        for node_id in nodes:
            if node_id in self.nodes and self.nodes[node_id].embeddings:
                embeddings.append(self.nodes[node_id].embeddings[0])  # Use first embedding
                valid_nodes.append(node_id)
        
        if len(embeddings) < 2:
            return [0] * len(valid_nodes)
        
        # M3's similarity matrix calculation
        try:
            similarities = cosine_similarity(embeddings)
            
            # M3's distance matrix conversion: distance = 1 - similarity
            distances = 1 - similarities
            distances[distances < 0] = 0  # M3's negative distance filtering
            
            # M3's DBSCAN clustering
            dbscan_model = DBSCAN(
                eps=(1 - threshold),  # M3's threshold conversion
                min_samples=1,        # M3's parameter
                metric='precomputed'  # M3's precomputed distance matrix
            )
            
            clusters = dbscan_model.fit_predict(distances)
            
            logger.debug(f"🧮 Clustered {len(valid_nodes)} semantic nodes into {len(set(clusters))} clusters")
            
            return clusters.tolist()
            
        except Exception as e:
            logger.error(f"Error in semantic node clustering: {e}")
            return [0] * len(valid_nodes)
    
    def fix_collisions(self, node_id: int, mode: str = 'eq_only') -> List[int]:
        """
        M3's collision resolution with voting mechanisms.
        
        Handles conflicts between semantic nodes connected to the same entity.
        
        Args:
            node_id: Target node ID to fix collisions for
            mode: Resolution mode ('eq_only', 'argmax', 'dropout')
            
        Returns:
            List of node IDs after collision resolution
        """
        # M3's connected semantic nodes detection
        connected_nodes = self.get_connected_nodes(node_id, type=['semantic'])
        
        if len(connected_nodes) == 0:
            return []
        
        filtered_nodes = []
        
        if mode == 'eq_only':
            # M3's equivalence-only mode for speaker identity
            equivalence_node = None
            max_edge_weight = 0
            
            for node in connected_nodes:
                if node not in self.nodes:
                    continue
                    
                content = self.nodes[node].metadata['contents']
                if content and len(content) > 0:
                    content_str = str(content[0]).lower()
                    
                    # M3's equivalence detection
                    if content_str.startswith("equivalence"):
                        # Parse equivalence relationships (simplified)
                        equal_nodes = self._parse_equivalence_content(content[0])
                        
                        # M3's voice/face mapping check (adapted for voice)
                        if not any('voice' in str(n) for n in equal_nodes):
                            filtered_nodes.append(node)
                        else:
                            # M3's edge weight voting mechanism
                            edge_key = (node_id, node)
                            if edge_key in self.edges:
                                edge_weight = self.edges[edge_key]
                                if edge_weight > max_edge_weight:
                                    max_edge_weight = edge_weight
                                    equivalence_node = node
                                elif edge_weight == max_edge_weight:
                                    # M3's random selection for ties
                                    if random.random() < 0.5:
                                        equivalence_node = node
                    else:
                        filtered_nodes.append(node)
            
            # M3's final equivalence node selection
            if equivalence_node is not None:
                filtered_nodes.append(equivalence_node)
                
            return filtered_nodes
        
        elif mode == 'argmax':
            # M3's argmax mode: select highest weight node from each cluster
            clusters = self._cluster_semantic_nodes(connected_nodes)
            cluster_ids = list(set(clusters))
            
            for cluster_id in cluster_ids:
                cluster_nodes = [connected_nodes[i] for i, c in enumerate(clusters) if c == cluster_id]
                
                # Find node with highest edge weight in cluster
                best_node = None
                max_weight = -1
                
                for node in cluster_nodes:
                    edge_key = (node_id, node)
                    if edge_key in self.edges:
                        weight = self.edges[edge_key]
                        if weight > max_weight:
                            max_weight = weight
                            best_node = node
                
                if best_node is not None:
                    filtered_nodes.append(best_node)
        
        elif mode == 'dropout':
            # M3's dropout mode: probabilistic selection based on relative weights
            total_weight = 0
            node_weights = {}
            
            for node in connected_nodes:
                edge_key = (node_id, node)
                if edge_key in self.edges:
                    weight = self.edges[edge_key]
                    node_weights[node] = weight
                    total_weight += weight
            
            # Probabilistic selection
            for node, weight in node_weights.items():
                probability = weight / total_weight if total_weight > 0 else 0.5
                if random.random() < probability:
                    filtered_nodes.append(node)
        
        logger.debug(f"🔧 Collision resolution ({mode}): {len(connected_nodes)} -> {len(filtered_nodes)} nodes")
        
        return filtered_nodes
    
    def _parse_equivalence_content(self, content: str) -> List[str]:
        """
        Parse equivalence relationships from content string.
        
        Args:
            content: Content string containing equivalence information
            
        Returns:
            List of entity references found in equivalence
        """
        # Simplified parsing for equivalence statements
        # Look for patterns like "Equivalence: <voice_1>, <speaker_2>"
        import re
        
        equivalence_pattern = r'<(\w+_\d+)>'
        matches = re.findall(equivalence_pattern, content)
        
        return matches
    
    # ============================================================================
    # M3's Equivalence Resolution (Speaker Identity)
    # ============================================================================
    
    def refresh_equivalences(self):
        """
        M3's disjoint set union-find for identity resolution.
        
        Processes all equivalence relationships and builds canonical identity mapping.
        """
        # M3's disjoint set data structure
        parent = {}
        rank = {}
        
        def find(x):
            """M3's path compression find operation."""
            if x not in parent:
                parent[x] = x
                rank[x] = 0
                return x
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            """M3's union by rank operation."""
            root_x = find(x)
            root_y = find(y)
            
            if root_x != root_y:
                # Union by rank
                if rank[root_x] < rank[root_y]:
                    parent[root_x] = root_y
                elif rank[root_x] > rank[root_y]:
                    parent[root_y] = root_x
                else:
                    parent[root_y] = root_x
                    rank[root_x] += 1
        
        # M3's equivalence relationship processing
        equivalence_count = 0
        
        for node_id, node in self.nodes.items():
            if node.type == 'semantic' and node.metadata['contents']:
                content = str(node.metadata['contents'][0]).lower()
                
                if 'equivalence' in content:
                    # Parse equivalence relationships
                    entities = self._parse_equivalence_content(node.metadata['contents'][0])
                    
                    # M3's union operations for equivalence pairs
                    for i in range(len(entities)):
                        for j in range(i + 1, len(entities)):
                            union(entities[i], entities[j])
                            equivalence_count += 1
        
        # M3's final equivalence set construction
        self.equivalences = {}
        all_entities = set()
        
        # Collect all entities from voice and text nodes
        for node_id, node in self.nodes.items():
            if node.type == 'voice' and node.metadata.get('speaker_id'):
                all_entities.add(f"speaker_{node.metadata['speaker_id']}")
            # Add other entity types as needed
        
        # Build equivalence sets
        for entity in all_entities:
            root = find(entity)
            if root not in self.equivalences:
                self.equivalences[root] = set()
            self.equivalences[root].add(entity)
        
        # Remove single-element sets (no equivalences)
        self.equivalences = {k: v for k, v in self.equivalences.items() if len(v) > 1}
        
        logger.info(f"🔗 Equivalence resolution: {equivalence_count} relationships -> {len(self.equivalences)} identity sets")
    
    def get_canonical_identity(self, entity: str) -> str:
        """
        Get canonical identity for an entity using equivalence mapping.
        
        Args:
            entity: Entity identifier
            
        Returns:
            Canonical entity identifier
        """
        for canonical, equivalence_set in self.equivalences.items():
            if entity in equivalence_set:
                return canonical
        
        return entity  # No equivalence found, return original
    
    # ============================================================================
    # M3's Advanced Graph Operations
    # ============================================================================
    
    def prune_weak_edges(self, threshold: float = 0.1) -> int:
        """
        Remove edges below weight threshold (M3-inspired).
        
        Args:
            threshold: Minimum edge weight to keep
            
        Returns:
            Number of edges removed
        """
        edges_to_remove = []
        
        for (n1, n2), weight in self.edges.items():
            if weight < threshold:
                edges_to_remove.append((n1, n2))
        
        # Remove weak edges (bidirectional)
        removed_count = 0
        processed_pairs = set()
        
        for n1, n2 in edges_to_remove:
            edge_pair = tuple(sorted([n1, n2]))
            if edge_pair not in processed_pairs:
                if (n1, n2) in self.edges:
                    del self.edges[(n1, n2)]
                if (n2, n1) in self.edges:
                    del self.edges[(n2, n1)]
                processed_pairs.add(edge_pair)
                removed_count += 1
        
        logger.info(f"🧹 Pruned {removed_count} weak edges below threshold {threshold}")
        
        return removed_count
    
    def get_node_importance(self, node_id: int) -> float:
        """
        Calculate node importance based on M3's patterns.
        
        Args:
            node_id: Target node ID
            
        Returns:
            Importance score (higher = more important)
        """
        if node_id not in self.nodes:
            return 0.0
        
        node = self.nodes[node_id]
        
        # Base importance from node properties
        importance = node.metadata.get('confidence', 1.0)
        
        # Add importance from connected edges (M3's pattern)
        edge_weight_sum = 0
        edge_count = 0
        
        for (n1, n2), weight in self.edges.items():
            if n1 == node_id or n2 == node_id:
                edge_weight_sum += weight
                edge_count += 1
        
        # Average edge weight as importance factor
        if edge_count > 0:
            importance *= (1 + edge_weight_sum / edge_count)
        
        # Boost for voice nodes (important for speaker identity)
        if node.type == 'voice':
            importance *= 1.5
        
        return importance
    
    def __repr__(self) -> str:
        """String representation with key statistics."""
        stats = self.get_stats()
        return (f"AudioGraph("
                f"nodes={stats['total_nodes']}, "
                f"edges={stats['total_edges']}, "
                f"voice={stats['voice_nodes']}, "
                f"episodic={stats['episodic_nodes']}, "
                f"semantic={stats['semantic_nodes']})")