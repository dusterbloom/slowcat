"""M3 Equivalence Resolver - Entity identity resolution with voting mechanism

Based on M3-Agent's meta-clip algorithm and weight-based voting for resolving
entity equivalences across modalities (face, voice, text).

Features:
- Meta-clip algorithm for high-confidence equivalence discovery
- Weight-based voting mechanism for conflict resolution
- Progressive identity annotation
- Multimodal entity consolidation
- Real-time equivalence updates
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import time

logger = logging.getLogger(__name__)

class EntityType(Enum):
    """Entity types for equivalence resolution"""
    SPEAKER = "speaker"
    PERSON = "person"
    OBJECT = "object"
    LOCATION = "location"
    CONCEPT = "concept"

class ModalityType(Enum):
    """Modality types for multimodal equivalence"""
    FACE = "face"
    VOICE = "voice"
    TEXT = "text"
    SEMANTIC = "semantic"

@dataclass
class EquivalenceCandidate:
    """Candidate for equivalence relationship"""
    node_id: int
    modality: ModalityType
    content: str
    confidence: float
    clip_id: int
    embedding: List[float]
    metadata: Dict[str, Any]

@dataclass
class EquivalenceEdge:
    """Edge representing potential equivalence"""
    source_node_id: int
    target_node_id: int
    source_modality: ModalityType
    target_modality: ModalityType
    weight: float
    evidence_count: int
    clips_observed: List[int]
    created_at: float
    last_reinforced: float

@dataclass
class MetaClip:
    """Meta-clip with single modality instances for clean equivalence"""
    clip_id: int
    face_node_id: Optional[int] = None
    voice_node_id: Optional[int] = None
    text_node_id: Optional[int] = None
    confidence: float = 0.8
    start_time: float = 0.0
    duration: float = 5.0  # M3 uses 5-second clips

class M3EquivalenceResolver:
    """
    M3-inspired equivalence resolver for entity identity across modalities
    
    Implements Algorithm 2 from M3-Agent paper: Progressive Identity Annotation
    with voting mechanism for conflict resolution.
    """
    
    # M3-Agent proven parameters
    MIN_VOTES_THRESHOLD = 2  # Minimum votes for equivalence
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for equivalence
    MAX_CLIP_DURATION = 5.0  # Seconds for meta-clips
    WEIGHT_DECAY_FACTOR = 0.95  # Weekly decay for old equivalences
    
    def __init__(self, m3_integration, similarity_search):
        """Initialize equivalence resolver
        
        Args:
            m3_integration: M3SurrealIntegration instance
            similarity_search: M3SimilaritySearch instance
        """
        self.m3_integration = m3_integration
        self.similarity_search = similarity_search
        
        # Equivalence tracking
        self.equivalence_edges: Dict[Tuple[int, int], EquivalenceEdge] = {}
        self.entity_clusters: Dict[str, Set[int]] = {}  # canonical_id -> node_ids
        self.meta_clips: List[MetaClip] = []
        
        # Performance tracking
        self.stats = {
            'total_equivalences': 0,
            'conflicts_resolved': 0,
            'meta_clips_processed': 0,
            'voting_rounds': 0
        }
        
        logger.info("🔗 M3EquivalenceResolver initialized")
    
    async def build_meta_dictionary(self, session_id: str = "default") -> Dict[int, int]:
        """
        Build meta-dictionary following M3-Agent Algorithm 2
        
        Finds high-confidence face-voice pairs from clean temporal segments
        
        Args:
            session_id: Session to analyze
            
        Returns:
            Dictionary mapping face_node_id -> voice_node_id
        """
        try:
            logger.info("🔍 Building meta-dictionary for equivalence resolution...")
            
            # Step 1: Get all clips for session
            clips_result = await self.m3_integration.query(
                "SELECT * FROM m3_clips WHERE session_id = $session_id ORDER BY start_time",
                {"session_id": session_id}
            )
            
            meta_clips = []
            
            # Step 2: Find meta-clips (1 face + 1 voice)
            for clip_info in clips_result:
                clip_id = clip_info['clip_id']
                
                # Get nodes in this clip
                clip_nodes = await self.m3_integration.get_clip_nodes(clip_id)
                
                # Separate by modality type
                face_nodes = [n for n in clip_nodes if n.get('node_type') == 'voice' and 'face' in str(n.get('contents', ''))]
                voice_nodes = [n for n in clip_nodes if n.get('node_type') == 'voice']
                text_nodes = [n for n in clip_nodes if n.get('node_type') == 'semantic']
                
                # Create meta-clip if we have clean single instances
                if len(voice_nodes) == 1:  # Focus on voice for now
                    meta_clip = MetaClip(
                        clip_id=clip_id,
                        voice_node_id=voice_nodes[0]['node_id'] if voice_nodes else None,
                        text_node_id=text_nodes[0]['node_id'] if len(text_nodes) == 1 else None,
                        confidence=self._calculate_clip_confidence(clip_nodes),
                        start_time=time.time(),  # Could use actual clip time
                        duration=5.0
                    )
                    meta_clips.append(meta_clip)
            
            # Step 3: Voting mechanism for final mapping
            vote_counts = defaultdict(lambda: defaultdict(int))
            confidence_sums = defaultdict(lambda: defaultdict(float))
            
            for meta_clip in meta_clips:
                if meta_clip.voice_node_id and meta_clip.text_node_id:
                    # Vote for voice->text equivalence
                    voice_id = meta_clip.voice_node_id
                    text_id = meta_clip.text_node_id
                    
                    vote_counts[voice_id][text_id] += 1
                    confidence_sums[voice_id][text_id] += meta_clip.confidence
            
            # Step 4: Create final dictionary (highest votes win)
            meta_dict = {}
            
            for voice_id, text_votes in vote_counts.items():
                if text_votes:  # Has votes
                    # Find best text match
                    best_text_id = max(text_votes.items(), key=lambda x: x[1])
                    
                    if best_text_id[1] >= self.MIN_VOTES_THRESHOLD:
                        avg_confidence = confidence_sums[voice_id][best_text_id[0]] / best_text_id[1]
                        
                        if avg_confidence >= self.CONFIDENCE_THRESHOLD:
                            meta_dict[voice_id] = best_text_id[0]
                            
                            # Store equivalence edge
                            await self._create_equivalence_edge(
                                voice_id, 
                                best_text_id[0], 
                                ModalityType.VOICE,
                                ModalityType.TEXT,
                                avg_confidence,
                                best_text_id[1]
                            )
            
            self.stats['meta_clips_processed'] += len(meta_clips)
            
            logger.info(f"✅ Built meta-dictionary with {len(meta_dict)} equivalences from {len(meta_clips)} meta-clips")
            return meta_dict
            
        except Exception as e:
            logger.error(f"Failed to build meta-dictionary: {e}")
            return {}
    
    async def resolve_entity_equivalence(self,
                                       candidates: List[EquivalenceCandidate],
                                       entity_type: EntityType = EntityType.SPEAKER) -> Optional[str]:
        """
        Resolve equivalence among candidate nodes using voting
        
        Args:
            candidates: List of equivalence candidates
            entity_type: Type of entity being resolved
            
        Returns:
            Canonical entity ID if resolved successfully
        """
        try:
            if len(candidates) < 2:
                return None
            
            # Group candidates by modality
            modality_groups = defaultdict(list)
            for candidate in candidates:
                modality_groups[candidate.modality].append(candidate)
            
            # Calculate pairwise similarities
            equivalence_votes = []
            
            for mod1, group1 in modality_groups.items():
                for mod2, group2 in modality_groups.items():
                    if mod1 != mod2:  # Cross-modal equivalence
                        for cand1 in group1:
                            for cand2 in group2:
                                # Calculate similarity
                                similarity = await self._calculate_cross_modal_similarity(cand1, cand2)
                                
                                if similarity > self.CONFIDENCE_THRESHOLD:
                                    equivalence_votes.append({
                                        'node1': cand1.node_id,
                                        'node2': cand2.node_id,
                                        'modality1': mod1,
                                        'modality2': mod2,
                                        'similarity': similarity,
                                        'weight': cand1.confidence * cand2.confidence
                                    })
            
            if not equivalence_votes:
                return None
            
            # Apply voting mechanism
            canonical_id = await self._apply_voting_mechanism(equivalence_votes, entity_type)
            
            if canonical_id:
                # Store resolved equivalence
                node_ids = [c.node_id for c in candidates]
                await self.m3_integration.resolve_equivalence(
                    canonical_id, 
                    node_ids, 
                    entity_type.value
                )
                
                self.stats['total_equivalences'] += 1
                logger.info(f"✅ Resolved equivalence: {canonical_id} with {len(node_ids)} nodes")
            
            return canonical_id
            
        except Exception as e:
            logger.error(f"Equivalence resolution failed: {e}")
            return None
    
    async def update_entity_equivalence(self,
                                      source_node_id: int,
                                      target_node_id: int,
                                      source_modality: ModalityType,
                                      target_modality: ModalityType,
                                      confidence: float = 0.8) -> bool:
        """
        Update equivalence relationship with new evidence
        
        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            source_modality: Source modality type
            target_modality: Target modality type
            confidence: Confidence in this equivalence
            
        Returns:
            True if update successful
        """
        try:
            edge_key = (min(source_node_id, target_node_id), max(source_node_id, target_node_id))
            
            if edge_key in self.equivalence_edges:
                # Strengthen existing equivalence
                edge = self.equivalence_edges[edge_key]
                edge.weight = min(1.0, edge.weight + 0.1)
                edge.evidence_count += 1
                edge.last_reinforced = time.time()
                
                logger.debug(f"🔗 Strengthened equivalence {edge_key} (weight: {edge.weight:.2f})")
            else:
                # Create new equivalence edge
                edge = EquivalenceEdge(
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    source_modality=source_modality,
                    target_modality=target_modality,
                    weight=confidence,
                    evidence_count=1,
                    clips_observed=[],
                    created_at=time.time(),
                    last_reinforced=time.time()
                )
                
                self.equivalence_edges[edge_key] = edge
                
                logger.debug(f"🆕 Created equivalence edge {edge_key} (weight: {confidence:.2f})")
            
            # Handle conflicts using weight-based voting
            await self._resolve_equivalence_conflicts(source_node_id, target_node_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update equivalence: {e}")
            return False
    
    async def _resolve_equivalence_conflicts(self, node1_id: int, node2_id: int):
        """
        Handle conflicting equivalences using weight-based voting
        
        Implements M3-Agent's conflict resolution strategy
        """
        try:
            # Find all edges involving node1
            node1_edges = [(k, e) for k, e in self.equivalence_edges.items() if node1_id in k]
            
            if len(node1_edges) <= 1:
                return  # No conflicts
            
            # Sort edges by weight (highest first)
            node1_edges.sort(key=lambda x: x[1].weight, reverse=True)
            
            # Keep strongest connection, evaluate others
            winner_edge = node1_edges[0]
            
            for edge_key, edge in node1_edges[1:]:
                # Remove weaker connections below threshold
                if edge.weight < winner_edge[1].weight * 0.7:  # 70% threshold from M3
                    logger.debug(f"🗑️ Pruning weak equivalence edge {edge_key} (weight: {edge.weight:.2f})")
                    del self.equivalence_edges[edge_key]
                    self.stats['conflicts_resolved'] += 1
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
    
    async def _calculate_cross_modal_similarity(self,
                                              cand1: EquivalenceCandidate,
                                              cand2: EquivalenceCandidate) -> float:
        """Calculate similarity between candidates from different modalities"""
        try:
            # Use embedding similarity if available
            if cand1.embedding and cand2.embedding and len(cand1.embedding) == len(cand2.embedding):
                import numpy as np
                
                vec1 = np.array(cand1.embedding, dtype=np.float32)
                vec2 = np.array(cand2.embedding, dtype=np.float32)
                
                # Cosine similarity
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    return float(np.dot(vec1, vec2) / (norm1 * norm2))
            
            # Fallback to content similarity
            if cand1.content and cand2.content:
                # Simple content overlap (could be enhanced)
                content1_words = set(cand1.content.lower().split())
                content2_words = set(cand2.content.lower().split())
                
                if content1_words and content2_words:
                    overlap = len(content1_words & content2_words)
                    total = len(content1_words | content2_words)
                    return overlap / total if total > 0 else 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Cross-modal similarity calculation failed: {e}")
            return 0.0
    
    async def _apply_voting_mechanism(self,
                                    votes: List[Dict],
                                    entity_type: EntityType) -> Optional[str]:
        """Apply M3-Agent voting mechanism to resolve entity identity"""
        try:
            if not votes:
                return None
            
            # Group votes by node pairs
            pair_votes = defaultdict(list)
            for vote in votes:
                pair_key = (min(vote['node1'], vote['node2']), max(vote['node1'], vote['node2']))
                pair_votes[pair_key].append(vote)
            
            # Find strongest equivalence pair
            best_pair = None
            best_score = 0.0
            
            for pair_key, pair_vote_list in pair_votes.items():
                # Calculate weighted score
                total_weight = sum(v['weight'] for v in pair_vote_list)
                avg_similarity = sum(v['similarity'] for v in pair_vote_list) / len(pair_vote_list)
                combined_score = total_weight * avg_similarity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_pair = pair_key
            
            if best_pair and best_score > self.CONFIDENCE_THRESHOLD:
                # Create canonical ID
                canonical_id = f"{entity_type.value}_{int(time.time())}_{best_pair[0]}_{best_pair[1]}"
                self.stats['voting_rounds'] += 1
                return canonical_id
            
            return None
            
        except Exception as e:
            logger.error(f"Voting mechanism failed: {e}")
            return None
    
    async def _create_equivalence_edge(self,
                                     node1_id: int,
                                     node2_id: int,
                                     mod1: ModalityType,
                                     mod2: ModalityType,
                                     confidence: float,
                                     evidence_count: int):
        """Create equivalence edge in memory and database"""
        try:
            # Create edge in M3 database
            success = await self.m3_integration.create_m3_edge(
                node1_id,
                node2_id,
                weight=confidence,
                edge_type='equivalence'
            )
            
            if success:
                # Track in local memory
                edge_key = (min(node1_id, node2_id), max(node1_id, node2_id))
                self.equivalence_edges[edge_key] = EquivalenceEdge(
                    source_node_id=node1_id,
                    target_node_id=node2_id,
                    source_modality=mod1,
                    target_modality=mod2,
                    weight=confidence,
                    evidence_count=evidence_count,
                    clips_observed=[],
                    created_at=time.time(),
                    last_reinforced=time.time()
                )
                
        except Exception as e:
            logger.error(f"Failed to create equivalence edge: {e}")
    
    def _calculate_clip_confidence(self, clip_nodes: List[Dict]) -> float:
        """Calculate confidence score for a meta-clip"""
        try:
            if not clip_nodes:
                return 0.0
            
            # Base confidence on node quality and consistency
            confidence_sum = 0.0
            for node in clip_nodes:
                metadata = node.get('metadata', {})
                node_confidence = metadata.get('confidence', 0.8)
                has_embeddings = bool(node.get('embeddings'))
                has_content = bool(node.get('contents'))
                
                # Higher confidence for nodes with complete data
                node_score = node_confidence
                if has_embeddings:
                    node_score *= 1.2
                if has_content:
                    node_score *= 1.1
                    
                confidence_sum += min(1.0, node_score)
            
            return confidence_sum / len(clip_nodes)
            
        except Exception as e:
            logger.error(f"Clip confidence calculation failed: {e}")
            return 0.5
    
    async def get_entity_equivalences(self, node_id: int) -> List[Dict[str, Any]]:
        """Get all known equivalences for a node"""
        try:
            equivalences = []
            
            # Check local edges
            for edge_key, edge in self.equivalence_edges.items():
                if node_id in edge_key:
                    other_node_id = edge_key[0] if edge_key[1] == node_id else edge_key[1]
                    
                    equivalences.append({
                        'node_id': other_node_id,
                        'weight': edge.weight,
                        'evidence_count': edge.evidence_count,
                        'modality': edge.target_modality.value if edge.source_node_id == node_id else edge.source_modality.value,
                        'created_at': edge.created_at,
                        'last_reinforced': edge.last_reinforced
                    })
            
            # Also check database
            db_equivalences = await self.m3_integration.query("""
                SELECT * FROM m3_equivalences 
                WHERE $node_id IN node_ids
            """, {"node_id": node_id})
            
            for eq in db_equivalences:
                equivalences.append({
                    'canonical_id': eq.get('canonical_id'),
                    'entity_type': eq.get('entity_type'),
                    'confidence': eq.get('confidence'),
                    'all_node_ids': eq.get('node_ids', [])
                })
            
            return equivalences
            
        except Exception as e:
            logger.error(f"Failed to get equivalences for node {node_id}: {e}")
            return []
    
    def get_equivalence_stats(self) -> Dict[str, Any]:
        """Get equivalence resolution statistics"""
        return {
            **self.stats,
            'active_edges': len(self.equivalence_edges),
            'entity_clusters': len(self.entity_clusters),
            'meta_clips_found': len(self.meta_clips)
        }