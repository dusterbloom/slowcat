"""
Memory Transition Manager

Handles automatic promotion of fragments between memory tiers based on:
- Age-based promotion (Working → Short → Long → Episodic)
- Importance scoring (access frequency, semantic uniqueness, emotional weight)
- Semantic compression during transitions
- Attractor weight calculation for consolidation
- Field state evolution across tiers

Integrates with existing consciousness/core.py and memory/surreal_memory.py
while providing sophisticated memory consolidation algorithms.
"""

import asyncio
import time
import math
import hashlib
import json
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from loguru import logger
import numpy as np

# Import existing components
from consciousness.core import SemanticEmbedder, MLX_AVAILABLE
from memory.surreal_memory import SurrealMemory, SURREALDB_AVAILABLE
from memory.hierarchical_manager import MemoryFragment, FieldState

@dataclass
class TransitionCandidate:
    """Fragment candidate for tier transition"""
    fragment: MemoryFragment
    current_tier: int
    target_tier: int
    importance_score: float
    age_seconds: float
    transition_confidence: float
    field_state: Optional[FieldState] = None

@dataclass
class ImportanceMetrics:
    """Metrics used for importance scoring"""
    access_frequency: float  # How often accessed
    recency_boost: float    # Recent access boost
    semantic_uniqueness: float  # How unique the content is
    cross_reference_density: float  # How connected to other memories
    emotional_weight: float  # Emotional/contextual significance
    user_attention: float   # User engagement indicators

class ImportanceCalculator:
    """Calculates importance scores for memory fragments"""
    
    def __init__(self):
        self.embedder = SemanticEmbedder()
        
        # Importance weights (sum to 1.0)
        self.weights = {
            "access_frequency": 0.25,
            "recency_boost": 0.15, 
            "semantic_uniqueness": 0.20,
            "cross_reference_density": 0.15,
            "emotional_weight": 0.15,
            "user_attention": 0.10
        }
    
    async def calculate_importance(self, 
                                 fragment: MemoryFragment,
                                 all_fragments: List[MemoryFragment] = None) -> ImportanceMetrics:
        """Calculate comprehensive importance score for fragment"""
        
        current_time = time.time()
        age_hours = (current_time - fragment.created_at) / 3600
        
        # Access frequency (normalized by age)
        access_frequency = fragment.access_count / max(1, age_hours)
        
        # Recency boost (higher if accessed recently)
        hours_since_access = (current_time - fragment.last_accessed) / 3600
        recency_boost = math.exp(-hours_since_access / 24)  # Decay over 24 hours
        
        # Semantic uniqueness (how different from other fragments)
        semantic_uniqueness = await self._calculate_semantic_uniqueness(
            fragment, all_fragments or []
        )
        
        # Cross-reference density (placeholder - would analyze relationships)
        cross_reference_density = len(fragment.context_tags) / 10  # Simple proxy
        
        # Emotional weight (based on context tags and content)
        emotional_weight = await self._calculate_emotional_weight(fragment)
        
        # User attention (based on interaction patterns)
        user_attention = await self._calculate_user_attention(fragment)
        
        return ImportanceMetrics(
            access_frequency=min(1.0, access_frequency),
            recency_boost=recency_boost,
            semantic_uniqueness=semantic_uniqueness,
            cross_reference_density=min(1.0, cross_reference_density),
            emotional_weight=emotional_weight,
            user_attention=user_attention
        )
    
    async def _calculate_semantic_uniqueness(self, 
                                           fragment: MemoryFragment,
                                           other_fragments: List[MemoryFragment]) -> float:
        """Calculate how semantically unique this fragment is"""
        if not other_fragments:
            return 0.8  # Default high uniqueness if no comparisons
        
        try:
            fragment_text = fragment.content.get("text", "")
            if not fragment_text:
                return 0.5
            
            fragment_embedding = self.embedder.embed(fragment_text)
            similarities = []
            
            # Compare with up to 50 other fragments for performance
            comparison_fragments = other_fragments[:50]
            
            for other in comparison_fragments:
                if other.fragment_id == fragment.fragment_id:
                    continue
                    
                other_text = other.content.get("text", "")
                if other_text:
                    other_embedding = self.embedder.embed(other_text)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(fragment_embedding, other_embedding) / (
                        np.linalg.norm(fragment_embedding) * np.linalg.norm(other_embedding)
                    )
                    similarities.append(similarity)
            
            if similarities:
                # Uniqueness is inverse of maximum similarity
                max_similarity = max(similarities)
                uniqueness = 1.0 - max_similarity
                return max(0.0, min(1.0, uniqueness))
            
            return 0.8  # High uniqueness if no similar fragments found
            
        except Exception as e:
            logger.debug(f"Semantic uniqueness calculation failed: {e}")
            return 0.5  # Default moderate uniqueness
    
    async def _calculate_emotional_weight(self, fragment: MemoryFragment) -> float:
        """Calculate emotional/contextual significance"""
        emotional_indicators = {
            "important", "urgent", "critical", "remember", "forget",
            "happy", "sad", "angry", "excited", "worried", "love",
            "meeting", "deadline", "appointment", "anniversary"
        }
        
        # Check context tags for emotional indicators
        tag_weight = 0.0
        for tag in fragment.context_tags:
            if any(indicator in tag.lower() for indicator in emotional_indicators):
                tag_weight += 0.1
        
        # Check content for emotional indicators
        content_text = fragment.content.get("text", "").lower()
        content_weight = 0.0
        for indicator in emotional_indicators:
            if indicator in content_text:
                content_weight += 0.05
        
        total_weight = min(1.0, tag_weight + content_weight)
        
        # Base emotional weight for certain fragment types
        type_weights = {
            "emotional": 0.8,
            "episodic": 0.6,
            "contextual": 0.4,
            "semantic": 0.2,
            "procedural": 0.3
        }
        
        base_weight = type_weights.get(fragment.type, 0.5)
        
        # Combine base weight with content-based weight
        final_weight = (base_weight * 0.7) + (total_weight * 0.3)
        return min(1.0, final_weight)
    
    async def _calculate_user_attention(self, fragment: MemoryFragment) -> float:
        """Calculate user attention indicators"""
        # This is a simplified implementation
        # Could analyze interaction patterns, conversation flow, etc.
        
        # Higher attention for recently created fragments
        age_hours = (time.time() - fragment.created_at) / 3600
        recency_attention = math.exp(-age_hours / 48)  # Decay over 48 hours
        
        # Higher attention for frequently accessed fragments
        access_attention = min(1.0, fragment.access_count / 10)
        
        # Combine factors
        return (recency_attention * 0.6) + (access_attention * 0.4)
    
    def calculate_final_importance_score(self, metrics: ImportanceMetrics) -> float:
        """Calculate final weighted importance score"""
        score = (
            metrics.access_frequency * self.weights["access_frequency"] +
            metrics.recency_boost * self.weights["recency_boost"] +
            metrics.semantic_uniqueness * self.weights["semantic_uniqueness"] +
            metrics.cross_reference_density * self.weights["cross_reference_density"] +
            metrics.emotional_weight * self.weights["emotional_weight"] +
            metrics.user_attention * self.weights["user_attention"]
        )
        
        return max(0.0, min(1.0, score))

class SemanticCompressor:
    """Handles semantic compression during tier transitions"""
    
    def __init__(self):
        self.embedder = SemanticEmbedder()
    
    async def compress_for_tier(self, 
                              fragment: MemoryFragment, 
                              target_tier: int) -> MemoryFragment:
        """Compress fragment content appropriate for target tier"""
        
        if target_tier == 2:  # Short-term: semantic compression
            return await self._compress_semantic(fragment)
        elif target_tier == 3:  # Long-term: graph node compression
            return await self._compress_to_graph_node(fragment)
        elif target_tier == 4:  # Episodic: importance-based compression
            return await self._compress_episodic(fragment)
        
        return fragment  # No compression needed
    
    async def _compress_semantic(self, fragment: MemoryFragment) -> MemoryFragment:
        """Compress for short-term memory with semantic preservation"""
        content = fragment.content.copy()
        
        # Keep essential semantic information
        text = content.get("text", "")
        if len(text) > 200:  # Compress long text
            # Simple compression: keep first and last sentences + key phrases
            sentences = text.split('. ')
            if len(sentences) > 3:
                compressed_text = f"{sentences[0]}. ... {sentences[-1]}"
                content["text"] = compressed_text
                content["original_length"] = len(text)
                content["compression_ratio"] = len(compressed_text) / len(text)
        
        # Preserve embedding for similarity searches
        content["compression_level"] = "semantic"
        
        compressed_fragment = MemoryFragment(
            fragment_id=fragment.fragment_id,
            type=fragment.type,
            content=content,
            context_tags=fragment.context_tags,
            strength=fragment.strength * 0.9,  # Slight strength reduction
            memory_tier=2,
            last_accessed=fragment.last_accessed,
            access_count=fragment.access_count,
            source_interactions=fragment.source_interactions,
            created_at=fragment.created_at
        )
        
        return compressed_fragment
    
    async def _compress_to_graph_node(self, fragment: MemoryFragment) -> MemoryFragment:
        """Compress for long-term memory as graph node"""
        content = fragment.content.copy()
        
        # Extract key entities and relationships
        text = content.get("text", "")
        entities = self._extract_entities(text)
        relationships = self._extract_relationships(text)
        
        content = {
            "summary": text[:100] + "..." if len(text) > 100 else text,
            "entities": entities,
            "relationships": relationships,
            "semantic_hash": hashlib.md5(text.encode()).hexdigest(),
            "compression_level": "graph_node"
        }
        
        compressed_fragment = MemoryFragment(
            fragment_id=fragment.fragment_id,
            type=fragment.type,
            content=content,
            context_tags=fragment.context_tags,
            strength=fragment.strength * 0.8,  # Moderate strength reduction
            memory_tier=3,
            last_accessed=fragment.last_accessed,
            access_count=fragment.access_count,
            source_interactions=fragment.source_interactions,
            created_at=fragment.created_at
        )
        
        return compressed_fragment
    
    async def _compress_episodic(self, fragment: MemoryFragment) -> MemoryFragment:
        """Compress for episodic memory with importance preservation"""
        content = fragment.content.copy()
        text = content.get("text", "")
        
        # Create episodic summary with key events and context
        content = {
            "episode_summary": self._create_episode_summary(text),
            "key_entities": self._extract_entities(text),
            "temporal_context": content.get("temporal_context", {}),
            "emotional_markers": self._extract_emotional_markers(text),
            "importance_indicators": fragment.context_tags,
            "compression_level": "episodic"
        }
        
        compressed_fragment = MemoryFragment(
            fragment_id=fragment.fragment_id,
            type="episodic",  # Force episodic type
            content=content,
            context_tags=fragment.context_tags,
            strength=fragment.strength * 0.7,  # Higher strength reduction
            memory_tier=4,
            last_accessed=fragment.last_accessed,
            access_count=fragment.access_count,
            source_interactions=fragment.source_interactions,
            created_at=fragment.created_at
        )
        
        return compressed_fragment
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified)"""
        # This would use NER or more sophisticated entity extraction
        # Simple implementation for now
        words = text.split()
        entities = [word for word in words if word[0].isupper() and len(word) > 2]
        return list(set(entities))[:10]  # Limit to 10 entities
    
    def _extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships from text (simplified)"""
        # This would use dependency parsing or relationship extraction
        return []  # Placeholder
    
    def _create_episode_summary(self, text: str) -> str:
        """Create episodic summary of text"""
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
        
        # Simple summarization: first sentence + key phrases
        return f"{sentences[0]}. [Episode with {len(sentences)} interactions]"
    
    def _extract_emotional_markers(self, text: str) -> List[str]:
        """Extract emotional markers from text"""
        emotional_words = [
            "happy", "sad", "angry", "excited", "worried", "love", "hate",
            "important", "urgent", "critical", "remember", "forget"
        ]
        
        markers = []
        text_lower = text.lower()
        for word in emotional_words:
            if word in text_lower:
                markers.append(word)
        
        return markers

class MemoryTransitionManager:
    """Main transition manager coordinating all tier promotions"""
    
    def __init__(self, 
                 surreal_memory: Optional[SurrealMemory] = None,
                 transition_interval: int = 300):  # 5 minutes
        
        self.surreal_memory = surreal_memory
        self.transition_interval = transition_interval
        self.last_transition_run = time.time()
        
        # Transition thresholds (in seconds)
        self.tier_thresholds = {
            1: 5 * 60,      # 5 minutes: Working → Short-term
            2: 2 * 3600,    # 2 hours: Short-term → Long-term  
            3: 24 * 3600,   # 24 hours: Long-term → Episodic
        }
        
        # Importance thresholds for early promotion
        self.importance_thresholds = {
            1: 0.8,  # High importance can promote from Working early
            2: 0.7,  # Medium-high for Short-term → Long-term
            3: 0.6,  # Medium for Long-term → Episodic
        }
        
        self.importance_calculator = ImportanceCalculator()
        self.semantic_compressor = SemanticCompressor()
        
        # Transition statistics
        self.transition_stats = {
            "total_transitions": 0,
            "transitions_by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
            "early_promotions": 0,
            "compression_ratios": []
        }
        
        logger.info("Memory transition manager initialized")
    
    async def run_transition_cycle(self, working_fragments: List[MemoryFragment]):
        """Run complete transition cycle across all tiers"""
        current_time = time.time()
        
        if current_time - self.last_transition_run < self.transition_interval:
            return  # Too soon for another cycle
        
        logger.info("Running memory transition cycle")
        transition_start = time.time()
        
        # Step 1: Identify transition candidates
        candidates = await self._identify_transition_candidates(working_fragments)
        
        if not candidates:
            logger.debug("No fragments ready for transition")
            self.last_transition_run = current_time
            return
        
        # Step 2: Process transitions by target tier
        transitions_by_tier = {}
        for candidate in candidates:
            tier = candidate.target_tier
            if tier not in transitions_by_tier:
                transitions_by_tier[tier] = []
            transitions_by_tier[tier].append(candidate)
        
        # Step 3: Execute transitions
        total_transitions = 0
        for tier, tier_candidates in transitions_by_tier.items():
            transitions = await self._execute_tier_transitions(tier_candidates, tier)
            total_transitions += transitions
            self.transition_stats["transitions_by_tier"][tier] += transitions
        
        # Update statistics
        self.transition_stats["total_transitions"] += total_transitions
        self.last_transition_run = current_time
        
        transition_time = (time.time() - transition_start) * 1000
        logger.info(f"Completed {total_transitions} transitions in {transition_time:.1f}ms")
    
    async def _identify_transition_candidates(self, 
                                           working_fragments: List[MemoryFragment]) -> List[TransitionCandidate]:
        """Identify fragments ready for tier transition"""
        candidates = []
        current_time = time.time()
        
        for fragment in working_fragments:
            age_seconds = current_time - fragment.created_at
            
            # Calculate importance score
            importance_metrics = await self.importance_calculator.calculate_importance(
                fragment, working_fragments
            )
            importance_score = self.importance_calculator.calculate_final_importance_score(
                importance_metrics
            )
            
            # Determine target tier based on age and importance
            target_tier = self._determine_target_tier(
                fragment.memory_tier, age_seconds, importance_score
            )
            
            if target_tier > fragment.memory_tier:
                # Calculate transition confidence
                confidence = self._calculate_transition_confidence(
                    fragment, target_tier, age_seconds, importance_score
                )
                
                candidate = TransitionCandidate(
                    fragment=fragment,
                    current_tier=fragment.memory_tier,
                    target_tier=target_tier,
                    importance_score=importance_score,
                    age_seconds=age_seconds,
                    transition_confidence=confidence
                )
                
                candidates.append(candidate)
                logger.debug(
                    f"Transition candidate: {fragment.fragment_id} "
                    f"Tier {fragment.memory_tier}→{target_tier} "
                    f"(importance: {importance_score:.3f})"
                )
        
        # Sort by transition confidence
        candidates.sort(key=lambda c: c.transition_confidence, reverse=True)
        
        return candidates
    
    def _determine_target_tier(self, 
                              current_tier: int, 
                              age_seconds: float, 
                              importance_score: float) -> int:
        """Determine appropriate target tier for fragment"""
        
        # Check for early promotion based on importance
        if importance_score >= self.importance_thresholds.get(current_tier, 1.0):
            early_tier = min(4, current_tier + 1)
            if age_seconds >= self.tier_thresholds[current_tier] * 0.5:  # At least 50% of normal time
                self.transition_stats["early_promotions"] += 1
                return early_tier
        
        # Normal age-based promotion
        for tier in [1, 2, 3]:
            if current_tier == tier and age_seconds >= self.tier_thresholds[tier]:
                return tier + 1
        
        return current_tier  # No promotion needed
    
    def _calculate_transition_confidence(self, 
                                       fragment: MemoryFragment,
                                       target_tier: int,
                                       age_seconds: float,
                                       importance_score: float) -> float:
        """Calculate confidence score for transition"""
        
        # Base confidence from age
        required_age = self.tier_thresholds.get(fragment.memory_tier, float('inf'))
        age_confidence = min(1.0, age_seconds / required_age)
        
        # Importance-based confidence boost
        importance_confidence = importance_score
        
        # Fragment strength consideration
        strength_confidence = fragment.strength
        
        # Access pattern confidence
        access_confidence = min(1.0, fragment.access_count / 5)  # Normalize to typical access pattern
        
        # Weighted combination
        confidence = (
            age_confidence * 0.4 +
            importance_confidence * 0.3 +
            strength_confidence * 0.2 +
            access_confidence * 0.1
        )
        
        return max(0.0, min(1.0, confidence))
    
    async def _execute_tier_transitions(self, 
                                      candidates: List[TransitionCandidate],
                                      target_tier: int) -> int:
        """Execute transitions for specific target tier"""
        
        if not candidates:
            return 0
        
        logger.info(f"Executing {len(candidates)} transitions to tier {target_tier}")
        
        successful_transitions = 0
        
        for candidate in candidates:
            try:
                # Step 1: Compress fragment for target tier
                compressed_fragment = await self.semantic_compressor.compress_for_tier(
                    candidate.fragment, target_tier
                )
                
                # Step 2: Store in appropriate tier (SurrealDB for tiers 2-4)
                if target_tier > 1 and self.surreal_memory and SURREALDB_AVAILABLE:
                    success = await self._store_in_surreal_tier(compressed_fragment, target_tier)
                    if success:
                        successful_transitions += 1
                        
                        # Calculate compression ratio if text was compressed
                        original_size = len(candidate.fragment.content.get("text", ""))
                        compressed_size = len(compressed_fragment.content.get("text", 
                                              compressed_fragment.content.get("summary", "")))
                        if original_size > 0:
                            ratio = compressed_size / original_size
                            self.transition_stats["compression_ratios"].append(ratio)
                else:
                    # For tier 1 or when SurrealDB unavailable, just update in memory
                    successful_transitions += 1
                
            except Exception as e:
                logger.warning(f"Transition failed for fragment {candidate.fragment.fragment_id}: {e}")
        
        return successful_transitions
    
    async def _store_in_surreal_tier(self, 
                                   fragment: MemoryFragment, 
                                   tier: int) -> bool:
        """Store fragment in SurrealDB tier"""
        try:
            if not self.surreal_memory:
                return False
            
            # Convert fragment to SurrealDB format
            fragment_data = fragment.to_dict()
            fragment_data["memory_tier"] = tier
            fragment_data["stored_at"] = time.time()
            
            # Insert into fragments table
            query = """
                INSERT INTO fragments {
                    fragment_id: $fragment_id,
                    type: $type,
                    content: $content,
                    context_tags: $context_tags,
                    strength: $strength,
                    memory_tier: $memory_tier,
                    last_accessed: $last_accessed,
                    access_count: $access_count,
                    source_interactions: $source_interactions,
                    created_at: $created_at,
                    stored_at: time::now()
                }
            """
            
            await self.surreal_memory.db.query(query, fragment_data)
            
            logger.debug(f"Stored fragment {fragment.fragment_id} in tier {tier}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to store fragment in SurrealDB tier {tier}: {e}")
            return False
    
    def get_transition_stats(self) -> Dict[str, Any]:
        """Get memory transition statistics"""
        stats = self.transition_stats.copy()
        
        # Calculate average compression ratio
        if self.transition_stats["compression_ratios"]:
            stats["avg_compression_ratio"] = np.mean(self.transition_stats["compression_ratios"])
            stats["compression_variance"] = np.var(self.transition_stats["compression_ratios"])
        else:
            stats["avg_compression_ratio"] = 1.0
            stats["compression_variance"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset transition statistics"""
        self.transition_stats = {
            "total_transitions": 0,
            "transitions_by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
            "early_promotions": 0,
            "compression_ratios": []
        }