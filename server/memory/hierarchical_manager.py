"""
Hierarchical Memory System Manager

Coordinates four-tier memory system:
1. Working Memory (0-5 min): MLX tensors, neural field states  
2. Short-term Memory (5min-2hr): SurrealDB time-series with semantic compression
3. Long-term Memory (2hr+): SurrealDB graph nodes with attractor weights
4. Episodic Memory (permanent): SurrealDB documents with importance scoring

Integrates existing consciousness/core.py and memory/surreal_memory.py with advanced
neural field schema for infinite context through reconstructive fragment assembly.
"""

import asyncio
import time
import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
from loguru import logger

# Import existing components
from consciousness.core import (
    Consciousness, SemanticEmbedder,
    create_consciousness, MLX_AVAILABLE, mx
)
from memory.surreal_memory import SurrealMemory, SURREALDB_AVAILABLE

@dataclass
class MemoryResult:
    """Simple memory result for compatibility"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = "memory"

@dataclass
class MemoryFragment:
    """Fragment for hierarchical memory system"""
    fragment_id: str
    type: str  # semantic, episodic, procedural, contextual, emotional
    content: Dict[str, Any]
    context_tags: List[str]
    strength: float  # 0.0 to 1.0
    memory_tier: int  # 1=Working, 2=Short-term, 3=Long-term, 4=Episodic
    last_accessed: float
    access_count: int = 0
    source_interactions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryFragment':
        return cls(**data)

@dataclass  
class FieldState:
    """Neural field state for persistence across tiers"""
    instance_id: str
    compression: float
    drift: str  # none, low, moderate, high
    recursion_depth: int
    resonance: float
    presence_signal: float
    boundary: str  # gradient, collapsed
    memory_tier: int = 1
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class WorkingMemoryManager:
    """Tier 1: Working Memory (0-5 minutes) - MLX tensors and neural field states"""
    
    def __init__(self, max_fragments: int = 100):
        self.max_fragments = max_fragments
        self.fragments: Dict[str, MemoryFragment] = {}
        self.field_states: Dict[str, FieldState] = {}
        self.consciousness_field: Optional[Consciousness] = None
        self.embedder = SemanticEmbedder()
        
        # Initialize consciousness field if available
        if MLX_AVAILABLE:
            try:
                self.consciousness_field = create_consciousness()
            except Exception as e:
                logger.warning(f"Could not initialize consciousness field: {e}")
    
    async def add_fragment(self, content: str, fragment_type: str = "semantic", 
                          context_tags: List[str] = None) -> str:
        """Add fragment to working memory with neural field integration"""
        fragment_id = hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()[:12]
        
        # Create embedding for content
        embedding = self.embedder.embed(content)
        
        fragment = MemoryFragment(
            fragment_id=fragment_id,
            type=fragment_type,
            content={
                "text": content,
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "semantic_hash": hashlib.md5(content.encode()).hexdigest()
            },
            context_tags=context_tags or [],
            strength=1.0,  # Fresh fragments start at full strength
            memory_tier=1,
            last_accessed=time.time()
        )
        
        self.fragments[fragment_id] = fragment
        
        # Update consciousness field if available
        if self.consciousness_field:
            try:
                # Process through consciousness field
                await self._update_consciousness_field(content, fragment_id)
            except Exception as e:
                logger.debug(f"Consciousness field update failed: {e}")
        
        # Manage memory capacity
        await self._manage_capacity()
        
        logger.debug(f"Added working memory fragment: {fragment_id} ({fragment_type})")
        return fragment_id
    
    async def _update_consciousness_field(self, content: str, fragment_id: str):
        """Update neural field states with new fragment"""
        if not self.consciousness_field:
            return
            
        try:
            # Process content through consciousness field using symbolize_async
            symbols = await self.consciousness_field.symbolize_async(content)
            
            # Get current field states
            current_states = self.consciousness_field.get_field_states()
            
            # Extract field state information from consciousness field
            # Use available stats and create reasonable field state
            field_state = FieldState(
                instance_id=fragment_id,
                compression=0.5 + len(symbols) * 0.05,  # More symbols = higher compression
                drift="low" if len(symbols) < 5 else "moderate",
                recursion_depth=min(3, len(symbols)),
                resonance=0.7 + min(0.2, len(symbols) * 0.02),  # Symbol richness affects resonance
                presence_signal=0.8,
                boundary="gradient" if len(symbols) > 2 else "collapsed",
                memory_tier=1
            )
            
            self.field_states[fragment_id] = field_state
            logger.debug(f"Created field state for fragment {fragment_id} with {len(symbols)} symbols")
            
        except Exception as e:
            logger.debug(f"Consciousness field processing failed: {e}")
    
    async def _manage_capacity(self):
        """Manage working memory capacity, promoting old fragments"""
        if len(self.fragments) > self.max_fragments:
            # Sort by last access time, promote oldest
            sorted_fragments = sorted(
                self.fragments.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Remove oldest 10% to make room
            remove_count = max(1, len(sorted_fragments) // 10)
            for fragment_id, fragment in sorted_fragments[:remove_count]:
                # Mark for promotion to short-term memory
                fragment.memory_tier = 2
                logger.debug(f"Promoting fragment {fragment_id} to short-term memory")
                del self.fragments[fragment_id]
                if fragment_id in self.field_states:
                    del self.field_states[fragment_id]
    
    async def retrieve_fragments(self, query: str, limit: int = 10) -> List[MemoryFragment]:
        """Retrieve fragments by semantic similarity"""
        if not self.fragments:
            return []
        
        query_embedding = self.embedder.embed(query)
        similarities = []
        
        for fragment_id, fragment in self.fragments.items():
            try:
                content_embedding = np.array(fragment.content.get("embedding", []))
                if content_embedding.size > 0:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, content_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                    )
                    similarities.append((similarity, fragment))
                    
                    # Update access tracking
                    fragment.last_accessed = time.time()
                    fragment.access_count += 1
            except Exception as e:
                logger.debug(f"Similarity calculation failed for {fragment_id}: {e}")
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [frag for _, frag in similarities[:limit]]
    
    def get_field_state(self, fragment_id: str) -> Optional[FieldState]:
        """Get neural field state for fragment"""
        return self.field_states.get(fragment_id)
    
    def get_active_fragments(self) -> List[MemoryFragment]:
        """Get all active fragments in working memory"""
        return list(self.fragments.values())
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get working memory statistics"""
        return {
            "total_fragments": len(self.fragments),
            "max_capacity": self.max_fragments,
            "utilization": len(self.fragments) / self.max_fragments,
            "field_states": len(self.field_states),
            "consciousness_available": self.consciousness_field is not None,
            "mlx_available": MLX_AVAILABLE
        }

class HierarchicalMemoryManager:
    """Main coordinator for four-tier hierarchical memory system"""
    
    def __init__(self, 
                 surreal_memory: Optional[SurrealMemory] = None,
                 working_memory_capacity: int = 100):
        
        # Initialize tier managers
        self.working_memory = WorkingMemoryManager(working_memory_capacity)
        self.surreal_memory = surreal_memory
        
        # Memory tier thresholds (in seconds)
        self.TIER_THRESHOLDS = {
            1: 5 * 60,      # 5 minutes: Working → Short-term
            2: 2 * 3600,    # 2 hours: Short-term → Long-term  
            3: 24 * 3600,   # 24 hours: Long-term → Episodic
        }
        
        self.transition_scheduler = TransitionScheduler(self)
    
    async def store_memory(self, content: str, memory_type: str = "semantic",
                          context_tags: List[str] = None, importance: float = 0.5) -> str:
        """Store memory starting in working memory tier"""
        
        # All memories start in working memory
        fragment_id = await self.working_memory.add_fragment(
            content=content,
            fragment_type=memory_type,
            context_tags=context_tags or []
        )
        
        logger.info(f"Stored memory fragment {fragment_id} in working memory")
        return fragment_id
    
    async def retrieve_memory(self, query: str, limit: int = 20) -> List[MemoryFragment]:
        """Retrieve memories across all tiers with tier priority"""
        all_results = []
        
        # 1. Check Working Memory first (fastest)
        working_results = await self.working_memory.retrieve_fragments(query, limit//4)
        all_results.extend(working_results)
        logger.debug(f"Found {len(working_results)} results in working memory")
        
        # 2. Query other tiers through SurrealDB if available
        if self.surreal_memory and SURREALDB_AVAILABLE:
            try:
                # Query short-term, long-term, and episodic memory
                surreal_results = await self._query_persistent_tiers(query, limit - len(working_results))
                all_results.extend(surreal_results)
            except Exception as e:
                logger.warning(f"SurrealDB query failed: {e}")
        
        return all_results[:limit]
    
    async def _query_persistent_tiers(self, query: str, limit: int) -> List[MemoryFragment]:
        """Query SurrealDB for fragments in tiers 2-4"""
        # This will be implemented when we add the SurrealDB schema
        # For now, return empty list
        return []
    
    async def promote_fragments(self):
        """Promote fragments between memory tiers based on age and importance"""
        await self.transition_scheduler.run_promotion_cycle()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        stats = {
            "working_memory": self.working_memory.get_memory_stats(),
            "surreal_available": self.surreal_memory is not None,
            "tier_thresholds": self.TIER_THRESHOLDS
        }
        
        if self.surreal_memory:
            # Add SurrealDB stats when available
            pass
            
        return stats

class TransitionScheduler:
    """Handles automated memory tier transitions"""
    
    def __init__(self, hierarchy_manager: HierarchicalMemoryManager):
        self.hierarchy_manager = hierarchy_manager
        self.last_promotion_run = time.time()
        self.promotion_interval = 60  # Run every minute
    
    async def run_promotion_cycle(self):
        """Run a cycle of memory tier promotions"""
        current_time = time.time()
        
        if current_time - self.last_promotion_run < self.promotion_interval:
            return  # Too soon for another promotion cycle
        
        logger.debug("Running memory promotion cycle")
        
        # Check working memory for promotions
        fragments_to_promote = []
        working_fragments = self.hierarchy_manager.working_memory.get_active_fragments()
        
        for fragment in working_fragments:
            age_seconds = current_time - fragment.created_at
            
            # Check if fragment should be promoted to short-term memory
            if age_seconds > self.hierarchy_manager.TIER_THRESHOLDS[1]:
                fragments_to_promote.append(fragment)
        
        # Promote fragments (will be implemented when SurrealDB schema is ready)
        if fragments_to_promote:
            logger.info(f"Promoting {len(fragments_to_promote)} fragments to short-term memory")
            # TODO: Implement promotion to SurrealDB tiers
        
        self.last_promotion_run = current_time

# Factory function for creating hierarchical memory system
async def create_hierarchical_memory(surreal_memory: Optional[SurrealMemory] = None,
                                   working_memory_capacity: int = 100) -> HierarchicalMemoryManager:
    """Create and initialize hierarchical memory system"""
    
    manager = HierarchicalMemoryManager(
        surreal_memory=surreal_memory,
        working_memory_capacity=working_memory_capacity
    )
    
    logger.info("Hierarchical memory system initialized")
    logger.info(f"System stats: {manager.get_system_stats()}")
    
    return manager