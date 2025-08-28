"""
Enhanced SurrealDB Memory System for Hierarchical Architecture

Extends the existing surreal_memory.py with hierarchical capabilities:
- Four-tier memory management (Working/Short-term/Long-term/Episodic)
- Fragment-based storage with neural field integration
- Pattern recognition and attractor dynamics
- Cross-tier relationships and transitions
- Backward compatibility with existing FactsGraph/TapeStore interfaces

This maintains all existing functionality while adding hierarchical features.
"""

import os
import time
import asyncio
import json
import math
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from loguru import logger

# Import existing components
from memory.surreal_memory import (
    SurrealMemory, SurrealFact, SURREALDB_AVAILABLE, AsyncSurreal,
    DECAY_HALF_LIFE_S, PROMOTE_THRESH, DEMOTE_THRESH, EMA_ALPHA, MAX_FACTS
)

@dataclass  
class MemoryResult:
    """Memory result for enhanced surreal memory compatibility"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = "enhanced_surreal"

from memory.hierarchical_manager import MemoryFragment, FieldState
from memory.transition_manager import MemoryTransitionManager
from memory.fragment_retrieval import FastFragmentRetriever

class HierarchicalSurrealMemory(SurrealMemory):
    """Enhanced SurrealMemory with hierarchical memory tiers and neural field integration"""
    
    def __init__(self, 
                 db_url: str = "ws://127.0.0.1:8000/rpc",
                 username: str = "root", 
                 password: str = "slowcat_secure_2024",
                 namespace: str = "slowcat", 
                 database: str = "memory",
                 enable_hierarchical: bool = True,
                 max_working_fragments: int = 100):
        
        # Initialize parent SurrealMemory
        super().__init__(db_url, username, password, namespace, database)
        
        self.enable_hierarchical = enable_hierarchical
        self.max_working_fragments = max_working_fragments
        
        # Hierarchical components (initialized after connection)
        self.transition_manager: Optional[MemoryTransitionManager] = None
        self.fragment_retriever: Optional[FastFragmentRetriever] = None
        
        # Working memory cache (Tier 1)
        self.working_memory_cache: Dict[str, MemoryFragment] = {}
        self.field_states_cache: Dict[str, FieldState] = {}
        
        # Enhanced statistics
        self.hierarchical_stats = {
            "fragments_by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
            "total_retrievals": 0,
            "avg_retrieval_time_ms": 0.0,
            "tier_hit_rates": {1: 0, 2: 0, 3: 0, 4: 0}
        }
        
        logger.info(f"HierarchicalSurrealMemory initialized (hierarchical: {enable_hierarchical})")
    
    async def connect(self):
        """Connect and initialize hierarchical components"""
        await super().connect()
        
        if self.enable_hierarchical:
            # Initialize hierarchical components
            self.transition_manager = MemoryTransitionManager(surreal_memory=self)
            self.fragment_retriever = FastFragmentRetriever(surreal_memory=self)
            
            # Initialize hierarchical schema if needed
            await self._ensure_hierarchical_schema()
            
            logger.info("Hierarchical memory components initialized")
    
    async def _ensure_hierarchical_schema(self):
        """Ensure hierarchical schema tables exist"""
        try:
            # Check if hierarchical tables exist
            tables_result = await self.db.query("INFO FOR DB")
            
            # If fragments table doesn't exist, we need to run migration
            # This is a simplified check - production would be more thorough
            fragments_check = await self.db.query("SELECT * FROM fragments LIMIT 1")
            
            if not fragments_check or not fragments_check[0]:
                logger.info("Hierarchical schema not found, would run migration here")
                # In production, would run schema_migration.py automatically
                # For now, just log that migration is needed
        
        except Exception as e:
            logger.debug(f"Schema check failed (expected if not migrated): {e}")
    
    # ===== Enhanced Fragment Management =====
    
    async def store_fragment(self, 
                           content: str, 
                           fragment_type: str = "semantic",
                           context_tags: List[str] = None,
                           importance_score: float = 0.5,
                           field_state: Optional[FieldState] = None) -> str:
        """Store fragment in hierarchical memory system"""
        
        if not self.enable_hierarchical:
            # Fallback to parent implementation for facts
            subject, predicate, obj = self._parse_simple_fact(content)
            return await super().store_fact(subject, predicate, obj)
        
        fragment_id = f"frag_{int(time.time() * 1000)}"
        
        # Create memory fragment
        fragment = MemoryFragment(
            fragment_id=fragment_id,
            type=fragment_type,
            content={
                "text": content,
                "importance_score": importance_score,
                "semantic_hash": hash(content)
            },
            context_tags=context_tags or [],
            strength=1.0,  # New fragments start at full strength
            memory_tier=1,  # Always start in Working Memory
            last_accessed=time.time(),
            access_count=0
        )
        
        # Store in working memory cache
        self.working_memory_cache[fragment_id] = fragment
        
        # Store field state if provided
        if field_state:
            self.field_states_cache[fragment_id] = field_state
        
        # Update statistics
        self.hierarchical_stats["fragments_by_tier"][1] += 1
        
        logger.debug(f"Stored fragment {fragment_id} in working memory")
        return fragment_id
    
    async def retrieve_fragments(self, 
                               query: str, 
                               limit: int = 20,
                               tier_preference: List[int] = None) -> List[MemoryFragment]:
        """Retrieve fragments across hierarchical memory tiers"""
        
        if not self.enable_hierarchical:
            # Fallback to parent fact retrieval
            facts = await super().search_facts(query, limit)
            return self._convert_facts_to_fragments(facts)
        
        start_time = time.time()
        
        # Use fast fragment retriever
        if self.fragment_retriever:
            retrieval_results = await self.fragment_retriever.retrieve_fragments(
                query, limit, tier_preference
            )
            fragments = [result.fragment for result in retrieval_results]
        else:
            # Fallback to simple working memory search
            fragments = await self._simple_working_memory_search(query, limit)
        
        # Update access tracking for retrieved fragments
        for fragment in fragments:
            fragment.last_accessed = time.time()
            fragment.access_count += 1
            
            # Update in cache if present
            if fragment.fragment_id in self.working_memory_cache:
                self.working_memory_cache[fragment.fragment_id] = fragment
        
        # Update statistics
        retrieval_time_ms = (time.time() - start_time) * 1000
        self._update_retrieval_stats(retrieval_time_ms, len(fragments))
        
        return fragments
    
    async def _simple_working_memory_search(self, query: str, limit: int) -> List[MemoryFragment]:
        """Simple working memory search fallback"""
        results = []
        query_lower = query.lower()
        
        for fragment in self.working_memory_cache.values():
            content_text = fragment.content.get("text", "").lower()
            if query_lower in content_text:
                results.append(fragment)
                if len(results) >= limit:
                    break
        
        return results
    
    def _convert_facts_to_fragments(self, facts: List[SurrealFact]) -> List[MemoryFragment]:
        """Convert legacy facts to fragment format"""
        fragments = []
        
        for fact in facts:
            fragment_id = f"fact_{fact.subject}_{fact.predicate}_{hash(fact.value or '')}"
            
            fragment = MemoryFragment(
                fragment_id=fragment_id,
                type="semantic",
                content={
                    "text": f"{fact.subject} {fact.predicate} {fact.value or ''}",
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "value": fact.value
                },
                context_tags=["legacy_fact"] + (fact.tags or []),
                strength=fact.strength,
                memory_tier=2,  # Treat existing facts as short-term
                last_accessed=time.time(),
                access_count=fact.access_count,
                created_at=fact.created
            )
            
            fragments.append(fragment)
        
        return fragments
    
    def _parse_simple_fact(self, content: str) -> Tuple[str, str, str]:
        """Parse simple fact from content for legacy compatibility"""
        # Very simple parsing - would be enhanced in production
        words = content.split()
        if len(words) >= 3:
            return words[0], words[1], " ".join(words[2:])
        return "user", "said", content
    
    # ===== Memory Tier Management =====
    
    async def run_memory_transitions(self):
        """Run memory tier transitions"""
        if not self.enable_hierarchical or not self.transition_manager:
            return
        
        # Get working memory fragments for transition analysis
        working_fragments = list(self.working_memory_cache.values())
        
        if working_fragments:
            await self.transition_manager.run_transition_cycle(working_fragments)
            
            # Clean up promoted fragments from working memory
            await self._cleanup_promoted_fragments()
    
    async def _cleanup_promoted_fragments(self):
        """Remove promoted fragments from working memory cache"""
        # This would identify fragments that have been promoted and remove them
        # For now, implement simple cleanup based on age and capacity
        
        current_time = time.time()
        fragments_to_remove = []
        
        # Remove fragments older than 5 minutes (promoted to tier 2)
        for fragment_id, fragment in self.working_memory_cache.items():
            age_seconds = current_time - fragment.created_at
            if age_seconds > 5 * 60:  # 5 minutes
                fragments_to_remove.append(fragment_id)
        
        # Capacity management - remove oldest if over limit
        if len(self.working_memory_cache) > self.max_working_fragments:
            sorted_fragments = sorted(
                self.working_memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            excess_count = len(self.working_memory_cache) - self.max_working_fragments
            for fragment_id, _ in sorted_fragments[:excess_count]:
                fragments_to_remove.append(fragment_id)
        
        # Remove identified fragments
        for fragment_id in set(fragments_to_remove):
            if fragment_id in self.working_memory_cache:
                del self.working_memory_cache[fragment_id]
            if fragment_id in self.field_states_cache:
                del self.field_states_cache[fragment_id]
        
        if fragments_to_remove:
            logger.debug(f"Cleaned up {len(fragments_to_remove)} promoted fragments")
    
    # ===== Enhanced Query Methods =====
    
    async def hierarchical_search(self, 
                                query: str, 
                                search_type: str = "hybrid",
                                limit: int = 20) -> List[MemoryResult]:
        """Enhanced search across hierarchical memory with multiple strategies"""
        
        if not self.enable_hierarchical:
            # Fallback to parent search
            return await super().search_facts(query, limit)
        
        fragments = await self.retrieve_fragments(query, limit)
        
        # Convert fragments to MemoryResult format for compatibility
        results = []
        for fragment in fragments:
            result = MemoryResult(
                content=fragment.content.get("text", ""),
                metadata={
                    "fragment_id": fragment.fragment_id,
                    "type": fragment.type,
                    "memory_tier": fragment.memory_tier,
                    "strength": fragment.strength,
                    "context_tags": fragment.context_tags,
                    "access_count": fragment.access_count
                },
                score=fragment.strength,
                source="hierarchical"
            )
            results.append(result)
        
        return results
    
    async def get_field_state(self, fragment_id: str) -> Optional[FieldState]:
        """Get neural field state for fragment"""
        return self.field_states_cache.get(fragment_id)
    
    async def update_field_state(self, fragment_id: str, field_state: FieldState):
        """Update neural field state for fragment"""
        self.field_states_cache[fragment_id] = field_state
        
        # If hierarchical schema is available, persist to database
        if self.connected and self.enable_hierarchical:
            try:
                await self._persist_field_state(fragment_id, field_state)
            except Exception as e:
                logger.debug(f"Field state persistence failed: {e}")
    
    async def _persist_field_state(self, fragment_id: str, field_state: FieldState):
        """Persist field state to SurrealDB"""
        field_data = field_state.to_dict()
        field_data["fragment_id"] = fragment_id
        
        query = """
            INSERT INTO field_states {
                instance_id: $instance_id,
                fragment_id: $fragment_id,
                compression: $compression,
                drift: $drift,
                recursion_depth: $recursion_depth,
                resonance: $resonance,
                presence_signal: $presence_signal,
                boundary: $boundary,
                memory_tier: $memory_tier,
                updated_at: time::now()
            }
        """
        
        await self.db.query(query, field_data)
    
    # ===== Statistics and Monitoring =====
    
    def _update_retrieval_stats(self, retrieval_time_ms: float, result_count: int):
        """Update retrieval performance statistics"""
        self.hierarchical_stats["total_retrievals"] += 1
        
        # Update rolling average
        total = self.hierarchical_stats["total_retrievals"]
        current_avg = self.hierarchical_stats["avg_retrieval_time_ms"]
        
        new_avg = ((current_avg * (total - 1)) + retrieval_time_ms) / total
        self.hierarchical_stats["avg_retrieval_time_ms"] = new_avg
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        base_stats = super().get_stats() if hasattr(super(), 'get_stats') else {}
        
        enhanced_stats = {
            **base_stats,
            "hierarchical_enabled": self.enable_hierarchical,
            "working_memory_fragments": len(self.working_memory_cache),
            "field_states_cached": len(self.field_states_cache),
            "fragments_by_tier": self.hierarchical_stats["fragments_by_tier"].copy(),
            "avg_retrieval_time_ms": self.hierarchical_stats["avg_retrieval_time_ms"],
            "total_hierarchical_retrievals": self.hierarchical_stats["total_retrievals"],
            "performance_target_met": self.hierarchical_stats["avg_retrieval_time_ms"] <= 50.0
        }
        
        # Add transition manager stats if available
        if self.transition_manager:
            enhanced_stats["transition_stats"] = self.transition_manager.get_transition_stats()
        
        # Add fragment retriever stats if available
        if self.fragment_retriever:
            enhanced_stats["retrieval_performance"] = self.fragment_retriever.get_performance_stats()
        
        return enhanced_stats
    
    # ===== Backward Compatibility =====
    
    async def store_fact(self, subject: str, predicate: str, obj: str, 
                        strength: float = 1.0) -> str:
        """Backward compatible fact storage"""
        if self.enable_hierarchical:
            # Store as semantic fragment
            content = f"{subject} {predicate} {obj}"
            return await self.store_fragment(
                content=content, 
                fragment_type="semantic",
                context_tags=["fact"],
                importance_score=strength
            )
        else:
            return await super().store_fact(subject, predicate, obj, strength)
    
    async def search_facts(self, query: str, limit: int = 50) -> List[MemoryResult]:
        """Backward compatible fact search"""
        if self.enable_hierarchical:
            return await self.hierarchical_search(query, limit=limit)
        else:
            return await super().search_facts(query, limit)

# Factory function for creating enhanced memory system
async def create_hierarchical_surreal_memory(
    db_url: str = None,
    username: str = None, 
    password: str = None,
    namespace: str = None,
    database: str = None,
    enable_hierarchical: bool = True,
    max_working_fragments: int = 100
) -> HierarchicalSurrealMemory:
    """Create and initialize hierarchical SurrealDB memory system"""
    
    # Use environment defaults if not provided
    db_url = db_url or os.getenv("SURREALDB_URL", "ws://127.0.0.1:8000/rpc")
    username = username or os.getenv("SURREALDB_USER", "root")
    password = password or os.getenv("SURREALDB_PASS", "slowcat_secure_2024")
    namespace = namespace or os.getenv("SURREALDB_NAMESPACE", "slowcat")
    database = database or os.getenv("SURREALDB_DATABASE", "memory")
    
    memory = HierarchicalSurrealMemory(
        db_url=db_url,
        username=username,
        password=password,
        namespace=namespace,
        database=database,
        enable_hierarchical=enable_hierarchical,
        max_working_fragments=max_working_fragments
    )
    
    await memory.connect()
    
    logger.info("Hierarchical SurrealDB memory system ready")
    return memory