"""
Knowledge Evolution Service - Background service for adaptive knowledge graph

This service runs in the background and continuously evolves the knowledge graph:
1. Learns predicate patterns from new conversations
2. Merges similar predicates automatically  
3. Cleans up weak or unused knowledge
4. Detects emerging conversation patterns
5. Adapts clustering thresholds based on usage

It integrates with the existing consciousness engine and memory decay systems.
"""

import asyncio
from typing import Optional
from loguru import logger
from dataclasses import dataclass

try:
    from memory.adaptive_knowledge_graph import get_adaptive_kg, run_evolution_cycle
    ADAPTIVE_KG_AVAILABLE = True
except ImportError:
    logger.error("Adaptive knowledge graph not available")
    ADAPTIVE_KG_AVAILABLE = False

@dataclass
class EvolutionConfig:
    """Configuration for knowledge evolution"""
    evolution_interval_minutes: int = 5  # How often to run evolution
    min_facts_for_evolution: int = 10   # Minimum new facts before evolution
    enable_background_cleanup: bool = True
    enable_pattern_detection: bool = True
    enable_predicate_clustering: bool = True
    
    # Adaptive thresholds
    max_clusters: int = 100  # Prevent over-clustering
    min_cluster_usage: int = 3
    semantic_threshold_min: float = 0.6
    semantic_threshold_max: float = 0.9

class KnowledgeEvolutionService:
    """Background service that evolves the knowledge graph"""
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.is_running = False
        self.evolution_task = None
        self.new_facts_count = 0
        self.last_evolution_time = None
        
        if ADAPTIVE_KG_AVAILABLE:
            self.adaptive_kg = get_adaptive_kg()
            logger.info("🧬 Knowledge evolution service initialized")
        else:
            self.adaptive_kg = None
            logger.warning("⚠️ Knowledge evolution disabled - adaptive KG not available")
    
    async def start(self):
        """Start the background evolution service"""
        if not ADAPTIVE_KG_AVAILABLE or self.is_running:
            return
            
        self.is_running = True
        self.evolution_task = asyncio.create_task(self._evolution_loop())
        logger.info("🚀 Knowledge evolution service started")
    
    async def stop(self):
        """Stop the background evolution service"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.evolution_task:
            self.evolution_task.cancel()
            try:
                await self.evolution_task
            except asyncio.CancelledError:
                pass
            
        logger.info("⏹️ Knowledge evolution service stopped")
    
    def notify_new_fact(self):
        """Notify the service that a new fact was added"""
        self.new_facts_count += 1
        
        # Trigger immediate evolution if we have many new facts
        if self.new_facts_count >= self.config.min_facts_for_evolution * 3:
            logger.info(f"📈 Triggering immediate evolution due to {self.new_facts_count} new facts")
            asyncio.create_task(self._run_evolution_now())
    
    async def _evolution_loop(self):
        """Main evolution loop"""
        logger.info(f"🔄 Starting evolution loop (interval: {self.config.evolution_interval_minutes}min)")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config.evolution_interval_minutes * 60)
                
                if not self.is_running:
                    break
                    
                # Only evolve if we have enough new facts
                if self.new_facts_count >= self.config.min_facts_for_evolution:
                    await self._run_evolution_now()
                else:
                    logger.debug(f"🔍 Evolution skipped: only {self.new_facts_count} new facts")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evolution loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _run_evolution_now(self):
        """Run evolution cycle now"""
        if not self.adaptive_kg:
            return
            
        try:
            logger.info(f"🧬 Running evolution cycle (processed {self.new_facts_count} new facts)")
            
            # Step 1: Evolve the graph structure
            if self.config.enable_predicate_clustering:
                await self.adaptive_kg.evolve_graph_structure()
            
            # Step 2: Detect emerging patterns if enabled
            if self.config.enable_pattern_detection:
                patterns = await self.adaptive_kg.detect_emerging_patterns()
                if patterns:
                    logger.info(f"🌟 Detected {len(patterns)} emerging patterns")
            
            # Step 3: Background cleanup if enabled
            if self.config.enable_background_cleanup:
                await self.adaptive_kg.cleanup_weak_patterns()
            
            # Reset counter
            self.new_facts_count = 0
            self.last_evolution_time = asyncio.get_event_loop().time()
            
            logger.info("✅ Evolution cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
    
    async def get_evolution_stats(self) -> dict:
        """Get current evolution statistics"""
        if not self.adaptive_kg:
            return {"status": "disabled", "reason": "adaptive_kg_not_available"}
        
        stats = {
            "status": "running" if self.is_running else "stopped",
            "new_facts_pending": self.new_facts_count,
            "cluster_count": len(self.adaptive_kg.predicate_clusters),
            "semantic_threshold": self.adaptive_kg.semantic_similarity_threshold,
            "last_evolution": self.last_evolution_time,
            "config": {
                "evolution_interval_min": self.config.evolution_interval_minutes,
                "min_facts_threshold": self.config.min_facts_for_evolution,
                "clustering_enabled": self.config.enable_predicate_clustering,
                "pattern_detection_enabled": self.config.enable_pattern_detection,
                "cleanup_enabled": self.config.enable_background_cleanup
            }
        }
        
        return stats
    
    async def force_evolution(self):
        """Force evolution to run immediately"""
        logger.info("🔥 Forcing immediate evolution cycle")
        await self._run_evolution_now()
    
    async def update_config(self, new_config: EvolutionConfig):
        """Update configuration at runtime"""
        old_interval = self.config.evolution_interval_minutes
        self.config = new_config
        
        logger.info(f"⚙️ Evolution config updated")
        
        # Restart service if interval changed and we're running
        if (old_interval != new_config.evolution_interval_minutes and 
            self.is_running):
            await self.stop()
            await self.start()

# Global evolution service instance
_evolution_service: Optional[KnowledgeEvolutionService] = None

def get_evolution_service() -> KnowledgeEvolutionService:
    """Get the global knowledge evolution service instance"""
    global _evolution_service
    if _evolution_service is None:
        _evolution_service = KnowledgeEvolutionService()
    return _evolution_service

async def start_evolution_service():
    """Start the global evolution service"""
    service = get_evolution_service()
    await service.start()

async def stop_evolution_service():
    """Stop the global evolution service"""
    if _evolution_service:
        await _evolution_service.stop()

def notify_new_knowledge_fact():
    """Notify the service that a new knowledge fact was created"""
    if _evolution_service:
        _evolution_service.notify_new_fact()

# Integration function for surreal_connection.py
def on_knowledge_stored():
    """Called when knowledge is stored - notifies evolution service"""
    notify_new_knowledge_fact()

if __name__ == "__main__":
    # Test the evolution service
    async def test():
        service = KnowledgeEvolutionService()
        await service.start()
        
        # Simulate new facts
        for i in range(15):
            service.notify_new_fact()
            await asyncio.sleep(1)
        
        # Get stats
        stats = await service.get_evolution_stats()
        print(f"Evolution stats: {stats}")
        
        # Force evolution
        await service.force_evolution()
        
        await service.stop()
    
    asyncio.run(test())