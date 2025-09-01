"""
Pillar 3 Enhancement: The Inference Engine Worker

Responsibility: Background reasoning and maintenance of the knowledge graph.
Runs inference, detects contradictions, and maintains data quality without
impacting real-time user interactions.

This runs as an independent background process that:
- Applies transitive inference rules
- Detects and resolves contradictions  
- Maintains memory decay
- Optimizes knowledge structure
- Reports on cognitive health
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger
from memory.surreal_connection import SurrealDBConnectionManager


class InferenceEngine:
    """
    Background reasoning engine for the knowledge graph
    
    Runs sophisticated inference and maintenance without blocking user interactions.
    Implements the "sleeping mind" - continuous processing during downtime.
    """
    
    def __init__(self, connection_manager: SurrealDBConnectionManager):
        self.db = connection_manager
        self.running = False
        self.cycle_count = 0
        
        # Configurable intervals (in seconds)
        self.intervals = {
            'quick_inference': 300,      # 5 minutes - light inference
            'deep_inference': 3600,      # 1 hour - comprehensive inference  
            'contradiction_check': 1800, # 30 minutes - detect conflicts
            'memory_maintenance': 7200,  # 2 hours - decay and cleanup
            'health_report': 86400       # 24 hours - system health report
        }
        
        # Performance tracking
        self.stats = {
            'inference_cycles': 0,
            'facts_inferred': 0,
            'contradictions_found': 0,
            'contradictions_resolved': 0,
            'memories_decayed': 0,
            'last_full_cycle': None,
            'average_cycle_time': 0.0
        }
        
        self.last_runs = {}
    
    async def start_background_inference(self):
        """
        Start the inference engine as a background task
        
        This should be called once when the system starts up.
        """
        if self.running:
            logger.warning("Inference engine already running")
            return
        
        self.running = True
        logger.info("🧠 Starting Cognitive Inference Engine")
        
        # Start all background tasks
        tasks = [
            asyncio.create_task(self._quick_inference_loop()),
            asyncio.create_task(self._deep_inference_loop()),
            asyncio.create_task(self._contradiction_check_loop()),
            asyncio.create_task(self._memory_maintenance_loop()),
            asyncio.create_task(self._health_report_loop())
        ]
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("🧠 Inference engine stopped")
            self.running = False
        except Exception as e:
            logger.error(f"🧠 Inference engine crashed: {e}")
            self.running = False
    
    async def stop_inference_engine(self):
        """Gracefully stop the inference engine"""
        logger.info("🧠 Stopping inference engine...")
        self.running = False
        
        # Give a moment for loops to exit gracefully
        await asyncio.sleep(1)
        
        # Report final stats
        logger.info("🧠 Final inference engine stats:")
        for key, value in self.stats.items():
            logger.info(f"   {key}: {value}")
    
    async def _quick_inference_loop(self):
        """
        Quick inference cycle - runs every 5 minutes
        
        Handles lightweight reasoning tasks that don't require deep analysis.
        """
        while self.running:
            try:
                if self._should_run('quick_inference'):
                    start_time = time.time()
                    logger.debug("🔄 Starting quick inference cycle...")
                    
                    # Run lightweight transitive inference
                    results = await self._run_quick_inference()
                    
                    cycle_time = time.time() - start_time
                    self.stats['inference_cycles'] += 1
                    self.stats['facts_inferred'] += results.get('new_facts', 0)
                    self._update_average_cycle_time(cycle_time)
                    
                    logger.debug(f"🔄 Quick inference complete: {results.get('new_facts', 0)} new facts in {cycle_time:.2f}s")
                    self.last_runs['quick_inference'] = time.time()
                
                # Sleep for a short interval to check if we should run
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Quick inference cycle failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _deep_inference_loop(self):
        """
        Deep inference cycle - runs every hour
        
        Comprehensive reasoning including complex transitive chains and semantic analysis.
        """
        while self.running:
            try:
                if self._should_run('deep_inference'):
                    start_time = time.time()
                    logger.info("🧠 Starting deep inference cycle...")
                    
                    results = await self._run_deep_inference()
                    
                    cycle_time = time.time() - start_time
                    self.stats['facts_inferred'] += results.get('new_facts', 0)
                    
                    logger.info(f"🧠 Deep inference complete: {results.get('new_facts', 0)} new facts in {cycle_time:.2f}s")
                    self.last_runs['deep_inference'] = time.time()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Deep inference cycle failed: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    async def _contradiction_check_loop(self):
        """
        Contradiction detection and resolution - runs every 30 minutes
        """
        while self.running:
            try:
                if self._should_run('contradiction_check'):
                    logger.debug("🔍 Checking for contradictions...")
                    
                    results = await self._detect_and_resolve_contradictions()
                    
                    self.stats['contradictions_found'] += results.get('found', 0)
                    self.stats['contradictions_resolved'] += results.get('resolved', 0)
                    
                    if results.get('found', 0) > 0:
                        logger.warning(f"🔍 Found {results['found']} contradictions, resolved {results['resolved']}")
                    else:
                        logger.debug("🔍 No contradictions found")
                    
                    self.last_runs['contradiction_check'] = time.time()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Contradiction check failed: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes before retry
    
    async def _memory_maintenance_loop(self):
        """
        Memory decay and cleanup - runs every 2 hours
        """
        while self.running:
            try:
                if self._should_run('memory_maintenance'):
                    logger.debug("🧹 Running memory maintenance...")
                    
                    results = await self._run_memory_maintenance()
                    
                    self.stats['memories_decayed'] += results.get('decayed', 0)
                    
                    logger.debug(f"🧹 Memory maintenance complete: {results.get('decayed', 0)} memories processed")
                    self.last_runs['memory_maintenance'] = time.time()
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Memory maintenance failed: {e}")
                await asyncio.sleep(7200)  # Wait 2 hours before retry
    
    async def _health_report_loop(self):
        """
        System health reporting - runs every 24 hours
        """
        while self.running:
            try:
                if self._should_run('health_report'):
                    logger.info("📊 Generating cognitive health report...")
                    
                    report = await self._generate_health_report()
                    
                    logger.info("📊 Cognitive System Health Report:")
                    for section, data in report.items():
                        logger.info(f"   {section}: {data}")
                    
                    self.stats['last_full_cycle'] = datetime.now().isoformat()
                    self.last_runs['health_report'] = time.time()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Health report failed: {e}")
                await asyncio.sleep(86400)  # Wait 24 hours before retry
    
    def _should_run(self, task_name: str) -> bool:
        """Check if enough time has passed to run a task"""
        last_run = self.last_runs.get(task_name, 0)
        interval = self.intervals[task_name]
        return (time.time() - last_run) >= interval
    
    def _update_average_cycle_time(self, cycle_time: float):
        """Update the rolling average cycle time"""
        if self.stats['average_cycle_time'] == 0:
            self.stats['average_cycle_time'] = cycle_time
        else:
            # Simple exponential moving average
            self.stats['average_cycle_time'] = (
                0.9 * self.stats['average_cycle_time'] + 
                0.1 * cycle_time
            )
    
    async def _run_quick_inference(self) -> Dict[str, int]:
        """Run lightweight transitive inference"""
        try:
            await self.db.ensure_connected()
            
            # Run transitive inference on key relations
            relations_to_infer = ['is_a', 'located_in', 'part_of']
            total_new_facts = 0
            
            for relation in relations_to_infer:
                try:
                    result = await self.db.db.query(f"RETURN fn::infer_transitive_facts('{relation}');")
                    new_facts = result[0]['result'] if result and result[0] else 0
                    total_new_facts += new_facts
                    
                    if new_facts > 0:
                        logger.debug(f"   Inferred {new_facts} new {relation} facts")
                        
                except Exception as e:
                    logger.warning(f"Failed to infer {relation} facts: {e}")
            
            return {'new_facts': total_new_facts}
            
        except Exception as e:
            logger.error(f"Quick inference failed: {e}")
            return {'new_facts': 0}
    
    async def _run_deep_inference(self) -> Dict[str, int]:
        """Run comprehensive inference including complex chains"""
        try:
            await self.db.ensure_connected()
            
            # Run all available transitive relations
            result = await self.db.db.query("""
                LET $relation_types = (SELECT name FROM relation_types WHERE properties.transitive = true);
                LET $total_inferred = 0;
                
                FOR $rel_type IN $relation_types {
                    LET $inferred = fn::infer_transitive_facts($rel_type.name);
                    LET $total_inferred = $total_inferred + $inferred;
                };
                
                RETURN $total_inferred;
            """)
            
            new_facts = result[0]['result'] if result and result[0] else 0
            
            return {'new_facts': new_facts}
            
        except Exception as e:
            logger.error(f"Deep inference failed: {e}")
            return {'new_facts': 0}
    
    async def _detect_and_resolve_contradictions(self) -> Dict[str, int]:
        """Detect and resolve contradictory facts"""
        try:
            await self.db.ensure_connected()
            
            # Detect contradictions
            contradictions_result = await self.db.db.query("RETURN fn::detect_contradictions();")
            contradictions = contradictions_result[0]['result'] if contradictions_result and contradictions_result[0] else []
            
            found_count = len(contradictions)
            
            if found_count == 0:
                return {'found': 0, 'resolved': 0}
            
            # Resolve contradictions
            resolved_result = await self.db.db.query("RETURN fn::resolve_contradictions();")
            resolved_count = resolved_result[0]['result'] if resolved_result and resolved_result[0] else 0
            
            return {'found': found_count, 'resolved': resolved_count}
            
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return {'found': 0, 'resolved': 0}
    
    async def _run_memory_maintenance(self) -> Dict[str, int]:
        """Run memory decay and cleanup processes"""
        try:
            await self.db.ensure_connected()
            
            # The advanced_memory_decay event should handle most of this automatically
            # But we can trigger manual cleanup for very old, unused memories
            
            result = await self.db.db.query("""
                LET $old_memories = (SELECT * FROM knowledge 
                    WHERE strength < 0.15 
                    AND time::now() - last_accessed > 7d 
                    AND extraction_method != 'manual'
                );
                
                LET $count = array::len($old_memories);
                
                FOR $memory IN $old_memories {
                    UPDATE $memory.id SET 
                        strength = 0.05,
                        metadata = object::set(metadata, 'archived_at', time::now());
                };
                
                RETURN $count;
            """)
            
            decayed_count = result[0]['result'] if result and result[0] else 0
            
            return {'decayed': decayed_count}
            
        except Exception as e:
            logger.error(f"Memory maintenance failed: {e}")
            return {'decayed': 0}
    
    async def _generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        try:
            await self.db.ensure_connected()
            
            # Get system statistics
            stats_query = """
                RETURN {
                    total_entities: (SELECT count() FROM entity)[0].count,
                    total_facts: (SELECT count() FROM knowledge)[0].count,
                    high_confidence_facts: (SELECT count() FROM knowledge WHERE confidence > 0.8)[0].count,
                    recent_facts: (SELECT count() FROM knowledge WHERE created_at > time::now() - 24h)[0].count,
                    weak_facts: (SELECT count() FROM knowledge WHERE strength < 0.3)[0].count,
                    source_breakdown: (SELECT extraction_method, count() as facts 
                                     FROM knowledge GROUP BY extraction_method),
                    avg_confidence: math::round((SELECT math::mean(confidence) FROM knowledge)[0] * 100) / 100,
                    relation_types_used: (SELECT predicate, count() as usage 
                                        FROM knowledge GROUP BY predicate ORDER BY usage DESC LIMIT 10)
                };
            """
            
            result = await self.db.db.query(stats_query)
            health_data = result[0]['result'] if result and result[0] else {}
            
            # Add inference engine stats
            health_data.update({
                'inference_stats': self.stats,
                'system_uptime': time.time() - (self.last_runs.get('health_report', time.time()) - 86400)
            })
            
            return health_data
            
        except Exception as e:
            logger.error(f"Health report generation failed: {e}")
            return {'error': str(e)}


# Global inference engine instance
_inference_engine = None

async def start_cognitive_inference(connection_manager: SurrealDBConnectionManager):
    """Start the global inference engine"""
    global _inference_engine
    
    if _inference_engine is None:
        _inference_engine = InferenceEngine(connection_manager)
    
    if not _inference_engine.running:
        # Start as a background task
        asyncio.create_task(_inference_engine.start_background_inference())
        logger.info("🧠 Cognitive inference engine started in background")
    else:
        logger.warning("🧠 Inference engine already running")

async def stop_cognitive_inference():
    """Stop the global inference engine"""
    global _inference_engine
    
    if _inference_engine and _inference_engine.running:
        await _inference_engine.stop_inference_engine()
        logger.info("🧠 Cognitive inference engine stopped")

def get_inference_stats() -> Dict[str, Any]:
    """Get current inference engine statistics"""
    global _inference_engine
    
    if _inference_engine:
        return _inference_engine.stats
    else:
        return {'error': 'Inference engine not initialized'}


# Integration hooks for the main application
async def initialize_cognitive_background(connection_manager: SurrealDBConnectionManager):
    """
    Initialize the cognitive background processes
    
    This should be called once during application startup.
    """
    logger.info("🧠 Initializing cognitive background systems...")
    
    try:
        # Start the inference engine
        await start_cognitive_inference(connection_manager)
        
        logger.info("🧠 Cognitive background systems initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize cognitive background: {e}")
        raise

async def shutdown_cognitive_background():
    """
    Shutdown cognitive background processes
    
    This should be called during application shutdown.
    """
    logger.info("🧠 Shutting down cognitive background systems...")
    
    try:
        await stop_cognitive_inference()
        logger.info("🧠 Cognitive background systems shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during cognitive background shutdown: {e}")