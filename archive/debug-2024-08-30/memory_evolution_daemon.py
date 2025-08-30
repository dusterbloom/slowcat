#!/usr/bin/env python3
"""
Memory Evolution Daemon - Background process for natural memory decay

This daemon runs periodically to:
1. Update memory strength based on decay calculations
2. Clean up extremely weak fragments
3. Maintain healthy memory ecosystem

Usage:
    python memory/memory_evolution_daemon.py --interval 3600  # Run every hour
    python memory/memory_evolution_daemon.py --once           # Run once and exit
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timezone
from loguru import logger
from memory.surreal_connection import SurrealConnectionManager


class MemoryEvolutionDaemon:
    """Background daemon for memory evolution and decay processing"""
    
    def __init__(self, interval_seconds: int = 3600, batch_size: int = 100):
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self.connection_manager = SurrealConnectionManager()
        self.running = True
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def process_memory_decay(self) -> dict:
        """Process memory decay for a batch of facts"""
        try:
            await self.connection_manager.ensure_connected()
            
            # Update memory strength based on decay calculations
            decay_result = await self.connection_manager.db.query(
                "RETURN fn::decay_background_memories($batch_size);",
                {'batch_size': self.batch_size}
            )
            
            # Clean up fragments (strength < 0.1)
            cleanup_result = await self.connection_manager.db.query(
                "RETURN fn::cleanup_fragments($limit);",
                {'limit': self.batch_size // 4}  # Clean fewer than we process
            )
            
            processed = decay_result[0] if decay_result else 0
            cleaned = cleanup_result[0] if cleanup_result else 0
            
            stats = {
                'processed': processed,
                'cleaned': cleaned,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'batch_size': self.batch_size
            }
            
            logger.info(f"🧠 Memory evolution: processed {processed} facts, cleaned {cleaned} fragments")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Memory decay processing failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now(timezone.utc).isoformat()}
    
    async def get_memory_stats(self) -> dict:
        """Get current memory system statistics"""
        try:
            await self.connection_manager.ensure_connected()
            
            # Get knowledge counts by strength category
            result = await self.connection_manager.db.query("""
                SELECT 
                    count() AS total_facts,
                    count(WHERE fn::calculate_memory_decay(created_at, last_accessed, access_count) > 0.7) AS strong_facts,
                    count(WHERE fn::calculate_memory_decay(created_at, last_accessed, access_count) BETWEEN 0.3 AND 0.7) AS weak_facts,
                    count(WHERE fn::calculate_memory_decay(created_at, last_accessed, access_count) < 0.3) AS fragment_facts
                FROM knowledge GROUP ALL;
            """)
            
            if result and result[0]:
                stats = result[0]
                logger.info(f"📊 Memory stats: {stats.get('total_facts', 0)} total, "
                          f"{stats.get('strong_facts', 0)} strong, "
                          f"{stats.get('weak_facts', 0)} weak, "
                          f"{stats.get('fragment_facts', 0)} fragments")
                return stats
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {'error': str(e)}
    
    async def run_once(self) -> dict:
        """Run memory evolution once and return results"""
        logger.info("🚀 Running memory evolution (single pass)...")
        
        # Get initial stats
        initial_stats = await self.get_memory_stats()
        
        # Process decay
        process_result = await self.process_memory_decay()
        
        # Get final stats
        final_stats = await self.get_memory_stats()
        
        return {
            'initial_stats': initial_stats,
            'process_result': process_result,
            'final_stats': final_stats
        }
    
    async def run_daemon(self):
        """Run memory evolution daemon continuously"""
        logger.info(f"🚀 Starting memory evolution daemon (interval: {self.interval_seconds}s)")
        
        while self.running:
            try:
                # Process memory evolution
                await self.process_memory_decay()
                
                # Show current memory stats every 10th iteration
                if hasattr(self, '_iteration_count'):
                    self._iteration_count += 1
                else:
                    self._iteration_count = 1
                
                if self._iteration_count % 10 == 0:
                    await self.get_memory_stats()
                
                # Wait for next iteration
                for _ in range(self.interval_seconds):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ Daemon iteration failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
        
        logger.info("🛑 Memory evolution daemon stopped")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.connection_manager.connected:
            await self.connection_manager.disconnect()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Memory Evolution Daemon")
    parser.add_argument('--interval', type=int, default=3600, help='Interval between decay processing (seconds)')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of facts to process per batch')
    parser.add_argument('--once', action='store_true', help='Run once and exit (don\'t run as daemon)')
    parser.add_argument('--stats', action='store_true', help='Show memory stats and exit')
    
    args = parser.parse_args()
    
    daemon = MemoryEvolutionDaemon(
        interval_seconds=args.interval,
        batch_size=args.batch_size
    )
    
    try:
        if args.stats:
            # Just show stats and exit
            stats = await daemon.get_memory_stats()
            print(f"Memory Statistics: {stats}")
            
        elif args.once:
            # Run once and show results
            result = await daemon.run_once()
            print(f"Evolution Results: {result}")
            
        else:
            # Run as daemon
            await daemon.run_daemon()
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Daemon failed: {e}")
        sys.exit(1)
    finally:
        await daemon.cleanup()


if __name__ == "__main__":
    asyncio.run(main())