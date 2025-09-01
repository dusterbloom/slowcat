#!/usr/bin/env python3
"""
Startup script for Adaptive Knowledge Graph

This script initializes and starts the adaptive knowledge graph system:
1. Loads existing predicate patterns from database  
2. Starts the background evolution service
3. Sets up automatic predicate normalization
4. Integrates with consciousness engine

Usage:
    python scripts/start_adaptive_kg.py
    
Or import and call:
    from scripts.start_adaptive_kg import initialize_adaptive_kg
    await initialize_adaptive_kg()
"""

import asyncio
import sys
import os
from pathlib import Path
from loguru import logger

# Add server directory to path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

async def initialize_adaptive_kg():
    """Initialize the adaptive knowledge graph system"""
    logger.info("🧬 Initializing Adaptive Knowledge Graph System...")
    
    try:
        # Import adaptive knowledge graph
        from memory.adaptive_knowledge_graph import get_adaptive_kg
        from services.knowledge_evolution_service import start_evolution_service
        
        # Step 1: Initialize the adaptive knowledge graph
        kg = get_adaptive_kg()
        logger.info("📚 Adaptive knowledge graph instance created")
        
        # Step 2: Load existing patterns from database
        await kg.refresh_clusters()
        logger.info(f"✅ Loaded {len(kg.predicate_clusters)} existing predicate clusters")
        
        # Step 3: Start evolution service
        await start_evolution_service()
        logger.info("🚀 Background evolution service started")
        
        # Step 4: Display initial statistics
        stats = await kg.analyze_current_predicates()
        if stats:
            total_predicates = len(stats)
            total_usage = sum(stats.values())
            logger.info(f"📊 Database contains {total_predicates} unique predicates with {total_usage} total usage")
            
            # Show top predicates
            top_predicates = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("🏆 Top predicates:")
            for pred, count in top_predicates:
                logger.info(f"  {pred}: {count} uses")
        
        logger.info("🎉 Adaptive Knowledge Graph System initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize adaptive KG system: {e}")
        return False

async def test_normalization():
    """Test predicate normalization with examples"""
    logger.info("🧪 Testing predicate normalization...")
    
    try:
        from memory.adaptive_knowledge_graph import normalize_predicate_adaptive
        
        # Test predicates that should be normalized
        test_cases = [
            "dog_name",
            "cat_name", 
            "pet_name",
            "works_at",
            "employed_at",
            "job_at",
            "location",
            "lives_in",
            "located_at",
            "age",
            "years_old",
            "born_in"
        ]
        
        logger.info("Testing normalization on sample predicates:")
        for pred in test_cases:
            normalized = await normalize_predicate_adaptive(pred)
            if normalized != pred:
                logger.info(f"  ✨ {pred} → {normalized}")
            else:
                logger.info(f"  ➡️  {pred} (no change)")
                
        logger.info("✅ Normalization test completed")
        
    except Exception as e:
        logger.error(f"Normalization test failed: {e}")

async def show_evolution_stats():
    """Display current evolution service statistics"""
    try:
        from services.knowledge_evolution_service import get_evolution_service
        
        service = get_evolution_service()
        stats = await service.get_evolution_stats()
        
        logger.info("📊 Evolution Service Statistics:")
        logger.info(f"  Status: {stats['status']}")
        logger.info(f"  Pending new facts: {stats['new_facts_pending']}")
        logger.info(f"  Current clusters: {stats['cluster_count']}")
        logger.info(f"  Semantic threshold: {stats['semantic_threshold']:.3f}")
        
        config = stats['config']
        logger.info(f"  Evolution interval: {config['evolution_interval_min']} minutes")
        logger.info(f"  Min facts threshold: {config['min_facts_threshold']}")
        logger.info(f"  Clustering enabled: {config['clustering_enabled']}")
        logger.info(f"  Pattern detection enabled: {config['pattern_detection_enabled']}")
        logger.info(f"  Cleanup enabled: {config['cleanup_enabled']}")
        
    except Exception as e:
        logger.error(f"Failed to get evolution stats: {e}")

async def main():
    """Main function for standalone execution"""
    logger.info("🚀 Starting Adaptive Knowledge Graph initialization...")
    
    # Initialize the system
    success = await initialize_adaptive_kg()
    
    if not success:
        logger.error("❌ Initialization failed")
        return 1
    
    # Test normalization
    await test_normalization()
    
    # Show stats
    await show_evolution_stats()
    
    # Keep running to demonstrate evolution service
    logger.info("🔄 System running... (Ctrl+C to stop)")
    try:
        while True:
            await asyncio.sleep(30)
            await show_evolution_stats()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        
        # Stop evolution service
        from services.knowledge_evolution_service import stop_evolution_service
        await stop_evolution_service()
        
        logger.info("✅ Shutdown complete")
    
    return 0

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)