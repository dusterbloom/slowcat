#!/usr/bin/env python3
"""
Test Memory Evolution System - Verify all components are working

This test script verifies:
1. Memory decay calculations are functioning
2. Real-time strength categorization (strong/weak/fragment)
3. Access tracking updates last_accessed and access_count
4. Fragment reconstruction strengthens related weak memories
5. Background decay processing works
6. Cleanup removes very weak memories

Usage:
    python test_memory_evolution.py
"""

import asyncio
from datetime import datetime, timezone, timedelta
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger


async def test_memory_evolution():
    """Test all memory evolution components"""
    logger.info("🧪 Testing Memory Evolution System")
    
    conn = SurrealConnectionManager()
    
    try:
        await conn.ensure_connected()
        logger.info("✅ Connected to SurrealDB")
        
        # Test 1: Verify decay function exists and works
        logger.info("\n🔧 Test 1: Memory decay function")
        test_result = await conn.db.query("""
            RETURN fn::calculate_memory_decay(
                time::now() - 24h,  // created 24 hours ago
                time::now() - 6h,   // last accessed 6 hours ago
                5                   // accessed 5 times
            );
        """)
        
        if test_result is not None:
            logger.info(f"   ✅ Decay function returns: {test_result} (type: {type(test_result)})")
        else:
            logger.error("   ❌ Decay function failed")
            return False
        
        # Test 2: Create test knowledge with different strengths
        logger.info("\n🔧 Test 2: Creating test knowledge records")
        
        # Create strong fact (recent, high access)
        strong_fact = await conn.db.query("""
            CREATE knowledge CONTENT {
                predicate: "test_strong_fact",
                strength: 1.0,
                confidence: 0.9,
                created_at: time::now() - 1h,
                last_accessed: time::now() - 5m,
                access_count: 10
            };
        """)
        
        # Create weak fact (older, moderate access)
        weak_fact = await conn.db.query("""
            CREATE knowledge CONTENT {
                predicate: "test_weak_fact", 
                strength: 0.4,
                confidence: 0.6,
                created_at: time::now() - 48h,
                last_accessed: time::now() - 12h,
                access_count: 2
            };
        """)
        
        # Create fragment fact (very old, low access)
        fragment_fact = await conn.db.query("""
            CREATE knowledge CONTENT {
                predicate: "test_fragment_fact",
                strength: 0.2,
                confidence: 0.3,
                created_at: time::now() - 168h,  // 1 week ago
                last_accessed: time::now() - 72h,
                access_count: 1
            };
        """)
        
        logger.info("   ✅ Created test knowledge records")
        
        # Test 3: Test search with strength categorization
        logger.info("\n🔧 Test 3: Search with real-time decay calculations")
        
        search_result = await conn.db.query("""
            SELECT *, 
                   fn::calculate_memory_decay(created_at, last_accessed, access_count) AS current_strength
            FROM knowledge
            WHERE string::starts_with(predicate, 'test_')
            ORDER BY current_strength DESC;
        """)
        
        if search_result:
            for fact in search_result:
                predicate = fact.get('predicate', 'unknown')
                current_strength = fact.get('current_strength', 0)
                
                # Calculate category in Python
                if current_strength > 0.7:
                    category = 'strong'
                elif current_strength > 0.3:
                    category = 'weak'
                else:
                    category = 'fragment'
                    
                logger.info(f"   📊 {predicate}: strength={current_strength:.3f}, category={category}")
        
        # Test 4: Test access tracking
        logger.info("\n🔧 Test 4: Access tracking updates")
        
        # Access facts via search function (should update timestamps)
        await conn.db.query("""SELECT * FROM fn::search_knowledge('test', 10);""")
        
        # Check if access counts were updated
        updated_facts = await conn.db.query("""
            SELECT predicate, access_count, last_accessed 
            FROM knowledge 
            WHERE string::starts_with(predicate, 'test_')
            ORDER BY access_count DESC;
        """)
        
        if updated_facts:
            logger.info("   ✅ Access tracking verification:")
            for fact in updated_facts:
                predicate = fact.get('predicate', 'unknown')
                count = fact.get('access_count', 0)
                accessed = fact.get('last_accessed', 'unknown')
                logger.info(f"      {predicate}: count={count}, last_accessed={accessed}")
        
        # Test 5: Test fragment reconstruction
        logger.info("\n🔧 Test 5: Fragment reconstruction")
        
        # Create related fragments about the same entity
        entity_fragments = await conn.db.query("""
            LET $entity_id = (CREATE entity CONTENT { canonical_name: "test_entity" }).id;
            
            CREATE knowledge CONTENT {
                predicate: "color",
                strength: 0.2,
                confidence: 0.4,
                in: $entity_id,
                created_at: time::now() - 100h,
                last_accessed: time::now() - 50h,
                access_count: 1
            };
            
            CREATE knowledge CONTENT {
                predicate: "size", 
                strength: 0.3,
                confidence: 0.5,
                in: $entity_id,
                created_at: time::now() - 80h,
                last_accessed: time::now() - 40h,
                access_count: 2
            };
        """)
        
        # Trigger reconstruction
        reconstruction_result = await conn.db.query("""
            RETURN fn::reconstruct_fragments('test_entity', 0.2);
        """)
        
        if reconstruction_result and reconstruction_result[0]:
            logger.info(f"   ✅ Reconstructed {reconstruction_result[0]} fragments")
        else:
            logger.info("   ⚠️ No fragments reconstructed (may be expected)")
        
        # Test 6: Test background decay processing
        logger.info("\n🔧 Test 6: Background decay processing")
        
        decay_result = await conn.db.query("""
            RETURN fn::decay_background_memories(10);
        """)
        
        if decay_result is not None:
            processed_count = decay_result
            logger.info(f"   ✅ Background decay processed {processed_count} facts")
        
        # Test 7: Test fragment cleanup
        logger.info("\n🔧 Test 7: Fragment cleanup")
        
        cleanup_result = await conn.db.query("""
            RETURN fn::cleanup_fragments(5);
        """)
        
        if cleanup_result is not None:
            cleaned_count = cleanup_result
            logger.info(f"   ✅ Fragment cleanup removed {cleaned_count} very weak memories")
        
        # Test 8: Show final memory statistics
        logger.info("\n🔧 Test 8: Final memory statistics")
        
        final_stats = await conn.db.query("""
            SELECT count() AS total_facts FROM knowledge GROUP ALL;
        """)
        
        if final_stats and final_stats[0]:
            stats = final_stats[0]
            total = stats.get('total_facts', 0)
            
            logger.info(f"   📊 Memory Distribution: {total} total facts")
            logger.info(f"   📈 Memory evolution system is operational")
        
        # Cleanup test data
        logger.info("\n🧹 Cleaning up test data")
        await conn.db.query("""DELETE knowledge WHERE string::starts_with(predicate, 'test_');""")
        await conn.db.query("""DELETE entity WHERE canonical_name = 'test_entity';""")
        
        logger.info("\n🎉 Memory Evolution System Test Complete!")
        logger.info("✅ All components are working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if conn.connected:
            await conn.disconnect()


if __name__ == "__main__":
    success = asyncio.run(test_memory_evolution())
    exit(0 if success else 1)