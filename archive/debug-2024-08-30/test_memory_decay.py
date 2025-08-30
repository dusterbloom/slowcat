#!/usr/bin/env python3
"""Test the existing memory decay functions in SurrealDB"""

import asyncio
from datetime import datetime, timezone, timedelta
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def test_memory_decay_function():
    """Test the existing fn::calculate_memory_decay function"""
    
    conn = SurrealConnectionManager()
    
    try:
        await conn.ensure_connected()
        logger.info("✅ Connected to SurrealDB")
        
        # Test 1: Direct function call with sample data
        logger.info("🧪 Test 1: Testing memory decay function directly")
        
        # Create test timestamps
        now = datetime.now(timezone.utc)
        created_1h_ago = now - timedelta(hours=1)
        created_1d_ago = now - timedelta(days=1) 
        created_30d_ago = now - timedelta(days=30)
        
        last_accessed_recent = now - timedelta(minutes=10)
        last_accessed_old = now - timedelta(days=7)
        
        test_cases = [
            {
                "name": "Fresh fact (1h old, accessed recently)",
                "created": created_1h_ago,
                "accessed": last_accessed_recent,
                "count": 5
            },
            {
                "name": "Day-old fact (accessed recently)", 
                "created": created_1d_ago,
                "accessed": last_accessed_recent,
                "count": 10
            },
            {
                "name": "Old fact (30d old, not accessed recently)",
                "created": created_30d_ago,
                "accessed": last_accessed_old,
                "count": 2
            },
            {
                "name": "Popular fact (old but frequently accessed)",
                "created": created_30d_ago,
                "accessed": last_accessed_recent, 
                "count": 50
            }
        ]
        
        for i, test in enumerate(test_cases):
            result = await conn.db.query("""
                RETURN fn::calculate_memory_decay($created, $accessed, $count);
            """, {
                'created': test['created'],
                'accessed': test['accessed'], 
                'count': test['count']
            })
            
            if result and len(result) > 0:
                strength = result[0]  # RETURN statement gives direct value
                logger.info(f"   {i+1}. {test['name']}: Strength = {strength} (type: {type(strength)})")
            else:
                logger.error(f"   {i+1}. {test['name']}: Failed to get strength")
        
        # Test 2: Check if we have actual knowledge data
        logger.info("\n🧪 Test 2: Checking existing knowledge data")
        
        knowledge_result = await conn.db.query("SELECT * FROM knowledge LIMIT 5;")
        if knowledge_result and knowledge_result[0].get('result'):
            knowledge = knowledge_result[0]['result']
            logger.info(f"   Found {len(knowledge)} knowledge records")
            
            if knowledge:
                # Test decay on actual data
                for i, fact in enumerate(knowledge[:2]):
                    if 'created_at' in fact and 'last_accessed' in fact:
                        result = await conn.db.query("""
                            RETURN fn::calculate_memory_decay($created, $accessed, $count);
                        """, {
                            'created': fact.get('created_at', now),
                            'accessed': fact.get('last_accessed', now),
                            'count': fact.get('access_count', 1)
                        })
                        
                        if result and len(result) > 0:
                            strength = result[0]  # RETURN statement gives direct value
                            subject = fact.get('subject', 'unknown')
                            predicate = fact.get('predicate', 'unknown')
                            logger.info(f"   Real fact {i+1}: {subject} {predicate} - Strength = {strength:.3f}")
            
        else:
            logger.info("   No existing knowledge data found")
        
        # Test 3: Test other memory functions
        logger.info("\n🧪 Test 3: Testing other memory functions")
        
        # Test search_knowledge function
        search_result = await conn.db.query("""
            RETURN fn::search_knowledge('dog', 5);
        """)
        if search_result and len(search_result) > 0:
            results = search_result[0]  # RETURN statement gives direct value
            logger.info(f"   Search for 'dog': {len(results) if results else 0} results")
        
        # Test get_entity_facts
        entity_result = await conn.db.query("""
            RETURN fn::get_entity_facts('user');
        """)
        if entity_result and len(entity_result) > 0:
            results = entity_result[0]  # RETURN statement gives direct value
            logger.info(f"   Entity facts for 'user': {len(results) if results else 0} results")
            
        logger.info("\n🎉 Memory decay function tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if conn.connected:
            await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(test_memory_decay_function())