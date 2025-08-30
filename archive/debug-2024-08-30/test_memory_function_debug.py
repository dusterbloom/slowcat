#!/usr/bin/env python3
"""Debug the memory decay function to understand what's happening"""

import asyncio
from datetime import datetime, timezone, timedelta
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def debug_memory_function():
    """Debug the memory decay function step by step"""
    
    conn = SurrealConnectionManager()
    
    try:
        await conn.ensure_connected()
        logger.info("✅ Connected to SurrealDB")
        
        # First, let's see what the function definition looks like
        logger.info("🔍 Checking function definition...")
        function_info = await conn.db.query("INFO FOR DB;")
        if function_info:
            functions = function_info.get('functions', {})
            if 'calculate_memory_decay' in functions:
                func_def = functions['calculate_memory_decay']
                logger.info(f"Function definition found: {func_def[:200]}...")
            else:
                logger.error("Memory decay function not found!")
                
        # Test with very simple values first
        logger.info("\n🧪 Testing with simple values...")
        
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        
        # Test just the time difference calculation
        logger.info("Testing individual components...")
        
        # Test 1: Simple math operations
        simple_test = await conn.db.query("""
            LET $now = time::now();
            LET $created = $now - 1h;
            LET $age_days = ($now - $created) / 1d;
            RETURN $age_days;
        """)
        
        if simple_test and len(simple_test) > 0:
            logger.info(f"Age days calculation: {simple_test[0]}")
        
        # Test 2: Try calling with explicit datetime strings
        result = await conn.db.query("""
            RETURN fn::calculate_memory_decay(
                d'2024-08-30T10:00:00Z',
                d'2024-08-30T13:00:00Z', 
                5
            );
        """)
        
        if result and len(result) > 0:
            logger.info(f"Function result with explicit dates: {result[0]} (type: {type(result[0])})")
        
        # Test 3: Check if there are any existing knowledge records to understand structure
        logger.info("\n🔍 Checking knowledge table structure...")
        sample_knowledge = await conn.db.query("SELECT * FROM knowledge LIMIT 1;")
        if sample_knowledge and sample_knowledge[0].get('result'):
            knowledge = sample_knowledge[0]['result']
            if knowledge:
                logger.info(f"Sample knowledge record: {knowledge[0]}")
            else:
                logger.info("Knowledge table is empty")
                
        # Let's try to create a test knowledge record to see the decay in action
        logger.info("\n🧪 Creating test knowledge record...")
        
        test_result = await conn.db.query("""
            CREATE knowledge CONTENT {
                predicate: "test_fact",
                strength: 1.0,
                confidence: 0.9,
                created_at: time::now() - 1h,
                last_accessed: time::now() - 30m,
                access_count: 3
            };
        """)
        
        if test_result and test_result[0].get('result'):
            test_record = test_result[0]['result'][0]
            logger.info(f"Created test record: {test_record}")
            
            # Now test the decay function on this record
            decay_result = await conn.db.query("""
                RETURN fn::calculate_memory_decay(
                    $record.created_at,
                    $record.last_accessed,
                    $record.access_count
                );
            """, {'record': test_record})
            
            if decay_result and len(decay_result) > 0:
                logger.info(f"Decay result for test record: {decay_result[0]} (type: {type(decay_result[0])})")
        
        logger.info("\n🎉 Debug completed!")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if conn.connected:
            await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_memory_function())