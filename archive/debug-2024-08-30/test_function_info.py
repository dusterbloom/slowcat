#!/usr/bin/env python3
"""Check what functions exist in the database"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def test_functions():
    conn = SurrealConnectionManager()
    try:
        await conn.ensure_connected()
        
        # Get database info
        result = await conn.db.query("INFO FOR DB;")
        if result and result[0] and 'result' in result[0]:
            info = result[0]['result']
            functions = info.get('functions', {})
            logger.info(f"Available functions: {list(functions.keys())}")
            
            # Check the decay function specifically
            if 'calculate_memory_decay' in functions:
                func_def = functions['calculate_memory_decay']
                logger.info(f"calculate_memory_decay definition: {func_def[:300]}")
        
        # Try to delete and recreate the function
        logger.info("Removing existing function...")
        await conn.db.query("REMOVE FUNCTION fn::calculate_memory_decay;")
        
        # Recreate with correct syntax
        logger.info("Creating corrected function...")
        await conn.db.query("""
            DEFINE FUNCTION fn::calculate_memory_decay(
                $created_at: datetime,
                $last_accessed: datetime, 
                $access_count: int
            ) {
                LET $age_days = (time::now() - $created_at) / 1d;
                LET $recency_hours = (time::now() - $last_accessed) / 1h;
                LET $base_decay = 1.0 / (1.0 + $age_days / 30.0);
                LET $access_boost = IF $access_count / 20.0 < 0.5 THEN $access_count / 20.0 ELSE 0.5 END;
                LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 168.0);
                
                LET $result = $base_decay + $access_boost * $recency_penalty;
                RETURN IF $result > 0.1 THEN $result ELSE 0.1 END;
            };
        """)
        
        # Test again
        result = await conn.db.query("""
            RETURN fn::calculate_memory_decay(
                d'2024-08-29T10:00:00Z',
                d'2024-08-30T12:00:00Z', 
                3
            );
        """)
        
        logger.info(f"After fix - Decay result: {result} (type: {type(result[0]) if result else 'None'})")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(test_functions())