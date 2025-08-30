#!/usr/bin/env python3
"""Fix the decay function directly"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def fix_function():
    conn = SurrealConnectionManager()
    try:
        await conn.ensure_connected()
        
        # Remove existing function
        logger.info("Removing existing function...")
        try:
            await conn.db.query("REMOVE FUNCTION fn::calculate_memory_decay;")
        except:
            pass
        
        # Create corrected function
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
        logger.info("Function created successfully")
        
        # Test the function
        result = await conn.db.query("""
            RETURN fn::calculate_memory_decay(
                d'2024-08-29T10:00:00Z',
                d'2024-08-30T12:00:00Z', 
                3
            );
        """)
        
        logger.info(f"Test result: {result}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(fix_function())