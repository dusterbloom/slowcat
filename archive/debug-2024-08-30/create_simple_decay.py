#!/usr/bin/env python3
"""Create a simplified decay function that just works"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def create_simple_function():
    conn = SurrealConnectionManager()
    try:
        await conn.ensure_connected()
        
        # Remove existing function
        try:
            await conn.db.query("REMOVE FUNCTION fn::calculate_memory_decay;")
        except:
            pass
        
        # Create very simple function that just returns a fixed decay based on age
        logger.info("Creating simplified decay function...")
        await conn.db.query("""
            DEFINE FUNCTION fn::calculate_memory_decay(
                $created_at: datetime,
                $last_accessed: datetime, 
                $access_count: int
            ) {
                LET $age_seconds = time::now() - $created_at;
                LET $hours_old = $age_seconds / 3600;
                
                // Simple linear decay: start at 1.0, lose 0.01 per hour, minimum 0.1
                LET $simple_decay = 1.0 - ($hours_old * 0.01);
                
                // Add access boost: +0.05 per access up to +0.3 maximum
                LET $access_boost = $access_count * 0.05;
                LET $capped_boost = IF $access_boost > 0.3 THEN 0.3 ELSE $access_boost END;
                
                LET $final_strength = $simple_decay + $capped_boost;
                
                // Ensure result is between 0.1 and 1.0
                RETURN IF $final_strength < 0.1 THEN 0.1 ELSE (IF $final_strength > 1.0 THEN 1.0 ELSE $final_strength END) END;
            };
        """)
        logger.info("Simplified function created")
        
        # Test it
        test_result = await conn.db.query("""
            RETURN fn::calculate_memory_decay(
                time::now() - 24h,
                time::now() - 6h, 
                5
            );
        """)
        
        logger.info(f"Test result: {test_result}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(create_simple_function())