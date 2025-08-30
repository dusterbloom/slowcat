#!/usr/bin/env python3
"""Debug the decay calculation step by step"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def debug_math():
    conn = SurrealConnectionManager()
    try:
        await conn.ensure_connected()
        
        # Test step by step
        steps = [
            "LET $created = d'2024-08-29T10:00:00Z'; RETURN $created;",
            "LET $accessed = d'2024-08-30T12:00:00Z'; RETURN $accessed;", 
            "LET $count = 3; RETURN $count;",
            "LET $age_days = (time::now() - d'2024-08-29T10:00:00Z') / 1d; RETURN $age_days;",
            "LET $recency_hours = (time::now() - d'2024-08-30T12:00:00Z') / 1h; RETURN $recency_hours;",
            "LET $age_days = (time::now() - d'2024-08-29T10:00:00Z') / 1d; LET $base_decay = 1.0 / (1.0 + $age_days / 30.0); RETURN $base_decay;",
            "LET $access_boost = IF 3 / 20.0 < 0.5 THEN 3 / 20.0 ELSE 0.5 END; RETURN $access_boost;",
            "LET $recency_hours = (time::now() - d'2024-08-30T12:00:00Z') / 1h; LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 168.0); RETURN $recency_penalty;",
        ]
        
        for i, query in enumerate(steps):
            try:
                result = await conn.db.query(query)
                logger.info(f"Step {i+1}: {result}")
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
        
        # Test the complete calculation manually
        logger.info("\nTesting complete calculation...")
        result = await conn.db.query("""
            LET $age_days = (time::now() - d'2024-08-29T10:00:00Z') / 1d;
            LET $recency_hours = (time::now() - d'2024-08-30T12:00:00Z') / 1h;
            LET $base_decay = 1.0 / (1.0 + $age_days / 30.0);
            LET $access_boost = IF 3 / 20.0 < 0.5 THEN 3 / 20.0 ELSE 0.5 END;
            LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 168.0);
            LET $result = $base_decay + $access_boost * $recency_penalty;
            RETURN {
                age_days: $age_days,
                recency_hours: $recency_hours, 
                base_decay: $base_decay,
                access_boost: $access_boost,
                recency_penalty: $recency_penalty,
                final_result: $result
            };
        """)
        logger.info(f"Complete calculation: {result}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_math())