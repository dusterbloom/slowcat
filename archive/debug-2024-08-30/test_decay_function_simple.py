#!/usr/bin/env python3
"""Simple test of just the decay function"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def test_decay_simple():
    conn = SurrealConnectionManager()
    try:
        await conn.ensure_connected()
        
        # Test with simple, explicit values
        result = await conn.db.query("""
            RETURN fn::calculate_memory_decay(
                d'2024-08-29T10:00:00Z',
                d'2024-08-30T12:00:00Z', 
                3
            );
        """)
        
        logger.info(f"Decay result: {result} (type: {type(result[0]) if result else 'None'})")
        
        # Test with time calculations
        result2 = await conn.db.query("""
            LET $created = time::now() - 24h;
            LET $accessed = time::now() - 6h;
            LET $count = 3;
            RETURN fn::calculate_memory_decay($created, $accessed, $count);
        """)
        
        logger.info(f"Decay result2: {result2} (type: {type(result2[0]) if result2 else 'None'})")
        
        # Check if function exists
        info_result = await conn.db.query("INFO FOR DB;")
        if info_result and info_result[0] and 'result' in info_result[0]:
            functions = info_result[0]['result'].get('functions', {})
            logger.info(f"Available functions: {list(functions.keys())}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(test_decay_simple())