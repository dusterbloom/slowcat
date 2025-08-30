#!/usr/bin/env python3
"""Reinitialize SurrealDB schema functions"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from memory.schema_init import ensure_schema_functions
from loguru import logger

async def reinit_schema():
    conn = SurrealConnectionManager()
    try:
        await conn.ensure_connected()
        logger.info("Reinitializing schema functions...")
        
        success = await ensure_schema_functions(conn)
        if success:
            logger.info("✅ Schema functions reinitialized successfully")
        else:
            logger.error("❌ Failed to reinitialize schema functions")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(reinit_schema())