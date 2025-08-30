#!/usr/bin/env python3
"""Debug script to check what tables exist in SurrealDB using proper connection manager"""

import asyncio
from loguru import logger
from memory.surreal_connection import SurrealConnectionManager

async def debug_db_tables():
    """Check database tables and contents using proper connection manager"""
    
    # Use the proper connection manager
    conn = SurrealConnectionManager()
    
    try:
        await conn.ensure_connected()
        logger.info("🔍 Connected to SurrealDB")
        
        # Check all potential tables
        potential_tables = [
            'facts', 'messages', 'sessions', 'speakers', 
            'memory_fragments', 'field_states', 'entity', 'knowledge'
        ]
        
        for table in potential_tables:
            try:
                count_result = await conn.db.query(f"SELECT * FROM {table};")
                if count_result and len(count_result) > 0 and count_result[0].get('result'):
                    records = count_result[0]['result']
                    count = len(records)
                    logger.info(f"✅ Table '{table}': {count} records")
                    
                    # Show sample records for non-empty tables
                    if count > 0:
                        print(f"   📋 Sample records (showing max 3):")
                        for i, record in enumerate(records[:3]):
                            print(f"      {i+1}: {record}")
                        print()
                            
                else:
                    logger.info(f"❓ Table '{table}': No data or query failed")
                    
            except Exception as e:
                logger.debug(f"❌ Table '{table}' error: {e}")
        
        # Show database info
        try:
            db_info = await conn.db.query("INFO FOR DB;")
            logger.info(f"🏗️ Database info: {db_info}")
        except Exception as e:
            logger.debug(f"Database info error: {e}")
            
        # Check specifically for facts table structure
        try:
            facts_info = await conn.db.query("INFO FOR TABLE facts;")
            logger.info(f"📊 Facts table info: {facts_info}")
        except Exception as e:
            logger.debug(f"Facts table info error: {e}")
            
    except Exception as e:
        logger.error(f"Connection error: {e}")
        
    finally:
        if conn.connected:
            await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_db_tables())