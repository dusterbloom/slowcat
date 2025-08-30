#!/usr/bin/env python3
"""Debug script to check what tables exist in SurrealDB and their contents"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def debug_db_tables():
    """Check database tables and contents"""
    
    # Connect to SurrealDB
    surreal = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    
    try:
        await surreal.signin({"user": "root", "pass": "slowcat_secure_2024"})
        await surreal.use("slowcat", "memory_graph")
        
        # Get all tables
        logger.info("🔍 Checking database tables...")
        tables_result = await surreal.query("INFO FOR DB;")
        print(f"Database info: {tables_result}")
        
        # List all tables by trying to query them
        potential_tables = [
            'facts', 'messages', 'sessions', 'speakers', 
            'memory_fragments', 'field_states', 'entity', 'knowledge'
        ]
        
        for table in potential_tables:
            try:
                count_result = await surreal.query(f"SELECT count() FROM {table} GROUP ALL;")
                if count_result and len(count_result) > 0 and count_result[0].get('result'):
                    count = count_result[0]['result'][0].get('count', 0) if count_result[0]['result'] else 0
                    logger.info(f"✅ Table '{table}': {count} records")
                    
                    # Show sample records for non-empty tables
                    if count > 0:
                        sample = await surreal.query(f"SELECT * FROM {table} LIMIT 3;")
                        if sample and sample[0].get('result'):
                            print(f"   Sample records: {sample[0]['result']}")
                            
                else:
                    logger.info(f"❓ Table '{table}': No data or doesn't exist")
                    
            except Exception as e:
                logger.debug(f"❌ Table '{table}' error: {e}")
        
        # Check specific facts table structure if it exists
        try:
            facts_structure = await surreal.query("INFO FOR TABLE facts;")
            if facts_structure:
                logger.info(f"🏗️ Facts table structure: {facts_structure}")
        except Exception as e:
            logger.debug(f"No facts table structure: {e}")
            
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        
    finally:
        try:
            await surreal.close()
        except NotImplementedError:
            pass  # Close not implemented for HTTP connection

if __name__ == "__main__":
    asyncio.run(debug_db_tables())