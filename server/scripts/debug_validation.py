#!/usr/bin/env python3
"""
Debug SurrealDB Query Results

This script debugs the SurrealDB query result format to understand
how to properly parse results in the validation script.
"""

import asyncio
import sys
sys.path.append('..')
from memory.graph_surreal_memory import GraphSurrealMemory
from loguru import logger

async def debug_queries():
    """Debug various SurrealDB queries to understand result format"""
    logger.info("🔍 Starting SurrealDB query debugging...")
    
    memory = GraphSurrealMemory()
    await memory.connect()
    
    # Test queries
    test_queries = [
        "SELECT count() FROM user",
        "SELECT * FROM user LIMIT 1", 
        "SELECT count() FROM session",
        "SELECT * FROM session LIMIT 1",
        "SELECT count() FROM message",
        "SELECT count() FROM concept",
        "SELECT count() FROM knows",
        "SELECT count() FROM contains",
        "INFO FOR DB"
    ]
    
    for query in test_queries:
        try:
            logger.info(f"Testing query: {query}")
            result = await memory.db.query(query)
            
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result length: {len(result) if result else 0}")
            
            if result:
                logger.info(f"First element: {result[0]}")
                logger.info(f"First element type: {type(result[0])}")
                
                if isinstance(result[0], dict):
                    logger.info(f"Keys in first element: {list(result[0].keys())}")
                    
                    if 'result' in result[0]:
                        result_data = result[0]['result'] 
                        logger.info(f"Result data: {result_data}")
                        logger.info(f"Result data type: {type(result_data)}")
                        logger.info(f"Result data length: {len(result_data) if hasattr(result_data, '__len__') else 'N/A'}")
                        
                        if result_data and len(result_data) > 0:
                            logger.info(f"First result item: {result_data[0]}")
                            logger.info(f"First result item type: {type(result_data[0])}")
            else:
                logger.info("Empty result")
            
            logger.info("---")
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.info("---")
    
    await memory.close()
    logger.info("✅ Debug session completed")

if __name__ == "__main__":
    asyncio.run(debug_queries())