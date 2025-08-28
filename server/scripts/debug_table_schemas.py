#!/usr/bin/env python3
"""
Debug Table Schema Requirements

Check schema constraints for all tables to understand why they're not being created.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def debug_table_schemas():
    """Debug schema constraints for all tables"""
    logger.info("🔍 Debugging table schemas...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Get detailed table info for each table
    tables = ['user', 'concept', 'message', 'thought']
    
    for table_name in tables:
        try:
            logger.info(f"\n📋 {table_name.upper()} TABLE SCHEMA:")
            table_info = await db.query(f"INFO FOR TABLE {table_name}")
            
            if table_info and len(table_info) > 0:
                info = table_info[0]
                logger.info(f"Schema definition: {info}")
                
                # Show field requirements
                if 'fields' in info:
                    logger.info(f"Required fields for {table_name}:")
                    for field_name, field_def in info['fields'].items():
                        logger.info(f"  {field_name}: {field_def}")
                        
        except Exception as e:
            logger.error(f"Failed to get schema for {table_name}: {e}")
    
    # Test creating each table type with minimal data
    logger.info("\n🧪 TESTING TABLE CREATION:")
    
    # Test 1: User creation
    try:
        logger.info("Testing user creation...")
        result = await db.create("user:test", {
            "name": "test_user"
        })
        logger.info(f"✅ User creation result: {result}")
    except Exception as e:
        logger.error(f"❌ User creation failed: {e}")
    
    # Test 2: Concept creation
    try:
        logger.info("Testing concept creation...")
        result = await db.create("concept:test", {
            "name": "test_concept"
        })
        logger.info(f"✅ Concept creation result: {result}")
    except Exception as e:
        logger.error(f"❌ Concept creation failed: {e}")
    
    # Test 3: Message creation
    try:
        logger.info("Testing message creation...")
        result = await db.create("message:test", {
            "content": "test message"
        })
        logger.info(f"✅ Message creation result: {result}")
    except Exception as e:
        logger.error(f"❌ Message creation failed: {e}")
    
    # Test 4: Thought creation  
    try:
        logger.info("Testing thought creation...")
        result = await db.create("thought:test", {
            "content": "test thought"
        })
        logger.info(f"✅ Thought creation result: {result}")
    except Exception as e:
        logger.error(f"❌ Thought creation failed: {e}")
    
    # Check what was created
    for table_name in tables:
        try:
            count = await db.query(f"SELECT count() FROM {table_name}")
            count_val = count[0].get('count', 0) if count else 0
            logger.info(f"📊 {table_name}: {count_val} records")
        except Exception as e:
            logger.error(f"Failed to count {table_name}: {e}")
    
    await db.close()
    logger.info("✅ Schema debugging completed")

if __name__ == "__main__":
    asyncio.run(debug_table_schemas())