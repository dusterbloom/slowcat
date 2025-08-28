#!/usr/bin/env python3
"""
Test Migration Query

Test a single migration query to understand why data isn't persisting.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def test_migration_query():
    """Test creating a single user to debug the issue"""
    logger.info("🔍 Testing migration query...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    logger.info("Connected to SurrealDB")
    
    # Test 1: Create user with parameterized query (like migration script)
    try:
        user_id = "user:peppi"
        result = await db.query("""
            CREATE $user_id SET
                name = $name,
                first_seen = time::now(),
                last_seen = time::now(),
                total_interactions = 0,
                metadata = {}
        """, {
            "user_id": user_id,
            "name": "peppi"
        })
        logger.info(f"Parameterized CREATE result: {result}")
    except Exception as e:
        logger.error(f"Parameterized CREATE failed: {e}")
    
    # Test 2: Create user with string interpolation
    try:
        result = await db.query("""
            CREATE user:test_user SET
                name = 'test_user',
                first_seen = time::now(),
                last_seen = time::now(),
                total_interactions = 0,
                metadata = {}
        """)
        logger.info(f"String interpolation CREATE result: {result}")
    except Exception as e:
        logger.error(f"String interpolation CREATE failed: {e}")
    
    # Test 3: Check if users exist
    try:
        result = await db.query("SELECT * FROM user")
        logger.info(f"Users after creation: {result}")
    except Exception as e:
        logger.error(f"SELECT users failed: {e}")
    
    # Test 4: Check specific user
    try:
        result = await db.query("SELECT * FROM user:peppi")
        logger.info(f"User peppi: {result}")
    except Exception as e:
        logger.error(f"SELECT user:peppi failed: {e}")
    
    await db.close()
    logger.info("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_migration_query())