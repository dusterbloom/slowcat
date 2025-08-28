#!/usr/bin/env python3
"""
Test Different SurrealDB CREATE Methods

Test various ways to create records with specific IDs to find the correct syntax.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def test_create_methods():
    """Test different CREATE methods to find what works"""
    logger.info("🔍 Testing CREATE methods...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Clear test data first
    try:
        await db.query("DELETE test_session")
        logger.info("Cleared test_session table")
    except Exception as e:
        logger.info(f"No test_session table to clear: {e}")
    
    # Test 1: Using db.create() method with specific ID
    try:
        logger.info("Test 1: db.create() with specific ID")
        result = await db.create("test_session:method_test", {
            "user_id": "user:peppi",
            "agent_id": "slowcat",
            "summary": "Test session using create method",
            "status": "active"
        })
        logger.info(f"✅ db.create() result: {result}")
    except Exception as e:
        logger.error(f"❌ db.create() failed: {e}")
    
    # Test 2: Using query() with f-string interpolation (current working method)
    try:
        logger.info("Test 2: query() with f-string interpolation")
        session_id = "test_session:fstring_test"
        result = await db.query(f"""
            CREATE {session_id} SET
                user_id = $user_id,
                agent_id = $agent_id,
                summary = $summary,
                status = $status
        """, {
            "user_id": "user:peppi",
            "agent_id": "slowcat", 
            "summary": "Test session using f-string",
            "status": "active"
        })
        logger.info(f"✅ f-string query() result: {result}")
    except Exception as e:
        logger.error(f"❌ f-string query() failed: {e}")
    
    # Test 3: Using query() with let() to set ID variable
    try:
        logger.info("Test 3: query() with let() for ID")
        await db.let("session_id", "test_session:let_test")
        result = await db.query("""
            CREATE $session_id SET
                user_id = $user_id,
                agent_id = $agent_id,
                summary = $summary,
                status = $status
        """, {
            "user_id": "user:peppi",
            "agent_id": "slowcat",
            "summary": "Test session using let()",
            "status": "active"
        })
        logger.info(f"✅ let() query() result: {result}")
    except Exception as e:
        logger.error(f"❌ let() query() failed: {e}")
    
    # Test 4: Using query() with type::thing() function
    try:
        logger.info("Test 4: query() with type::thing()")
        result = await db.query("""
            CREATE type::thing($table, $id) SET
                user_id = $user_id,
                agent_id = $agent_id,
                summary = $summary,
                status = $status
        """, {
            "table": "test_session",
            "id": "thing_test",
            "user_id": "user:peppi",
            "agent_id": "slowcat",
            "summary": "Test session using type::thing()",
            "status": "active"
        })
        logger.info(f"✅ type::thing() result: {result}")
    except Exception as e:
        logger.error(f"❌ type::thing() failed: {e}")
    
    # Check what was actually created
    try:
        result = await db.query("SELECT * FROM test_session")
        logger.info(f"📊 Created test sessions: {result}")
        
        if result:
            logger.info(f"Total sessions created: {len(result)}")
            for session in result:
                logger.info(f"  Session: {session}")
    except Exception as e:
        logger.error(f"Failed to check created sessions: {e}")
    
    await db.close()
    logger.info("✅ CREATE method tests completed")

if __name__ == "__main__":
    asyncio.run(test_create_methods())