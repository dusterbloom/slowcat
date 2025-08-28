#!/usr/bin/env python3
"""
Test Final Session Creation

Test creating sessions with proper SurrealDB types and functions.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def test_final_session():
    """Test creating sessions with proper SurrealDB syntax"""
    logger.info("🔍 Testing final session creation...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Clear sessions
    await db.query("DELETE session")
    
    # Test 1: Using query method with proper SurrealDB functions
    try:
        logger.info("Test 1: Query method with SurrealDB functions...")
        result = await db.query("""
            CREATE session:final_test SET
                user_id = user:peppi,
                agent_id = 'slowcat',
                summary = 'Test session with proper syntax',
                keywords = ['test', 'final'],
                status = 'active',
                started_at = time::now(),
                ended_at = time::now(),
                turn_count = 0,
                duration_secs = 0
        """)
        logger.info(f"✅ Query method result: {result}")
        
    except Exception as e:
        logger.error(f"❌ Query method failed: {e}")
    
    # Test 2: Using type::thing() for record reference
    try:
        logger.info("Test 2: Using type::thing() for user reference...")
        result = await db.query("""
            CREATE session:type_test SET
                user_id = type::thing('user', 'peppi'),
                agent_id = 'slowcat',
                summary = 'Test session with type::thing',
                keywords = ['test', 'type'],
                status = 'ended',
                started_at = time::now(),
                ended_at = time::now(),
                turn_count = 5,
                duration_secs = 120
        """)
        logger.info(f"✅ type::thing() result: {result}")
        
    except Exception as e:
        logger.error(f"❌ type::thing() failed: {e}")
    
    # Test 3: Mixed parameterized + function approach
    try:
        logger.info("Test 3: Mixed parameterized approach...")
        result = await db.query("""
            CREATE session:mixed_test SET
                user_id = type::thing('user', $user_name),
                agent_id = $agent_id,
                summary = $summary,
                keywords = $keywords,
                status = $status,
                started_at = time::now(),
                ended_at = time::now(),
                turn_count = $turn_count,
                duration_secs = $duration_secs
        """, {
            "user_name": "peppi",
            "agent_id": "slowcat",
            "summary": "Test session with mixed approach",
            "keywords": ["test", "mixed"],
            "status": "active",
            "turn_count": 0,
            "duration_secs": 0
        })
        logger.info(f"✅ Mixed approach result: {result}")
        
    except Exception as e:
        logger.error(f"❌ Mixed approach failed: {e}")
    
    # Check results
    try:
        all_sessions = await db.query("SELECT count() FROM session")
        count = all_sessions[0].get('count', 0) if all_sessions else 0
        logger.info(f"📊 Total sessions created: {count}")
        
        if count > 0:
            sessions = await db.query("SELECT * FROM session")
            for session in sessions:
                logger.info(f"  ✅ Session: {session['id']} - User: {session['user_id']} - Status: {session['status']}")
                
    except Exception as e:
        logger.error(f"Failed to check sessions: {e}")
    
    await db.close()
    logger.info("✅ Final session test completed")

if __name__ == "__main__":
    asyncio.run(test_final_session())