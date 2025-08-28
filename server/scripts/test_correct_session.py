#!/usr/bin/env python3
"""
Test Correct Session Creation

Test creating sessions with all required fields.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def test_correct_session():
    """Test creating sessions with proper schema compliance"""
    logger.info("🔍 Testing correct session creation...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Clear sessions
    await db.query("DELETE session")
    
    # Test with all required fields
    try:
        logger.info("Creating session with all required fields...")
        result = await db.create("session:correct_test", {
            "user_id": "user:peppi",  # record<user> type
            "agent_id": "slowcat",
            "summary": "Test session with correct schema",
            "keywords": ["test", "schema"],
            "status": "active",
            "started_at": "time::now()",  # Will be handled by SurrealDB
            "ended_at": "time::now()",    # Required field!
            "turn_count": 0,
            "duration_secs": 0
        })
        logger.info(f"✅ Correct session result: {result}")
        
        # Check persistence
        check = await db.query("SELECT * FROM session:correct_test")
        logger.info(f"Persistence check: {check}")
        
    except Exception as e:
        logger.error(f"❌ Correct session failed: {e}")
    
    # Test using query method with proper datetime
    try:
        logger.info("Creating session using query method...")
        result = await db.query(f"""
            CREATE session:query_test SET
                user_id = $user_id,
                agent_id = $agent_id,
                summary = $summary,
                keywords = $keywords,
                status = $status,
                started_at = time::now(),
                ended_at = time::now(),
                turn_count = $turn_count,
                duration_secs = $duration_secs
        """, {
            "user_id": "user:peppi",
            "agent_id": "slowcat",
            "summary": "Test session using query method",
            "keywords": ["test", "query"],
            "status": "active",
            "turn_count": 0,
            "duration_secs": 0
        })
        logger.info(f"✅ Query method result: {result}")
        
    except Exception as e:
        logger.error(f"❌ Query method failed: {e}")
    
    # Check final state
    try:
        all_sessions = await db.query("SELECT count() FROM session")
        count = all_sessions[0].get('count', 0) if all_sessions else 0
        logger.info(f"📊 Total sessions created: {count}")
        
        if count > 0:
            sessions = await db.query("SELECT * FROM session")
            for session in sessions:
                logger.info(f"  Session: {session['id']} - Status: {session['status']}")
                
    except Exception as e:
        logger.error(f"Failed to check sessions: {e}")
    
    await db.close()
    logger.info("✅ Correct session test completed")

if __name__ == "__main__":
    asyncio.run(test_correct_session())