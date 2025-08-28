#!/usr/bin/env python3
"""
Debug Session Table Constraints

Check what's causing the turn_count constraint issue.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def debug_constraints():
    """Debug session table constraints"""
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Get session table schema
    schema_result = await db.query("INFO FOR TABLE session")
    logger.info(f"Session table schema: {schema_result}")
    
    # Check session data that exists
    sessions_result = await db.query("SELECT * FROM session")
    logger.info(f"Existing sessions: {sessions_result}")
    
    # Try creating a simple message without relationships
    try:
        message_result = await db.query("""
            INSERT INTO message {
                id: message:test123,
                session_id: session:fed7c0cd5c92,
                speaker_type: 'user',
                content: 'test',
                timestamp: time::now(),
                sequence_num: 1,
                embedding: [],
                metadata: {}
            }
        """)
        logger.info(f"Message creation result: {message_result}")
    except Exception as e:
        logger.error(f"Message creation failed: {e}")
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(debug_constraints())
