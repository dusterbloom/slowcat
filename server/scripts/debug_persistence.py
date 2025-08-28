#!/usr/bin/env python3
"""
Debug Data Persistence Issue

Test different approaches to creating records and verify they persist.
"""

import asyncio
import uuid
from surrealdb import AsyncSurreal
from loguru import logger

async def debug_persistence():
    """Debug why data isn't persisting"""
    logger.info("🔍 Debugging data persistence...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Clear test data (use proper SurrealDB syntax)
    await db.query("DELETE user:test_create")
    await db.query("DELETE user:test_query") 
    await db.query("DELETE user:test_tx")
    await db.query("DELETE concept:test_create")
    await db.query("DELETE message:test_create")
    
    logger.info("=== TESTING db.create() METHOD ===")
    
    # Test 1: User creation with db.create()
    try:
        logger.info("Testing user creation with db.create()...")
        user_result = await db.create("user:test_create", {
            "name": "test_user",
            "first_seen": "time::now()",
            "last_seen": "time::now()",
            "total_interactions": 0,
            "metadata": {}
        })
        logger.info(f"✅ User create result: {user_result}")
        
        # Verify it exists immediately
        check_result = await db.query("SELECT * FROM user:test_create")
        logger.info(f"📋 User verification: {check_result}")
        
    except Exception as e:
        logger.error(f"❌ User creation with db.create() failed: {e}")
    
    # Test 2: Concept creation with db.create()
    try:
        logger.info("Testing concept creation with db.create()...")
        concept_result = await db.create("concept:test_create", {
            "name": "test_concept",
            "kind": "general",
            "properties": {},
            "mentioned_count": 0,
            "first_mentioned": "time::now()",
            "last_mentioned": "time::now()"
        })
        logger.info(f"✅ Concept create result: {concept_result}")
        
        # Verify it exists immediately
        check_result = await db.query("SELECT * FROM concept:test_create")
        logger.info(f"📋 Concept verification: {check_result}")
        
    except Exception as e:
        logger.error(f"❌ Concept creation with db.create() failed: {e}")
    
    # Test 3: Message creation with db.create()
    try:
        logger.info("Testing message creation with db.create()...")
        message_result = await db.create("message:test_create", {
            "session_id": "session:test123",
            "speaker_type": "user",
            "content": "test message",
            "timestamp": "time::now()",
            "sequence_num": 1,
            "embedding": [],
            "metadata": {}
        })
        logger.info(f"✅ Message create result: {message_result}")
        
        # Verify it exists immediately
        check_result = await db.query("SELECT * FROM message:test_create")
        logger.info(f"📋 Message verification: {check_result}")
        
    except Exception as e:
        logger.error(f"❌ Message creation with db.create() failed: {e}")
    
    logger.info("\n=== TESTING db.query() WITH CREATE ===")
    
    # Test 4: User creation with db.query()
    try:
        logger.info("Testing user creation with db.query()...")
        query_result = await db.query(f"""
            CREATE user:test_query SET
                name = 'test_user_query',
                first_seen = time::now(),
                last_seen = time::now(),
                total_interactions = 0,
                metadata = {{}}
        """)
        logger.info(f"✅ User query result: {query_result}")
        
        # Verify it exists immediately
        check_result = await db.query("SELECT * FROM user:test_query")
        logger.info(f"📋 User query verification: {check_result}")
        
    except Exception as e:
        logger.error(f"❌ User creation with db.query() failed: {e}")
    
    # Test 5: Check what's actually in tables now
    logger.info("\n=== CHECKING TABLE CONTENTS ===")
    
    tables = ['user', 'concept', 'message', 'session']
    for table in tables:
        try:
            count_result = await db.query(f"SELECT count() FROM {table}")
            count = count_result[0].get('count', 0) if count_result else 0
            logger.info(f"📊 {table}: {count} records")
            
            if count > 0:
                # Show samples
                samples = await db.query(f"SELECT * FROM {table} LIMIT 3")
                for i, record in enumerate(samples):
                    logger.info(f"  Sample {i+1}: {record}")
            
        except Exception as e:
            logger.error(f"Failed to check {table}: {e}")
    
    # Test 6: Transaction test
    logger.info("\n=== TESTING TRANSACTION ===")
    try:
        logger.info("Testing transaction...")
        tx_result = await db.query("""
            BEGIN TRANSACTION;
            CREATE user:test_tx SET name = 'transaction_user', first_seen = time::now(), last_seen = time::now(), total_interactions = 0, metadata = {};
            COMMIT TRANSACTION;
        """)
        logger.info(f"✅ Transaction result: {tx_result}")
        
        # Verify
        check_result = await db.query("SELECT * FROM user:test_tx")
        logger.info(f"📋 Transaction verification: {check_result}")
        
    except Exception as e:
        logger.error(f"❌ Transaction failed: {e}")
    
    await db.close()
    logger.info("✅ Persistence debugging completed")

if __name__ == "__main__":
    asyncio.run(debug_persistence())