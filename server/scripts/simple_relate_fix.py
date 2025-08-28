#!/usr/bin/env python3
"""
Simple RELATE Fix - Create relationships using basic RELATE statements that actually work
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def create_relationships():
    """Create relationships using simple RELATE statements"""
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory_graph")
    
    logger.info("🔗 Creating relationships with simple RELATE statements...")
    
    # 1. Create user->knows->concept relationships
    logger.info("Creating knows relationships...")
    users = ["user:alex", "user:peppi", "user:integration_test", "user:test_user"]
    concepts = ["concept:blue", "concept:peppy", "concept:potola", "concept:serramanna__sardinia__italy"]
    
    knows_count = 0
    for user in users:
        for concept in concepts:
            try:
                result = await db.query(f"""
                    RELATE {user}->knows->{concept} SET
                        relationship = 'personal_knowledge',
                        strength = 0.8,
                        fidelity = 3,
                        access_count = 1,
                        decay_rate = 1.0,
                        learned_at = time::now(),
                        reinforced_at = time::now(),
                        source_message = NONE
                """)
                knows_count += 1
            except Exception as e:
                logger.error(f"Failed to create knows {user} -> {concept}: {e}")
    
    logger.info(f"✅ Created {knows_count} knows relationships")
    
    # 2. Create session->contains->message relationships (sample)
    logger.info("Creating contains relationships...")
    
    # Get a few sessions and their messages
    sessions_result = await db.query("SELECT * FROM session LIMIT 3")
    messages_result = await db.query("SELECT * FROM message WHERE session_id = 'session:10e9b4672c73' LIMIT 5")
    
    contains_count = 0
    if sessions_result and messages_result:
        session_id = str(sessions_result[0]['id'])  # Use first session
        
        for message in messages_result:
            message_id = str(message['id'])
            try:
                result = await db.query(f"""
                    RELATE {session_id}->contains->{message_id} SET
                        sequence_num = {message.get('sequence_num', 1)},
                        created_at = time::now()
                """)
                contains_count += 1
            except Exception as e:
                logger.error(f"Failed to create contains {session_id} -> {message_id}: {e}")
    
    logger.info(f"✅ Created {contains_count} contains relationships")
    
    # 3. Validate relationships
    logger.info("🔍 Validating relationships...")
    
    for table in ['knows', 'contains']:
        try:
            result = await db.query(f"SELECT count() FROM {table}")
            count = result[0]['count'] if result and len(result) > 0 else 0
            logger.info(f"📊 {table}: {count} relationships")
        except Exception as e:
            logger.error(f"Failed to count {table}: {e}")
    
    # Test a graph query
    try:
        peppi_concepts = await db.query("SELECT out.name FROM knows WHERE in = user:peppi")
        concept_names = [c.get('name', 'unknown') for c in peppi_concepts]
        logger.info(f"✅ user:peppi knows concepts: {concept_names}")
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
    
    await db.close()
    logger.info("✅ Relationship creation completed")

if __name__ == "__main__":
    asyncio.run(create_relationships())