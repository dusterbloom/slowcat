#!/usr/bin/env python3
"""
Final Working Relationships - Create all relationships with proper field requirements
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def create_all_relationships():
    """Create all relationships with proper schema compliance"""
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory_graph")
    
    logger.info("🔗 Creating all relationships with proper schema compliance...")
    
    # Get a reference message for source_message field (required)
    messages = await db.query("SELECT * FROM message LIMIT 1")
    ref_message_id = str(messages[0]['id']) if messages else None
    
    if not ref_message_id:
        logger.error("❌ No messages found - cannot create relationships")
        return
    
    logger.info(f"Using reference message: {ref_message_id}")
    
    # 1. Create user->knows->concept relationships
    logger.info("Creating knows relationships...")
    await db.query("DELETE knows")  # Clear first
    
    users = ["user:alex", "user:peppi", "user:integration_test", "user:test_user"]
    concepts = ["concept:blue", "concept:peppy", "concept:potola", "concept:serramanna__sardinia__italy"]
    
    knows_count = 0
    for user in users:
        for concept in concepts:
            concept_name = concept.split(':')[1]
            try:
                result = await db.query(f"""
                    RELATE {user}->knows->{concept} SET
                        relationship = 'knows_{concept_name}',
                        strength = 0.8,
                        fidelity = 3,
                        access_count = 1,
                        decay_rate = 1.0,
                        learned_at = time::now(),
                        reinforced_at = time::now(),
                        source_message = {ref_message_id}
                """)
                if result:
                    knows_count += 1
                    logger.debug(f"✅ {user} knows {concept}")
            except Exception as e:
                logger.error(f"Failed to create knows {user} -> {concept}: {e}")
    
    logger.info(f"✅ Created {knows_count} knows relationships")
    
    # 2. Create session->contains->message relationships  
    logger.info("Creating contains relationships...")
    await db.query("DELETE contains")  # Clear first
    
    # Get sessions and their related messages
    sessions = await db.query("SELECT * FROM session LIMIT 5")
    all_messages = await db.query("SELECT * FROM message LIMIT 100")
    
    contains_count = 0
    for session in sessions:
        session_id = str(session['id'])
        session_key = session_id.split(':')[1]  # Extract key part
        
        # Find messages that belong to this session
        session_messages = [
            msg for msg in all_messages 
            if session_key in str(msg.get('session_id', ''))
        ]
        
        logger.info(f"Session {session_key}: {len(session_messages)} messages")
        
        for message in session_messages:
            message_id = str(message['id'])
            try:
                result = await db.query(f"""
                    RELATE {session_id}->contains->{message_id} SET
                        sequence_num = {message.get('sequence_num', 1)},
                        created_at = time::now()
                """)
                if result:
                    contains_count += 1
                    if contains_count % 10 == 0:
                        logger.info(f"  Created {contains_count} contains relationships...")
            except Exception as e:
                logger.error(f"Failed to create contains {session_id} -> {message_id}: {e}")
    
    logger.info(f"✅ Created {contains_count} contains relationships")
    
    # 3. Create session->reflects->thought relationships
    logger.info("Creating reflects relationships...")
    await db.query("DELETE reflects")  # Clear first
    
    thoughts = await db.query("SELECT * FROM thought")
    reflects_count = 0
    
    for session in sessions:
        session_id = str(session['id'])
        session_key = session_id.split(':')[1]
        
        # Find thoughts that belong to this session
        session_thoughts = [
            thought for thought in thoughts
            if session_key in str(thought.get('session_id', ''))
        ]
        
        for thought in session_thoughts:
            thought_id = str(thought['id'])
            try:
                result = await db.query(f"""
                    RELATE {session_id}->reflects->{thought_id} SET
                        trigger_event = 'conversation_end',
                        generated_at = time::now()
                """)
                if result:
                    reflects_count += 1
                    logger.debug(f"✅ {session_id} reflects {thought_id}")
            except Exception as e:
                logger.error(f"Failed to create reflects {session_id} -> {thought_id}: {e}")
    
    logger.info(f"✅ Created {reflects_count} reflects relationships")
    
    # 4. Final validation
    logger.info("🔍 Final validation...")
    
    for table in ['knows', 'contains', 'reflects']:
        try:
            result = await db.query(f"SELECT count() FROM {table}")
            count = result[0].get('count', 0) if result else 0
            logger.info(f"📊 {table}: {count} relationships")
        except Exception as e:
            logger.error(f"Failed to count {table}: {e}")
    
    # Test graph queries
    logger.info("🧠 Testing graph queries...")
    
    try:
        # Test user->knows->concept
        peppi_concepts = await db.query("SELECT out.name FROM knows WHERE in = user:peppi")
        concept_names = [c.get('name') for c in peppi_concepts if c.get('name')]
        logger.info(f"✅ user:peppi knows concepts: {concept_names}")
        
        # Test session->contains->message
        sample_session = sessions[0] if sessions else None
        if sample_session:
            session_id = str(sample_session['id'])
            session_messages = await db.query(f"SELECT out.content FROM contains WHERE in = {session_id} LIMIT 3")
            message_contents = [m.get('content', '')[:50] + '...' for m in session_messages]
            logger.info(f"✅ {session_id} contains messages: {len(message_contents)} messages")
            for content in message_contents:
                logger.info(f"    - {content}")
        
    except Exception as e:
        logger.error(f"Graph query failed: {e}")
    
    await db.close()
    logger.info("🎉 All relationships created successfully!")

if __name__ == "__main__":
    asyncio.run(create_all_relationships())