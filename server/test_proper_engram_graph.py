#!/usr/bin/env python3
"""
Test the new proper graph-based engram system
"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def create_test_data():
    """Create test knowledge and sessions for engram testing"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    logger.info("🧪 Creating test data for engram system...")
    
    # Create test sessions
    await conn.db.query("""
        CREATE sessions:test_session1 SET 
            session_id = "test_session1",
            start_time = time::now() - 1h,
            is_active = false;
        
        CREATE sessions:test_session2 SET 
            session_id = "test_session2", 
            start_time = time::now() - 30m,
            is_active = true;
    """)
    
    # Create test knowledge with good confidence/strength
    test_knowledge = [
        {
            "session_id": "test_session1",
            "predicate": "has_pet",
            "confidence": 0.9,
            "strength": 0.8,
            "in": {"canonical_name": "user"},
            "out": {"canonical_name": "Potola"}
        },
        {
            "session_id": "test_session1", 
            "predicate": "loves",
            "confidence": 0.85,
            "strength": 0.7,
            "in": {"canonical_name": "user"},
            "out": {"canonical_name": "dogs"}
        },
        {
            "session_id": "test_session1",
            "predicate": "is_type", 
            "confidence": 0.95,
            "strength": 0.9,
            "in": {"canonical_name": "Potola"},
            "out": {"canonical_name": "dog"}
        },
        {
            "session_id": "test_session2",
            "predicate": "enjoys",
            "confidence": 0.8,
            "strength": 0.6,
            "in": {"canonical_name": "user"},
            "out": {"canonical_name": "programming"}
        },
        {
            "session_id": "test_session2",
            "predicate": "uses",
            "confidence": 0.75,
            "strength": 0.65,
            "in": {"canonical_name": "user"}, 
            "out": {"canonical_name": "Python"}
        }
    ]
    
    for i, knowledge in enumerate(test_knowledge):
        await conn.db.query(f"""
            CREATE knowledge:test_fact_{i} SET
                session_id = "{knowledge['session_id']}",
                predicate = "{knowledge['predicate']}",
                confidence = {knowledge['confidence']},
                strength = {knowledge['strength']},
                in = {{ canonical_name: "{knowledge['in']['canonical_name']}" }},
                out = {{ canonical_name: "{knowledge['out']['canonical_name']}" }},
                created_at = time::now();
        """)
    
    logger.info("✅ Test data created")
    return conn

async def test_engram_detection():
    """Test the engram detection function with real data"""
    
    conn = await create_test_data()
    
    logger.info("🎯 Testing engram detection...")
    
    # Test with session 1 (should create engram)
    result1 = await conn.db.query("""
        RETURN fn::detect_engrams_graph("test_session1", 0.6, 2);
    """)
    
    logger.info(f"🔍 Raw result1: {result1}")
    
    if result1:
        # SurrealDB RETURN statement returns the result directly
        result = result1
        logger.info(f"📋 Session 1 result: {result}")
        
        if result.get('success'):
            logger.info(f"✅ SUCCESS! Created engram: '{result.get('narrative')}'")
            logger.info(f"🔍 Coherence: {result.get('coherence', 0):.2f}")
            logger.info(f"📊 Pattern hash: {result.get('pattern_hash')}")
            
            # Test graph queries
            engram_id = result.get('engram_id')
            if engram_id:
                # Test getting knowledge from engram
                knowledge_result = await conn.db.query("""
                    RETURN fn::get_engram_knowledge($engram_id);
                """, {"engram_id": engram_id})
                
                if knowledge_result and len(knowledge_result) > 0:
                    knowledge_count = len(knowledge_result[0]) if knowledge_result[0] else 0
                    logger.info(f"🔗 Engram linked to {knowledge_count} knowledge facts via RELATE edges")
                
                # Test getting sessions from engram
                session_result = await conn.db.query("""
                    RETURN fn::get_engram_sessions($engram_id);
                """, {"engram_id": engram_id})
                
                if session_result and len(session_result) > 0:
                    session_count = len(session_result[0]) if session_result[0] else 0
                    logger.info(f"🔗 Engram appears in {session_count} sessions via RELATE edges")
        else:
            logger.warning(f"⚠️ Session 1 failed: {result.get('reason')}")
    
    # Test with session 2 (should create different engram)
    result2 = await conn.db.query("""
        RETURN fn::detect_engrams_graph("test_session2", 0.6, 2);
    """)
    
    if result2:
        result = result2
        logger.info(f"📋 Session 2 result: {result}")
        
        if result.get('success'):
            logger.info(f"✅ SUCCESS! Created engram: '{result.get('narrative')}'")
            logger.info(f"🔍 Coherence: {result.get('coherence', 0):.2f}")
        else:
            logger.warning(f"⚠️ Session 2 failed: {result.get('reason')}")
    
    # Test reinforcement - run session 1 again (should reinforce existing engram)
    logger.info("🔄 Testing engram reinforcement...")
    result3 = await conn.db.query("""
        RETURN fn::detect_engrams_graph("test_session1", 0.6, 2);
    """)
    
    if result3:
        result = result3
        if result.get('success') and result.get('action') == 'reinforced':
            logger.info(f"✅ SUCCESS! Reinforced existing engram")
            logger.info(f"🔍 Updated coherence: {result.get('coherence', 0):.2f}")
        else:
            logger.info(f"📋 Reinforcement result: {result}")
    
    # Verify engrams table has proper structure
    logger.info("📊 Checking engrams table...")
    engrams = await conn.db.query("SELECT * FROM engrams;")
    
    if engrams:
        logger.info(f"🧠 Found {len(engrams)} engrams total")
        for i, engram in enumerate(engrams):
            logger.info(f"   Engram {i+1}: '{engram.get('narrative_summary', 'NO NARRATIVE')}'")
            logger.info(f"      Symbols: {engram.get('dominant_symbols', [])}")
            logger.info(f"      Pattern: {engram.get('pattern_hash', 'NO_HASH')}")
    
    # Test graph relation tables
    logger.info("🔗 Checking graph relations...")
    
    contains_relations = await conn.db.query("SELECT count() FROM engram_contains;")
    appears_relations = await conn.db.query("SELECT count() FROM engram_appears_in;")
    
    if contains_relations:
        logger.info(f"📊 engram_contains relations: {contains_relations[0]}")
    if appears_relations:
        logger.info(f"📊 engram_appears_in relations: {appears_relations[0]}")
    
    logger.info("🎉 Engram graph testing complete!")

async def main():
    logger.info("🚀 Testing Proper Graph-Based Engram System...")
    await test_engram_detection() 
    logger.info("✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())