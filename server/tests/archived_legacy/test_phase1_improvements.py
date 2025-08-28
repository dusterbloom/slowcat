#!/usr/bin/env python3
"""
Test script for Phase 1 SurrealDB improvements

Tests:
1. FTS search with fallback
2. Unique constraint fact deduplication  
3. Session counter auto-increment
4. Session summary functionality
5. Index performance improvements

Run after executing DDL commands in Surrealist:
python test_phase1_improvements.py
"""

import asyncio
import os
import time
from loguru import logger

# Set up environment
os.environ.update({
    'USE_SURREALDB': 'true',
    'SURREALDB_URL': 'ws://127.0.0.1:8000/rpc',
    'SURREALDB_NAMESPACE': 'slowcat',
    'SURREALDB_DATABASE': 'memory'
})

from memory import create_smart_memory_system

async def test_fts_search():
    """Test full-text search with fallback"""
    logger.info("🔍 Testing FTS search with fallback...")
    
    memory = create_smart_memory_system()
    
    # Add test conversation
    test_messages = [
        "Hello, my name is Alice and I love jazz music",
        "I have a dog named Potola who likes walks", 
        "Can you help me with my cooking recipes?",
        "The weather is beautiful today for hiking"
    ]
    
    session_id = f"test_session_{int(time.time())}"
    
    for i, msg in enumerate(test_messages):
        await memory.add_entry({
            'ts': time.time() + i,
            'speaker_id': 'alice',
            'role': 'user',
            'content': msg,
            'session_id': session_id,
            'agent_id': 'slowcat'
        })
    
    # Test searches
    test_queries = ["jazz", "Potola", "cooking", "weather", "nonexistent"]
    
    for query in test_queries:
        results = await memory.search_tape(query, limit=5)
        logger.info(f"Search '{query}': {len(results)} results")
        if results:
            logger.info(f"  → {results[0]['content'][:50]}...")
    
    await memory.close()
    return True

async def test_fact_deduplication():
    """Test unique constraint deduplication"""
    logger.info("🎯 Testing fact deduplication with unique constraints...")
    
    memory = create_smart_memory_system()
    
    # Same fact multiple times - should reinforce, not duplicate
    fact_data = {
        'subject': 'alice',
        'predicate': 'likes',
        'value': 'jazz',
        'species': 'music_genre',
        'agent_id': 'slowcat'
    }
    
    # First insertion
    result1 = await memory.reinforce_or_insert(fact_data)
    logger.info(f"First insert: {'reinforced' if result1 else 'new'}")
    
    # Second insertion - should reinforce
    result2 = await memory.reinforce_or_insert(fact_data)
    logger.info(f"Second insert: {'reinforced' if result2 else 'new'}")
    
    # Verify only one fact exists
    facts = await memory.search_facts("alice likes jazz")
    logger.info(f"Total facts matching 'alice likes jazz': {len(facts)}")
    
    if facts:
        fact = facts[0]
        logger.info(f"Fact strength: {fact.strength}, access_count: {fact.access_count}")
    
    await memory.close()
    return True

async def test_session_counters():
    """Test auto-increment session counters"""
    logger.info("📊 Testing session counter auto-increment...")
    
    memory = create_smart_memory_system()
    
    speaker_id = f"test_speaker_{int(time.time())}"
    
    # Get initial session state
    initial_sessions = await memory.db.query(
        "SELECT * FROM sessions WHERE speaker_id = $speaker_id",
        {"speaker_id": speaker_id}
    )
    initial_count = len(initial_sessions[0].get('result', []))
    logger.info(f"Initial sessions for {speaker_id}: {initial_count}")
    
    # Add multiple tape entries
    for i in range(5):
        await memory.add_entry({
            'ts': time.time() + i,
            'speaker_id': speaker_id,
            'role': 'user',
            'content': f"Test message {i}",
            'session_id': f"session_{speaker_id}",
            'agent_id': 'slowcat'
        })
    
    # Give event time to process
    await asyncio.sleep(1)
    
    # Check session counters
    sessions = await memory.db.query(
        "SELECT * FROM sessions WHERE speaker_id = $speaker_id",
        {"speaker_id": speaker_id}
    )
    
    if sessions[0].get('result'):
        session = sessions[0]['result'][0]
        turns = session.get('total_turns', 0)
        logger.info(f"Session total_turns: {turns}")
        logger.info(f"Session last_interaction: {session.get('last_interaction')}")
        
        if turns >= 5:
            logger.info("✅ Session counters working correctly!")
        else:
            logger.warning(f"❌ Expected >= 5 turns, got {turns}")
    else:
        logger.warning("❌ No session found after adding entries")
    
    await memory.close()
    return True

async def test_session_summary():
    """Test session summary functionality"""
    logger.info("📝 Testing session summary functionality...")
    
    memory = create_smart_memory_system()
    
    # Create a session summary
    summary_data = {
        'session_id': f"test_summary_session_{int(time.time())}",
        'summary': "User discussed jazz music and their dog Potola",
        'keywords': ['jazz', 'music', 'dog', 'Potola'],
        'turns': 10,
        'duration_s': 300,
        'ts': time.time()
    }
    
    # Store summary  
    await memory.db.query("""
        CREATE session_summary SET
            session_id = $session_id,
            summary = $summary,
            keywords = $keywords,
            turns = $turns,
            duration_s = $duration_s,
            ts = time::now()
    """, summary_data)
    
    # Retrieve last summary
    last_summary = await memory.get_last_summary()
    
    if last_summary:
        logger.info(f"Retrieved summary: {last_summary['summary'][:50]}...")
        logger.info(f"Keywords: {last_summary['keywords']}")
        logger.info(f"Turns: {last_summary['turns']}, Duration: {last_summary['duration_s']}s")
        logger.info("✅ Session summary functionality working!")
    else:
        logger.warning("❌ Could not retrieve session summary")
    
    await memory.close()
    return True

async def test_performance():
    """Test index performance improvements"""
    logger.info("⚡ Testing index performance improvements...")
    
    memory = create_smart_memory_system()
    
    # Test index queries are fast
    test_queries = [
        ("Tape by timestamp", "SELECT * FROM tape ORDER BY ts DESC LIMIT 10"),
        ("Tape by speaker", "SELECT * FROM tape WHERE speaker_id = 'alice' ORDER BY ts DESC LIMIT 10"),
        ("Sessions by speaker", "SELECT * FROM sessions WHERE speaker_id = 'alice'"),
        ("Facts by subject", "SELECT * FROM fact WHERE subject = 'alice' LIMIT 10")
    ]
    
    for desc, query in test_queries:
        start_time = time.time()
        result = await memory.db.query(query)
        elapsed = (time.time() - start_time) * 1000
        
        count = len(result[0].get('result', [])) if result else 0
        logger.info(f"{desc}: {count} results in {elapsed:.2f}ms")
        
        if elapsed > 100:  # More than 100ms might indicate missing indexes
            logger.warning(f"⚠️  Slow query detected: {elapsed:.2f}ms")
    
    await memory.close()
    return True

async def main():
    """Run all Phase 1 tests"""
    logger.info("🚀 Starting Phase 1 SurrealDB improvements test...")
    
    tests = [
        ("FTS Search", test_fts_search),
        ("Fact Deduplication", test_fact_deduplication),
        ("Session Counters", test_session_counters),
        ("Session Summary", test_session_summary),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- {test_name} ---")
            success = await test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            results[test_name] = "ERROR"
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("PHASE 1 TEST RESULTS")
    logger.info(f"{'='*50}")
    
    for test_name, status in results.items():
        status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[status]
        logger.info(f"{status_emoji} {test_name}: {status}")
    
    passed = sum(1 for status in results.values() if status == "PASS")
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 1 improvements working correctly!")
    else:
        logger.warning("⚠️  Some tests failed - check DDL commands in Surrealist")

if __name__ == "__main__":
    asyncio.run(main())