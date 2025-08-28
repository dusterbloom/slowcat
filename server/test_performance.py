#!/usr/bin/env python3
"""
Quick performance test to validate indexes are working
"""
import asyncio
import time
from dotenv import load_dotenv
from memory.surreal_connection import SurrealConnectionManager

load_dotenv()

async def test_query_performance():
    """Test query performance with indexes"""
    print("⚡ Testing SurrealDB query performance...")
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        # Test 1: Session lookup by speaker_id (should use idx_sessions_speaker_id)
        start = time.time()
        sessions = await conn.db.query("SELECT * FROM sessions WHERE speaker_id = 'test_user';")
        session_time = time.time() - start
        print(f"✅ Session lookup: {len(sessions)} results in {session_time:.3f}s")
        
        # Test 2: Message lookup by session_id (should use idx_messages_session_id)  
        start = time.time()
        if sessions:
            session_id = sessions[0]['session_id']
            messages = await conn.db.query("SELECT * FROM messages WHERE session_id = $session_id;", 
                                         {'session_id': session_id})
            message_time = time.time() - start
            print(f"✅ Message lookup: {len(messages)} results in {message_time:.3f}s")
        
        # Test 3: Recent messages by timestamp (should use idx_messages_timestamp)
        start = time.time()
        recent = await conn.db.query("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 10;")
        recent_time = time.time() - start
        print(f"✅ Recent messages: {len(recent)} results in {recent_time:.3f}s")
        
        # Test 4: Speaker-role composite query (should use idx_messages_speaker_role)
        start = time.time()
        user_msgs = await conn.db.query("SELECT * FROM messages WHERE speaker_id = 'test_user' AND role = 'user' LIMIT 5;")
        composite_time = time.time() - start
        print(f"✅ Composite query: {len(user_msgs)} results in {composite_time:.3f}s")
        
        total_queries = 4
        total_time = session_time + message_time + recent_time + composite_time
        avg_time = total_time / total_queries
        
        print(f"\n📊 Performance Summary:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average per query: {avg_time:.3f}s")
        
        if avg_time < 0.1:
            print("🚀 EXCELLENT - Sub-100ms average query time!")
        elif avg_time < 0.5:
            print("✅ GOOD - Sub-500ms average query time")
        else:
            print("⚠️ SLOW - Consider optimizing queries or adding more indexes")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await conn.disconnect()

if __name__ == '__main__':
    asyncio.run(test_query_performance())