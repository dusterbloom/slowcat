#!/usr/bin/env python3
"""
Simple test to verify SurrealDB integration is working
"""
import asyncio
import os
from datetime import datetime, timezone
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def test_surreal_integration():
    """Test the complete SurrealDB integration"""
    print("🧪 Testing SurrealDB Integration...")
    
    try:
        # Test 1: Basic connection
        print("\n1. Testing SurrealDB connection...")
        from memory.surreal_connection import SurrealConnectionManager
        
        conn = SurrealConnectionManager()
        await conn.ensure_connected()
        print("✅ Connection successful")
        
        # Test 2: Session creation and counting
        print("\n2. Testing session creation...")
        session_id = await conn.start_session('test_user')
        print(f"✅ Created session: {session_id}")
        
        # Get session info
        info = await conn.get_session_info('test_user')
        print(f"✅ Session info: {info}")
        
        # Test 3: Message storage
        print("\n3. Testing message storage...")
        from memory.surreal_connection import Message
        
        # Store user message
        user_msg = Message(
            role='user',
            content='Hello, this is a test message',
            speaker_id='test_user',
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            tokens=8
        )
        
        user_id = await conn.store_message(user_msg)
        print(f"✅ Stored user message: {user_id}")
        
        # Store assistant message
        assistant_msg = Message(
            role='assistant', 
            content='Hi there! I received your test message.',
            speaker_id='assistant',
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            tokens=10
        )
        
        assistant_id = await conn.store_message(assistant_msg)
        print(f"✅ Stored assistant message: {assistant_id}")
        
        # Test 4: Message retrieval
        print("\n4. Testing message retrieval...")
        messages = await conn.search_messages('test message', limit=5)
        print(f"✅ Found {len(messages)} messages")
        for msg in messages:
            print(f"   - {msg.get('role')}: {msg.get('content')[:50]}...")
            
        # Test 5: Session count increment
        print("\n5. Testing session count increment...")
        session_id2 = await conn.start_session('test_user')
        info2 = await conn.get_session_info('test_user')
        print(f"✅ New session info: {info2}")
        
        if info2.get('session_count', 0) > info.get('session_count', 0):
            print("✅ Session count incremented correctly!")
        else:
            print("❌ Session count did not increment")
            
        # Test 6: Facts storage (if available)
        print("\n6. Testing facts storage...")
        try:
            facts_stored = await conn.store_facts([
                {'subject': 'test_user', 'predicate': 'name', 'value': 'Test User'},
                {'subject': 'test_user', 'predicate': 'likes', 'value': 'testing'}
            ])
            print(f"✅ Stored {facts_stored} facts")
        except Exception as e:
            print(f"⚠️ Facts storage failed: {e}")
        
        await conn.disconnect()
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(test_surreal_integration())