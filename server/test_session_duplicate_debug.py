#!/usr/bin/env python3
"""
Test to debug session ID and duplicate message issues
Simulates a simple conversation and checks exactly what gets stored
"""

import asyncio
import os
import sys
import time

# Set environment like the real bot
os.environ["USE_SURREALDB"] = "true"
os.environ["USER_ID"] = "peppi"
os.environ["ASSISTANT_ID"] = "slowcat"

async def test_conversation_storage():
    """Test actual conversation flow to debug storage issues"""
    
    print("🔍 Session & Storage Debug Test")
    print("=" * 50)
    
    try:
        # Import after setting environment
        from memory.surreal_connection import get_surreal_connection
        
        # Get SurrealDB connection
        surreal = get_surreal_connection()
        if not surreal:
            print("❌ Failed to get SurrealDB connection")
            return
            
        print("✅ SurrealDB connection established")
        
        # Clear existing test data
        print("\n🧹 Clearing existing test sessions...")
        try:
            # Delete all sessions for peppi to start clean
            result = await surreal.db.delete("messages", where="speaker_id = 'peppi'")
            print(f"   Deleted {len(result) if result else 0} existing peppi messages")
            
            result = await surreal.db.delete("sessions", where="speaker_id = 'peppi'")  
            print(f"   Deleted {len(result) if result else 0} existing peppi sessions")
        except Exception as e:
            print(f"   Cleanup warning: {e}")
        
        # Test 1: Create SmartContextManager like the real pipeline
        print("\n🧠 Creating SmartContextManager...")
        from processors.smart_context_manager import create_smart_context_manager
        from pipecat.services.openai import OpenAILLMContext
        
        context = OpenAILLMContext([{"role": "system", "content": "You are Slowcat."}])
        smart_ctx = create_smart_context_manager(
            context=context,
            facts_db_path="/tmp/test_facts.db",
            max_tokens=4096,
            enable_consciousness=False,
            user_id="peppi"
        )
        
        print(f"✅ SmartContextManager created")
        print(f"   user_id: {getattr(smart_ctx, '_user_id', 'NOT_SET')}")
        print(f"   surreal_store: {smart_ctx.surreal_store}")
        
        # Test 2: Simulate user message
        print("\n🎤 Simulating user message: 'Hello'")
        from pipecat.frames.frames import TranscriptionFrame
        
        user_frame = TranscriptionFrame("Hello", "", time.time())
        await smart_ctx.process_frame(user_frame, None)
        
        # Wait for async storage
        await asyncio.sleep(1)
        
        # Test 3: Check what sessions exist in DB
        print("\n📊 Checking sessions in database...")
        try:
            sessions = await surreal.db.select("sessions")
            print(f"   Total sessions: {len(sessions) if sessions else 0}")
            
            if sessions:
                for session in sessions:
                    print(f"   Session: {session.get('id', 'no-id')}")
                    print(f"     Speaker: {session.get('speaker_id', 'no-speaker')}")
                    print(f"     Created: {session.get('created_at', 'no-time')}")
                    print(f"     Status: {session.get('status', 'no-status')}")
                    print()
        except Exception as e:
            print(f"   Error checking sessions: {e}")
        
        # Test 4: Check what messages exist in DB
        print("\n💬 Checking messages in database...")
        try:
            messages = await surreal.db.select("messages")
            print(f"   Total messages: {len(messages) if messages else 0}")
            
            if messages:
                for msg in messages:
                    print(f"   Message: {msg.get('id', 'no-id')}")
                    print(f"     Role: {msg.get('role', 'no-role')}")
                    print(f"     Speaker: {msg.get('speaker_id', 'no-speaker')}")
                    print(f"     Session: {msg.get('session_id', 'no-session')}")
                    print(f"     Content: {msg.get('content', 'no-content')[:50]}...")
                    print(f"     Timestamp: {msg.get('timestamp', 'no-time')}")
                    print()
        except Exception as e:
            print(f"   Error checking messages: {e}")
            
        # Test 5: Ensure session is created first
        print("\n🔧 Ensuring session is created...")
        if smart_ctx.surreal_store:
            await smart_ctx.surreal_store._ensure_session()
            print(f"   Session ID: {smart_ctx.surreal_store.session_id}")
        
        # Test 6: Simulate assistant response  
        print("\n🤖 Simulating assistant response: 'Hello! I am Slowcat.'")
        
        # Simulate what happens when assistant responds
        assistant_response = "Hello! I am Slowcat."
        await smart_ctx._store_assistant_message(assistant_response)
        
        # Wait for async storage
        await asyncio.sleep(1)
        
        # Test 7: Check final state
        print("\n📊 Final database state...")
        try:
            sessions = await surreal.db.select("sessions")
            messages = await surreal.db.select("messages")
            
            print(f"   Final sessions: {len(sessions) if sessions else 0}")
            print(f"   Final messages: {len(messages) if messages else 0}")
            
            # Count by role
            if messages:
                user_count = sum(1 for m in messages if m.get('role') == 'user')
                assistant_count = sum(1 for m in messages if m.get('role') == 'assistant')
                print(f"   User messages: {user_count}")
                print(f"   Assistant messages: {assistant_count}")
                
                # Check for duplicates
                session_ids = set()
                for msg in messages:
                    session_ids.add(msg.get('session_id'))
                print(f"   Unique session IDs: {len(session_ids)}")
                print(f"   Session IDs: {list(session_ids)}")
                
        except Exception as e:
            print(f"   Error checking final state: {e}")
        
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_conversation_storage())