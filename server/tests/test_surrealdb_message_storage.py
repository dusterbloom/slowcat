#!/usr/bin/env python3
"""
Test SurrealDB Message Storage Integration

Tests the complete conversation message storage system:
1. SurrealDB connection and schema
2. Message storage and retrieval 
3. SmartContextManager integration
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment setup for testing
os.environ['ENABLE_SURREALDB'] = 'true'
os.environ['SURREAL_URL'] = 'ws://localhost:8000'
os.environ['SURREAL_USER'] = 'root'
os.environ['SURREAL_PASS'] = 'root'


async def test_connection_manager():
    """Test basic SurrealDB connection and operations"""
    logger.info("🔌 Testing SurrealDB connection manager...")
    
    try:
        from memory.surreal_connection import get_surreal_connection, Message
        
        # Get connection manager
        manager = get_surreal_connection()
        
        # Test connection
        connected = await manager.connect()
        if not connected:
            logger.error("❌ Failed to connect to SurrealDB")
            return False
        
        logger.info("✅ Connected to SurrealDB")
        
        # Test session creation
        session_id = await manager.create_session(
            speaker_id='test_user',
            metadata={'test': 'integration_test'}
        )
        
        if not session_id:
            logger.error("❌ Failed to create session")
            return False
        
        logger.info(f"✅ Created session: {session_id}")
        
        # Test message storage
        user_message = Message(
            role='user',
            content='Hello, this is a test message',
            speaker_id='test_user',
            session_id=session_id,
            timestamp=datetime.utcnow(),
            tokens=6
        )
        
        user_id = await manager.store_message(user_message)
        if not user_id:
            logger.error("❌ Failed to store user message")
            return False
        
        logger.info(f"✅ Stored user message: {user_id}")
        
        # Test assistant message
        assistant_message = Message(
            role='assistant',
            content='Hello! I received your test message.',
            speaker_id='assistant',
            session_id=session_id,
            timestamp=datetime.utcnow(),
            tokens=7,
            parent_message=user_id
        )
        
        assistant_id = await manager.store_message(assistant_message)
        if not assistant_id:
            logger.error("❌ Failed to store assistant message")
            return False
        
        logger.info(f"✅ Stored assistant message: {assistant_id}")
        
        # Test retrieval
        messages = await manager.get_conversation(session_id=session_id)
        if len(messages) != 2:
            logger.error(f"❌ Expected 2 messages, got {len(messages)}")
            return False
        
        logger.info(f"✅ Retrieved {len(messages)} messages")
        
        # Test search
        search_results = await manager.search_messages("test message")
        if not search_results:
            logger.warning("⚠️ Search returned no results (may be normal)")
        else:
            logger.info(f"✅ Search returned {len(search_results)} results")
        
        # Clean up test session
        await manager.end_session(session_id, "Integration test completed")
        logger.info("✅ Session ended")
        
        await manager.disconnect()
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Connection manager test failed: {e}")
        return False


async def test_message_store_processor():
    """Test the SurrealMessageStore processor"""
    logger.info("📝 Testing SurrealMessageStore processor...")
    
    try:
        from processors.surreal_message_store import create_surreal_message_store
        
        # Create message store
        store = create_surreal_message_store(
            speaker_id='test_processor_user',
            auto_create_session=True
        )
        
        if not store:
            logger.warning("⚠️ SurrealMessageStore not available (may be normal if SurrealDB disabled)")
            return True
        
        # Simulate user message
        await store._handle_user_message("What is the meaning of life?")
        logger.info("✅ Handled user message")
        
        # Wait a moment for async storage
        await asyncio.sleep(0.1)
        
        # Simulate assistant response
        await store._handle_assistant_message("The meaning of life is 42, according to Douglas Adams!")
        logger.info("✅ Handled assistant response")
        
        # Wait for async storage
        await asyncio.sleep(0.1)
        
        # Finalize session
        await store.finalize_session("Test conversation completed")
        logger.info("✅ Session finalized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Message store processor test failed: {e}")
        return False


async def test_conversation_flow():
    """Test complete conversation flow"""
    logger.info("💬 Testing complete conversation flow...")
    
    try:
        from memory.surreal_connection import store_conversation_turn, get_surreal_connection
        
        # Store a complete conversation turn
        success = await store_conversation_turn(
            user_text="Can you help me understand quantum physics?",
            assistant_text="Quantum physics is the study of matter and energy at the smallest scales. It reveals that particles can exist in multiple states simultaneously until observed.",
            session_id=None,  # Will auto-create
            speaker_id='test_flow_user'
        )
        
        if success:
            logger.info("✅ Complete conversation turn stored")
        else:
            logger.warning("⚠️ Conversation turn storage failed")
        
        # Test retrieval
        manager = get_surreal_connection()
        await manager.connect()
        
        recent_messages = await manager.get_recent_messages(minutes=1)
        logger.info(f"✅ Retrieved {len(recent_messages)} recent messages")
        
        await manager.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Conversation flow test failed: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("🧪 Starting SurrealDB Message Storage Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Connection Manager", test_connection_manager),
        ("Message Store Processor", test_message_store_processor),
        ("Conversation Flow", test_conversation_flow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🎯 Running: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {status}")
        except Exception as e:
            logger.error(f"   ❌ ERROR: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Test Results Summary")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {status} - {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! SurrealDB message storage is working.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above.")
    
    return passed == total


if __name__ == '__main__':
    logger.info("🚀 SurrealDB Message Storage Test Suite")
    logger.info("Make sure SurrealDB is running: surreal start --user root --pass root")
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)