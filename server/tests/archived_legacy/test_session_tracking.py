#!/usr/bin/env python3
"""
Test script to verify session count tracking in SmartContextManager
"""

import os
import sys
import asyncio
from pathlib import Path
import tempfile
from loguru import logger

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processors.smart_context_manager import SmartContextManager, SessionMetadata
from memory import create_smart_memory_system
from pipecat.services.openai import OpenAILLMContext

async def test_session_tracking():
    """Test session count persistence and tracking"""
    logger.info("🧪 Starting session tracking test...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        logger.info(f"📁 Using test database: {db_path}")
        
        # Set USER_ID for consistent testing
        os.environ['USER_ID'] = 'test_user'
        
        # Create memory system
        memory_system = create_smart_memory_system(db_path)
        
        # Create mock context
        context = OpenAILLMContext([{"role": "system", "content": "Test"}])
        
        # Create SmartContextManager
        manager = SmartContextManager(
            context=context,
            facts_db_path=db_path,
            max_tokens=4096
        )
        
        # Test 1: Initial session info (should be 0)
        logger.info("🧪 Test 1: Initial session info")
        speaker_key = manager._speaker_key()
        logger.info(f"Speaker key: {speaker_key}")
        
        initial_info = await manager._maybe_await(
            memory_system.facts_graph.get_session_info(speaker_key)
        )
        logger.info(f"Initial session info: {initial_info}")
        
        # Test 2: Start first session
        logger.info("🧪 Test 2: Start first session")
        initial_frame = await manager.get_initial_context_frame()
        logger.info(f"Initial context frame created")
        
        # Check session info after start
        after_start = await manager._maybe_await(
            memory_system.facts_graph.get_session_info(speaker_key)
        )
        logger.info(f"Session info after start: {after_start}")
        
        # Test 3: Generate dynamic prompt with session count
        logger.info("🧪 Test 3: Dynamic prompt generation")
        dynamic_prompt = await manager._generate_dynamic_prompt()
        logger.info(f"Dynamic prompt snippet (first 200 chars): {dynamic_prompt[:200]}...")
        
        # Look for session count in the prompt
        if 'Sessions:' in dynamic_prompt:
            session_line = [line for line in dynamic_prompt.split('\n') if 'Sessions:' in line][0]
            logger.info(f"✅ Found session count in prompt: {session_line}")
        else:
            logger.warning("⚠️ Session count not found in dynamic prompt")
            
        # Test 4: Create a second SmartContextManager (simulating restart)
        logger.info("🧪 Test 4: Simulate restart with new manager")
        manager2 = SmartContextManager(
            context=OpenAILLMContext([{"role": "system", "content": "Test2"}]),
            facts_db_path=db_path,
            max_tokens=4096
        )
        
        # Start second session
        second_frame = await manager2.get_initial_context_frame()
        
        # Check session count after second start
        final_info = await manager2._maybe_await(
            memory_system.facts_graph.get_session_info(speaker_key)
        )
        logger.info(f"Final session info after second start: {final_info}")
        
        # Test 5: Verify session count incremented
        expected_count = initial_info.get('session_count', 0) + 2
        actual_count = final_info.get('session_count', 0)
        
        if actual_count == expected_count:
            logger.info(f"✅ Session count correctly incremented: {initial_info.get('session_count', 0)} → {actual_count}")
        else:
            logger.error(f"❌ Session count mismatch! Expected: {expected_count}, Actual: {actual_count}")
            
        # Test 6: Generate final dynamic prompt to see updated count
        final_prompt = await manager2._generate_dynamic_prompt()
        if 'Sessions:' in final_prompt:
            session_line = [line for line in final_prompt.split('\n') if 'Sessions:' in line][0]
            logger.info(f"✅ Final session count in prompt: {session_line}")
        
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
            logger.info(f"🗑️ Cleaned up test database: {db_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {db_path}: {e}")
            
    logger.info("🧪 Session tracking test completed!")

if __name__ == "__main__":
    asyncio.run(test_session_tracking())