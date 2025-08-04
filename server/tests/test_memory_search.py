#!/usr/bin/env python3
"""Quick test script to verify memory search functionality"""

import asyncio
import sys
import os

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from tools.handlers import execute_tool_call, set_memory_processor
from processors.local_memory import LocalMemoryProcessor
from pipecat.frames.frames import TranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection
from loguru import logger


async def quick_test():
    """Quick test of memory search functionality"""
    logger.info("🚀 Quick Memory Search Test")
    
    # Create memory processor
    memory = LocalMemoryProcessor(
        data_dir="data/quick_test",
        user_id="test_user",
        max_history_items=100,
        include_in_context=10
    )
    
    # Set memory processor for tools
    set_memory_processor(memory)
    
    # Add some test data
    logger.info("📝 Adding test conversations...")
    test_data = [
        ("user", "My favorite color is blue"),
        ("assistant", "Blue is a great color! It's calming and peaceful."),
        ("user", "I live in San Francisco"),
        ("assistant", "San Francisco is a beautiful city! How do you like living there?"),
        ("user", "I love the weather here"),
        ("assistant", "San Francisco does have great weather, especially the mild temperatures year-round."),
    ]
    
    for role, content in test_data:
        if role == "user":
            frame = TranscriptionFrame(text=content, user_id="test_user", timestamp=0)
        else:
            frame = TextFrame(text=content)
        await memory.process_frame(frame, FrameDirection.UPSTREAM if role == "user" else FrameDirection.DOWNSTREAM)
    
    await asyncio.sleep(0.5)  # Wait for async writes
    
    # Test search_conversations tool
    logger.info("\n🔍 Testing search_conversations tool...")
    result = await execute_tool_call("search_conversations", {"query": "favorite color"})
    logger.info(f"Search result: {result}")
    
    # Test get_conversation_summary tool
    logger.info("\n📊 Testing get_conversation_summary tool...")
    result = await execute_tool_call("get_conversation_summary", {"days_back": 1})
    logger.info(f"Summary result: {result}")
    
    # Test searching for location
    logger.info("\n🏙️ Testing location search...")
    result = await execute_tool_call("search_conversations", {"query": "San Francisco"})
    logger.info(f"Location search result: {result}")
    
    logger.info("\n✅ Quick test completed!")


if __name__ == "__main__":
    asyncio.run(quick_test())