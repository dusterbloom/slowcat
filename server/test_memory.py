#!/usr/bin/env python3
"""Test script for local memory functionality"""

import asyncio
import sys
import os
from pathlib import Path

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from processors.local_memory import LocalMemoryProcessor
from processors.memory_context_injector import MemoryContextInjector
from pipecat.frames.frames import TextFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection
from loguru import logger


async def test_memory():
    """Test the local memory processor"""
    logger.info("Testing Local Memory Processor...")
    
    # Create memory processor
    memory = LocalMemoryProcessor(
        data_dir="data/test_memory",
        user_id="test_user",
        max_history_items=10,
        include_in_context=5
    )
    
    # Simulate some conversations
    logger.info("Simulating conversation...")
    
    # User message 1
    user_frame1 = TextFrame(text="Hello, my name is John")
    await memory.process_frame(user_frame1, FrameDirection.UPSTREAM)
    
    # Assistant response 1
    assistant_frame1 = TextFrame(text="Hello John! Nice to meet you.")
    await memory.process_frame(assistant_frame1, FrameDirection.DOWNSTREAM)
    
    # User message 2
    user_frame2 = TextFrame(text="What's the weather like?")
    await memory.process_frame(user_frame2, FrameDirection.UPSTREAM)
    
    # Assistant response 2
    assistant_frame2 = TextFrame(text="I'm sorry, I don't have access to weather data.")
    await memory.process_frame(assistant_frame2, FrameDirection.DOWNSTREAM)
    
    # Save memory
    memory._save_memory()
    
    # Check saved memory
    logger.info(f"Memory items saved: {len(memory.memory)}")
    logger.info("Recent context messages:")
    for msg in memory.get_context_messages():
        logger.info(f"  {msg['role']}: {msg['content']}")
    
    # Test memory persistence
    logger.info("\nTesting memory persistence...")
    
    # Create new memory instance
    memory2 = LocalMemoryProcessor(
        data_dir="data/test_memory",
        user_id="test_user",
        max_history_items=10,
        include_in_context=5
    )
    
    logger.info(f"Loaded memory items: {len(memory2.memory)}")
    logger.info("Loaded context messages:")
    for msg in memory2.get_context_messages():
        logger.info(f"  {msg['role']}: {msg['content']}")
    
    # Test memory context injector
    logger.info("\nTesting Memory Context Injector...")
    
    injector = MemoryContextInjector(
        memory_processor=memory2,
        system_prompt="Based on our previous conversations:",
        inject_as_system=True
    )
    
    # Create a test LLM messages frame
    test_messages = [
        {"role": "user", "content": "Do you remember my name?"}
    ]
    test_frame = LLMMessagesFrame(messages=test_messages)
    
    # Process through injector
    processed_frames = []
    
    async def capture_frame(frame, direction):
        processed_frames.append(frame)
    
    injector.push_frame = capture_frame
    await injector.process_frame(test_frame, FrameDirection.UPSTREAM)
    
    # Check enhanced messages
    if processed_frames:
        enhanced_frame = processed_frames[0]
        logger.info("\nEnhanced messages:")
        for msg in enhanced_frame.messages:
            logger.info(f"  {msg['role']}: {msg['content'][:100]}...")
    
    # Clean up test data
    test_dir = Path("data/test_memory")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        logger.info("\nTest data cleaned up")
    
    logger.info("\n✅ Memory test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_memory())