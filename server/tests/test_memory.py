#!/usr/bin/env python3
"""Test script for local memory functionality"""

import asyncio
import sys
import os
from pathlib import Path
import shutil

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from processors.local_memory import LocalMemoryProcessor
from processors.memory_context_injector import MemoryContextInjector
from pipecat.frames.frames import TextFrame, TranscriptionFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection
from loguru import logger


async def test_memory():
    """Test the local memory processor with SQLite storage"""
    logger.info("🧪 Testing Local Memory Processor...")
    
    # Clean up test directory
    test_dir = Path("data/test_memory")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create memory processor
    async with LocalMemoryProcessor(
        data_dir="data/test_memory",
        user_id="test_user",
        max_history_items=20,
        include_in_context=5
    ) as memory:
        
        # Simulate some conversations
        logger.info("\n📝 Simulating conversations...")
        
        # User message 1 - Use TranscriptionFrame for user input
        user_frame1 = TranscriptionFrame(text="Hello, my name is John", user_id="test_user")
        await memory.process_frame(user_frame1, FrameDirection.UPSTREAM)
        
        # Assistant response 1
        assistant_frame1 = TextFrame(text="Hello John! Nice to meet you.")
        await memory.process_frame(assistant_frame1, FrameDirection.DOWNSTREAM)
        
        # User message 2
        user_frame2 = TranscriptionFrame(text="What's the weather like today?", user_id="test_user")
        await memory.process_frame(user_frame2, FrameDirection.UPSTREAM)
        
        # Assistant response 2
        assistant_frame2 = TextFrame(text="I can help you check the weather. Let me look that up for you.")
        await memory.process_frame(assistant_frame2, FrameDirection.DOWNSTREAM)
        
        # User message 3
        user_frame3 = TranscriptionFrame(text="Can you tell me about Python programming?", user_id="test_user")
        await memory.process_frame(user_frame3, FrameDirection.UPSTREAM)
        
        # Assistant response 3
        assistant_frame3 = TextFrame(text="Python is a versatile programming language known for its simplicity.")
        await memory.process_frame(assistant_frame3, FrameDirection.DOWNSTREAM)
        
        # Give time for async writes to complete
        await asyncio.sleep(0.5)
        
        # Test context retrieval
        logger.info("\n🔍 Testing context retrieval...")
        context_messages = await memory.get_context_messages()
        logger.info(f"Context messages (last {len(context_messages)}):")
        for msg in context_messages:
            logger.info(f"  {msg['role']}: {msg['content'][:50]}...")
        
        # Test conversation search
        logger.info("\n🔎 Testing conversation search...")
        
        # Search for "weather"
        weather_results = await memory.search_conversations("weather", limit=5)
        logger.info(f"Search for 'weather' found {len(weather_results)} results:")
        for result in weather_results:
            logger.info(f"  [{result['role']}] {result['content'][:50]}...")
        
        # Search for "John"
        john_results = await memory.search_conversations("John", limit=5)
        logger.info(f"\nSearch for 'John' found {len(john_results)} results:")
        for result in john_results:
            logger.info(f"  [{result['role']}] {result['content'][:50]}...")
        
        # Test conversation summary
        logger.info("\n📊 Testing conversation summary...")
        summary = await memory.get_conversation_summary(days_back=1)
        logger.info(f"Conversation summary (last 1 day):")
        logger.info(f"  Total messages: {summary['total_messages']}")
        logger.info(f"  User messages: {summary['user_messages']}")
        logger.info(f"  Assistant messages: {summary['assistant_messages']}")
        logger.info(f"  Recent topics: {summary['recent_topics']}")
    
    # Test persistence - create new instance
    logger.info("\n💾 Testing memory persistence...")
    async with LocalMemoryProcessor(
        data_dir="data/test_memory",
        user_id="test_user",
        max_history_items=20,
        include_in_context=5
    ) as memory2:
        
        # Get context from persisted memory
        persisted_context = await memory2.get_context_messages()
        logger.info(f"Persisted context messages: {len(persisted_context)}")
        
        # Search persisted memory
        persisted_search = await memory2.search_conversations("Python", limit=5)
        logger.info(f"Search persisted memory for 'Python': {len(persisted_search)} results")
        
        # Test memory context injector
        logger.info("\n🔗 Testing Memory Context Injector...")
        
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
                content = msg['content'][:200] if len(msg['content']) > 200 else msg['content']
                logger.info(f"  [{msg['role']}] {content}...")
        else:
            logger.warning("No enhanced frame captured")
    
    logger.info("\n✅ All memory tests completed!")


async def test_tool_integration():
    """Test the integration of memory search with tool handlers"""
    logger.info("\n🔧 Testing Tool Integration...")
    
    from tools.handlers import ToolHandlers, set_memory_processor
    
    # Clean up and setup
    test_dir = Path("data/test_memory_tools")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create memory processor with some test data
    async with LocalMemoryProcessor(
        data_dir=str(test_dir),
        user_id="tool_test_user",
        max_history_items=50,
        include_in_context=10
    ) as memory:
        
        # Add some test conversations
        test_conversations = [
            ("user", "What's the capital of France?"),
            ("assistant", "The capital of France is Paris."),
            ("user", "Tell me about the Eiffel Tower"),
            ("assistant", "The Eiffel Tower is an iconic iron lattice tower in Paris, built in 1889."),
            ("user", "How tall is it?"),
            ("assistant", "The Eiffel Tower is 330 meters (1,083 feet) tall."),
        ]
        
        for role, content in test_conversations:
            if role == "user":
                frame = TranscriptionFrame(text=content, user_id="tool_test_user")
            else:
                frame = TextFrame(text=content)
            await memory.process_frame(frame, FrameDirection.UPSTREAM if role == "user" else FrameDirection.DOWNSTREAM)
        
        await asyncio.sleep(0.5)  # Wait for async writes
        
        # Set memory processor for tools
        set_memory_processor(memory)
        
        # Create tool handlers
        tool_handlers = ToolHandlers(memory_processor=memory)
        
        # Test search_conversations tool
        logger.info("\n🔍 Testing search_conversations tool...")
        search_result = await tool_handlers.search_conversations("Eiffel Tower", limit=3)
        logger.info(f"Search result: {search_result}")
        
        # Test get_conversation_summary tool
        logger.info("\n📊 Testing get_conversation_summary tool...")
        summary_result = await tool_handlers.get_conversation_summary(days_back=1)
        logger.info(f"Summary result: {summary_result}")
    
    logger.info("\n✅ Tool integration tests completed!")


if __name__ == "__main__":
    asyncio.run(test_memory())
    asyncio.run(test_tool_integration())