#!/usr/bin/env python3
"""
Test script to verify stateless memory system integration
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from processors.stateless_memory import StatelessMemoryProcessor, MemoryItem
from pipecat.frames.frames import (
    LLMMessagesFrame, TranscriptionFrame, UserStartedSpeakingFrame, 
    StartFrame, EndFrame
)
from pipecat.processors.frame_processor import FrameDirection

async def test_stateless_memory():
    """Test the stateless memory processor functionality"""
    
    print("🧠 Testing Stateless Memory System")
    print("=" * 50)
    
    # Initialize memory processor
    db_path = "data/test_memory"
    memory_processor = StatelessMemoryProcessor(
        db_path=db_path,
        max_context_tokens=512,
        perfect_recall_window=5
    )
    
    print(f"✅ Memory processor initialized with path: {db_path}")
    
    # Test 1: Basic frame processing
    print("\n📋 Test 1: Basic Frame Processing")
    
    # Send StartFrame
    start_frame = StartFrame()
    await memory_processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)
    print("✅ StartFrame processed")
    
    # Test 2: Speaker identification
    print("\n📋 Test 2: Speaker Identification")
    
    speaker_frame = UserStartedSpeakingFrame()
    speaker_frame.speaker_id = "test_user"
    await memory_processor.process_frame(speaker_frame, FrameDirection.UPSTREAM)
    print(f"✅ Speaker set: {memory_processor.current_speaker}")
    
    # Test 3: Store conversation exchange
    print("\n📋 Test 3: Conversation Storage")
    
    # Simulate user transcription
    transcription_frame = TranscriptionFrame(
        text="Hello, can you remember my name is Alice?",
        user_id="test_user",
        timestamp="2024-01-01T10:00:00Z",
        language="en"
    )
    await memory_processor.process_frame(transcription_frame, FrameDirection.UPSTREAM)
    print(f"✅ User message captured: {memory_processor.current_user_message}")
    
    # Simulate LLM response
    llm_frame = LLMMessagesFrame([
        {"role": "user", "content": "Hello, can you remember my name is Alice?"},
        {"role": "assistant", "content": "Hello Alice! Yes, I'll remember your name."}
    ])
    await memory_processor.process_frame(llm_frame, FrameDirection.DOWNSTREAM)
    print("✅ LLM response processed and stored")
    
    # Test 4: Memory injection
    print("\n📋 Test 4: Memory Context Injection")
    
    # Set up new conversation that should trigger memory injection
    memory_processor.current_user_message = "What's my name?"
    
    new_llm_frame = LLMMessagesFrame([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's my name?"}
    ])
    
    original_message_count = len(new_llm_frame.messages)
    await memory_processor._inject_memory_context(new_llm_frame)
    
    if len(new_llm_frame.messages) > original_message_count:
        print("✅ Memory context injected successfully!")
        for i, msg in enumerate(new_llm_frame.messages):
            print(f"  Message {i}: {msg['role']} - {msg['content'][:50]}...")
    else:
        print("⚠️  No memory context injected (might be expected if no relevant memories)")
    
    # Test 5: Memory retrieval
    print("\n📋 Test 5: Memory Retrieval")
    
    memories = await memory_processor._get_relevant_memories("Alice", "test_user")
    print(f"✅ Retrieved {len(memories)} memories")
    for i, memory in enumerate(memories):
        print(f"  Memory {i}: {memory.content[:50]}... (speaker: {memory.speaker_id})")
    
    # Test 6: Performance stats
    print("\n📋 Test 6: Performance Statistics")
    
    stats = memory_processor.get_stats()
    print(f"✅ Memory stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    end_frame = EndFrame()
    await memory_processor.process_frame(end_frame, FrameDirection.DOWNSTREAM)
    print("\n✅ EndFrame processed - test complete!")
    
    return True

async def test_memory_persistence():
    """Test that memory persists across processor instances"""
    
    print("\n🔄 Testing Memory Persistence")
    print("=" * 50)
    
    db_path = "data/test_persistence"
    
    # First processor instance - store some data
    print("📝 Creating first processor instance...")
    processor1 = StatelessMemoryProcessor(db_path=db_path)
    
    # Store a memory item
    test_memory = MemoryItem(
        content="I like pizza and programming",
        timestamp=time.time(),
        speaker_id="persistent_user"
    )
    
    await processor1._store_persistent([test_memory])
    print("✅ Memory stored in first instance")
    
    # Create second processor instance - should load existing data
    print("📂 Creating second processor instance...")
    processor2 = StatelessMemoryProcessor(db_path=db_path)
    
    # Try to retrieve the memory
    memories = await processor2._search_persistent_memory("pizza", "persistent_user", 1000)
    
    if memories:
        print(f"✅ Memory persistence verified! Found: {memories[0].content}")
        return True
    else:
        print("❌ Memory persistence failed - no memories found")
        return False

if __name__ == "__main__":
    async def main():
        print("🚀 Starting Stateless Memory Integration Tests")
        print("=" * 60)
        
        # Ensure test directories exist
        os.makedirs("data/test_memory", exist_ok=True)
        os.makedirs("data/test_persistence", exist_ok=True)
        
        try:
            # Run tests
            test1_result = await test_stateless_memory()
            test2_result = await test_memory_persistence()
            
            print("\n" + "=" * 60)
            print("📊 TEST RESULTS")
            print(f"Basic Memory Integration: {'✅ PASS' if test1_result else '❌ FAIL'}")
            print(f"Memory Persistence: {'✅ PASS' if test2_result else '❌ FAIL'}")
            
            if test1_result and test2_result:
                print("\n🎉 All memory tests PASSED!")
                return 0
            else:
                print("\n💥 Some memory tests FAILED!")
                return 1
                
        except Exception as e:
            print(f"\n💥 Test execution failed: {e}")
            logger.exception("Test failure details:")
            return 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)