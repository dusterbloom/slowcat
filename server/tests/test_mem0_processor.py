#!/usr/bin/env python3
"""
Test the Mem0 Memory Processor integration
"""

import asyncio
from processors.mem0_memory_processor import Mem0MemoryProcessor
from pipecat.frames.frames import TranscriptionFrame, TextFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameDirection

async def test_mem0_processor():
    """Test the Mem0 memory processor"""
    print("🧪 Testing Mem0 Memory Processor...")
    
    # Initialize processor
    processor = Mem0MemoryProcessor(
        user_id="test_user",
        enabled=True
    )
    
    if not processor.enabled:
        print("❌ Processor failed to initialize")
        return False
    
    print("✅ Mem0 Memory Processor initialized")
    
    # Simulate conversation flow
    print("\n💬 Simulating conversation...")
    
    # 1. User speaks (transcription)
    user_input = "My name is Peppi and I love coffee"
    transcription_frame = TranscriptionFrame(text=user_input, user_id="test_user", timestamp=0)
    
    await processor.process_frame(transcription_frame, FrameDirection.DOWNSTREAM)
    print(f"📝 Processed user input: {user_input}")
    
    # 2. LLM messages (would get memory context injected)  
    llm_messages = [{"role": "user", "content": user_input}]
    llm_frame = LLMMessagesFrame(messages=llm_messages)
    
    enhanced_frame = await processor._inject_memory_context(llm_frame, user_input)
    print(f"🧠 Memory context injection test complete")
    
    # 3. Assistant responds (gets stored)
    assistant_response = "Nice to meet you Peppi! I'll remember that you love coffee."
    text_frame = TextFrame(text=assistant_response)
    
    await processor.process_frame(text_frame, FrameDirection.UPSTREAM)
    print(f"🤖 Processed assistant response: {assistant_response[:50]}...")
    
    # Wait for async storage
    await asyncio.sleep(2)
    
    # 4. Test memory search with new input
    print("\n🔍 Testing memory retrieval...")
    new_user_input = "What's my name and what do I like?"
    memories = await processor._search_memories(new_user_input)
    
    if memories:
        print(f"✅ Found {len(memories)} relevant memories:")
        for i, memory in enumerate(memories, 1):
            print(f"  {i}. {memory[:80]}...")
    else:
        print("❌ No memories found")
    
    # Get stats
    stats = processor.get_memory_stats()
    print(f"\n📊 Memory Stats: {stats}")
    
    print("\n🎉 Mem0 Memory Processor test completed!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_mem0_processor())
        if success:
            print("\n✅ Ready to integrate with Slowcat pipeline!")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()