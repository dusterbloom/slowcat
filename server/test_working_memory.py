#!/usr/bin/env python3
"""
Test memory exactly as the voice agent does it
"""

import asyncio
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
import os
from dotenv import load_dotenv
load_dotenv()

from core.service_factory import service_factory

async def test_memory_like_voice_agent():
    print("🧪 Testing Memory Like Voice Agent...")
    
    try:
        # Get memory service exactly like pipeline does
        memory_service = await service_factory.get_service("memory_service")
        
        print(f"✅ Memory service ready: {type(memory_service)}")
        
        # Create context like the voice agent does
        context = OpenAILLMContext()
        context.add_message({"role": "user", "content": "My favorite color is purple"})
        
        # Create context frame
        context_frame = OpenAILLMContextFrame(context=context)
        
        print("🔄 Processing frame through memory service...")
        
        # Process like pipeline does
        await memory_service.process_frame(context_frame, None)
        
        print("✅ Frame processed successfully")
        
        # Now test retrieval with another context
        print("\n🔍 Testing memory retrieval...")
        
        retrieval_context = OpenAILLMContext()
        retrieval_context.add_message({"role": "user", "content": "Do you remember my favorite color?"})
        
        retrieval_frame = OpenAILLMContextFrame(context=retrieval_context)
        
        # Process retrieval
        await memory_service.process_frame(retrieval_frame, None)
        
        print("✅ Retrieval frame processed")
        
        # Check if context was enhanced
        messages = retrieval_context.get_messages()
        print(f"\n📋 Final context messages ({len(messages)}):")
        for i, msg in enumerate(messages):
            print(f"  {i+1}. Role: {msg['role']}")
            print(f"     Content: {msg['content'][:100]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_like_voice_agent())