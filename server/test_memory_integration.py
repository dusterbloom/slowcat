#!/usr/bin/env python3
"""
Integration test for memory system using actual bot pipeline
"""

import os
import asyncio
import time
from unittest.mock import Mock

# Load environment from .env file first
from dotenv import load_dotenv
load_dotenv(override=True)

from core.pipeline_builder import PipelineBuilder  
from core.service_factory import ServiceFactory
from pipecat.frames.frames import TranscriptionFrame, TextFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection

async def test_memory_integration():
    """Test memory system through the actual pipeline"""
    
    print("🧪 Memory Integration Test")
    print("=" * 50)
    
    # Show environment
    print(f"USE_STATELESS_MEMORY: {os.getenv('USE_STATELESS_MEMORY')}")
    print(f"USE_ENHANCED_MEMORY: {os.getenv('USE_ENHANCED_MEMORY')}")
    
    try:
        # Create service factory and pipeline builder
        service_factory = ServiceFactory()
        pipeline_builder = PipelineBuilder(service_factory)
        
        # Get memory service
        memory_service = await service_factory.get_service("memory_service")
        
        if memory_service is None:
            print("❌ No memory service found")
            return False
        
        memory_class = memory_service.__class__.__name__
        print(f"✅ Memory service active: {memory_class}")
        
        # Test message capture and storage
        print("\n📝 Testing message capture...")
        
        # CRITICAL: Send StartFrame first to initialize processor
        start_frame = StartFrame()
        await memory_service.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        print("✅ StartFrame sent to initialize processor")
        
        # Simulate user transcription
        user_message = "What is machine learning?"
        transcription_frame = TranscriptionFrame(
            text=user_message, 
            user_id="default_user", 
            timestamp=time.time()
        )
        
        print(f"Simulating transcription: '{user_message}'")
        
        # Process the frame
        await memory_service.process_frame(transcription_frame, FrameDirection.DOWNSTREAM)
        
        # Check if message was captured
        if hasattr(memory_service, 'current_user_message'):
            captured = memory_service.current_user_message
            print(f"Captured message: '{captured}'")
            
            if captured == user_message.strip():
                print("✅ User message captured successfully!")
            else:
                print(f"⚠️ Message mismatch: expected '{user_message}', got '{captured}'")
        
        # Test memory injection
        print("\n🧠 Testing memory injection...")
        
        # Get memories (should be empty for new system)
        if hasattr(memory_service, '_get_relevant_memories'):
            memories = await memory_service._get_relevant_memories(user_message, "default_user")
            print(f"Retrieved {len(memories)} memories")
        
        # Test storage
        print("\n💾 Testing memory storage...")
        
        assistant_response = "Machine learning is a subset of AI that uses algorithms to learn patterns from data."
        
        # Simulate assistant response
        response_frame = TextFrame(text=assistant_response)
        await memory_service.process_frame(response_frame, FrameDirection.DOWNSTREAM)
        
        # For enhanced memory, check hot tier
        if hasattr(memory_service, 'hot_tier'):
            print(f"Hot tier now has {len(memory_service.hot_tier)} items")
            
            if memory_service.hot_tier:
                latest = memory_service.hot_tier[-1]
                print(f"Latest memory: '{latest.content[:50]}...'")
        
        # For standard memory, check cache
        elif hasattr(memory_service, 'perfect_recall_cache'):
            print(f"Recall cache now has {len(memory_service.perfect_recall_cache)} items")
            
            if memory_service.perfect_recall_cache:
                latest = memory_service.perfect_recall_cache[-1]
                print(f"Latest memory: '{latest.content[:50]}...'")
        
        print("\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_memory_integration())
    exit(0 if success else 1)