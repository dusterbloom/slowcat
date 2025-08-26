#!/usr/bin/env python3
"""
Simple test of SmartContextManager with actual query
"""

import asyncio
import os
import sys
from pathlib import Path

# Add server path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

async def test_smart_context_simple():
    """Test SmartContextManager with simple query"""
    
    print("🧠 SmartContextManager Simple Test")
    print("=" * 40)
    
    try:
        from processors.smart_context_manager import SmartContextManager
        from memory import create_smart_memory_system
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.processors.frame_processor import FrameDirection
        
        # Create memory system
        memory_system = create_smart_memory_system()
        if hasattr(memory_system, 'surreal_memory'):
            await memory_system.surreal_memory.connect()
        
        # Create a mock context object
        class MockContext:
            def __init__(self):
                self.messages = []
        
        # Create SmartContextManager  
        context_manager = SmartContextManager(
            context=MockContext(),
            max_tokens=4400
        )
        
        print(f"✅ SmartContextManager created")
        print(f"   Has query router: {context_manager.query_router is not None}")
        
        if context_manager.query_router:
            router_type = str(type(context_manager.query_router).__name__)
            print(f"   Router type: {router_type}")
        
        # Test with simple query
        test_frame = TranscriptionFrame(text="what is my dog's name", user_id="peppi", timestamp=1756127600.0)
        
        print(f"\n🔍 Testing with query: '{test_frame.text}'")
        
        # Process the frame (this calls the memory retrieval)
        await context_manager.process_frame(test_frame, FrameDirection.DOWNSTREAM)
        
        print(f"✅ Frame processed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_smart_context_simple())