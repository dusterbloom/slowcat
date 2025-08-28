#!/usr/bin/env python3
"""
Test the SmartContextManager recent conversation fix
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

async def test_context_fix():
    """Test that recent conversation is now included in context"""
    
    print("🧠 SmartContextManager Recent Context Fix Test")
    print("=" * 50)
    
    try:
        from processors.smart_context_manager import SmartContextManager
        from memory import create_smart_memory_system
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.processors.frame_processor import FrameDirection
        
        # Create memory system
        memory_system = create_smart_memory_system()
        if hasattr(memory_system, 'surreal_memory'):
            await memory_system.surreal_memory.connect()
        
        # Create a mock context object that captures messages
        class MockContext:
            def __init__(self):
                self.messages = []
                
            def set_messages(self, messages):
                self.messages = messages
                print(f"📨 Context received {len(messages)} messages:")
                for i, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:100]
                    print(f"   [{i}] {role}: {content}...")
        
        mock_context = MockContext()
        
        # Create SmartContextManager  
        context_manager = SmartContextManager(
            context=mock_context,
            max_tokens=4400
        )
        
        print(f"✅ SmartContextManager created")
        
        # Simulate a few conversation turns
        print(f"\n🎭 Simulating conversation turns...")
        
        # Turn 1
        await context_manager.process_frame(
            TranscriptionFrame(text="Hello, who are you?", user_id="peppi", timestamp=1000), 
            FrameDirection.DOWNSTREAM
        )
        await context_manager.add_assistant_response("I'm Slowcat, your AI assistant.")
        
        # Turn 2 
        await context_manager.process_frame(
            TranscriptionFrame(text="What's the weather like?", user_id="peppi", timestamp=2000), 
            FrameDirection.DOWNSTREAM
        )
        await context_manager.add_assistant_response("I don't have access to current weather data.")
        
        # Turn 3 - This should now have recent context
        print(f"\n🔍 Final test with recent context...")
        await context_manager.process_frame(
            TranscriptionFrame(text="What did I just ask about?", user_id="peppi", timestamp=3000), 
            FrameDirection.DOWNSTREAM
        )
        
        print(f"\n📊 Recent exchanges tracked: {len(context_manager.recent_exchanges)}")
        for i, exchange in enumerate(context_manager.recent_exchanges):
            user_msg = exchange[0] if len(exchange) > 0 else "No user msg"
            asst_msg = exchange[1] if len(exchange) > 1 else "No assistant msg"
            print(f"   [{i}] User: {user_msg[:50]}...")
            print(f"       Assistant: {asst_msg[:50]}...")
        
        # Check if recent context was included in messages
        recent_context_found = any(
            msg.get('role') in ['user', 'assistant'] and 
            msg.get('content') != "What did I just ask about?"  # Not the current input
            for msg in mock_context.messages
        )
        
        if recent_context_found:
            print(f"\n✅ SUCCESS: Recent conversation context is now included in LLM messages!")
        else:
            print(f"\n❌ FAILURE: Recent conversation context still missing from LLM messages")
            
        return recent_context_found
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_context_fix())
    exit(0 if success else 1)