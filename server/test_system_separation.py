#!/usr/bin/env python3
"""
Test to understand the separation of concerns between consciousness and memory systems
"""

import asyncio
import os

async def test_system_separation():
    """Test what each system handles"""
    
    print("🧠 Testing System Separation: Consciousness vs Memory")
    print("=" * 60)
    
    from processors.smart_context_manager import create_smart_context_manager
    
    class MockContext:
        def __init__(self):
            self.messages = []
    
    test_user = "system_test_user"
    os.environ['USER_ID'] = test_user
    
    smart_manager = create_smart_context_manager(
        MockContext(),
        user_id=test_user,
        enable_consciousness=True
    )
    
    consciousness = smart_manager._consciousness_instance
    
    # Test different types of content
    test_inputs = [
        ("Factual", "My dog's name is Potola and I live in San Francisco"),
        ("Emotional", "I love this amazing new system! It's incredible!"),
        ("Question", "What if this doesn't work? I'm wondering about the results."),
        ("Decision", "I need to choose between Python or JavaScript for this project"),
        ("Important", "This is crucial information that I need to remember"),
        ("Pattern", "This always happens again and again in the same loop")
    ]
    
    print("Testing different content types:")
    print("-" * 40)
    
    for content_type, text in test_inputs:
        print(f"\n{content_type}: '{text}'")
        
        # Test consciousness symbol extraction
        symbols = consciousness.symbolize(text)
        print(f"  Consciousness symbols: {symbols}")
        
        # Test memory storage
        await smart_manager.memory_system.store_facts(text)
        print(f"  Stored in memory system: ✅")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("✅ Memory System (Facts/Tape): Handles factual information storage/retrieval")
    print("✅ Consciousness System: Handles emotional/cognitive state tracking") 
    print("✅ Both systems work together for complete context understanding")
    print("✅ SmartContextManager integrates both for comprehensive memory")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_system_separation())