#!/usr/bin/env python3
"""
Debug what's actually being sent to the LLM
"""

import os
os.environ["USE_STATELESS_MEMORY"] = "true"
os.environ["USE_ENHANCED_MEMORY"] = "true"

from processors.enhanced_stateless_memory import EnhancedStatelessMemoryProcessor
from processors.memory_context_aggregator import create_memory_context

def debug_actual_context():
    """Debug what context is actually sent to LLM"""
    
    print("🔍 Debugging Actual LLM Context")
    print("=" * 50)
    
    # Use the real database path from config
    from config import config
    
    memory_processor = EnhancedStatelessMemoryProcessor(
        db_path=config.stateless_memory.db_path,
        hot_tier_size=10,
        warm_tier_size=100,
        cold_tier_size=1000
    )
    
    print(f"📂 Using real database: {memory_processor.db_path}")
    print(f"🔥 Hot tier has: {len(memory_processor.hot_tier)} items")
    
    # Print all hot tier memories
    print(f"\n📋 Hot Tier Contents:")
    for i, item in enumerate(memory_processor.hot_tier):
        print(f"  {i+1}. [{item.speaker_id}]: '{item.content}'")
        if "Potola" in item.content:
            print(f"     ⭐ CONTAINS 'Potola'!")
    
    # Test memory search
    print(f"\n🔍 Testing memory search for 'dog name':")
    
    # Create context
    context = create_memory_context(
        initial_messages=[
            {"role": "system", "content": "You are Slowcat, a helpful voice assistant."}
        ],
        memory_processor=memory_processor,
        max_context_tokens=1024
    )
    
    # Set up a test conversation
    test_messages = [
        {"role": "system", "content": "You are Slowcat, a helpful voice assistant."},
        {"role": "user", "content": "Can you recall my dog's name"}
    ]
    context._messages = test_messages
    
    # Get the actual context that would be sent to LLM
    final_context = context.get_messages()
    
    print(f"\n📤 Final Context Sent to LLM:")
    print(f"   Total messages: {len(final_context)}")
    
    contains_potola = False
    for i, msg in enumerate(final_context):
        print(f"\n   Message {i+1} ({msg['role']}):")
        print(f"   '{msg['content']}'")
        
        if "Potola" in msg['content']:
            contains_potola = True
            print(f"   ⭐⭐⭐ CONTAINS 'Potola'! ⭐⭐⭐")
    
    print(f"\n🎯 DIAGNOSIS:")
    if contains_potola:
        print(f"   ✅ Context DOES contain 'Potola' - LLM should know the answer")
        print(f"   🚨 Problem: LLM is ignoring the context or can't understand the format")
    else:
        print(f"   ❌ Context does NOT contain 'Potola' - Memory retrieval failed")
        print(f"   🚨 Problem: Memory search not finding the right memories")
    
    # Test direct memory search
    print(f"\n🔬 Direct Memory Search Test:")
    memories = context._get_memories_sync("dog name")
    print(f"   Found {len(memories)} memories:")
    
    for memory in memories:
        print(f"   - [{memory.speaker_id}]: '{memory.content}'")
        if "Potola" in memory.content:
            print(f"     ⭐ This memory contains 'Potola'!")

if __name__ == "__main__":
    debug_actual_context()