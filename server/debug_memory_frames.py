#!/usr/bin/env python3
"""
Debug script to check if memory processor is receiving frames
"""

import os
import tempfile
import shutil
from pathlib import Path

# Set environment before importing
os.environ["USE_STATELESS_MEMORY"] = "true"
os.environ["USE_ENHANCED_MEMORY"] = "true"

from processors.enhanced_stateless_memory import EnhancedStatelessMemoryProcessor

def debug_memory_frames():
    """Debug memory frame reception"""
    
    print("🔍 Memory Frame Debug")
    print("=" * 40)
    
    # Create a clean temporary database
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create enhanced memory processor
        memory_processor = EnhancedStatelessMemoryProcessor(
            db_path=temp_dir,
            hot_tier_size=5,
            warm_tier_size=10,
            cold_tier_size=20
        )
        
        print(f"✅ Enhanced memory processor created")
        print(f"   Hot tier size: {len(memory_processor.hot_tier)}")
        print(f"   Database path: {memory_processor.db_path}")
        
        # Test direct storage (bypass frame processing)
        print("\n📝 Testing direct message storage...")
        
        user_message = "My favorite color is blue"
        assistant_response = "I'll remember that your favorite color is blue!"
        
        # Store directly in hot tier
        from processors.enhanced_stateless_memory import MemoryItem, MemoryTier
        import time
        
        user_memory = MemoryItem(
            content=user_message,
            timestamp=time.time(),
            speaker_id="default_user",
            tier=MemoryTier.HOT
        )
        
        assistant_memory = MemoryItem(
            content=assistant_response,
            timestamp=time.time() + 0.001,
            speaker_id="assistant",
            tier=MemoryTier.HOT
        )
        
        memory_processor.hot_tier.extend([user_memory, assistant_memory])
        
        print(f"✅ Stored memories directly in hot tier")
        print(f"   Hot tier now has: {len(memory_processor.hot_tier)} items")
        
        # Test search
        print(f"\n🔍 Testing memory search...")
        
        # Create a simple async context for testing
        import asyncio
        
        async def test_search():
            memories = await memory_processor._search_hot_tier(
                "favorite color", 
                "default_user", 
                500
            )
            return memories
        
        found_memories = asyncio.run(test_search())
        print(f"   Found {len(found_memories)} relevant memories")
        
        for memory in found_memories:
            print(f"   - {memory.speaker_id}: '{memory.content}'")
        
        if found_memories:
            print("✅ Memory search is working!")
        else:
            print("❌ Memory search failed")
            
            # Test the relevance function directly
            print("\n🔍 Testing relevance function...")
            is_relevant = memory_processor._is_relevant_enhanced(user_message, "favorite color")
            print(f"   Is '{user_message}' relevant to 'favorite color'? {is_relevant}")
            
            # Test with exact match
            is_relevant_exact = memory_processor._is_relevant_enhanced(user_message, "blue")
            print(f"   Is '{user_message}' relevant to 'blue'? {is_relevant_exact}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up temporary directory")

if __name__ == "__main__":
    debug_memory_frames()