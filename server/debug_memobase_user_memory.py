#!/usr/bin/env python3
"""
Debug script to check MemoBase memory for the default_user.
This will help identify why memories are being lost or overwritten.
"""

import asyncio
import sys
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent))

from config import config
from processors.memobase_memory_processor import MemobaseMemoryProcessor


async def debug_memobase_memory():
    """Debug MemoBase memory state for default_user."""
    print("🔍 Debugging MemoBase memory for default_user...")
    
    processor = MemobaseMemoryProcessor(user_id="default_user")
    
    if not processor.is_enabled:
        print("❌ MemoBase is not enabled")
        return
    
    try:
        # Get memory stats
        stats = await processor.get_memory_stats()
        print(f"📊 Memory Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        print()
        
        # Try to get current memory context
        print("🧠 Current memory context for default_user:")
        if hasattr(processor.patched_sync_client, 'get_memory_prompt'):
            try:
                memory_prompt = await asyncio.to_thread(
                    processor.patched_sync_client.get_memory_prompt,
                    "default_user"
                )
                print(f"Raw memory ({len(memory_prompt)} chars):")
                print("=" * 50)
                print(memory_prompt)
                print("=" * 50)
                print()
                
                # Test deduplication
                deduplicated = processor._deduplicate_memory_content(memory_prompt)
                print(f"After deduplication ({len(deduplicated)} chars):")
                print("=" * 30)
                print(deduplicated)
                print("=" * 30)
                
            except Exception as e:
                print(f"❌ Failed to get memory prompt: {e}")
        
        # Check if we can manually add a test memory
        print("\n🧪 Testing memory addition...")
        await processor._add_to_conversation_buffer(
            "user", 
            "TEST: My dog's name is Bobby and I want to remember this",
            {"test": True}
        )
        print("✅ Test memory added")
        
        # Wait a moment for async operations
        await asyncio.sleep(1)
        
        # Try to retrieve memory again
        print("\n🔍 Memory after test addition:")
        if hasattr(processor.patched_sync_client, 'get_memory_prompt'):
            try:
                updated_memory = await asyncio.to_thread(
                    processor.patched_sync_client.get_memory_prompt,
                    "default_user"
                )
                print(f"Updated memory ({len(updated_memory)} chars):")
                print("=" * 40)
                print(updated_memory)
                print("=" * 40)
                
            except Exception as e:
                print(f"❌ Failed to get updated memory: {e}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await processor.cleanup()


if __name__ == "__main__":
    asyncio.run(debug_memobase_memory())