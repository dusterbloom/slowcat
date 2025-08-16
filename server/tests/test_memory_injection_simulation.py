#!/usr/bin/env python3
"""
Simulate the memory injection process to see what's going wrong.
"""

import asyncio
import sys
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent))

from processors.memobase_memory_processor import MemobaseMemoryProcessor

async def simulate_memory_injection():
    """Simulate memory injection for dog query"""
    print("🧪 Simulating memory injection for dog query...")
    
    processor = MemobaseMemoryProcessor(user_id="default_user")
    
    if not processor.is_enabled:
        print("❌ MemoBase processor not enabled")
        return
    
    user_query = "Do you remember my dog's name?"
    
    print(f"📝 User query: '{user_query}'")
    
    # Test memory retrieval
    try:
        memory_context = await processor._get_contextual_memory(user_query)
        print(f"🧠 Retrieved memory ({len(memory_context)} chars)")
        
        # Test deduplication
        deduplicated = processor._deduplicate_memory_content(memory_context)
        print(f"🔄 After deduplication ({len(deduplicated)} chars)")
        
        # Test compression
        compressed = processor._compress_memory_content(memory_context)
        print(f"🗜️ After compression ({len(compressed)} chars)")
        
        # Check if dog info survives the process
        dog_keywords = ['Bobby', 'dog', 'pet', 'orangey']
        
        print(f"\n🔍 Dog info presence check:")
        print(f"   Original: {any(word in memory_context.lower() for word in dog_keywords)}")
        print(f"   Deduplicated: {any(word in deduplicated.lower() for word in dog_keywords)}")
        print(f"   Compressed: {any(word in compressed.lower() for word in dog_keywords)}")
        
        if any(word in compressed.lower() for word in dog_keywords):
            print(f"\n✅ Dog info preserved in final memory:")
            lines = compressed.split('\n')
            for line in lines:
                if any(word in line.lower() for word in dog_keywords):
                    print(f"   📄 {line.strip()}")
        else:
            print(f"\n❌ Dog info LOST during processing!")
            print(f"Final compressed memory:")
            print("=" * 50)
            print(compressed)
            print("=" * 50)
        
    except Exception as e:
        print(f"❌ Memory injection simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simulate_memory_injection())