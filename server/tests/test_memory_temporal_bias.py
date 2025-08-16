#!/usr/bin/env python3
"""
Test if MemoBase has temporal bias preventing old memories from surfacing.
"""

import asyncio
import sys
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from memobase import MemoBaseClient
    from memobase.patch.openai import openai_memory
    from openai import OpenAI
    from memobase.utils import string_to_uuid
    
    async def test_temporal_bias():
        """Test if older memories are being overshadowed by newer ones"""
        print("📅 Testing temporal bias in memory retrieval...")
        
        mb_client = MemoBaseClient(
            project_url="http://localhost:8019",
            api_key="secret"
        )
        
        user_id = "default_user"
        
        openai_client = OpenAI(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )
        
        patched_client = openai_memory(openai_client, mb_client, max_context_size=1000)
        
        # Get full memory
        memory_prompt = await asyncio.to_thread(
            patched_client.get_memory_prompt,
            user_id
        )
        
        print(f"📊 Full memory length: {len(memory_prompt)} chars")
        
        # Analyze memory by date
        lines = memory_prompt.split('\n')
        
        memories_by_date = {
            '2025/08/15': [],
            '2025/08/16': [],
            'no_date': []
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if '2025/08/15' in line:
                memories_by_date['2025/08/15'].append(line)
            elif '2025/08/16' in line:
                memories_by_date['2025/08/16'].append(line)
            else:
                memories_by_date['no_date'].append(line)
        
        print(f"\n📅 Memory distribution by date:")
        for date, items in memories_by_date.items():
            print(f"   {date}: {len(items)} items")
            
            # Show dog-related memories specifically
            dog_items = [item for item in items if any(word in item.lower() for word in ['dog', 'pet', 'Bobby'])]
            if dog_items:
                print(f"      🐕 Dog-related: {len(dog_items)} items")
                for item in dog_items[:3]:  # Show first 3
                    print(f"         📄 {item[:80]}...")
        
        # Test specific queries with different memory limits
        test_queries = [
            "Tell me about my dog",
            "What is my dog's name?", 
            "Do you remember Bobby?",
            "What pets do I have?"
        ]
        
        # Test with different max_context_size to see if that affects retrieval
        for context_size in [200, 500, 1000]:
            print(f"\n🧪 Testing with max_context_size={context_size}")
            
            limited_client = openai_memory(openai_client, mb_client, max_context_size=context_size)
            
            limited_memory = await asyncio.to_thread(
                limited_client.get_memory_prompt,
                user_id
            )
            
            has_dog_info = any(word in limited_memory.lower() for word in ['Bobby', 'dog named'])
            has_recent_info = 'favorite number' in limited_memory.lower()
            
            print(f"   📊 Memory size: {len(limited_memory)} chars")
            print(f"   🐕 Contains dog info: {has_dog_info}")
            print(f"   🔢 Contains recent number: {has_recent_info}")
            
            if has_dog_info:
                # Extract dog lines
                dog_lines = [line for line in limited_memory.split('\n') 
                           if any(word in line.lower() for word in ['Bobby', 'dog named'])]
                for line in dog_lines[:2]:
                    print(f"      📄 {line.strip()}")

    if __name__ == "__main__":
        asyncio.run(test_temporal_bias())
        
except ImportError as e:
    print(f"❌ MemoBase not available: {e}")