#!/usr/bin/env python3
"""
Test specific memory retrieval for dog-related queries.
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
    
    async def test_dog_memory_queries():
        """Test memory retrieval for dog-specific queries"""
        print("🐕 Testing dog memory retrieval...")
        
        # Initialize MemoBase client
        mb_client = MemoBaseClient(
            project_url="http://localhost:8019",
            api_key="secret"
        )
        
        user_id = "default_user"
        uuid_for_user = string_to_uuid(user_id)
        
        # Test OpenAI client with MemoBase patching
        openai_client = OpenAI(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )
        
        # Apply MemoBase patching
        patched_client = openai_memory(openai_client, mb_client, max_context_size=500)
        
        # Test different dog-related queries
        dog_queries = [
            "Do you remember my dog's name?",
            "What do you know about my dog?", 
            "Tell me about my pet",
            "What is my dog called?",
            "I'm asking about Bobby"
        ]
        
        print("\n🔍 Testing dog-related queries:")
        for query in dog_queries:
            print(f"\n📝 Query: '{query}'")
            
            try:
                # Test context method with query
                if hasattr(patched_client, 'context'):
                    context_result = await asyncio.to_thread(
                        patched_client.context,
                        user_id=user_id,
                        max_tokens=300,
                        query=query
                    )
                    
                    # Check if dog/Bobby information is in result
                    has_dog_info = any(word in context_result.lower() for word in ['Bobby', 'dog', 'pet', 'orangey'])
                    status = "✅ Found" if has_dog_info else "❌ Missing"
                    
                    print(f"   {status} dog info in context")
                    if has_dog_info:
                        # Extract the relevant lines
                        lines = context_result.split('\n')
                        for line in lines:
                            if any(word in line.lower() for word in ['Bobby', 'dog', 'pet', 'orangey']):
                                print(f"   📄 {line.strip()}")
                    
                    print(f"   📊 Context length: {len(context_result)} chars")
                else:
                    print("   ⚠️ Context method not available")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Test general memory retrieval
        print(f"\n🧠 General memory retrieval:")
        try:
            if hasattr(patched_client, 'get_memory_prompt'):
                general_memory = await asyncio.to_thread(
                    patched_client.get_memory_prompt,
                    user_id
                )
                
                has_dog_info = any(word in general_memory.lower() for word in ['Bobby', 'dog', 'pet', 'orangey'])
                status = "✅ Found" if has_dog_info else "❌ Missing"
                print(f"   {status} dog info in general memory")
                print(f"   📊 General memory length: {len(general_memory)} chars")
                
                if has_dog_info:
                    print("   📄 Dog-related lines:")
                    lines = general_memory.split('\n')
                    for line in lines:
                        if any(word in line.lower() for word in ['Bobby', 'dog', 'pet', 'orangey']):
                            print(f"      {line.strip()}")
                            
        except Exception as e:
            print(f"   ❌ General memory failed: {e}")

    if __name__ == "__main__":
        asyncio.run(test_dog_memory_queries())
        
except ImportError as e:
    print(f"❌ MemoBase not available: {e}")