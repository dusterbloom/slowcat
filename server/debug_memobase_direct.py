#!/usr/bin/env python3
"""
Debug MemoBase memory retrieval directly using the MemoBase client.
This tests if the issue is in our processor or in MemoBase itself.
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
    
    async def test_memobase_direct():
        """Test MemoBase memory retrieval directly"""
        print("🔍 Testing MemoBase direct memory retrieval...")
        
        # Initialize MemoBase client
        mb_client = MemoBaseClient(
            project_url="http://localhost:8019",
            api_key="secret"
        )
        
        user_id = "default_user"
        uuid_for_user = string_to_uuid(user_id)
        print(f"User ID: {user_id} -> UUID: {uuid_for_user}")
        
        # Test connection
        if mb_client.ping():
            print("✅ MemoBase connection successful")
        else:
            print("❌ MemoBase connection failed")
            return
        
        # Test OpenAI client with MemoBase patching
        openai_client = OpenAI(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )
        
        # Apply MemoBase patching
        patched_client = openai_memory(openai_client, mb_client, max_context_size=500)
        
        # Test memory retrieval methods
        print("\n🧠 Testing memory retrieval methods...")
        
        # Method 1: get_memory_prompt
        if hasattr(patched_client, 'get_memory_prompt'):
            try:
                memory_prompt = await asyncio.to_thread(
                    patched_client.get_memory_prompt,
                    user_id
                )
                print(f"📝 get_memory_prompt result ({len(memory_prompt)} chars):")
                print("=" * 50)
                print(memory_prompt)
                print("=" * 50)
            except Exception as e:
                print(f"❌ get_memory_prompt failed: {e}")
        
        # Method 2: context method  
        if hasattr(patched_client, 'context'):
            try:
                context_result = await asyncio.to_thread(
                    patched_client.context,
                    user_id=user_id,
                    max_tokens=500,
                    query="What do you remember about my dog?"
                )
                print(f"\n🔍 context method result ({len(context_result)} chars):")
                print("=" * 50)
                print(context_result)
                print("=" * 50)
            except Exception as e:
                print(f"❌ context method failed: {e}")
        
        # Method 3: Direct API test
        print(f"\n🌐 Testing direct API call...")
        import requests
        try:
            response = requests.get(
                f"http://localhost:8019/api/v1/users/{uuid_for_user}/memory",
                headers={"Authorization": "Bearer secret"},
                timeout=5
            )
            print(f"API Response: {response.status_code}")
            if response.status_code == 200:
                print(f"API Result: {response.json()}")
            else:
                print(f"API Error: {response.text}")
        except Exception as e:
            print(f"❌ Direct API call failed: {e}")
            
        print("\n✅ MemoBase direct test completed")

    if __name__ == "__main__":
        asyncio.run(test_memobase_direct())
        
except ImportError as e:
    print(f"❌ MemoBase not available: {e}")