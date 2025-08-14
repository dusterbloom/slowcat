#!/usr/bin/env python3
"""
Debug what Pipecat's Mem0MemoryService actually receives
"""

import asyncio
import sys
sys.path.append('.')
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from core.service_factory import service_factory

async def debug_pipecat_memory():
    print("🔍 Debugging Pipecat's Mem0MemoryService...")
    
    try:
        # Get the memory service from our factory
        memory_service = await service_factory.get_service("memory_service")
        
        if memory_service is None:
            print("❌ Memory service is None (disabled)")
            return
            
        print(f"✅ Memory service type: {type(memory_service)}")
        print(f"   User ID: {memory_service.user_id}")
        print(f"   Agent ID: {memory_service.agent_id}")
        print(f"   Search limit: {memory_service.search_limit}")
        print(f"   System prompt: '{memory_service.system_prompt}'")
        
        # Test direct memory operations
        print("\n📝 Testing memory add...")
        if hasattr(memory_service.memory_client, 'add'):
            add_result = memory_service.memory_client.add("My favorite food is sushi", user_id=memory_service.user_id)
            print(f"   Add result: {add_result}")
        
        print("\n🔍 Testing memory search...")
        if hasattr(memory_service.memory_client, 'search'):
            search_result = memory_service.memory_client.search("What food do I like?", user_id=memory_service.user_id)
            print(f"   Search result type: {type(search_result)}")
            print(f"   Search result: {search_result}")
            
            if isinstance(search_result, dict) and 'results' in search_result:
                results = search_result['results']
                print(f"   Results count: {len(results)}")
                for i, result in enumerate(results):
                    print(f"     {i+1}. {result}")
        
        # Test the internal retrieve method
        print("\n🔍 Testing _retrieve_memories...")
        if hasattr(memory_service, '_retrieve_memories'):
            retrieve_result = memory_service._retrieve_memories("What food do I like?")
            print(f"   Retrieve result type: {type(retrieve_result)}")
            print(f"   Retrieve result: {retrieve_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_pipecat_memory())