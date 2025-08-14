#!/usr/bin/env python3
"""
Full test of Mem0 storage and retrieval cycle with structured output
"""

import asyncio
import os
from mem0 import Memory

async def test_mem0_full_cycle():
    print("🧪 Testing Mem0 Full Storage & Retrieval Cycle...")
    
    # Use the exact same config as our voice agent
    local_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 768,
                "collection_name": "test_slowcat_cycle"
            }
        },
        "embedder": {
            "provider": "lmstudio", 
            "config": {
                "model": "text-embedding-nomic-embed-text-v1.5",
                "lmstudio_base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "embedding_dims": 768
            }
        },
        "llm": {
            "provider": "lmstudio",
            "config": {
                "model": "qwen/qwen3-1.7b",
                "lmstudio_base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "temperature": 0.1,
                "max_tokens": 500,
                "lmstudio_response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object", 
                        "schema": {
                            "type": "object",
                            "properties": {
                                "facts": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "description": "A factual statement about the user"
                                    }
                                }
                            },
                            "required": ["facts"]
                        }
                    }
                }
            }
        },
        "version": "v1.1"
    }
    
    try:
        # Initialize Mem0
        print("📝 Initializing Mem0 with structured output...")
        memory = Memory.from_config(local_config)
        user_id = "test_user_cycle"
        
        # Test 1: Store some facts
        print("\n🔹 Test 1: Storing personal facts...")
        test_messages = [
            "My name is Alice Johnson and I'm 28 years old",
            "I live in San Francisco and work as a software engineer", 
            "My favorite programming language is Python",
            "I have a golden retriever named Max"
        ]
        
        for message in test_messages:
            print(f"   Storing: '{message[:50]}...'")
            result = memory.add(message, user_id=user_id)
            print(f"   ✅ Stored: {result}")
        
        print(f"\n📊 Checking Qdrant collections...")
        import requests
        collections = requests.get("http://localhost:6333/collections").json()
        print(f"   Collections: {collections}")
        
        # Test 2: Search for specific memories  
        print(f"\n🔹 Test 2: Searching for memories...")
        search_queries = [
            "What is my name?",
            "Where do I live?", 
            "What programming language do I like?",
            "Tell me about my pet"
        ]
        
        for query in search_queries:
            print(f"   Query: '{query}'")
            results = memory.search(query, user_id=user_id)
            print(f"   Results: {len(results)} memories found")
            for i, result in enumerate(results, 1):
                if hasattr(result, 'memory'):
                    print(f"      {i}. {result.memory}")
                elif isinstance(result, dict) and 'memory' in result:
                    print(f"      {i}. {result['memory']}")
                else:
                    print(f"      {i}. {result}")
        
        # Test 3: Get all memories
        print(f"\n🔹 Test 3: Getting all stored memories...")
        all_memories = memory.get_all(user_id=user_id)
        print(f"   Total memories for {user_id}: {len(all_memories)}")
        for i, mem in enumerate(all_memories, 1):
            if hasattr(mem, 'memory'):
                print(f"      {i}. {mem.memory}")
            elif isinstance(mem, dict) and 'memory' in mem:
                print(f"      {i}. {mem['memory']}")
        
        print(f"\n✅ Full cycle test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in full cycle test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mem0_full_cycle())
    if success:
        print("🎉 Memory system is working properly!")
    else:
        print("🚨 Memory system has issues - check the errors above")