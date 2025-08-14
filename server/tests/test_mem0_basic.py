#!/usr/bin/env python3
"""
Quick test of Mem0 basic functionality with LM Studio
"""

import asyncio
import os
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig

async def test_mem0_basic():
    """Test basic Mem0 functionality"""
    print("🚀 Testing Mem0 basic functionality...")
    
    # Configure Mem0 to use LM Studio for both LLM and embeddings
    config = MemoryConfig(
        llm=LlmConfig(
            provider="lmstudio",
            config={
                "model": "qwen2.5-14b-instruct",  # Your chat model
                "lmstudio_base_url": "http://localhost:1234/v1",
                "api_key": "not-required",
                "temperature": 0.1,
                "max_tokens": 1000,
                "lmstudio_response_format": {"type": "text"}  # Avoid JSON issues
            }
        ),
        embedder=EmbedderConfig(
            provider="lmstudio",
            config={
                "model": "text-embedding-nomic-embed-text-v1.5",  # Your embedding model
                "lmstudio_base_url": "http://localhost:1234/v1",
                "api_key": "not-required",
                "embedding_dims": 768  # Actual dimensions from the model
            }
        ),
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "embedding_model_dims": 768,  # Must match embedding dimensions
                "collection_name": "mem0_test"
            }
        )
    )
    
    # Initialize memory with LM Studio configuration
    m = Memory(config=config)
    print("✅ Mem0 Memory initialized with LM Studio")
    
    # Add some test memories
    test_messages = [
        {"role": "user", "content": "My name is Peppi and I live in Serramanna"},
        {"role": "assistant", "content": "Nice to meet you Peppi! Serramanna sounds lovely."},
        {"role": "user", "content": "I love coffee in the morning"},
        {"role": "assistant", "content": "Morning coffee is a great way to start the day!"},
        {"role": "user", "content": "I work on AI voice agents"},
        {"role": "assistant", "content": "That's fascinating! Voice AI is such an exciting field."}
    ]
    
    print("\n📝 Adding test memories...")
    for i, message in enumerate(test_messages):
        result = m.add(message["content"], user_id="peppi")
        print(f"  Added memory {i+1}: {message['content'][:50]}...")
    
    print("✅ All memories added")
    
    # Test searches
    search_queries = [
        "What's my name?",
        "Where do I live?", 
        "What do I drink in the morning?",
        "What do I work on?"
    ]
    
    print("\n🔍 Testing memory searches...")
    for query in search_queries:
        print(f"\n❓ Query: {query}")
        results = m.search(query, user_id="peppi")
        
        if results:
            print(f"✅ Found {len(results)} relevant memories:")
            # Handle results list properly
            for i, result in enumerate(results[:2] if isinstance(results, list) else [results]):
                if isinstance(result, dict):
                    print(f"  {i+1}. {result.get('memory', result.get('text', 'N/A'))}")
                else:
                    print(f"  {i+1}. {str(result)}")
        else:
            print("❌ No relevant memories found")
    
    # Test getting all memories for user
    print(f"\n📊 All memories for user 'peppi':")
    all_memories = m.get_all(user_id="peppi")
    print(f"Total memories stored: {len(all_memories) if all_memories else 0}")
    
    if all_memories:
        for i, memory in enumerate(all_memories[:3]):  # Show first 3
            print(f"  {i+1}. {memory.get('memory', 'N/A')}")
    
    print("\n🎉 Mem0 basic test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_mem0_basic())
        if success:
            print("\n✅ Ready to integrate Mem0 with Slowcat!")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()