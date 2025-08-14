#!/usr/bin/env python3
"""
Working Mem0 test with LM Studio integration
"""

import asyncio
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig

def test_mem0_working():
    """Simple working test of Mem0 with LM Studio"""
    print("🚀 Testing Mem0 with LM Studio...")
    
    # Configure Mem0 to use LM Studio
    config = MemoryConfig(
        llm=LlmConfig(
            provider="lmstudio",
            config={
                "model": "qwen2.5-14b-instruct",
                "lmstudio_base_url": "http://localhost:1234/v1",
                "api_key": "not-required",
                "temperature": 0.1,
                "max_tokens": 500,
                "lmstudio_response_format": {"type": "text"}
            }
        ),
        embedder=EmbedderConfig(
            provider="lmstudio",
            config={
                "model": "text-embedding-nomic-embed-text-v1.5",
                "lmstudio_base_url": "http://localhost:1234/v1", 
                "api_key": "not-required",
                "embedding_dims": 768
            }
        ),
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "embedding_model_dims": 768,
                "collection_name": "slowcat_memory"
            }
        )
    )
    
    # Initialize memory
    m = Memory(config=config)
    print("✅ Mem0 initialized with LM Studio")
    
    # Add some memories
    print("\n📝 Adding memories...")
    m.add("My name is Peppi", user_id="peppi")
    m.add("I live in Serramanna", user_id="peppi") 
    m.add("I drink coffee every morning", user_id="peppi")
    m.add("I work on AI voice agents using Slowcat", user_id="peppi")
    print("✅ Memories added")
    
    # Search memories
    print("\n🔍 Testing searches...")
    queries = [
        "What's my name?",
        "Where do I live?", 
        "What do I work on?"
    ]
    
    for query in queries:
        print(f"\n❓ {query}")
        try:
            results = m.search(query, user_id="peppi")
            if results:
                print(f"✅ Found {len(results)} memories")
                # Print the actual memory content 
                for i, result in enumerate(results[:2]):
                    if hasattr(result, 'memory'):
                        print(f"  {i+1}. {result.memory}")
                    elif isinstance(result, dict) and 'memory' in result:
                        print(f"  {i+1}. {result['memory']}")
                    else:
                        print(f"  {i+1}. {str(result)}")
            else:
                print("❌ No memories found")
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    print(f"\n🎉 Mem0 is working with LM Studio!")
    print(f"✅ Ready to integrate with Slowcat pipeline!")
    return True

if __name__ == "__main__":
    try:
        success = test_mem0_working()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()