#!/usr/bin/env python3
"""
Test Mem0 storage by adding some test data
"""

import os
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.vector_stores.configs import VectorStoreConfig

def create_mem0_instance():
    """Create Mem0 instance with same config as the bot"""
    config = MemoryConfig(
        llm=LlmConfig(
            provider="lmstudio",
            config={
                "model": "qwen2.5-14b-instruct",
                "lmstudio_base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
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
                "api_key": "lm-studio",
                "embedding_dims": 768
            }
        ),
        vector_store=VectorStoreConfig(
            provider="qdrant",
            config={
                "embedding_model_dims": 768,
                "collection_name": "slowcat_default_user"
            }
        )
    )
    
    return Memory(config=config)

def test_memory_storage():
    """Test storing and retrieving memories"""
    print("🧪 Testing Mem0 Storage...")
    
    try:
        # Create Mem0 instance
        memory = create_mem0_instance()
        print("✅ Connected to Mem0")
        
        # Add some test conversations
        test_conversations = [
            "User said: 'My dog's name is Bobby' and I responded: 'That's a wonderful name for a dog!'",
            "User said: 'She is five years old' and I responded: 'That's awesome! Five years old - she must be full of energy!'",
            "User said: 'I live in Serramanna' and I responded: 'Serramanna sounds like a beautiful place!'"
        ]
        
        print("\n📝 Adding test memories...")
        for i, conv in enumerate(test_conversations, 1):
            try:
                result = memory.add(conv, user_id="default_user")
                print(f"✅ Added memory {i}: {conv[:60]}...")
            except Exception as e:
                print(f"❌ Failed to add memory {i}: {e}")
        
        print("\n🔍 Testing searches...")
        test_searches = ["Bobby", "dog", "five years", "Serramanna"]
        
        for query in test_searches:
            try:
                results = memory.search(query, user_id="default_user")
                if results:
                    print(f"✅ Search '{query}': {len(results)} results")
                    for j, result in enumerate(results[:1], 1):
                        if isinstance(result, dict):
                            content = result.get('memory', str(result))
                        elif hasattr(result, 'memory'):
                            content = result.memory
                        else:
                            content = str(result)
                        print(f"   → {content[:80]}...")
                else:
                    print(f"❌ Search '{query}': No results")
            except Exception as e:
                print(f"❌ Search '{query}' error: {e}")
        
        # Get total count
        all_memories = memory.get_all(user_id="default_user")
        print(f"\n📊 Total memories stored: {len(all_memories) if all_memories else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧠 Mem0 Storage Test")
    print("=" * 40)
    
    success = test_memory_storage()
    
    if success:
        print("\n✅ Mem0 storage is working!")
        print("💡 Now try talking to your voice agent about Bobby")
    else:
        print("\n❌ Mem0 storage test failed")
        print("💡 Check that LM Studio is running with both models loaded")