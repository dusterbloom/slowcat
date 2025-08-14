#!/usr/bin/env python3
"""
Test Huggingface embeddings to bypass LM Studio bug
"""

import asyncio
from mem0 import Memory

async def test_huggingface_embeddings():
    print("🧪 Testing Huggingface Embeddings (bypassing LM Studio bug)...")
    
    # Configuration with Huggingface embeddings
    local_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 384,
                "collection_name": "test_huggingface"
            }
        },
        "embedder": {
            "provider": "huggingface", 
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384
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
                "lmstudio_response_format": {"type": "text"}
            }
        },
        "version": "v1.1"
    }
    
    try:
        print("📝 Initializing Mem0 with HuggingFace embeddings...")
        memory = Memory.from_config(local_config)
        print("✅ Mem0 initialized successfully")
        
        # Test simple memory storage
        user_id = "test_hf_user"
        test_message = "My name is Bob and I love pizza"
        
        print(f"🔄 Storing: '{test_message}'")
        result = memory.add(test_message, user_id=user_id)
        print(f"✅ Storage result: {result}")
        
        # Check if collection was created
        print("📊 Checking Qdrant collections...")
        import requests
        try:
            collections = requests.get("http://localhost:6333/collections").json()
            print(f"   Collections: {collections}")
            
            if collections.get('result', {}).get('collections'):
                print("🎉 SUCCESS: Collection created!")
                collection_name = collections['result']['collections'][0]['name']
                
                # Get collection info
                info = requests.get(f"http://localhost:6333/collections/{collection_name}").json()
                points_count = info.get('result', {}).get('points_count', 0)
                print(f"   Points stored: {points_count}")
            else:
                print("❌ No collections created")
        except Exception as e:
            print(f"❌ Could not check Qdrant: {e}")
        
        # Test memory search
        print("🔍 Testing memory search...")
        search_results = memory.search("What is my name?", user_id=user_id)
        print(f"   Found {len(search_results)} memories:")
        for i, result in enumerate(search_results, 1):
            print(f"      {i}. Raw result: {result}")
            if hasattr(result, 'memory'):
                print(f"         Memory: {result.memory}")
            elif isinstance(result, dict) and 'memory' in result:
                print(f"         Memory: {result['memory']}")
        
        # Test getting all memories for user
        print("🗄️ Getting all memories for user...")
        try:
            all_memories = memory.get_all(user_id=user_id)
            print(f"   All memories: {all_memories}")
        except Exception as e:
            print(f"   ❌ Error getting all memories: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_huggingface_embeddings())
    if success:
        print("\n🎉 HuggingFace embeddings working! Memory system should work now.")
    else:
        print("\n🚨 Still having issues - check the errors above")