#!/usr/bin/env python3
"""
Check what data is stored in Mem0 memory system
"""

import os
import asyncio
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
                "collection_name": "slowcat_default_user"  # Same as bot uses
            }
        )
    )
    
    return Memory(config=config)

def check_mem0_data():
    """Check what data is stored in Mem0"""
    print("🔍 Checking Mem0 Memory Data...")
    
    try:
        # Create Mem0 instance
        memory = create_mem0_instance()
        print("✅ Connected to Mem0")
        
        # Get all memories for default user
        print("\n📊 Getting all stored memories...")
        all_memories = memory.get_all(user_id="default_user")
        
        if all_memories:
            print(f"✅ Found {len(all_memories)} memories stored!")
            print("\n📝 Stored Memories:")
            print("-" * 60)
            
            for i, memory_item in enumerate(all_memories, 1):
                # Handle different memory formats
                if isinstance(memory_item, dict):
                    mem_content = memory_item.get('memory', str(memory_item))
                    mem_id = memory_item.get('id', f'mem_{i}')
                    created = memory_item.get('created_at', 'unknown')
                    
                    print(f"{i}. [{mem_id}] {mem_content}")
                    print(f"   Created: {created}")
                else:
                    print(f"{i}. {str(memory_item)}")
                
                print()
        else:
            print("❌ No memories found in storage")
            print("\nThis could mean:")
            print("1. No conversations have happened yet")
            print("2. Memory storage is not working properly")
            print("3. Different user_id or collection name is being used")
        
        # Test search functionality
        print("\n🔍 Testing memory search...")
        search_queries = ["Bobby", "dog", "name", "five", "years"]
        
        for query in search_queries:
            try:
                results = memory.search(query, user_id="default_user")
                if results:
                    print(f"✅ Search '{query}': Found {len(results)} results")
                    for j, result in enumerate(results[:2], 1):
                        if isinstance(result, dict):
                            content = result.get('memory', str(result))
                        else:
                            content = str(result)
                        print(f"   {j}. {content[:80]}...")
                else:
                    print(f"❌ Search '{query}': No results")
            except Exception as e:
                print(f"❌ Search '{query}' failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking Mem0 data: {e}")
        return False

def check_qdrant_storage():
    """Check Qdrant vector database directly"""
    print("\n🗄️ Checking Qdrant storage...")
    
    # Check if Qdrant data directory exists
    possible_paths = [
        "./qdrant_storage",
        "~/.mem0",
        "./data/qdrant",
        "~/.qdrant"
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            print(f"✅ Found Qdrant data at: {expanded_path}")
            
            # List contents
            try:
                contents = os.listdir(expanded_path)
                if contents:
                    print(f"   Contents: {contents}")
                else:
                    print("   Directory is empty")
            except Exception as e:
                print(f"   Could not list contents: {e}")
        else:
            print(f"❌ No data found at: {expanded_path}")

if __name__ == "__main__":
    print("🧠 Mem0 Data Inspector")
    print("=" * 50)
    
    success = check_mem0_data()
    check_qdrant_storage()
    
    if success:
        print("\n💡 Tips:")
        print("- If no memories found, try having a conversation first")
        print("- Memories are stored after the assistant responds")
        print("- Check that ENABLE_MEMORY=true in your .env file")
        print("- Make sure LM Studio is running with the embedding model")
    else:
        print("\n❌ Could not connect to Mem0 - check your configuration")