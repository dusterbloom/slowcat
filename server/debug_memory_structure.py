#!/usr/bin/env python3
"""
Debug the exact structure returned by Mem0 memory search
"""

import asyncio
from mem0 import Memory
import os

async def debug_memory_structure():
    print("🔍 Debugging Memory Search Result Structure...")
    
    # Use same config as service_factory
    local_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 384,
                "collection_name": f"slowcat_default_user"
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
        memory = Memory.from_config(local_config)
        
        # Add a test memory
        print("📝 Adding test memory...")
        add_result = memory.add("My favorite color is blue", user_id="default_user")
        print(f"Add result type: {type(add_result)}")
        print(f"Add result content: {add_result}")
        
        # Search for memory  
        print("\n🔍 Searching for memory...")
        search_result = memory.search("What is my favorite color?", user_id="default_user")
        print(f"Search result type: {type(search_result)}")
        print(f"Search result content: {search_result}")
        
        # Check if it has 'results' key
        if isinstance(search_result, dict) and 'results' in search_result:
            print(f"\n✅ Has 'results' key with {len(search_result['results'])} items:")
            for i, result in enumerate(search_result['results']):
                print(f"  {i+1}. Type: {type(result)}, Content: {result}")
        else:
            print(f"\n❌ No 'results' key. Direct results: {len(search_result) if hasattr(search_result, '__len__') else 'N/A'} items")
            if hasattr(search_result, '__iter__'):
                for i, result in enumerate(search_result):
                    print(f"  {i+1}. Type: {type(result)}, Content: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_memory_structure())