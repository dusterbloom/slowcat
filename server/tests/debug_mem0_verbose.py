#!/usr/bin/env python3
"""
Debug Mem0 with verbose logging to see what's failing
"""

import logging
import asyncio
from mem0 import Memory

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('mem0').setLevel(logging.DEBUG)

async def debug_mem0_verbose():
    print("🔍 Debugging Mem0 with Verbose Logging...")
    
    local_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 384,
                "collection_name": "debug_verbose_test"
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
        print("📝 Creating Mem0 instance...")
        memory = Memory.from_config(local_config)
        print("✅ Mem0 instance created")
        
        print("\n📝 Adding memory with full debugging...")
        result = memory.add("My favorite color is blue and I like pizza", user_id="debug_user")
        print(f"✅ Add result: {result}")
        
        # Wait a moment for async operations
        await asyncio.sleep(2)
        
        print("\n🔍 Searching...")
        search_result = memory.search("What color do I like?", user_id="debug_user")
        print(f"✅ Search result: {search_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_mem0_verbose())