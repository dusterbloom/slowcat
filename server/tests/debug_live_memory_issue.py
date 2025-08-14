#!/usr/bin/env python3
"""
Debug script to test the exact same Mem0 configuration as the live voice agent
"""
import os
import asyncio
from dotenv import load_dotenv

# Load environment exactly like the live system
load_dotenv()

async def test_live_memory_config():
    """Test with the exact same configuration as service_factory.py"""
    print("🧠 Testing live memory configuration...")
    
    # Check environment variables
    print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"MEM0_CHAT_MODEL: {os.getenv('MEM0_CHAT_MODEL')}")
    print(f"ENABLE_MEM0: {os.getenv('ENABLE_MEM0')}")
    
    # Import after setting environment
    from mem0 import Memory
    
    # Use EXACT config from service_factory.py
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 384,
                "collection_name": "slowcat_defaultuser"
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
            "provider": "openai",
            "config": {
                "model": os.getenv("MEM0_CHAT_MODEL", "llama-3.2-3b-instruct-uncensored"),
                "openai_base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
                "temperature": 0.1,
                "max_tokens": 500
            }
        },
        "version": "v1.1"
    }
    
    print(f"Using model: {config['llm']['config']['model']}")
    print(f"Using base URL: {config['llm']['config']['openai_base_url']}")
    
    try:
        print("Creating Mem0 instance...")
        m = Memory.from_config(config)
        
        print("💾 Testing memory storage with dog name...")
        # Test with exact phrase from your conversation
        result = m.add("My dog's name is Buddy and I live in California", user_id="defaultuser")
        print(f"📝 Storage result: {result}")
        
        if result.get('results'):
            print("✅ Memory storage successful!")
            for memory in result['results']:
                print(f"  - {memory}")
        else:
            print("❌ Memory storage returned empty results")
            
        print("🔍 Testing immediate search...")
        search_result = m.search("dog name", user_id="defaultuser", limit=3)
        print(f"📋 Search result: {search_result}")
        
        print("📚 Getting all memories...")
        all_memories = m.get_all(user_id="defaultuser")
        print(f"🗃️ All memories: {all_memories}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_live_memory_config())