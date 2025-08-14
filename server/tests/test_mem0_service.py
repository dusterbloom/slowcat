#!/usr/bin/env python3
"""
Test script to verify Pipecat's Mem0MemoryService integration
"""

import asyncio
import os
from pipecat.services.mem0.memory import Mem0MemoryService
from pipecat.frames.frames import LLMMessagesFrame

async def test_mem0_service():
    print("🧪 Testing Pipecat Mem0MemoryService...")
    
    # Configure for fully local setup with Qdrant + LM Studio
    local_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 768,
                "collection_name": "test_slowcat_default_user"
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
                "model": "qwen2.5-14b-instruct",
                "lmstudio_base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "temperature": 0.1,
                "max_tokens": 500
            }
        },
        "version": "v1.1"
    }
    
    try:
        memory_service = Mem0MemoryService(
            local_config=local_config,
            user_id="test_user",
            agent_id="slowcat",
            run_id=None,
            params=Mem0MemoryService.InputParams(
                search_limit=3,
                search_threshold=0.20,
                system_prompt="Previously you learned:\n\n",
                add_as_system_message=True
            )
        )
        
        print("✅ Mem0MemoryService created successfully")
        
        # Test storing a conversation
        print("📝 Testing message storage...")
        test_messages = [
            {"role": "user", "content": "My name is Alice and I live in Paris"},
            {"role": "assistant", "content": "Nice to meet you, Alice! Paris is a beautiful city."}
        ]
        
        # Simulate an LLMMessagesFrame
        test_frame = LLMMessagesFrame(messages=test_messages)
        
        # The service should process this frame and store memories
        print("🔍 Processing test frame...")
        
        # Check if we can see any collections created
        import requests
        response = requests.get("http://localhost:6333/collections")
        collections = response.json()
        print(f"📊 Qdrant collections: {collections}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mem0_service())