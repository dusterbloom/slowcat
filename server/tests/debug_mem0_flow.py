#!/usr/bin/env python3
"""
Debug the exact Mem0 flow to find the broken link
"""

import requests
import json
from mem0 import Memory

def debug_mem0_step_by_step():
    print("🔍 Debugging Mem0 Flow Step by Step...")
    
    # Step 1: Test Qdrant connection
    print("\n1️⃣ Testing Qdrant Connection...")
    try:
        response = requests.get("http://localhost:6333/collections")
        print(f"   ✅ Qdrant responsive: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Qdrant connection failed: {e}")
        return
    
    # Step 2: Test LM Studio embedding
    print("\n2️⃣ Testing LM Studio Embedding...")
    try:
        embed_response = requests.post(
            "http://localhost:1234/v1/embeddings",
            headers={"Content-Type": "application/json"},
            json={
                "model": "text-embedding-nomic-embed-text-v1.5",
                "input": "Test embedding for debugging"
            }
        )
        if embed_response.status_code == 200:
            embedding = embed_response.json()['data'][0]['embedding']
            print(f"   ✅ Embedding works: {len(embedding)} dimensions")
        else:
            print(f"   ❌ Embedding failed: {embed_response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Embedding error: {e}")
        return
    
    # Step 3: Test LM Studio LLM with structured output
    print("\n3️⃣ Testing LM Studio LLM...")
    try:
        llm_response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "qwen/qwen3-1.7b",
                "messages": [
                    {"role": "system", "content": "Extract facts from the following conversation."},
                    {"role": "user", "content": "My name is Alice and I live in Paris"}
                ],
                # Remove response_format to test with text
                "temperature": 0.1
            }
        )
        if llm_response.status_code == 200:
            llm_result = llm_response.json()
            content = llm_result['choices'][0]['message']['content']
            print(f"   ✅ LLM response: {content[:100]}...")
            
            # Try to parse as JSON
            try:
                parsed = json.loads(content)
                print(f"   ✅ JSON valid: {parsed}")
            except:
                print(f"   ❌ JSON invalid: {content}")
        else:
            print(f"   ❌ LLM failed: {llm_response.status_code}")
            print(f"   Error details: {llm_response.text}")
            return
    except Exception as e:
        print(f"   ❌ LLM error: {e}")
        return
    
    # Step 4: Test minimal Mem0 config
    print("\n4️⃣ Testing Minimal Mem0 Configuration...")
    minimal_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": 768,
                "collection_name": "debug_test"
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
                "lmstudio_response_format": {"type": "text"}  # Simplest format
            }
        },
        "version": "v1.1"
    }
    
    try:
        print("   Creating Mem0 instance...")
        memory = Memory.from_config(minimal_config)
        print("   ✅ Mem0 instance created")
        
        print("   Adding simple memory...")
        result = memory.add("My name is Alice", user_id="debug_user")
        print(f"   Add result: {result}")
        
        print("   Checking Qdrant after add...")
        collections_after = requests.get("http://localhost:6333/collections").json()
        print(f"   Collections now: {collections_after}")
        
        if collections_after.get('result', {}).get('collections'):
            print("   🎉 SUCCESS: Collection created!")
        else:
            print("   ❌ FAILED: Still no collections")
            
    except Exception as e:
        print(f"   ❌ Mem0 error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mem0_step_by_step()