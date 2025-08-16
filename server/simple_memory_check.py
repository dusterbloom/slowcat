#!/usr/bin/env python3
"""
Simple memory check using the exact same config as service_factory
"""

import os
import sys
from pathlib import Path

# Set environment variables from .env
os.environ["ENABLE_MEM0"] = "true"
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1"

def check_memories_simple():
    """Check memories using Pipecat's Mem0MemoryService config"""
    
    try:
        # Try to import Pipecat's Mem0 service
        from pipecat.services.mem0.memory import Mem0MemoryService
        
        print("🔍 Creating Mem0MemoryService with same config as bot...")
        
        # Same config as in your service_factory.py
        local_config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "slowcat_default_user",
                    "path": "./data/chroma_db"
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
                    "model": "qwen2.5-7b-instruct",
                    "openai_base_url": "http://localhost:1234/v1",
                    "temperature": 0.0,
                    "max_tokens": 100    
                }
            },
            "version": "v1.1",
            "custom_prompt": None,
            "memory_deduplication": True,
            "memory_threshold": 0.9
        }
        
        # Create the service
        memory_service = Mem0MemoryService(
            local_config=local_config,
            user_id="default_user",
            agent_id="slowcat",
            run_id=None,
            params=Mem0MemoryService.InputParams(
                search_limit=3,
                search_threshold=0.10,
                system_prompt="You previously learned about the user:\n\n",
                add_as_system_message=True
            )
        )
        
        print("✅ Memory service created successfully!")
        
        # Check if there's a way to access stored memories
        if hasattr(memory_service, 'memory'):
            print("🧠 Checking underlying memory client...")
            mem_client = memory_service.memory
            
            # Try to get all memories
            try:
                all_memories = mem_client.get_all(user_id="default_user")
                
                if all_memories:
                    print(f"✅ Found {len(all_memories)} stored memories!")
                    print("\n📝 Your stored memories:")
                    print("-" * 50)
                    
                    for i, memory_item in enumerate(all_memories, 1):
                        if isinstance(memory_item, dict):
                            content = memory_item.get('memory', str(memory_item))
                            print(f"{i}. {content}")
                            
                            # Check for dog name specifically
                            if any(word in content.lower() for word in ['Bobby', 'dog', 'pet']):
                                print(f"   🐕 DOG INFO FOUND: {content}")
                        else:
                            print(f"{i}. {str(memory_item)}")
                else:
                    print("❌ No memories found")
                    print("This means either:")
                    print("1. No conversations have been stored yet")
                    print("2. Memory is not working in the pipeline")
                    
            except Exception as e:
                print(f"❌ Error getting memories: {e}")
            
            # Try searching for dog-related info
            print("\n🔍 Searching for dog-related memories...")
            search_terms = ["Bobby", "dog", "pet", "name", "five", "years"]
            
            for term in search_terms:
                try:
                    results = mem_client.search(term, user_id="default_user")
                    if results:
                        print(f"✅ Found {len(results)} results for '{term}':")
                        for result in results[:2]:
                            if isinstance(result, dict):
                                content = result.get('memory', str(result))
                            else:
                                content = str(result)
                            print(f"   - {content}")
                    else:
                        print(f"❌ No results for '{term}'")
                except Exception as e:
                    print(f"❌ Search error for '{term}': {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import Pipecat Mem0MemoryService: {e}")
        print("Make sure pipecat-ai is installed with mem0 support")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧠 Simple Memory Check")
    print("=" * 30)
    check_memories_simple()