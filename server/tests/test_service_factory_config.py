#!/usr/bin/env python3
"""
Test that the service factory is using the correct environment
"""
import asyncio
import os

async def test_service_factory():
    print("🧠 Testing service factory memory configuration...")
    
    # Import after setting up environment
    from core.service_factory import service_factory
    
    # Check what environment variables the service factory sees
    print(f"Service factory sees:")
    print(f"  OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
    print(f"  OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"  MEM0_CHAT_MODEL: {os.getenv('MEM0_CHAT_MODEL')}")
    
    try:
        # Get the memory service
        memory_service = await service_factory.get_service('memory_service')
        
        if memory_service is None:
            print("❌ Memory service is None - Mem0 might be disabled")
            return
            
        print(f"✅ Memory service created: {type(memory_service)}")
        
        # Check the underlying Mem0 configuration
        if hasattr(memory_service, '_mem0_instance'):
            mem0_instance = memory_service._mem0_instance
            if hasattr(mem0_instance, 'llm') and hasattr(mem0_instance.llm, 'client'):
                client = mem0_instance.llm.client
                print(f"  Mem0 LLM client base_url: {client.base_url}")
                print(f"  Mem0 LLM client api_key: {client.api_key}")
        
    except Exception as e:
        print(f"❌ Error creating memory service: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_service_factory())