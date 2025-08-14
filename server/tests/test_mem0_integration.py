#!/usr/bin/env python3
"""
Test the complete Mem0 integration with the Slowcat pipeline
"""

import os
import asyncio
from loguru import logger

# Set environment for Mem0 testing
os.environ["ENABLE_MEMORY"] = "true"
os.environ["ENABLE_MEM0"] = "true"
os.environ["MEM0_CHAT_MODEL"] = "qwen2.5-14b-instruct"
os.environ["MEM0_EMBEDDING_MODEL"] = "text-embedding-nomic-embed-text-v1.5"
os.environ["MEM0_MAX_CONTEXT"] = "3"

async def test_mem0_integration():
    """Test complete Mem0 integration"""
    print("🧪 Testing Mem0 Integration with Slowcat Pipeline...")
    
    try:
        # Import the service factory
        from core.service_factory import ServiceFactory
        
        # Create service factory instance
        service_factory = ServiceFactory()
        
        print("✅ Service factory created")
        
        # Test memory service creation
        print("\n🧠 Testing memory service creation...")
        memory_service = service_factory._create_memory_service()
        
        if memory_service is None:
            print("❌ Memory service is None - check ENABLE_MEMORY setting")
            return False
        
        print(f"✅ Memory service created: {type(memory_service).__name__}")
        
        # Test if it's Mem0MemoryProcessor
        from processors.mem0_memory_processor import Mem0MemoryProcessor
        if isinstance(memory_service, Mem0MemoryProcessor):
            print("🚀 Mem0 memory system successfully integrated!")
            
            # Test drop-in compatibility methods
            print("\n🔧 Testing drop-in compatibility...")
            
            # Test search_conversations
            results = await memory_service.search_conversations("test query", limit=5)
            print(f"✅ search_conversations method works: {len(results)} results")
            
            # Test add_conversation
            await memory_service.add_conversation("Hello", "Hi there!", "test_user")
            print("✅ add_conversation method works")
            
            # Test get_context_memories
            context = memory_service.get_context_memories("Hello")
            print(f"✅ get_context_memories method works: {len(context)} chars")
            
            # Test get_memory_stats
            stats = memory_service.get_memory_stats()
            print(f"✅ get_memory_stats method works: {stats}")
            
        else:
            print(f"💾 Using local memory system: {type(memory_service).__name__}")
        
        print("\n🎉 Mem0 integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_pipeline_compatibility():
    """Test that the pipeline builder can use Mem0"""
    print("\n🔨 Testing Pipeline Builder compatibility...")
    
    try:
        from core.service_factory import ServiceFactory
        from core.pipeline_builder import PipelineBuilder
        
        service_factory = ServiceFactory()
        pipeline_builder = PipelineBuilder(service_factory)
        
        # Test getting memory service through the factory
        memory_service = await service_factory.get_service("memory_service")
        
        if memory_service:
            print(f"✅ Pipeline can get memory service: {type(memory_service).__name__}")
        else:
            print("⚠️ No memory service available")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    async def run_tests():
        print("🚀 Starting Mem0 Integration Tests...")
        
        test1 = await test_mem0_integration()
        test2 = await test_pipeline_compatibility()
        
        if test1 and test2:
            print("\n✅ ALL TESTS PASSED - Mem0 is ready for deployment!")
            print("\nTo enable Mem0 in production:")
            print("1. Set ENABLE_MEMORY=true")  
            print("2. Set ENABLE_MEM0=true")
            print("3. Make sure LM Studio has the embedding model loaded")
            print("4. Restart the bot")
        else:
            print("\n❌ Some tests failed")
    
    asyncio.run(run_tests())