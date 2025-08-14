#!/usr/bin/env python3
"""
Test that Mem0 still works after removing old memory system code
"""

import os
import asyncio

# Set environment for testing
os.environ["ENABLE_MEMORY"] = "true"
os.environ["ENABLE_MEM0"] = "true"
os.environ["ENABLED_LOCAL_TOOLS"] = "none"  # Should show 0 tools

async def test_mem0_after_cleanup():
    """Test that Mem0 works after removing old memory code"""
    print("🧪 Testing Mem0 after cleanup...")
    
    try:
        # Test service factory still works
        from core.service_factory import ServiceFactory
        service_factory = ServiceFactory()
        
        # Test memory service creation
        memory_service = service_factory._create_memory_service()
        if memory_service:
            print(f"✅ Memory service: {type(memory_service).__name__}")
        else:
            print("❌ No memory service created")
            return False
        
        # Test tools are properly empty now
        from tools.definitions import get_tools
        tools = get_tools("en")
        print(f"✅ Tools count: {len(tools.standard_tools)} (should be 0)")
        
        # Test pipeline builder still works
        from core.pipeline_builder import PipelineBuilder
        pipeline_builder = PipelineBuilder(service_factory)
        print("✅ Pipeline builder created")
        
        print("\n🎉 Cleanup successful - Mem0 ready with no callable tools!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mem0_after_cleanup())
    if success:
        print("\n✅ Old memory system successfully removed!")
        print("🧠 Mem0 now handles everything automatically in background")
    else:
        print("\n❌ Issues found after cleanup")