#!/usr/bin/env python3
"""
Enhanced integration test for stateless memory system with all fixes
Run this to verify that the stateless memory system is working correctly
"""

import asyncio
import sys
import os
import tempfile
import shutil
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment for testing
os.environ['USE_STATELESS_MEMORY'] = 'true'
os.environ['ENABLE_MEMORY'] = 'true'

try:
    from processors.stateless_memory import StatelessMemoryProcessor
    from config import config
    from core.service_factory import ServiceFactory
    from pipecat.frames.frames import (
        LLMMessagesFrame, 
        TextFrame, 
        TranscriptionFrame, 
        UserStartedSpeakingFrame
    )
    from pipecat.processors.frame_processor import FrameDirection
    
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running from the server directory with venv activated")
    print("Also check that dependencies are installed: pip install lmdb lz4")
    sys.exit(1)


async def test_stateless_memory_integration():
    """Test the complete stateless memory integration"""
    
    print("🧪 Testing Stateless Memory Integration")
    print("=" * 50)
    
    # Test 1: Configuration
    print("\n1. Testing Configuration...")
    print(f"   Stateless memory enabled: {config.stateless_memory.enabled}")
    print(f"   Memory enabled: {config.memory.enabled}")
    print(f"   Database path: {config.stateless_memory.db_path}")
    print(f"   Max context tokens: {config.stateless_memory.max_context_tokens}")
    
    assert config.stateless_memory.enabled, "Stateless memory should be enabled"
    assert config.memory.enabled, "Memory should be enabled"
    print("   ✅ Configuration test passed")
    
    # Test 2: Service Factory
    print("\n2. Testing Service Factory...")
    service_factory = ServiceFactory()
    memory_service = service_factory._create_memory_service()
    
    assert memory_service is not None, "Memory service should be created"
    assert isinstance(memory_service, StatelessMemoryProcessor), "Should create StatelessMemoryProcessor"
    print(f"   ✅ Created: {type(memory_service).__name__}")
    
    # Test 3: Basic Memory Operations
    print("\n3. Testing Basic Memory Operations...")
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    
    try:
        processor = StatelessMemoryProcessor(
            db_path=temp_dir,
            max_context_tokens=512,
            perfect_recall_window=5
        )
        
        # Test message injection
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'My dog name is Potola'}
        ]
        
        enhanced_messages = await processor._inject_memory_context(messages, 'test_user')
        print(f"   Original messages: {len(messages)}")
        print(f"   Enhanced messages: {len(enhanced_messages)}")
        
        # Test storage
        await processor._store_exchange(
            'My dog name is Potola',
            'That\'s a lovely name for your dog!'
        )
        
        # Test retrieval with context
        messages2 = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'What do you know about my pet?'}
        ]
        
        enhanced_messages2 = await processor._inject_memory_context(messages2, 'test_user')
        
        # Check if memory context was injected
        has_memory_context = any(
            '[Memory Context' in msg.get('content', '') 
            for msg in enhanced_messages2
        )
        
        assert has_memory_context, "Memory context should be injected"
        
        # Check if dog's name is in context
        memory_content = ""
        for msg in enhanced_messages2:
            if '[Memory Context' in msg.get('content', ''):
                memory_content = msg['content'].lower()
                break
        
        assert 'potola' in memory_content, "Dog's name should be in memory context"
        print("   ✅ Memory operations test passed")
        
        # Test 4: Performance
        print("\n4. Testing Performance...")
        
        import time
        latencies = []
        
        for i in range(10):
            start = time.perf_counter()
            await processor._inject_memory_context(messages2, 'test_user')
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"   Average injection latency: {avg_latency:.2f}ms")
        print(f"   Maximum injection latency: {max_latency:.2f}ms")
        
        assert avg_latency < 20.0, f"Average latency too high: {avg_latency:.2f}ms"
        assert max_latency < 50.0, f"Maximum latency too high: {max_latency:.2f}ms"
        print("   ✅ Performance test passed")
        
        # Test 5: Statistics
        print("\n5. Testing Statistics...")
        stats = processor.get_performance_stats()
        print(f"   Total conversations: {stats['total_conversations']}")
        print(f"   Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
        print(f"   Average injection time: {stats['avg_injection_time_ms']:.2f}ms")
        print(f"   Reconstruction failures: {stats['reconstruction_failures']}")
        
        assert stats['reconstruction_failures'] == 0, "Should have no reconstruction failures"
        print("   ✅ Statistics test passed")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n🎉 All integration tests passed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install lmdb lz4")
    print("2. Enable stateless memory: export USE_STATELESS_MEMORY=true")
    print("3. Run the bot: ./run_bot.sh")
    print("4. Run A/B test: python tests/ab_test_memory.py")


async def test_service_factory_integration():
    """Test that service factory creates the right memory processor"""
    
    print("\n🏭 Testing Service Factory Integration")
    print("=" * 50)
    
    # Test with stateless enabled
    print("\n1. Testing with stateless memory enabled...")
    original_value = config.stateless_memory.enabled
    config.stateless_memory.enabled = True
    
    service_factory = ServiceFactory()
    memory_service = service_factory._create_memory_service()
    
    assert isinstance(memory_service, StatelessMemoryProcessor), "Should create stateless processor"
    print("   ✅ Created StatelessMemoryProcessor")
    
    # Test with stateless disabled
    print("\n2. Testing with stateless memory disabled...")
    config.stateless_memory.enabled = False
    
    from processors.local_memory import LocalMemoryProcessor
    memory_service = service_factory._create_memory_service()
    
    assert isinstance(memory_service, LocalMemoryProcessor), "Should create traditional processor"
    print("   ✅ Created LocalMemoryProcessor")
    
    # Restore original value
    config.stateless_memory.enabled = original_value
    
    print("\n✅ Service factory integration test passed!")


if __name__ == "__main__":
    print("🚀 Stateless Memory Integration Test")
    print("This test verifies that the stateless memory system integrates correctly")
    print("with the existing Slowcat architecture.\n")
    
    # Check dependencies
    missing_deps = []
    try:
        import lmdb
    except ImportError:
        missing_deps.append("lmdb")
    
    try:
        import lz4
    except ImportError:
        missing_deps.append("lz4")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        sys.exit(1)
    
    # Run tests
    try:
        asyncio.run(test_stateless_memory_integration())
        asyncio.run(test_service_factory_integration())
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\nStateless memory system is ready for deployment!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)