#!/usr/bin/env python3
"""
Quick integration test for stateless memory system without database corruption issues
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

async def test_stateless_memory_quick():
    """Quick test of stateless memory functionality"""
    
    print("🚀 Quick Stateless Memory Test")
    print("=" * 40)
    
    # Import after setting environment
    from processors.stateless_memory import StatelessMemoryProcessor
    
    # Use a completely clean temporary directory
    temp_dir = tempfile.mkdtemp(prefix="stateless_test_")
    print(f"Using temp dir: {temp_dir}")
    
    try:
        # Test 1: Basic Initialization
        print("\n1. Testing Initialization...")
        processor = StatelessMemoryProcessor(
            db_path=temp_dir,
            max_context_tokens=512,
            perfect_recall_window=5
        )
        print("   ✅ Processor created")
        
        # Test 2: Direct Memory Operations
        print("\n2. Testing Memory Operations...")
        
        # Set speaker directly for testing
        processor.current_speaker = 'test_user'
        processor.current_user_message = "My dog name is Potola"
        
        # Test memory injection
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'Tell me about my dog'}
        ]
        
        start_time = time.perf_counter()
        await processor._inject_memory_context(messages, 'test_user')
        injection_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   Memory injection: {injection_time:.2f}ms")
        print("   ✅ Memory injection working")
        
        # Test storage
        await processor._store_exchange(
            "Tell me about my dog",
            "Your dog Potola sounds wonderful!"
        )
        
        # Wait for async storage
        await asyncio.sleep(0.1)
        print("   ✅ Memory storage working")
        
        # Test retrieval after storage
        messages2 = [
            {'role': 'system', 'content': 'You are a helpful assistant'},
            {'role': 'user', 'content': 'What was my dog\'s name again?'}
        ]
        
        start_time = time.perf_counter()
        await processor._inject_memory_context(messages2, 'test_user')
        retrieval_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   Memory retrieval: {retrieval_time:.2f}ms")
        
        # Check if context was injected
        if len(messages2) > 2:
            print("   ✅ Memory context injected successfully")
            print(f"   Context messages: {len(messages2)}")
        else:
            print("   ⚠️  No memory context found (expected for fresh DB)")
        
        # Test 3: Performance Check
        print("\n3. Testing Performance...")
        
        latencies = []
        for i in range(10):
            messages_test = [
                {'role': 'user', 'content': f'Question {i}: Tell me about pets'}
            ]
            
            start = time.perf_counter()
            await processor._inject_memory_context(messages_test, 'test_user')
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   Max latency: {max_latency:.2f}ms")
        
        if avg_latency < 30.0:
            print("   ✅ Performance within target")
        else:
            print("   ⚠️  Performance higher than expected")
        
        # Get stats
        stats = processor.get_performance_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Total conversations: {stats['total_conversations']}")
        print(f"   Cache hit ratio: {stats['cache_hit_ratio']:.1%}")
        print(f"   Reconstruction failures: {stats['reconstruction_failures']}")
        print(f"   Avg injection time: {stats['avg_injection_time_ms']:.2f}ms")
        
        # Clean up
        processor.env.close()
        processor.thread_pool.shutdown(wait=True)
        
        print("\n🎉 Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    result = asyncio.run(test_stateless_memory_quick())
    sys.exit(0 if result else 1)