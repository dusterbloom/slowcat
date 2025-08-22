#!/usr/bin/env python3
"""
Quick latency test to optimize performance for the failing test
"""

import asyncio
import time
import statistics
from pathlib import Path
import sys

# Add server path for imports
sys.path.append(str(Path(__file__).parent))

async def test_latency_performance():
    """Test DTH latency with different configurations"""
    
    print("⚡ LATENCY OPTIMIZATION TEST")
    print("=" * 50)
    
    try:
        from memory.dynamic_tape_head import DynamicTapeHead
        
        # Create mock memory
        class FastMockMemory:
            async def knn_tape(self, query, limit=10, scan=100):
                # Return minimal data quickly
                now = time.time()
                return [
                    {
                        "ts": now - 100,
                        "speaker_id": "user",
                        "role": "user",
                        "content": f"Test content {query}"
                    }
                ]
        
        memory = FastMockMemory()
        
        # Test with ultra-fast policy
        print("🚀 Testing with ultra-fast policy...")
        dth = DynamicTapeHead(memory, policy_path="config/test_tape_head_policy.json")
        
        # Run multiple iterations to get stable measurements
        latencies = []
        for i in range(10):
            start = time.time()
            bundle = await dth.seek(f"test query {i}", budget=500)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"📊 Latency Statistics:")
        print(f"   Average: {avg_latency:.1f}ms")
        print(f"   P95: {p95_latency:.1f}ms")
        print(f"   Min: {min_latency:.1f}ms")
        print(f"   Max: {max_latency:.1f}ms")
        
        # Check if we meet the target
        target = 30.0
        if p95_latency <= target:
            print(f"✅ PASSED: P95 latency {p95_latency:.1f}ms <= {target}ms")
            return True
        else:
            print(f"❌ FAILED: P95 latency {p95_latency:.1f}ms > {target}ms")
            
            # Suggest optimizations
            print(f"\n🔧 Optimization suggestions:")
            if p95_latency > 50:
                print("   - Disable more features in test policy")
                print("   - Reduce knn_k further")
                print("   - Disable embedding model loading")
            elif p95_latency > 35:
                print("   - Fine-tune policy parameters")
                print("   - Check for slow imports")
            
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_minimal_dth():
    """Test absolutely minimal DTH configuration"""
    
    print("\n🏃 MINIMAL DTH TEST")
    print("=" * 30)
    
    try:
        from memory.dynamic_tape_head import DynamicTapeHead
        
        # Ultra-minimal mock memory
        class MinimalMemory:
            async def knn_tape(self, query, limit=10, scan=100):
                return []  # Return nothing for maximum speed
        
        memory = MinimalMemory()
        dth = DynamicTapeHead(memory, policy_path="config/test_tape_head_policy.json")
        
        # Disable everything possible
        dth.encoder = None
        dth.nlp = None
        dth.cross_encoder = None
        
        # Single fast test
        start = time.time()
        bundle = await dth.seek("test", budget=100)
        latency_ms = (time.time() - start) * 1000
        
        print(f"📊 Minimal latency: {latency_ms:.1f}ms")
        
        if latency_ms <= 15:
            print("✅ Minimal configuration very fast")
        elif latency_ms <= 30:
            print("✅ Minimal configuration acceptable")
        else:
            print("❌ Still too slow even with minimal config")
            
        return latency_ms <= 30
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False

async def main():
    """Run latency optimization tests"""
    
    results = [
        await test_latency_performance(),
        await test_minimal_dth()
    ]
    
    if all(results):
        print(f"\n🎉 ALL LATENCY TESTS PASSED!")
        print(f"Ready to run pytest again.")
    else:
        print(f"\n⚠️ Some latency tests failed - need more optimization")

if __name__ == "__main__":
    asyncio.run(main())
