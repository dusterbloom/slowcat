#!/usr/bin/env python3
"""
Final test validation before running pytest
"""

import asyncio
import time
import sys
from pathlib import Path

# Add server path for imports
sys.path.append(str(Path(__file__).parent))

async def test_final_optimizations():
    """Test all our optimizations work together"""
    
    print("🔧 FINAL OPTIMIZATION VALIDATION")
    print("=" * 50)
    
    try:
        from memory.dynamic_tape_head import DynamicTapeHead
        
        # Create the exact same mock memory as in tests
        class TestMockMemory:
            def __init__(self):
                self.tape_store = None
                
            async def knn_tape(self, query, limit=10, scan=100):
                # Return minimal test data like the actual test
                now = time.time()
                return [
                    {"ts": now - 10, "speaker_id": "u", "role": "user", "content": "Test content"},
                    {"ts": now - 20, "speaker_id": "u", "role": "user", "content": "Another test"},
                ]
        
        memory = TestMockMemory()
        
        # Create DTH exactly like in the failing test
        dth = DynamicTapeHead(memory, policy_path="config/test_tape_head_policy.json")
        
        print("📋 DTH Configuration:")
        print(f"   Semantic search: {dth.policy['ablation']['use_semantic']}")
        print(f"   Entity extraction: {dth.policy['ablation']['use_entities']}")
        print(f"   Shadows: {dth.policy['ablation']['use_shadows']}")
        print(f"   Symbols: {dth.policy['ablation']['use_symbols']}")
        print(f"   Encoder loaded: {dth.encoder is not None}")
        print(f"   NLP loaded: {dth.nlp is not None}")
        print(f"   Symbol detector: {dth.symbol_detector is not None}")
        
        # Test the exact same pattern as the failing test: 10 iterations
        print(f"\n⚡ Running 10 iterations (same as test)...")
        latencies = []
        
        for i in range(10):
            start = time.time()
            bundle = await dth.seek(f"Quick test query {i}", budget=1000)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate P95 exactly like pytest does
        sorted_latencies = sorted(latencies)
        p95_index = int(0.95 * len(sorted_latencies))
        p95_latency = sorted_latencies[p95_index]
        
        print(f"📊 Results:")
        print(f"   Average: {sum(latencies)/len(latencies):.1f}ms")
        print(f"   P95: {p95_latency:.1f}ms")
        print(f"   Max: {max(latencies):.1f}ms")
        print(f"   Min: {min(latencies):.1f}ms")
        
        # Test the exact assertion that was failing
        target = 30.0
        success = p95_latency <= target
        
        if success:
            print(f"✅ SUCCESS: P95 latency {p95_latency:.1f}ms <= {target}ms")
        else:
            print(f"❌ FAILURE: P95 latency {p95_latency:.1f}ms > {target}ms")
        
        # Test backward compatibility methods
        print(f"\n🔧 Testing backward compatibility...")
        
        from memory.dynamic_tape_head import MemorySpan
        test_memory = MemorySpan("Test", time.time(), "user", "test")
        test_memory.tokens = 10
        
        # Test the methods that were failing
        methods_ok = True
        try:
            score, components = dth._score_memory(test_memory, None, [], [])
            print(f"   ✅ _score_memory: {score:.3f}")
        except Exception as e:
            print(f"   ❌ _score_memory: {e}")
            methods_ok = False
        
        try:
            shadow = dth._compress_to_shadow(test_memory)
            print(f"   ✅ _compress_to_shadow: {len(shadow.content)} chars")
        except Exception as e:
            print(f"   ❌ _compress_to_shadow: {e}")
            methods_ok = False
        
        try:
            bundle = dth._select_within_budget([test_memory], 100)
            print(f"   ✅ _select_within_budget: {bundle.token_count} tokens")
        except Exception as e:
            print(f"   ❌ _select_within_budget: {e}")
            methods_ok = False
        
        try:
            diverse = dth._apply_diversity([test_memory], 5)
            print(f"   ✅ _apply_diversity: {len(diverse)} items")
        except Exception as e:
            print(f"   ❌ _apply_diversity: {e}")
            methods_ok = False
        
        overall_success = success and methods_ok
        
        print(f"\n{'🎉 READY FOR PYTEST!' if overall_success else '❌ NEEDS MORE WORK'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_final_optimizations())
    if success:
        print(f"\n🚀 Run pytest now:")
        print(f"   cd /Users/peppi/Dev/macos-local-voice-agents/server")
        print(f"   python -m pytest tests/test_dynamic_tape_head.py::TestDynamicTapeHead::test_retrieval_latency -v")
    sys.exit(0 if success else 1)
