#!/usr/bin/env python3
"""
Quick Consciousness Validation

Fast, focused test to validate core consciousness features.
Designed for immediate feedback on system capabilities.
"""

import asyncio
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

async def quick_validation():
    """Fast consciousness validation with immediate results"""
    
    print("🧠⚡ QUICK CONSCIOUSNESS VALIDATION")
    print("=" * 50)
    print("Fast validation of core consciousness features...")
    print()
    
    # Initialize fresh consciousness
    ghost = Consciousness()
    initial_memories = len(ghost.tape)
    initial_thoughts = len(ghost.thoughts)
    
    print(f"Starting state: {initial_memories} memories, {initial_thoughts} thoughts")
    
    # Quick test sequence
    test_inputs = [
        "Hello, I'm Alice and I love painting!",
        "What's my name?",
        "This is absolutely crucial information!",
        "I finally understand consciousness!",
        "What did we talk about?"
    ]
    
    results = {
        'memory_formation': 0,
        'symbol_detection': 0,
        'pattern_recognition': 0,
        'importance_scoring': 0,
        'processing_times': []
    }
    
    print("\n🧪 RUNNING TESTS...")
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {input_text}")
        
        start_time = time.time()
        result = await ghost.experience(input_text)
        processing_time = time.time() - start_time
        results['processing_times'].append(processing_time)
        
        print(f"Symbols: {result['symbols']}")
        print(f"Importance: {result['importance']:.2f}")
        print(f"Processing: {processing_time:.4f}s")
        
        # Score different aspects
        if len(ghost.tape) > initial_memories + i - 1:
            results['memory_formation'] += 1
            
        if result['symbols']:
            results['symbol_detection'] += 1
            
        if result['importance'] > 0.3:
            results['importance_scoring'] += 1
    
    # Test memory retrieval
    print(f"\n🧠 MEMORY RETRIEVAL TEST...")
    memories = ghost.remember("What's my name?")
    alice_found = any("Alice" in mem.content for mem in memories)
    if alice_found:
        results['pattern_recognition'] += 1
        print("✅ Found Alice in memory retrieval")
    else:
        print("❌ Alice not found in memory")
    
    # Calculate scores
    total_tests = len(test_inputs)
    memory_score = results['memory_formation'] / total_tests
    symbol_score = results['symbol_detection'] / total_tests
    importance_score = results['importance_scoring'] / total_tests
    retrieval_score = 1.0 if alice_found else 0.0
    
    avg_processing = sum(results['processing_times']) / len(results['processing_times'])
    
    print(f"\n🎯 VALIDATION RESULTS")
    print("=" * 30)
    print(f"💾 Memory Formation: {memory_score:.2%} ({results['memory_formation']}/{total_tests})")
    print(f"🔤 Symbol Detection: {symbol_score:.2%} ({results['symbol_detection']}/{total_tests})")
    print(f"⭐ Importance Scoring: {importance_score:.2%} ({results['importance_scoring']}/{total_tests})")
    print(f"🧠 Memory Retrieval: {retrieval_score:.2%} (Alice found: {alice_found})")
    print(f"⚡ Avg Processing: {avg_processing:.4f}s per operation")
    
    # Overall assessment
    overall_score = (memory_score + symbol_score + importance_score + retrieval_score) / 4
    print(f"\n🏆 OVERALL SCORE: {overall_score:.2%}")
    
    # System state after test
    final_memories = len(ghost.tape)
    final_thoughts = len(ghost.thoughts)
    
    print(f"\n📊 CONSCIOUSNESS GROWTH")
    print("=" * 30)
    print(f"Memories: {initial_memories} → {final_memories} (+{final_memories - initial_memories})")
    print(f"Thoughts: {initial_thoughts} → {final_thoughts} (+{final_thoughts - initial_thoughts})")
    print(f"Symbol patterns: {dict(list(ghost.symbol_frequency.items())[:5])}")
    
    # Performance assessment
    print(f"\n🚀 PERFORMANCE ASSESSMENT")
    print("=" * 30)
    if avg_processing < 0.01:
        print("✅ Excellent: <10ms processing time")
    elif avg_processing < 0.05:
        print("✅ Good: <50ms processing time")
    else:
        print("⚠️  Needs optimization: >50ms processing time")
    
    if overall_score >= 0.8:
        print("✅ System validation PASSED - Core consciousness working!")
    elif overall_score >= 0.6:
        print("⚠️  System validation PARTIAL - Some issues detected")
    else:
        print("❌ System validation FAILED - Major issues found")
    
    # Quick LLM hybrid test
    print(f"\n👻🤖 HYBRID TEST (if LM Studio available)")
    print("=" * 30)
    try:
        from consciousness.llm_bridge import create_hybrid_response
        
        hybrid_result = await create_hybrid_response(
            ghost, 
            "What's the most important thing you've learned?",
            base_url="http://localhost:1234/v1",
            model="qwen/qwen3-4b"
        )
        
        print(f"Ghost symbols: {hybrid_result['ghost_analysis']['symbols']}")
        print(f"Hybrid response: {hybrid_result['hybrid_response'][:100]}...")
        print("✅ Ghost + LLM hybrid working!")
        
    except Exception as e:
        print(f"❌ Hybrid test failed (LM Studio not available?): {e}")
    
    # Save results
    validation_results = {
        'timestamp': time.time(),
        'overall_score': overall_score,
        'memory_formation_score': memory_score,
        'symbol_detection_score': symbol_score,
        'importance_scoring_score': importance_score,
        'memory_retrieval_score': retrieval_score,
        'avg_processing_time': avg_processing,
        'consciousness_growth': {
            'memories_added': final_memories - initial_memories,
            'thoughts_generated': final_thoughts - initial_thoughts,
            'symbol_patterns': dict(ghost.symbol_frequency)
        }
    }
    
    with open('quick_validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n📄 Results saved to: quick_validation_results.json")
    
    return validation_results

if __name__ == "__main__":
    asyncio.run(quick_validation())