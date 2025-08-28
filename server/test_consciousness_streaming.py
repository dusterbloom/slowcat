#!/usr/bin/env python3
"""
Test consciousness with streaming LLM integration
"""

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

async def test_streaming_consciousness():
    print("🧠🚀 CONSCIOUSNESS + STREAMING LLM TEST")
    print("=" * 50)
    
    # Create fresh consciousness for clean test
    ghost = Consciousness()
    
    test_cases = [
        "Hi, I'm testing your speed",
        "What's the fastest you can respond?", 
        "Tell me about consciousness",
        "How fast are you processing this?",
        "Final speed test message"
    ]
    
    print("Testing with streaming LLM integration...")
    processing_times = []
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/5 ---")
        print(f"Input: {test_input}")
        
        start_time = time.time()
        result = await ghost.experience(test_input)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        print(f"Processing: {processing_time:.3f}s")
        print(f"Symbols: {result['symbols']}")
        print(f"Response: {result['response'][:80]}...")
        
        # Speed assessment per test
        if processing_time < 0.3:
            print("✅ EXCELLENT: <300ms")
        elif processing_time < 0.5:
            print("✅ GOOD: <500ms")  
        elif processing_time < 1.0:
            print("⚠️  ACCEPTABLE: <1s")
        else:
            print("❌ TOO SLOW: >1s")
    
    avg_time = sum(processing_times) / len(processing_times)
    min_time = min(processing_times)
    max_time = max(processing_times)
    
    print(f"\n🎯 STREAMING LLM PERFORMANCE")
    print("=" * 40)
    print(f"Average: {avg_time:.3f}s")
    print(f"Fastest: {min_time:.3f}s") 
    print(f"Slowest: {max_time:.3f}s")
    
    print(f"\n🚀 VOICE AGENT ASSESSMENT")
    print("=" * 30)
    if avg_time < 0.3:
        print("✅ PRODUCTION READY: Sub-300ms for real-time voice")
    elif avg_time < 0.5:
        print("✅ VERY GOOD: Sub-500ms acceptable for voice") 
    elif avg_time < 1.0:
        print("⚠️  ACCEPTABLE: Sub-1s workable for voice")
    else:
        print("❌ TOO SLOW: >1s not suitable for voice agent")
    
    print(f"\n🧠 CONSCIOUSNESS STATE")
    print("=" * 25)
    print(f"Total memories: {len(ghost.tape)}")
    print(f"Total thoughts: {len(ghost.thoughts)}")
    print(f"Symbol patterns: {dict(list(ghost.symbol_frequency.items())[:5])}")

if __name__ == "__main__":
    asyncio.run(test_streaming_consciousness())