#!/usr/bin/env python3
"""
Profile consciousness performance to identify bottlenecks
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

async def profile_consciousness():
    """Profile individual consciousness operations"""
    
    print("🔍 CONSCIOUSNESS PERFORMANCE PROFILER")
    print("=" * 50)
    
    ghost = Consciousness()
    print(f"Initial state: {len(ghost.tape)} memories, {len(ghost.thoughts)} thoughts")
    
    test_input = "Hello, can you remember this important message?"
    
    # Profile each step of the experience method
    print(f"\nProfiling input: '{test_input}'")
    
    # 1. Embedding generation
    start = time.time()
    from consciousness.core import simple_hash_embed
    embedding = simple_hash_embed(test_input)
    embed_time = time.time() - start
    print(f"1. Embedding generation: {embed_time:.4f}s")
    
    # 2. Symbolization
    start = time.time()
    symbols = ghost.symbolize(test_input)
    symbol_time = time.time() - start
    print(f"2. Symbolization: {symbol_time:.4f}s")
    
    # 3. Memory object creation
    start = time.time()
    from consciousness.core import Memory
    memory = Memory(
        content=test_input,
        timestamp=time.time(),
        role='user',
        tokens=len(test_input.split()) * 1.3,
        embedding=embedding,
        symbols=symbols
    )
    memory_create_time = time.time() - start
    print(f"3. Memory creation: {memory_create_time:.4f}s")
    
    # 4. Importance calculation
    start = time.time()
    importance = ghost.calculate_importance(memory)
    importance_time = time.time() - start
    print(f"4. Importance calculation: {importance_time:.4f}s")
    
    # 5. Memory retrieval (likely the bottleneck)
    start = time.time()
    relevant_memories = ghost.remember(test_input)
    remember_time = time.time() - start
    print(f"5. Memory retrieval: {remember_time:.4f}s ⚠️")
    
    # 6. Response generation
    start = time.time()
    response = await ghost.respond(relevant_memories)
    respond_time = time.time() - start
    print(f"6. Response generation: {respond_time:.4f}s")
    
    # 7. Reflection
    start = time.time()
    thought = ghost.reflect(memory, memory, ["test"])
    reflect_time = time.time() - start
    print(f"7. Reflection: {reflect_time:.4f}s")
    
    # 8. Weight evolution
    start = time.time()
    ghost.evolve(0.5)
    evolve_time = time.time() - start
    print(f"8. Weight evolution: {evolve_time:.4f}s")
    
    # 9. State persistence
    start = time.time()
    ghost.save_state()
    save_time = time.time() - start
    print(f"9. State persistence: {save_time:.4f}s")
    
    total_time = (embed_time + symbol_time + memory_create_time + importance_time + 
                  remember_time + respond_time + reflect_time + evolve_time + save_time)
    
    print(f"\n📊 PERFORMANCE BREAKDOWN")
    print("=" * 30)
    print(f"Total profiled time: {total_time:.4f}s")
    print(f"Biggest bottleneck: Memory retrieval ({remember_time:.4f}s)")
    
    # Now test the full experience method
    print(f"\n🧠 FULL EXPERIENCE METHOD TEST")
    print("=" * 30)
    start = time.time()
    result = await ghost.experience(test_input)
    full_time = time.time() - start
    print(f"Full experience() time: {full_time:.4f}s")
    
    if full_time > total_time * 2:
        print(f"⚠️  Full method is {full_time/total_time:.1f}x slower than individual components!")
        print("Likely cause: Async overhead or hidden operations")

if __name__ == "__main__":
    asyncio.run(profile_consciousness())