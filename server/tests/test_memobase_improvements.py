#!/usr/bin/env python3
"""
Test script for MemoBase memory processor improvements.
Tests deduplication, compression, and async performance.
"""

import asyncio
import time
import os
import sys
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent))

from processors.memobase_memory_processor import MemobaseMemoryProcessor
from config import config


async def test_deduplication():
    """Test memory content deduplication."""
    print("🧪 Testing memory content deduplication...")
    
    processor = MemobaseMemoryProcessor(user_id="test_user")
    
    # Create test memory content with duplicates
    duplicate_memory = """
User's alias: TestUser
The user likes coffee.
User's alias: TestUser
The user prefers tea over coffee.
The user likes coffee.
Important: User has a meeting at 3 PM.
The user prefers tea over coffee.
User's alias: TestUser
Recent conversation about weather.
Important: User has a meeting at 3 PM.
"""
    
    print(f"📊 Original memory ({len(duplicate_memory)} chars):")
    print(f"   Estimated tokens: {processor._estimate_tokens(duplicate_memory)}")
    
    deduplicated = processor._deduplicate_memory_content(duplicate_memory)
    
    print(f"📊 Deduplicated memory ({len(deduplicated)} chars):")
    print(f"   Estimated tokens: {processor._estimate_tokens(deduplicated)}")
    print(f"   Savings: {len(duplicate_memory) - len(deduplicated)} chars")
    
    print("🔍 Deduplicated content:")
    print(deduplicated)
    print()


async def test_compression_with_aliases():
    """Test compression while preserving user aliases."""
    print("🧪 Testing compression with user alias preservation...")
    
    processor = MemobaseMemoryProcessor(user_id="test_user")
    
    # Create large memory content with user aliases
    large_memory = """
User's alias: ImportantUser
The user is a software engineer working on AI systems.
They have extensive experience with Python and machine learning.
User's alias: ImportantUser
Recently discussed: Natural language processing and voice recognition systems.
The user mentioned they work with OpenAI's API frequently.
Technical preference: Prefers clean, readable code with good documentation.
User's alias: ImportantUser
Meeting scheduled for tomorrow at 2 PM to discuss project roadmap.
The user expressed interest in real-time audio processing capabilities.
Background: Has been working on voice agents for macOS applications.
They mentioned using MLX for optimal performance on Apple Silicon.
User's alias: ImportantUser
Previous conversation topics included Pipecat framework integration.
The user is particularly interested in sub-second voice-to-voice latency.
Technical notes: Currently using LM Studio with Ollama for different models.
""" * 10  # Make it large enough to trigger compression
    
    print(f"📊 Large memory ({len(large_memory)} chars):")
    print(f"   Estimated tokens: {processor._estimate_tokens(large_memory)}")
    
    compressed = processor._compress_memory_content(large_memory)
    
    print(f"📊 Compressed memory ({len(compressed)} chars):")
    print(f"   Estimated tokens: {processor._estimate_tokens(compressed)}")
    print(f"   Compression ratio: {len(compressed) / len(large_memory):.2%}")
    
    # Check if user aliases are preserved
    alias_count_original = large_memory.count("User's alias:")
    alias_count_compressed = compressed.count("User's alias:")
    
    print(f"🏷️ User aliases preserved: {alias_count_compressed}/{alias_count_original}")
    print()


async def test_async_performance():
    """Test async memory operations performance."""
    print("🧪 Testing async memory operations performance...")
    
    processor = MemobaseMemoryProcessor(user_id="test_user")
    
    # Simulate multiple rapid memory operations
    start_time = time.time()
    
    tasks = []
    for i in range(10):
        # Create async tasks for memory operations
        task = processor._add_to_conversation_buffer(
            "user", 
            f"Test message {i} with some content to simulate real usage",
            {"test_id": i}
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️ Processed 10 memory operations in {duration:.3f} seconds")
    print(f"   Average per operation: {duration/10:.3f} seconds")
    print(f"   Operations per second: {10/duration:.1f}")
    print()


async def test_cache_key_generation():
    """Test Redis cache key generation."""
    print("🧪 Testing cache key generation...")
    
    processor = MemobaseMemoryProcessor(user_id="test_user")
    
    test_message = "What's the weather like today?"
    
    memory_key = processor._generate_cache_key(test_message, "memory")
    injection_key = processor._generate_injection_key()
    
    print(f"🔑 Memory cache key: {memory_key}")
    print(f"🔑 Injection tracking key: {injection_key}")
    
    # Test key consistency
    key2 = processor._generate_cache_key(test_message, "memory")
    print(f"✅ Key consistency: {memory_key == key2}")
    print()


async def main():
    """Run all tests."""
    print("🚀 Starting MemoBase processor improvement tests...\n")
    
    try:
        await test_deduplication()
        await test_compression_with_aliases()
        await test_async_performance()
        await test_cache_key_generation()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())