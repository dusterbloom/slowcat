#!/usr/bin/env python3
"""
Test script for Enhanced Stateless Memory System

This tests the three-tier memory architecture:
- Hot tier: Perfect recall
- Warm tier: LZ4 compressed 
- Cold tier: Zstd compressed with importance scoring
"""

import asyncio
import time
import tempfile
import shutil
from pathlib import Path
import json

from processors.enhanced_stateless_memory import (
    EnhancedStatelessMemoryProcessor, 
    MemoryItem, 
    MemoryTier
)
from loguru import logger

class MockFrame:
    """Mock frame for testing"""
    def __init__(self, text):
        self.text = text

async def test_three_tier_memory():
    """Test the enhanced three-tier memory system"""
    
    print("🧠 Enhanced Memory System Test")
    print("=" * 50)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize enhanced memory processor
        memory_processor = EnhancedStatelessMemoryProcessor(
            db_path=temp_dir,
            max_context_tokens=512,
            hot_tier_size=5,     # Small for testing
            warm_tier_size=10,   # Small for testing  
            cold_tier_size=20,   # Small for testing
            degradation_interval=2  # Fast degradation for testing
        )
        
        print(f"✅ Memory processor initialized at {temp_dir}")
        
        # Test 1: Hot tier storage and retrieval
        print("\n🔥 Testing Hot Tier...")
        
        conversations = [
            ("What is machine learning?", "Machine learning is a subset of AI that uses algorithms to learn patterns."),
            ("Explain neural networks", "Neural networks are computing systems inspired by biological networks."),
            ("What is deep learning?", "Deep learning uses neural networks with multiple layers."),
            ("Tell me about Python", "Python is a versatile programming language."),
            ("How does training work?", "Training involves feeding data to algorithms to improve performance."),
        ]
        
        # Store conversations in hot tier
        for i, (user_msg, assistant_msg) in enumerate(conversations):
            memory_processor.current_user_message = user_msg
            await memory_processor._store_exchange_three_tier(user_msg, assistant_msg)
            print(f"   Stored conversation {i+1}: '{user_msg[:30]}...'")
        
        # Test hot tier retrieval
        memories = await memory_processor._get_relevant_memories_three_tier(
            "What is machine learning?", 
            "default_user", 
            max_tokens=200
        )
        print(f"   Retrieved {len(memories)} memories from hot tier")
        
        # Test 2: Force degradation to warm tier
        print("\n🌡️ Testing Degradation to Warm Tier...")
        
        # Add more conversations to trigger degradation
        additional_conversations = [
            ("What is TensorFlow?", "TensorFlow is an open-source machine learning framework."),
            ("Explain gradient descent", "Gradient descent is an optimization algorithm."),
            ("What are transformers?", "Transformers are a neural network architecture."),
        ]
        
        for user_msg, assistant_msg in additional_conversations:
            memory_processor.current_user_message = user_msg
            await memory_processor._store_exchange_three_tier(user_msg, assistant_msg)
        
        # Force degradation
        await memory_processor._perform_degradation()
        
        # Check tier distribution
        stats = memory_processor.get_enhanced_stats()
        print(f"   Hot tier: {stats['tier_distribution']['hot']} items")
        print(f"   Warm tier: {stats['tier_distribution']['warm']} items")
        print(f"   Cold tier: {stats['tier_distribution']['cold']} items")
        
        # Test 3: Cross-tier retrieval
        print("\n🔍 Testing Cross-Tier Retrieval...")
        
        # Search for something that should span tiers
        memories = await memory_processor._get_relevant_memories_three_tier(
            "machine learning algorithms", 
            "default_user",
            max_tokens=300
        )
        
        print(f"   Found {len(memories)} memories across tiers:")
        tier_counts = {"HOT": 0, "WARM": 0, "COLD": 0}
        for memory in memories:
            tier_counts[memory.tier.value.upper()] += 1
            print(f"     {memory.tier.value.upper()}: '{memory.content[:40]}...' (score: {memory.importance_score:.2f})")
        
        print(f"   Tier distribution: {tier_counts}")
        
        # Test 4: Importance scoring
        print("\n⭐ Testing Importance Scoring...")
        
        # Add conversations with different importance levels
        important_conversations = [
            ("How do I fix this critical bug?", "Critical bugs require immediate debugging steps."),  # High importance
            ("Hello", "Hello! How can I help you today?"),  # Low importance  
            ("Explain the algorithm complexity", "Algorithm complexity analysis involves Big O notation."),  # Medium importance
            ("What's the weather?", "I don't have access to weather data."),  # Low importance
        ]
        
        for user_msg, assistant_msg in important_conversations:
            memory_processor.current_user_message = user_msg
            await memory_processor._store_exchange_three_tier(user_msg, assistant_msg)
        
        # Check importance scores in hot tier
        print("   Importance scores in hot tier:")
        for item in memory_processor.hot_tier:
            if isinstance(item, MemoryItem):
                print(f"     '{item.content[:30]}...' -> {item.importance_score:.2f}")
        
        # Test 5: Performance metrics
        print("\n📊 Testing Performance Metrics...")
        
        # Perform several retrieval operations
        queries = [
            "machine learning",
            "neural networks", 
            "algorithms",
            "Python programming",
            "debugging"
        ]
        
        start_time = time.perf_counter()
        total_memories = 0
        
        for query in queries:
            memories = await memory_processor._get_relevant_memories_three_tier(
                query, "default_user", max_tokens=200
            )
            total_memories += len(memories)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        print(f"   Retrieved {total_memories} memories in {elapsed_ms:.2f}ms")
        print(f"   Average: {elapsed_ms / len(queries):.2f}ms per query")
        
        # Final statistics
        final_stats = memory_processor.get_enhanced_stats()
        print(f"\n📈 Final Statistics:")
        print(f"   Total items: {final_stats['storage']['total_items']}")
        print(f"   Hot: {final_stats['tier_distribution']['hot']}")
        print(f"   Warm: {final_stats['tier_distribution']['warm']}")
        print(f"   Cold: {final_stats['tier_distribution']['cold']}")
        print(f"   Storage type: {final_stats['storage']['storage_type']}")
        print(f"   Compression available: {final_stats['storage']['compression_available']}")
        print(f"   Degradation events: {final_stats['performance']['degradation_events']}")
        print(f"   Avg retrieval time: {final_stats['performance']['avg_retrieval_time_ms']:.2f}ms")
        
        # Test 6: Memory context injection
        print("\n💉 Testing Memory Context Injection...")
        
        # Set up a query that should trigger memory injection
        memory_processor.current_user_message = "Tell me about machine learning again"
        memory_processor.last_injected_message = ""  # Reset to allow injection
        
        # Test injection
        await memory_processor._inject_memory_for_transcription("Tell me about machine learning again")
        
        print("   Memory injection test completed")
        
        print("\n✅ All Enhanced Memory Tests Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 Cleaned up test directory: {temp_dir}")

async def test_degradation_timing():
    """Test automatic degradation timing"""
    
    print("\n⏰ Testing Automatic Degradation Timing...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create processor with very fast degradation for testing
        memory_processor = EnhancedStatelessMemoryProcessor(
            db_path=temp_dir,
            hot_tier_size=3,
            degradation_interval=1  # 1 second for testing
        )
        
        # Add conversations
        conversations = [
            ("Question 1", "Answer 1"),
            ("Question 2", "Answer 2"),
            ("Question 3", "Answer 3"),
            ("Question 4", "Answer 4"),  # This should trigger degradation
        ]
        
        for i, (user_msg, assistant_msg) in enumerate(conversations):
            memory_processor.current_user_message = user_msg
            await memory_processor._store_exchange_three_tier(user_msg, assistant_msg)
            
            stats = memory_processor.get_enhanced_stats()
            print(f"   After conversation {i+1}: Hot={stats['tier_distribution']['hot']}, Warm={stats['tier_distribution']['warm']}")
            
            # Wait for potential degradation
            if i == 2:  # After filling hot tier
                print("   Waiting for automatic degradation...")
                await asyncio.sleep(2)
        
        print("   Final tier distribution:")
        final_stats = memory_processor.get_enhanced_stats()
        print(f"     Hot: {final_stats['tier_distribution']['hot']}")
        print(f"     Warm: {final_stats['tier_distribution']['warm']}")
        print(f"     Cold: {final_stats['tier_distribution']['cold']}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def test_compression_ratios():
    """Test compression effectiveness"""
    
    print("\n🗜️ Testing Compression Ratios...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        memory_processor = EnhancedStatelessMemoryProcessor(
            db_path=temp_dir,
            hot_tier_size=2,
            warm_tier_size=5
        )
        
        # Create conversations with varying lengths
        long_conversations = [
            ("Tell me everything about machine learning", 
             "Machine learning is a vast field that encompasses many different algorithms, techniques, and approaches. It includes supervised learning where we train models on labeled data, unsupervised learning where we find patterns in unlabeled data, and reinforcement learning where agents learn through interaction with environments. The field has evolved significantly over the past few decades."),
            ("Explain deep learning in detail",
             "Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers. These networks can learn hierarchical representations of data, where early layers learn simple features and deeper layers learn more complex, abstract features. Deep learning has been particularly successful in computer vision, natural language processing, and speech recognition tasks."),
        ]
        
        for user_msg, assistant_msg in long_conversations:
            memory_processor.current_user_message = user_msg
            await memory_processor._store_exchange_three_tier(user_msg, assistant_msg)
        
        # Force degradation to test compression
        await memory_processor._perform_degradation()
        
        # Calculate compression ratios
        total_original = 0
        total_compressed = 0
        
        for item in memory_processor.hot_tier:
            if isinstance(item, MemoryItem):
                total_original += item.original_size
                total_compressed += item.compressed_size
        
        if total_original > 0:
            compression_ratio = total_compressed / total_original
            print(f"   Compression ratio: {compression_ratio:.2f}")
            print(f"   Space saved: {(1 - compression_ratio) * 100:.1f}%")
        else:
            print("   No compression data available")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}")
    
    async def run_all_tests():
        await test_three_tier_memory()
        await test_degradation_timing()
        await test_compression_ratios()
        
        print("\n🎉 All Enhanced Memory Tests Completed!")
    
    asyncio.run(run_all_tests())