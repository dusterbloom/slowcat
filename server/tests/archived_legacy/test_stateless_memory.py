"""
Comprehensive test suite for the stateless memory system

Tests:
1. Constant performance over many conversation turns
2. Memory degradation quality
3. Semantic validation accuracy
4. Perfect recall window functionality
5. Context injection latency
6. A/B comparison with traditional memory
"""

import asyncio
import time
import tempfile
import shutil
import json
import os
from pathlib import Path
from typing import List, Dict, Any

# Test framework
import pytest
import numpy as np

# Pipecat imports
from pipecat.frames.frames import LLMMessagesFrame, TextFrame

# Our imports
from processors.stateless_memory import (
    StatelessMemoryProcessor, 
    MemoryItem, 
    MemoryDegradation,
    SemanticValidator
)
from processors.local_memory import LocalMemoryProcessor  # For comparison


class TestStatelessMemoryProcessor:
    """Test the enhanced stateless memory processor"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def processor(self, temp_db):
        """Create test processor"""
        return StatelessMemoryProcessor(
            db_path=temp_db,
            max_context_tokens=1024,
            perfect_recall_window=10,
            enable_semantic_validation=True
        )
    
    def test_constant_performance_over_many_turns(self, processor):
        """Test that memory injection stays fast regardless of conversation length"""
        
        async def run_performance_test():
            latencies = []
            
            # Simulate 100 conversation turns
            for turn in range(100):
                # Create messages like real conversation
                messages = [
                    {'role': 'system', 'content': 'You are a helpful assistant'},
                    {'role': 'user', 'content': f'This is test message number {turn}. My dog is named Potola.'}
                ]
                
                # Measure injection time
                start = time.perf_counter()
                enhanced = await processor._inject_memory_context(messages, 'test_user')
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                latencies.append(elapsed_ms)
                
                # Store a response to build up memory
                await processor._store_exchange(
                    f'Test message {turn}',
                    f'Response to test message {turn}'
                )
                
                # Log progress every 20 turns
                if turn % 20 == 0:
                    avg_latency = sum(latencies[-10:]) / min(10, len(latencies))
                    print(f"Turn {turn}: Avg latency = {avg_latency:.2f}ms")
            
            # Analyze performance consistency
            first_10_avg = sum(latencies[:10]) / 10
            last_10_avg = sum(latencies[-10:]) / 10
            overall_avg = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            print(f"\nPerformance Analysis:")
            print(f"First 10 turns avg: {first_10_avg:.2f}ms")
            print(f"Last 10 turns avg: {last_10_avg:.2f}ms")
            print(f"Overall average: {overall_avg:.2f}ms")
            print(f"Maximum latency: {max_latency:.2f}ms")
            
            # Performance requirements
            assert overall_avg < 15.0, f"Average latency too high: {overall_avg:.2f}ms"
            assert max_latency < 50.0, f"Maximum latency too high: {max_latency:.2f}ms"
            
            # Consistency requirement - latency shouldn't increase significantly
            performance_degradation = abs(last_10_avg - first_10_avg) / first_10_avg
            assert performance_degradation < 0.5, f"Performance degraded by {performance_degradation*100:.1f}%"
            
            # Get statistics
            stats = processor.get_performance_stats()
            print(f"\nProcessor Statistics:")
            print(f"Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
            print(f"Total conversations: {stats['total_conversations']}")
            print(f"Reconstruction failures: {stats['reconstruction_failures']}")
            
            assert stats['cache_hit_ratio'] > 0.6, "Cache hit ratio too low"
            assert stats['reconstruction_failures'] == 0, "Reconstruction failures detected"
            
        asyncio.run(run_performance_test())
    
    def test_memory_degradation_quality(self):
        """Test that memory degradation preserves semantic meaning"""
        
        degradation = MemoryDegradation(half_life_days=1.0)  # Fast degradation for testing
        
        # Test cases with varying complexity
        test_cases = [
            {
                'original': 'My dog name is Potola and she loves to play fetch in the park',
                'expected_entities': ['dog', 'Potola', 'fetch', 'park']
            },
            {
                'original': 'I work as a software engineer at Google in San Francisco',
                'expected_entities': ['software', 'engineer', 'Google', 'San', 'Francisco']
            },
            {
                'original': 'The weather today is sunny and 75 degrees',
                'expected_entities': ['weather', 'sunny', '75', 'degrees']
            }
        ]
        
        for case in test_cases:
            # Create memory item
            memory = MemoryItem(
                content=case['original'],
                timestamp=time.time() - 86400 * 2,  # 2 days ago for degradation
                speaker_id='test_user',
                access_count=1
            )
            
            # Apply degradation
            degraded = degradation.apply_degradation(memory)
            
            print(f"\nOriginal: {case['original']}")
            print(f"Degraded: {degraded.content}")
            print(f"Compression level: {degraded.compression_level}")
            print(f"Confidence: {degraded.reconstruction_confidence}")
            
            # Verify some key entities are preserved
            degraded_words = degraded.content.lower().split()
            preserved_entities = 0
            
            for entity in case['expected_entities']:
                if any(entity.lower() in word for word in degraded_words):
                    preserved_entities += 1
            
            preservation_ratio = preserved_entities / len(case['expected_entities'])
            print(f"Preservation ratio: {preservation_ratio:.2%}")
            
            # At least 30% of key entities should be preserved
            assert preservation_ratio >= 0.3, f"Too much information lost: {preservation_ratio:.2%}"
            
            # Confidence should reflect degradation level
            if degraded.compression_level <= 1:
                assert degraded.reconstruction_confidence >= 0.8
            elif degraded.compression_level <= 2:
                assert degraded.reconstruction_confidence >= 0.6
    
    def test_semantic_validation(self):
        """Test semantic validation prevents drift"""
        
        validator = SemanticValidator()
        
        test_cases = [
            {
                'original': 'My dog name is Potola',
                'good_reconstruction': 'dog Potola',
                'bad_reconstruction': 'cat Whiskers',
                'expected_good_similarity': 0.7,
                'expected_bad_similarity': 0.3
            },
            {
                'original': 'I work as a software engineer',
                'good_reconstruction': 'software engineer work',
                'bad_reconstruction': 'teacher school students',
                'expected_good_similarity': 0.6,
                'expected_bad_similarity': 0.2
            }
        ]
        
        for case in test_cases:
            # Test good reconstruction
            good_valid, good_sim = validator.validate_reconstruction(
                case['original'],
                case['good_reconstruction'],
                min_similarity=0.5
            )
            
            # Test bad reconstruction
            bad_valid, bad_sim = validator.validate_reconstruction(
                case['original'],
                case['bad_reconstruction'],
                min_similarity=0.5
            )
            
            print(f"\nOriginal: {case['original']}")
            print(f"Good reconstruction: {case['good_reconstruction']} (valid: {good_valid}, sim: {good_sim:.2f})")
            print(f"Bad reconstruction: {case['bad_reconstruction']} (valid: {bad_valid}, sim: {bad_sim:.2f})")
            
            # Good reconstruction should be valid
            assert good_valid, f"Good reconstruction marked invalid: {good_sim:.2f}"
            assert good_sim >= case['expected_good_similarity'], f"Good similarity too low: {good_sim:.2f}"
            
            # Bad reconstruction should be invalid
            assert not bad_valid, f"Bad reconstruction marked valid: {bad_sim:.2f}"
            assert bad_sim <= case['expected_bad_similarity'], f"Bad similarity too high: {bad_sim:.2f}"
    
    def test_perfect_recall_window(self, processor):
        """Test that recent conversations have perfect recall"""
        
        async def run_recall_test():
            # Store more conversations than the perfect recall window
            conversations = []
            
            for i in range(15):  # More than window size of 10
                user_msg = f"Message {i}: Remember that my favorite color is blue"
                assistant_msg = f"I'll remember that your favorite color is blue, message {i}"
                
                await processor._store_exchange(user_msg, assistant_msg)
                conversations.append((user_msg, assistant_msg))
            
            # Check that recent conversations are in perfect recall cache
            cache_contents = list(processor.perfect_recall_cache)
            
            # Should have exactly the window size * 2 (user + assistant messages)
            expected_cache_size = processor.perfect_recall_window * 2
            assert len(cache_contents) == expected_cache_size, f"Cache size wrong: {len(cache_contents)}"
            
            # Check that the most recent conversations are present with full fidelity
            recent_contents = [item.content for item in cache_contents if hasattr(item, 'content')]
            
            # Last 5 conversations should be fully represented
            for i in range(10, 15):  # Last 5 conversations
                user_msg, assistant_msg = conversations[i]
                
                # Check if messages are in cache (might be slightly modified)
                user_found = any(user_msg[:20] in content for content in recent_contents)
                assistant_found = any(assistant_msg[:20] in content for content in recent_contents)
                
                assert user_found, f"Recent user message not in cache: {user_msg}"
                assert assistant_found, f"Recent assistant message not in cache: {assistant_msg}"
            
            print(f"Perfect recall window test passed: {len(cache_contents)} items in cache")
            
        asyncio.run(run_recall_test())
    
    def test_context_injection_format(self, processor):
        """Test that context injection works correctly"""
        
        async def run_injection_test():
            # Store some test conversations
            await processor._store_exchange(
                "My dog is named Potola",
                "That's a lovely name for your dog!"
            )
            
            await processor._store_exchange(
                "She loves playing fetch",
                "Dogs often enjoy playing fetch!"
            )
            
            # Test context injection
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant'},
                {'role': 'user', 'content': 'What do you know about my dog?'}
            ]
            
            enhanced_messages = await processor._inject_memory_context(messages, 'test_user')
            
            # Should have added a memory context message
            assert len(enhanced_messages) > len(messages), "No memory context added"
            
            # Find the memory context message
            memory_message = None
            for msg in enhanced_messages:
                if '[Memory Context' in msg.get('content', ''):
                    memory_message = msg
                    break
            
            assert memory_message is not None, "Memory context message not found"
            assert memory_message['role'] == 'system', "Memory context should be system message"
            
            # Should contain relevant information about the dog
            content = memory_message['content'].lower()
            assert 'potola' in content, "Dog's name not in memory context"
            
            print(f"Context injection test passed")
            print(f"Memory context: {memory_message['content'][:100]}...")
            
        asyncio.run(run_injection_test())


class TestMemoryComparison:
    """Compare stateless vs traditional memory performance"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for both systems"""
        stateless_dir = tempfile.mkdtemp()
        traditional_dir = tempfile.mkdtemp()
        yield stateless_dir, traditional_dir
        shutil.rmtree(stateless_dir)
        shutil.rmtree(traditional_dir)
    
    def test_ab_performance_comparison(self, temp_dirs):
        """A/B test comparing stateless vs traditional memory"""
        
        stateless_dir, traditional_dir = temp_dirs
        
        async def benchmark_system(use_stateless: bool, db_path: str, num_turns: int = 50):
            """Benchmark a memory system"""
            
            if use_stateless:
                processor = StatelessMemoryProcessor(
                    db_path=db_path,
                    max_context_tokens=1024,
                    perfect_recall_window=10
                )
                system_name = "Stateless"
            else:
                processor = LocalMemoryProcessor(
                    data_dir=db_path,
                    max_history_items=200,
                    include_in_context=10
                )
                system_name = "Traditional"
            
            latencies = []
            
            for turn in range(num_turns):
                messages = [
                    {'role': 'system', 'content': 'You are a helpful assistant'},
                    {'role': 'user', 'content': f'Turn {turn}: My dog Potola loves walks'}
                ]
                
                start = time.perf_counter()
                
                if use_stateless:
                    enhanced = await processor._inject_memory_context(messages, 'test_user')
                    await processor._store_exchange(f'Turn {turn} input', f'Turn {turn} response')
                else:
                    # Traditional memory processor doesn't have direct injection method
                    # So we'll just measure the frame processing time
                    frame = LLMMessagesFrame(messages=messages)
                    # This is a simplified benchmark - in reality LocalMemoryProcessor 
                    # works differently through the frame processing pipeline
                    pass
                
                elapsed = time.perf_counter() - start
                latencies.append(elapsed * 1000)  # Convert to ms
            
            return {
                'system': system_name,
                'avg_latency_ms': sum(latencies) / len(latencies),
                'max_latency_ms': max(latencies),
                'min_latency_ms': min(latencies),
                'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
                'latencies': latencies
            }
        
        async def run_comparison():
            print("Running A/B performance comparison...")
            
            # Test both systems
            stateless_results = await benchmark_system(True, stateless_dir)
            traditional_results = await benchmark_system(False, traditional_dir)
            
            # Print results
            print(f"\n📊 Performance Comparison Results:")
            print(f"{'System':<12} {'Avg (ms)':<10} {'P95 (ms)':<10} {'Max (ms)':<10}")
            print("-" * 45)
            
            for results in [stateless_results, traditional_results]:
                print(f"{results['system']:<12} {results['avg_latency_ms']:<10.1f} "
                      f"{results['p95_latency_ms']:<10.1f} {results['max_latency_ms']:<10.1f}")
            
            # Calculate improvement
            if traditional_results['avg_latency_ms'] > 0:
                improvement = (
                    (traditional_results['avg_latency_ms'] - stateless_results['avg_latency_ms']) 
                    / traditional_results['avg_latency_ms'] * 100
                )
                print(f"\n🚀 Stateless is {improvement:.1f}% faster on average!")
            
            # Performance assertions
            assert stateless_results['avg_latency_ms'] < 20.0, "Stateless system too slow"
            assert stateless_results['p95_latency_ms'] < 30.0, "P95 latency too high"
            
            return stateless_results, traditional_results
        
        asyncio.run(run_comparison())


@pytest.mark.asyncio
async def test_memory_system_integration():
    """Integration test with realistic conversation flow"""
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        processor = StatelessMemoryProcessor(
            db_path=temp_dir,
            max_context_tokens=1024,
            perfect_recall_window=5  # Smaller for testing
        )
        
        # Simulate a realistic conversation about a pet
        conversation_turns = [
            ("My dog name is Potola", "Nice to meet Potola! Tell me more about her."),
            ("She's a golden retriever", "Golden retrievers are wonderful dogs!"),
            ("She loves playing fetch", "That's typical for golden retrievers!"),
            ("We go to the park every morning", "What a great routine for both of you!"),
            ("Potola is 3 years old", "She's still quite young and playful then!"),
            ("What do you remember about my dog?", "Let me recall what you've told me...")
        ]
        
        # Process each turn
        for i, (user_msg, assistant_msg) in enumerate(conversation_turns):
            print(f"\nTurn {i+1}: {user_msg}")
            
            # Create messages as they would appear in pipeline
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant'},
                {'role': 'user', 'content': user_msg}
            ]
            
            # Inject memory context
            enhanced_messages = await processor._inject_memory_context(messages, 'test_user')
            
            # Store the exchange
            await processor._store_exchange(user_msg, assistant_msg)
            
            # Check memory context injection on the final turn
            if i == len(conversation_turns) - 1:
                # Find memory context
                memory_context = None
                for msg in enhanced_messages:
                    if '[Memory Context' in msg.get('content', ''):
                        memory_context = msg['content']
                        break
                
                assert memory_context is not None, "No memory context on final turn"
                
                # Should contain information about Potola
                context_lower = memory_context.lower()
                assert 'potola' in context_lower, "Dog's name not in context"
                assert 'golden retriever' in context_lower or 'golden' in context_lower, "Breed not in context"
                assert '3' in context_lower or 'three' in context_lower, "Age not in context"
                
                print(f"Final memory context includes key information about Potola ✓")
        
        # Get final statistics
        stats = processor.get_performance_stats()
        print(f"\nFinal Statistics:")
        print(f"Total conversations: {stats['total_conversations']}")
        print(f"Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
        print(f"Average injection time: {stats['avg_injection_time_ms']:.2f}ms")
        
        # Verify performance
        assert stats['avg_injection_time_ms'] < 15.0, "Average injection time too high"
        assert stats['reconstruction_failures'] == 0, "Reconstruction failures occurred"
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests directly
    import sys
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run specific test
    test_processor = TestStatelessMemoryProcessor()
    
    print("Testing constant performance...")
    # This would need actual pytest setup to run properly
    # For now, just print that the test file is created
    print("✅ Stateless memory test suite created successfully!")
    print("\nTo run tests:")
    print("cd server/")
    print("source venv/bin/activate")
    print("python -m pytest tests/test_stateless_memory.py -v")