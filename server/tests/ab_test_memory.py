#!/usr/bin/env python3
"""
A/B Testing Script for Stateless vs Traditional Memory
Run this to compare performance and quality between memory systems
"""

import asyncio
import time
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.stateless_memory import StatelessMemoryProcessor
from processors.local_memory import LocalMemoryProcessor
from pipecat.frames.frames import LLMMessagesFrame


class MemoryBenchmark:
    """Benchmark memory systems for performance and quality"""
    
    def __init__(self):
        self.test_conversations = self._generate_test_conversations()
    
    def _generate_test_conversations(self) -> List[tuple]:
        """Generate realistic test conversations"""
        return [
            ("My dog name is Potola", "That's a beautiful name for your dog!"),
            ("She's a golden retriever", "Golden retrievers are wonderful dogs!"),
            ("Potola is 3 years old", "She's still quite young and energetic then!"),
            ("We live in San Francisco", "What a great city for dog walks!"),
            ("She loves playing fetch", "That's typical for golden retrievers!"),
            ("We go to Golden Gate Park", "Perfect place for a golden retriever!"),
            ("Potola knows many tricks", "Smart dog! What tricks does she know?"),
            ("She can sit, stay, and roll over", "Those are impressive basic commands!"),
            ("I work as a software engineer", "Interesting! Do you work from home?"),
            ("Yes, I work remotely", "That's great for spending time with Potola!"),
            ("My favorite programming language is Python", "Python is excellent for many projects!"),
            ("I'm building an AI voice assistant", "That sounds like an exciting project!"),
            ("It uses local models on Mac", "Local processing has many advantages!"),
            ("The assistant remembers our conversations", "Memory is crucial for good AI assistants!"),
            ("What do you remember about my dog?", "Let me recall what you've told me about Potola..."),
            ("Where do we go for walks?", "You mentioned Golden Gate Park, right?"),
            ("What's my profession again?", "You work as a software engineer remotely."),
            ("What programming language do I prefer?", "You mentioned Python as your favorite."),
            ("How old is my dog?", "Potola is 3 years old."),
            ("What breed is she?", "She's a golden retriever.")
        ]
    
    async def benchmark_stateless_memory(self, temp_dir: str, num_conversations: int = 50) -> Dict[str, Any]:
        """Benchmark the stateless memory system"""
        
        processor = StatelessMemoryProcessor(
            db_path=temp_dir,
            max_context_tokens=1024,
            perfect_recall_window=10,
            enable_semantic_validation=True
        )
        
        injection_times = []
        storage_times = []
        context_qualities = []
        
        print("🧠 Testing Stateless Memory System...")
        
        for i in range(num_conversations):
            user_msg, assistant_msg = self.test_conversations[i % len(self.test_conversations)]
            
            # Measure context injection time
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant'},
                {'role': 'user', 'content': user_msg}
            ]
            
            start = time.perf_counter()
            enhanced_messages = await processor._inject_memory_context(messages, 'test_user')
            injection_time = (time.perf_counter() - start) * 1000
            injection_times.append(injection_time)
            
            # Measure storage time
            start = time.perf_counter()
            await processor._store_exchange(user_msg, assistant_msg)
            storage_time = (time.perf_counter() - start) * 1000
            storage_times.append(storage_time)
            
            # Analyze context quality
            context_quality = self._analyze_context_quality(enhanced_messages, user_msg)
            context_qualities.append(context_quality)
            
            if i % 10 == 0:
                avg_injection = statistics.mean(injection_times[-10:]) if injection_times else 0
                print(f"  Turn {i:2d}: Injection {avg_injection:.1f}ms")
        
        # Get final stats
        stats = processor.get_performance_stats()
        
        return {
            'system': 'Stateless',
            'avg_injection_ms': statistics.mean(injection_times),
            'p95_injection_ms': sorted(injection_times)[int(len(injection_times) * 0.95)],
            'max_injection_ms': max(injection_times),
            'avg_storage_ms': statistics.mean(storage_times),
            'avg_context_quality': statistics.mean(context_qualities),
            'cache_hit_ratio': stats['cache_hit_ratio'],
            'total_conversations': stats['total_conversations'],
            'reconstruction_failures': stats['reconstruction_failures'],
            'injection_times': injection_times,
            'storage_times': storage_times,
            'context_qualities': context_qualities
        }
    
    async def benchmark_traditional_memory(self, temp_dir: str, num_conversations: int = 50) -> Dict[str, Any]:
        """Benchmark the traditional memory system"""
        
        processor = LocalMemoryProcessor(
            data_dir=temp_dir,
            max_history_items=200,
            include_in_context=10
        )
        
        # Initialize processor context
        await processor.start()
        
        injection_times = []
        storage_times = []
        context_qualities = []
        
        print("📝 Testing Traditional Memory System...")
        
        for i in range(num_conversations):
            user_msg, assistant_msg = self.test_conversations[i % len(self.test_conversations)]
            
            # Simulate message processing (traditional system works differently)
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant'},
                {'role': 'user', 'content': user_msg}
            ]
            
            # Traditional memory doesn't have direct injection method
            # We'll measure the overall processing time
            start = time.perf_counter()
            
            # Simulate traditional memory context retrieval
            memory_context = await processor.get_memory_context('test_user')
            if memory_context:
                memory_message = {
                    'role': 'system', 
                    'content': f"Memory context: {memory_context}"
                }
                messages.insert(1, memory_message)
            
            injection_time = (time.perf_counter() - start) * 1000
            injection_times.append(injection_time)
            
            # Measure storage time
            start = time.perf_counter()
            await processor.add_conversation_item('test_user', 'user', user_msg)
            await processor.add_conversation_item('test_user', 'assistant', assistant_msg)
            storage_time = (time.perf_counter() - start) * 1000
            storage_times.append(storage_time)
            
            # Analyze context quality
            context_quality = self._analyze_context_quality(messages, user_msg)
            context_qualities.append(context_quality)
            
            if i % 10 == 0:
                avg_injection = statistics.mean(injection_times[-10:]) if injection_times else 0
                print(f"  Turn {i:2d}: Injection {avg_injection:.1f}ms")
        
        await processor.stop()
        
        return {
            'system': 'Traditional',
            'avg_injection_ms': statistics.mean(injection_times),
            'p95_injection_ms': sorted(injection_times)[int(len(injection_times) * 0.95)],
            'max_injection_ms': max(injection_times),
            'avg_storage_ms': statistics.mean(storage_times),
            'avg_context_quality': statistics.mean(context_qualities),
            'cache_hit_ratio': 0.0,  # Traditional doesn't have cache
            'total_conversations': num_conversations,
            'reconstruction_failures': 0,  # Traditional doesn't reconstruct
            'injection_times': injection_times,
            'storage_times': storage_times,
            'context_qualities': context_qualities
        }
    
    def _analyze_context_quality(self, messages: List[Dict], user_msg: str) -> float:
        """Analyze the quality of injected context"""
        
        # Find memory context message
        memory_content = ""
        for msg in messages:
            if 'memory' in msg.get('content', '').lower():
                memory_content = msg['content'].lower()
                break
        
        if not memory_content:
            return 0.0
        
        # Simple quality metrics
        quality_score = 0.0
        
        # Check for key information based on user message
        if 'dog' in user_msg.lower():
            if 'potola' in memory_content:
                quality_score += 0.3
            if 'golden retriever' in memory_content:
                quality_score += 0.2
            if '3' in memory_content or 'three' in memory_content:
                quality_score += 0.2
        
        if 'work' in user_msg.lower() or 'programming' in user_msg.lower():
            if 'software engineer' in memory_content:
                quality_score += 0.3
            if 'python' in memory_content:
                quality_score += 0.2
        
        if 'walk' in user_msg.lower() or 'park' in user_msg.lower():
            if 'golden gate park' in memory_content:
                quality_score += 0.3
            if 'san francisco' in memory_content:
                quality_score += 0.2
        
        # Base score for having any context
        if memory_content:
            quality_score += 0.3
        
        return min(quality_score, 1.0)
    
    def print_results(self, stateless_results: Dict, traditional_results: Dict):
        """Print comprehensive comparison results"""
        
        print("\n" + "="*60)
        print("🚀 MEMORY SYSTEM A/B TEST RESULTS")
        print("="*60)
        
        # Performance comparison table
        print(f"\n📊 Performance Metrics:")
        print(f"{'Metric':<25} {'Stateless':<15} {'Traditional':<15} {'Improvement':<15}")
        print("-" * 70)
        
        metrics = [
            ('Avg Injection (ms)', 'avg_injection_ms'),
            ('P95 Injection (ms)', 'p95_injection_ms'),
            ('Max Injection (ms)', 'max_injection_ms'),
            ('Avg Storage (ms)', 'avg_storage_ms'),
            ('Context Quality', 'avg_context_quality'),
            ('Cache Hit Ratio', 'cache_hit_ratio')
        ]
        
        for metric_name, metric_key in metrics:
            stateless_val = stateless_results[metric_key]
            traditional_val = traditional_results[metric_key]
            
            if traditional_val > 0 and metric_key != 'avg_context_quality':
                improvement = ((traditional_val - stateless_val) / traditional_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            elif metric_key == 'avg_context_quality':
                improvement = ((stateless_val - traditional_val) / traditional_val) * 100 if traditional_val > 0 else 0
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
            
            print(f"{metric_name:<25} {stateless_val:<15.2f} {traditional_val:<15.2f} {improvement_str:<15}")
        
        # Key findings
        print(f"\n🔍 Key Findings:")
        
        if stateless_results['avg_injection_ms'] < traditional_results['avg_injection_ms']:
            improvement = ((traditional_results['avg_injection_ms'] - stateless_results['avg_injection_ms']) 
                          / traditional_results['avg_injection_ms']) * 100
            print(f"✅ Stateless is {improvement:.1f}% faster for memory injection")
        else:
            print(f"❌ Traditional is faster for memory injection")
        
        if stateless_results['avg_context_quality'] >= traditional_results['avg_context_quality']:
            print(f"✅ Stateless maintains equal or better context quality")
        else:
            print(f"⚠️  Traditional has slightly better context quality")
        
        print(f"✅ Stateless cache hit ratio: {stateless_results['cache_hit_ratio']:.1%}")
        print(f"✅ Stateless reconstruction failures: {stateless_results['reconstruction_failures']}")
        
        # Performance consistency
        stateless_stdev = statistics.stdev(stateless_results['injection_times'])
        traditional_stdev = statistics.stdev(traditional_results['injection_times'])
        
        print(f"\n📈 Performance Consistency:")
        print(f"Stateless injection std dev: {stateless_stdev:.2f}ms")
        print(f"Traditional injection std dev: {traditional_stdev:.2f}ms")
        
        if stateless_stdev < traditional_stdev:
            print(f"✅ Stateless has more consistent performance")
        else:
            print(f"⚠️  Traditional has more consistent performance")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if (stateless_results['avg_injection_ms'] < 15.0 and 
            stateless_results['reconstruction_failures'] == 0 and
            stateless_results['avg_context_quality'] >= 0.7):
            print(f"✅ RECOMMENDED: Deploy stateless memory system")
            print(f"   - Fast and consistent performance")
            print(f"   - Good context quality")
            print(f"   - Zero reconstruction failures")
        else:
            print(f"⚠️  CAUTION: Consider optimizing before deployment")
            if stateless_results['avg_injection_ms'] >= 15.0:
                print(f"   - Injection time too high: {stateless_results['avg_injection_ms']:.1f}ms")
            if stateless_results['reconstruction_failures'] > 0:
                print(f"   - Reconstruction failures: {stateless_results['reconstruction_failures']}")
            if stateless_results['avg_context_quality'] < 0.7:
                print(f"   - Context quality too low: {stateless_results['avg_context_quality']:.2f}")


async def main():
    """Run the A/B test"""
    
    print("🧪 Starting Memory System A/B Test")
    print("This will compare stateless vs traditional memory performance\n")
    
    # Create temporary directories
    stateless_dir = tempfile.mkdtemp(prefix="stateless_memory_")
    traditional_dir = tempfile.mkdtemp(prefix="traditional_memory_")
    
    try:
        benchmark = MemoryBenchmark()
        
        # Run benchmarks
        print("Running benchmarks with 30 conversation turns each...\n")
        
        stateless_results = await benchmark.benchmark_stateless_memory(stateless_dir, 30)
        traditional_results = await benchmark.benchmark_traditional_memory(traditional_dir, 30)
        
        # Print results
        benchmark.print_results(stateless_results, traditional_results)
        
        # Save results
        results_file = Path("data/ab_test_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'stateless': stateless_results,
                'traditional': traditional_results,
                'timestamp': time.time()
            }, f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
    finally:
        # Cleanup
        shutil.rmtree(stateless_dir)
        shutil.rmtree(traditional_dir)


if __name__ == "__main__":
    # Check dependencies
    try:
        import lmdb
        import lz4
    except ImportError:
        print("❌ Missing dependencies. Please install:")
        print("   pip install lmdb lz4")
        sys.exit(1)
    
    print("🚀 Memory System A/B Testing Tool")
    print("This script compares stateless vs traditional memory systems\n")
    
    # Run the test
    asyncio.run(main())