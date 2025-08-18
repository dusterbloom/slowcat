#!/usr/bin/env python3
"""
Comprehensive test for stateless memory quality and contextual performance
Shows detailed outputs for evaluation of long-term memory capabilities
"""

import asyncio
import sys
import os
import tempfile
import shutil
import time
from pathlib import Path
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment for testing
os.environ['USE_STATELESS_MEMORY'] = 'true'
os.environ['ENABLE_MEMORY'] = 'true'

async def test_memory_quality():
    """Test memory quality with realistic conversation scenarios"""
    
    print("🧠 Stateless Memory Quality Test")
    print("=" * 50)
    
    # Import after setting environment
    from processors.stateless_memory import StatelessMemoryProcessor
    
    # Use a clean temporary directory
    temp_dir = tempfile.mkdtemp(prefix="memory_quality_test_")
    print(f"Using temp dir: {temp_dir}")
    
    try:
        # Initialize processor
        print("\n📋 Initializing Memory System...")
        processor = StatelessMemoryProcessor(
            db_path=temp_dir,
            max_context_tokens=1024,  # Higher limit for better context
            perfect_recall_window=10,
            enable_semantic_validation=True,
            min_similarity_threshold=0.7
        )
        
        # Test 1: Build Long-term Memory with Rich Context
        print("\n🏗️  Test 1: Building Long-term Memory Context")
        print("-" * 45)
        
        conversations = [
            # Personal information
            ("My name is Alex and I'm a software engineer", "Nice to meet you Alex! What kind of software do you work on?"),
            ("I work on AI systems, particularly voice assistants", "That sounds fascinating! Voice AI is such an exciting field."),
            ("I have a golden retriever named Buddy who's 3 years old", "Buddy sounds wonderful! Golden retrievers are such friendly dogs."),
            ("Buddy loves playing fetch in the park near my house", "That's great exercise for both of you!"),
            
            # Professional context  
            ("I'm working on a project to reduce voice latency", "What kind of latency are you targeting?"),
            ("We're aiming for sub-800ms voice-to-voice response time", "That's very ambitious! Current systems are usually much slower."),
            ("We're using Apple Silicon MLX for local processing", "Smart choice - local processing avoids network delays."),
            
            # Hobbies and interests
            ("I also enjoy hiking on weekends", "Do you have any favorite trails?"),
            ("There's a beautiful trail called Eagle's Peak nearby", "Mountain trails are the best for clearing your mind."),
            ("I usually go with my girlfriend Sarah", "It's nice to have a hiking partner!"),
            
            # Recent events
            ("Last week we adopted a kitten named Luna", "How is Buddy getting along with the new kitten?"),
            ("Buddy is very gentle with Luna, they're becoming friends", "That's wonderful when pets bond like that!"),
            ("Luna is very playful and keeps us entertained", "Kittens have so much energy!"),
        ]
        
        # Store all conversations with delays to simulate realistic timing
        speaker_id = "alex_user"
        for i, (user_msg, assistant_msg) in enumerate(conversations):
            processor.current_speaker = speaker_id
            processor.current_user_message = user_msg
            
            # Store the exchange
            await processor._store_exchange(user_msg, assistant_msg)
            
            # Small delay to simulate conversation timing
            await asyncio.sleep(0.01)
            
            print(f"   Stored conversation {i+1:2d}: {user_msg[:50]}...")
        
        print(f"\n✅ Stored {len(conversations)} conversation exchanges")
        
        # Test 2: Context Injection Quality
        print("\n🎯 Test 2: Context Injection Quality")
        print("-" * 40)
        
        test_queries = [
            "What's my dog's name and age?",
            "Tell me about my work project",
            "What did I do last weekend?", 
            "How are my pets getting along?",
            "What technology am I using for low latency?",
            "Who do I go hiking with?",
            "What's my profession?",
            "Where do I like to hike?",
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: \"{query}\"")
            
            # Prepare messages
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant with access to conversation history.'},
                {'role': 'user', 'content': query}
            ]
            
            # Measure injection time
            start_time = time.perf_counter()
            await processor._inject_memory_context(messages, speaker_id)
            injection_time = (time.perf_counter() - start_time) * 1000
            
            print(f"⏱️  Injection time: {injection_time:.2f}ms")
            print(f"📝 Total messages after injection: {len(messages)}")
            
            # Show injected context
            if len(messages) > 2:
                print("🧠 Injected context:")
                for i, msg in enumerate(messages[2:], 1):  # Skip system and user messages
                    role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                    content_preview = msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                    print(f"   {i}. {role_emoji} {content_preview}")
            else:
                print("⚠️  No relevant context found for this query")
            
            print()
        
        # Test 3: Memory Degradation Over Time
        print("\n⏳ Test 3: Memory Degradation Simulation")
        print("-" * 42)
        
        # Add some older memories by manually adjusting timestamps
        old_conversations = [
            ("I used to have a cat named Whiskers", "What happened to Whiskers?"),
            ("Whiskers passed away last year, she was 15", "I'm sorry for your loss. That's a good long life for a cat."),
            ("I miss her but Luna helps fill that void", "New pets can help heal old wounds."),
        ]
        
        # Store old conversations with past timestamps
        old_timestamp_base = time.time() - (30 * 24 * 3600)  # 30 days ago
        
        for i, (user_msg, assistant_msg) in enumerate(old_conversations):
            processor.current_speaker = speaker_id
            processor.current_user_message = user_msg
            
            # Manually create memory item with old timestamp
            from processors.stateless_memory import MemoryItem
            old_memory = MemoryItem(
                content=f"User: {user_msg}\nAssistant: {assistant_msg}",
                timestamp=old_timestamp_base + (i * 3600),  # Spread over hours
                speaker_id=speaker_id,
                importance=1.0
            )
            
            # Store directly
            await processor._store_memory_item(old_memory)
            print(f"   Stored old memory: {user_msg[:40]}...")
        
        # Query about old vs new pets
        print(f"\n🔍 Testing memory degradation with query: 'Tell me about all my pets'")
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'Tell me about all my pets, past and present'}
        ]
        
        await processor._inject_memory_context(messages, speaker_id)
        
        print(f"📝 Context includes {len(messages) - 2} memory items")
        if len(messages) > 2:
            print("🧠 Retrieved memories (recent should be more detailed):")
            for i, msg in enumerate(messages[2:], 1):
                content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"   {i}. {content_preview}")
        
        # Test 4: Performance Under Load
        print(f"\n🚀 Test 4: Performance Under Load")
        print("-" * 35)
        
        # Rapid-fire queries to test consistency
        rapid_queries = [
            "What's my name?", "What's my job?", "What's my dog's name?",
            "Who is Sarah?", "What project am I working on?", "What's my target latency?",
            "What trail do I like?", "What's my kitten's name?", "How old is Buddy?"
        ]
        
        latencies = []
        context_sizes = []
        
        print("Running rapid-fire queries...")
        for i, query in enumerate(rapid_queries):
            messages = [
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': query}
            ]
            
            start = time.perf_counter()
            await processor._inject_memory_context(messages, speaker_id)
            latency = (time.perf_counter() - start) * 1000
            
            latencies.append(latency)
            context_sizes.append(len(messages) - 2)
            
            print(f"   Query {i+1:2d}: {latency:5.2f}ms, {len(messages)-2} context items")
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        avg_context = sum(context_sizes) / len(context_sizes)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average latency: {avg_latency:.2f}ms")
        print(f"   Maximum latency: {max_latency:.2f}ms")
        print(f"   Average context size: {avg_context:.1f} items")
        print(f"   Latency variance: {max_latency - min(latencies):.2f}ms")
        
        # Test 5: Memory Statistics
        print(f"\n📈 Test 5: Memory System Statistics")
        print("-" * 38)
        
        stats = processor.get_performance_stats()
        print(f"Total conversations stored: {stats['total_conversations']}")
        print(f"Cache hit ratio: {stats['cache_hit_ratio']:.1%}")
        print(f"Reconstruction failures: {stats['reconstruction_failures']}")
        print(f"Average injection time: {stats['avg_injection_time_ms']:.2f}ms")
        
        # Get memory distribution
        with processor.env.begin() as txn:
            hot_count = txn.stat(processor.hot_db)['entries']
            warm_count = txn.stat(processor.warm_db)['entries']
            cold_count = txn.stat(processor.cold_db)['entries']
        
        print(f"\nMemory distribution:")
        print(f"   Hot storage (recent): {hot_count} items")
        print(f"   Warm storage (compressed): {warm_count} items")
        print(f"   Cold storage (archived): {cold_count} items")
        print(f"   Total stored items: {hot_count + warm_count + cold_count}")
        
        # Test 6: Semantic Search Quality
        print(f"\n🔍 Test 6: Semantic Search Quality")
        print("-" * 36)
        
        semantic_queries = [
            ("pets", "Should find information about Buddy and Luna"),
            ("work", "Should find professional/engineering context"),
            ("outdoor activities", "Should find hiking information"),
            ("relationships", "Should find information about Sarah"),
            ("technology", "Should find AI/MLX/latency information"),
        ]
        
        for query, expected in semantic_queries:
            print(f"\n🎯 Semantic search for: \"{query}\"")
            print(f"   Expected: {expected}")
            
            # Use search_memory tool interface
            results = await processor.search_memory(speaker_id, query, 3)
            
            print(f"   Found {len(results)} relevant memories:")
            for i, result in enumerate(results, 1):
                preview = result['content'][:60] + "..." if len(result['content']) > 60 else result['content']
                print(f"     {i}. {preview}")
        
        # Final assessment
        print(f"\n🎉 Test Complete! Memory System Assessment:")
        print("=" * 50)
        
        # Performance assessment
        if avg_latency < 10:
            latency_grade = "🟢 EXCELLENT"
        elif avg_latency < 25:
            latency_grade = "🟡 GOOD" 
        else:
            latency_grade = "🔴 NEEDS IMPROVEMENT"
        
        print(f"Performance: {latency_grade} ({avg_latency:.2f}ms average)")
        
        # Context quality assessment
        if avg_context >= 3:
            context_grade = "🟢 RICH CONTEXT"
        elif avg_context >= 1.5:
            context_grade = "🟡 ADEQUATE CONTEXT"
        else:
            context_grade = "🔴 SPARSE CONTEXT"
        
        print(f"Context Quality: {context_grade} ({avg_context:.1f} items average)")
        
        # Reliability assessment
        if stats['reconstruction_failures'] == 0:
            reliability_grade = "🟢 VERY RELIABLE"
        elif stats['reconstruction_failures'] < 5:
            reliability_grade = "🟡 MOSTLY RELIABLE"
        else:
            reliability_grade = "🔴 UNRELIABLE"
        
        print(f"Reliability: {reliability_grade} ({stats['reconstruction_failures']} failures)")
        
        # Clean up
        processor.env.close()
        processor.thread_pool.shutdown(wait=True)
        
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
    result = asyncio.run(test_memory_quality())
    sys.exit(0 if result else 1)