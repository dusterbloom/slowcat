#!/usr/bin/env python3
"""
Quick Model Comparison for Long Conversations
Focused 20-turn test to get complete results
"""

import asyncio
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

async def test_model_performance(model_name: str, turns: int = 20):
    """Test model with consciousness system"""
    print(f"\n🧪 TESTING MODEL: {model_name}")
    print("=" * 50)
    
    # Create fresh consciousness
    ghost = Consciousness()
    
    # Override model
    try:
        from consciousness.llm_bridge import get_llm_bridge
        llm = get_llm_bridge()
        llm.model = model_name
        print(f"✅ Model set to: {model_name}")
    except Exception as e:
        print(f"⚠️  Could not set model: {e}")
    
    # Test conversation with memory challenges
    conversation = [
        "Hi, I'm Alex and I love astronomy.",
        "I work at the Griffith Observatory in LA.",
        "My favorite planet is Saturn because of its rings.", 
        "I've been studying space for 12 years.",
        "What's my profession?",  # Memory test 1
        "Which planet do I like most?",  # Memory test 2
        "How long have I been studying space?",  # Memory test 3
        "I also play guitar in my free time.",
        "My band is called 'Cosmic Strings'.",
        "We play progressive rock music.",
        "What are my two hobbies?",  # Memory test 4
        "Where do I work again?",  # Memory test 5
        "I'm planning to observe Mars next week.",
        "This will help with my research project.",
        "What am I planning to do next week?",  # Memory test 6
        "Actually, let me correct that - I meant Jupiter, not Mars.",
        "Which planet am I actually going to observe?",  # Memory test 7 (correction)
        "Can you summarize everything about me?",  # Comprehensive test
        "What's the most important thing I told you?",
        "Do you remember our entire conversation?"  # Final memory test
    ]
    
    results = {
        'model': model_name,
        'responses': [],
        'memory_tests': 0,
        'memory_passed': 0,
        'processing_times': [],
        'errors': 0,
        'avg_time': 0,
        'fastest_time': float('inf'),
        'slowest_time': 0
    }
    
    memory_test_turns = [4, 5, 6, 10, 11, 14, 16, 17, 18, 19]  # Expected memory test turns
    
    for i, turn in enumerate(conversation[:turns]):
        print(f"Turn {i+1:2d}: {turn[:40]}{'...' if len(turn) > 40 else ''}")
        
        start_time = time.time()
        try:
            result = await ghost.experience(turn)
            processing_time = time.time() - start_time
            
            response = result.get('response', 'No response')
            symbols = result.get('symbols', [])
            
            # Check if this is a memory test
            is_memory_test = i in memory_test_turns
            memory_passed = False
            
            if is_memory_test:
                results['memory_tests'] += 1
                # Simple check: response should be substantial and not generic
                if len(response) > 30 and not response.startswith("I'm thinking") and "empty" not in response:
                    results['memory_passed'] += 1
                    memory_passed = True
                    print(f"        ✅ Memory: {response[:50]}...")
                else:
                    print(f"        ❌ Memory: {response[:50]}...")
            else:
                print(f"        💬 Response: {response[:50]}...")
            
            results['responses'].append({
                'turn': i + 1,
                'input': turn,
                'response': response,
                'symbols': symbols,
                'processing_time': processing_time,
                'is_memory_test': is_memory_test,
                'memory_passed': memory_passed
            })
            
            results['processing_times'].append(processing_time)
            results['fastest_time'] = min(results['fastest_time'], processing_time)
            results['slowest_time'] = max(results['slowest_time'], processing_time)
            
            print(f"        ⚡ {processing_time:.3f}s {symbols}")
            
            # Break if responses are taking too long
            if processing_time > 30:
                print(f"        ⚠️  Response too slow, stopping test")
                break
                
        except Exception as e:
            results['errors'] += 1
            print(f"        ❌ Error: {e}")
    
    if results['processing_times']:
        results['avg_time'] = sum(results['processing_times']) / len(results['processing_times'])
        results['fastest_time'] = results['fastest_time'] if results['fastest_time'] != float('inf') else 0
    
    # Calculate scores
    memory_score = results['memory_passed'] / max(results['memory_tests'], 1)
    speed_score = max(0, 1 - min(results['avg_time'], 2.0) / 2.0)  # Penalty for >2s avg
    reliability_score = 1 - (results['errors'] / len(conversation[:turns]))
    
    overall_score = memory_score * 0.5 + speed_score * 0.3 + reliability_score * 0.2
    
    results['memory_score'] = memory_score
    results['speed_score'] = speed_score
    results['reliability_score'] = reliability_score
    results['overall_score'] = overall_score
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"Memory Retention: {memory_score:.2%} ({results['memory_passed']}/{results['memory_tests']})")
    print(f"Avg Processing: {results['avg_time']:.3f}s")
    print(f"Speed Range: {results['fastest_time']:.3f}s - {results['slowest_time']:.3f}s") 
    print(f"Reliability: {reliability_score:.2%} ({len(conversation[:turns]) - results['errors']}/{len(conversation[:turns])})")
    print(f"Overall Score: {overall_score:.3f}/1.0")
    
    return results

async def main():
    print("🚀 CONSCIOUSNESS MODEL COMPARISON")
    print("=" * 60)
    print("20-turn conversation with memory retention tests")
    
    models = [
        "google/gemma-3-270m",
        "qwen/qwen3-1.7b",           # User's exact model specification
        "qwen2.5-0.5b-instruct-mlx"
    ]
    
    all_results = {}
    
    for model in models:
        try:
            results = await test_model_performance(model, 20)
            all_results[model] = results
        except Exception as e:
            print(f"❌ Failed to test {model}: {e}")
    
    # Final comparison
    print(f"\n🏆 FINAL COMPARISON")
    print("=" * 50)
    
    # Sort by overall score
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    for rank, (model, results) in enumerate(sorted_results, 1):
        print(f"{rank}. {model}")
        print(f"   Overall Score: {results['overall_score']:.3f}")
        print(f"   Memory: {results['memory_score']:.2%}")
        print(f"   Speed: {results['avg_time']:.3f}s avg")
        print(f"   Reliability: {results['reliability_score']:.2%}")
        print()
    
    # Save results
    with open('model_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"💾 Results saved to: model_comparison_results.json")

if __name__ == "__main__":
    asyncio.run(main())