"""
Visualization and analysis tools for Dynamic Tape Head

This helps understand how DTH scores and selects memories.
"""

import json
import time
import numpy as np
import asyncio
from pathlib import Path
from typing import List, Dict
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))

from memory.dynamic_tape_head import DynamicTapeHead, MemorySpan


def visualize_scoring(memories: List[MemorySpan]):
    """
    Create ASCII visualization of memory scoring
    """
    
    print("\n" + "="*80)
    print("MEMORY SCORING VISUALIZATION")
    print("="*80)
    
    for i, memory in enumerate(memories[:10]):  # Top 10
        print(f"\n[{i+1}] {memory.content[:50]}...")
        print(f"    Age: {(time.time() - memory.ts)/3600:.1f} hours")
        
        # Score bar chart
        components = memory.score_components
        total = memory.score
        
        print(f"    Total Score: {total:.3f}")
        print("    Components:")
        
        # Recency
        r_bar = "█" * int(components['R'] * 20)
        print(f"      R (recency):  {components['R']:.3f} |{r_bar}")
        
        # Semantic
        s_bar = "█" * int(components['S'] * 20)
        print(f"      S (semantic): {components['S']:.3f} |{s_bar}")
        
        # Entity
        e_bar = "█" * int(components['E'] * 20)
        print(f"      E (entity):   {components['E']:.3f} |{e_bar}")
        
        # Novelty (penalty)
        d_bar = "█" * int(components['D'] * 20)
        print(f"      D (novelty):  {components['D']:.3f} |{d_bar} (penalty)")


def analyze_policy_impact(dth: DynamicTapeHead):
    """
    Show how different policy weights affect selection
    """
    
    print("\n" + "="*80)
    print("POLICY WEIGHT ANALYSIS")
    print("="*80)
    
    weights = dth.policy['weights']
    
    print("\nCurrent Weights:")
    for key, value in weights.items():
        bar = "█" * int(value * 40)
        print(f"  {key:12} = {value:.2f} |{bar}")
    
    print("\nFormula: Score = ", end="")
    print(f"{weights['w_recency']:.2f}×R + ", end="")
    print(f"{weights['w_semantic']:.2f}×S + ", end="")
    print(f"{weights['w_entity']:.2f}×E - ", end="")
    print(f"{weights['w_novelty']:.2f}×D")
    
    print("\nInterpretation:")
    
    # Find dominant factor
    max_weight = max(weights.items(), key=lambda x: x[1] if 'novelty' not in x[0] else 0)
    print(f"  🎯 Primary factor: {max_weight[0].replace('w_', '').title()} ({max_weight[1]:.2f})")
    
    if weights['w_recency'] > 0.5:
        print("  ⏰ Strong recency bias - prefers recent memories")
    elif weights['w_semantic'] > 0.5:
        print("  🧠 Semantic focus - prefers topically relevant memories")
    elif weights['w_entity'] > 0.3:
        print("  🏷️ Entity matching - tracks named entities across conversation")
    
    if weights['w_novelty'] > 0.15:
        print("  🔄 High novelty penalty - avoids repetition")
    elif weights['w_novelty'] < 0.05:
        print("  ♻️ Low novelty penalty - may repeat information")


async def tune_weights_experiment(dth: DynamicTapeHead, test_queries: List[str]):
    """
    Experiment with different weight configurations
    """
    
    print("\n" + "="*80)
    print("WEIGHT TUNING EXPERIMENT")
    print("="*80)
    
    # Different weight configurations to test
    configs = [
        {"name": "Balanced", "weights": {"w_recency": 0.40, "w_semantic": 0.35, "w_entity": 0.15, "w_novelty": 0.10}},
        {"name": "Recency-focused", "weights": {"w_recency": 0.70, "w_semantic": 0.20, "w_entity": 0.05, "w_novelty": 0.05}},
        {"name": "Semantic-focused", "weights": {"w_recency": 0.20, "w_semantic": 0.60, "w_entity": 0.10, "w_novelty": 0.10}},
        {"name": "Entity-tracking", "weights": {"w_recency": 0.30, "w_semantic": 0.30, "w_entity": 0.30, "w_novelty": 0.10}},
        {"name": "Anti-repetition", "weights": {"w_recency": 0.35, "w_semantic": 0.30, "w_entity": 0.10, "w_novelty": 0.25}},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        
        # Update weights
        original_weights = dth.policy['weights'].copy()
        dth.policy['weights'] = config['weights']
        
        config_results = {
            "name": config['name'],
            "avg_score": 0,
            "avg_memories": 0,
            "avg_tokens": 0
        }
        
        for query in test_queries:
            bundle = await dth.seek(query, budget=1000)
            
            if bundle.verbatim:
                avg_score = np.mean([m.score for m in bundle.verbatim])
                config_results["avg_score"] += avg_score
            
            config_results["avg_memories"] += len(bundle.verbatim) + len(bundle.shadows)
            config_results["avg_tokens"] += bundle.token_count
        
        # Average across queries
        n_queries = len(test_queries)
        config_results["avg_score"] /= n_queries
        config_results["avg_memories"] /= n_queries
        config_results["avg_tokens"] /= n_queries
        
        results.append(config_results)
        
        print(f"  Avg score: {config_results['avg_score']:.3f}")
        print(f"  Avg memories: {config_results['avg_memories']:.1f}")
        print(f"  Avg tokens: {config_results['avg_tokens']:.0f}")
        
        # Restore original weights
        dth.policy['weights'] = original_weights
    
    # Find best configuration
    print("\n🏆 RESULTS:")
    best_score = max(results, key=lambda x: x['avg_score'])
    print(f"  Highest avg score: {best_score['name']} ({best_score['avg_score']:.3f})")
    
    most_memories = max(results, key=lambda x: x['avg_memories'])
    print(f"  Most memories selected: {most_memories['name']} ({most_memories['avg_memories']:.1f})")
    
    return results


async def main():
    """Run visualization and analysis"""
    
    import time
    from memory import create_smart_memory_system
    
    # Create memory system and DTH
    memory_system = create_smart_memory_system("data/facts.db")
    dth = DynamicTapeHead(memory_system)
    
    print("\n🧠 DYNAMIC TAPE HEAD ANALYSIS TOOL\n")
    
    # Add some test memories
    if hasattr(memory_system, 'tape_store'):
        tape = memory_system.tape_store
        
        # Add varied test data
        test_memories = [
            ("What's the weather in Sardinia?", time.time() - 3600),
            ("Tell me about Python programming", time.time() - 7200),
            ("I'm working on SlowCat, an AI assistant", time.time() - 1800),
            ("Can you help with consciousness architecture?", time.time() - 900),
            ("The Dynamic Tape Head uses a scoring formula", time.time() - 300),
        ]
        
        for content, ts in test_memories:
            tape.add_entry("user", content, "test_user", ts)
    
    # Test queries
    test_queries = [
        "Tell me about SlowCat",
        "What's the weather like?",
        "Help with Python code",
        "Consciousness and AI",
    ]
    
    # 1. Analyze current policy
    analyze_policy_impact(dth)
    
    # 2. Run a test query and visualize scoring
    print("\n📝 Test Query: 'Tell me about consciousness and SlowCat'")
    bundle = await dth.seek("Tell me about consciousness and SlowCat", budget=1000)
    
    if bundle.verbatim:
        visualize_scoring(bundle.verbatim)
    
    # 3. Weight tuning experiment
    results = await tune_weights_experiment(dth, test_queries)
    
    # 4. Save results
    output_path = "logs/dth_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "policy": dth.policy,
            "tuning_results": results,
            "metrics": dth.get_metrics_summary()
        }, f, indent=2)
    
    print(f"\n📊 Analysis saved to {output_path}")
    
    # 5. Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("  1. Current policy seems balanced for general use")
    print("  2. Increase w_semantic if you want more topical relevance")
    print("  3. Increase w_recency for more conversational flow")
    print("  4. Increase w_entity for better fact tracking")
    print("  5. Adjust w_novelty based on repetition tolerance")


if __name__ == "__main__":
    asyncio.run(main())
