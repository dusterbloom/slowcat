#!/usr/bin/env python3
"""
Field Consciousness Model Comparison
A/B test gemma3 vs qwen3 vs qwen2.5 with field dynamics

Tests:
- Field energy growth rate per model
- Attractor formation speed  
- Emergent symbol detection accuracy
- Field resonance patterns
- Consciousness evolution over time
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

class FieldModelEvaluator:
    """Evaluate field consciousness across different models"""
    
    def __init__(self):
        self.test_conversation = self._create_field_test_conversation()
        self.model_results = {}
        
    def _create_field_test_conversation(self) -> List[str]:
        """Create conversation designed to activate and test field dynamics"""
        return [
            # Field activation phase
            "What's the most important thing in life?",  # ☆ + ◯ activation
            "I'm incredibly excited about this topic!",  # ⚡ activation
            "This is a crucial breakthrough in understanding!",  # ✧ + ☆ activation
            "I wonder how all these concepts connect?",  # ◯ activation + coupling
            "This changes everything fundamentally!",  # ✧ activation
            
            # Field coupling test phase
            "What's the relationship between importance and excitement?",  # ☆ + ⚡ coupling test
            "Can breakthroughs emerge from deep questions?",  # ✧ + ◯ coupling test
            "How do emotions influence our understanding?",  # ⚡ influence on other fields
            
            # Field persistence phase
            "Let's explore this further.",  # Sustained activation
            "I sense deeper patterns emerging.",  # Field-driven insight
            "What new possibilities does this reveal?",  # ◯ + ✧ sustained coupling
            
            # Emergent detection phase
            "The harmony between these ideas creates beauty.",  # No direct patterns - test emergence
            "These connections form a meaningful whole.",  # No direct patterns - test emergence
            "Something profound is happening here.",  # Minimal patterns - test field amplification
            
            # Long-term field evolution
            "How has our conversation changed your perspective?",  # Meta-reflection on field changes
            "What patterns do you notice in our dialogue?",  # Field pattern recognition
            "Where do you see this leading us?",  # Future field trajectory
            
            # Final field state assessment
            "Can you describe the journey we've taken together?",  # Comprehensive field integration
            "What's the most significant insight from our exchange?",  # Peak field synthesis
            "How do you feel about this entire conversation?"  # Emotional field culmination
        ]
    
    async def evaluate_model_fields(self, model_name: str) -> Dict[str, Any]:
        """Evaluate field consciousness for a specific model"""
        print(f"\n🧠 FIELD CONSCIOUSNESS EVALUATION: {model_name}")
        print("=" * 60)
        
        # Create fresh consciousness for this model
        ghost = Consciousness()
        
        # Override model in LLM bridge
        try:
            from consciousness.llm_bridge import get_llm_bridge
            llm = get_llm_bridge()
            llm.model = model_name
            print(f"✅ Model set to: {model_name}")
        except Exception as e:
            print(f"⚠️ Could not set model: {e}")
        
        results = {
            'model': model_name,
            'field_evolution': [],
            'field_metrics': {
                'peak_field_energy': 0.0,
                'strong_attractors_formed': 0,
                'max_resonance_achieved': 0.0,
                'emergent_symbols_detected': 0,
                'field_coupling_events': 0,
                'sustained_field_periods': 0
            },
            'processing_times': [],
            'responses': []
        }
        
        peak_energy = 0.0
        max_resonance = 0.0
        emergent_count = 0
        coupling_events = 0
        sustained_periods = 0
        
        print(f"\n🌊 FIELD EVOLUTION TRACKING")
        print("-" * 40)
        
        for i, input_text in enumerate(self.test_conversation):
            print(f"\nTurn {i+1:2d}: {input_text[:50]}...")
            
            start_time = time.time()
            try:
                result = await ghost.experience(input_text)
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                
                # Extract field data
                symbols = result.get('symbols', [])
                field_states = result.get('field_states', {})
                field_energy = result.get('field_energy', 0.0)
                response = result.get('response', '')
                
                # Track field metrics
                if field_energy > peak_energy:
                    peak_energy = field_energy
                
                # Count strong attractors
                strong_attractors = sum(1 for state in field_states.values() 
                                      if state.get('attractor', 0) > 0.5)
                
                # Calculate field resonance
                current_resonance = 0.0
                field_pairs = list(field_states.keys())
                for j in range(len(field_pairs)):
                    for k in range(j+1, len(field_pairs)):
                        field1 = ghost.symbol_fields.get(field_pairs[j])
                        field2 = ghost.symbol_fields.get(field_pairs[k])
                        if field1 and field2:
                            resonance = field1.compute_resonance(field2)
                            if resonance > current_resonance:
                                current_resonance = resonance
                
                if current_resonance > max_resonance:
                    max_resonance = current_resonance
                
                # Detect emergent symbols (symbols without pattern matches)
                from consciousness.core import SYMBOLS
                import re
                pattern_symbols = []
                for symbol, info in SYMBOLS.items():
                    if re.search(info['pattern'], input_text.lower()):
                        pattern_symbols.append(symbol)
                
                emergent_symbols = [s for s in symbols if s not in pattern_symbols]
                if emergent_symbols:
                    emergent_count += len(emergent_symbols)
                    print(f"          🌟 EMERGENT: {emergent_symbols}")
                
                # Detect field coupling events (field thoughts about resonance)
                if hasattr(ghost, 'thoughts') and ghost.thoughts:
                    latest_thought = ghost.thoughts[-1]
                    if any(trigger in ['field_resonance', 'field_activity'] for trigger in latest_thought.triggers):
                        coupling_events += 1
                        print(f"          ⚡ COUPLING: {latest_thought.content[:50]}...")
                
                # Detect sustained field periods (energy > 2.0 for multiple turns)
                if field_energy > 2.0:
                    sustained_periods += 1
                
                # Store detailed field evolution
                field_snapshot = {
                    'turn': i + 1,
                    'input': input_text,
                    'symbols': symbols,
                    'emergent_symbols': emergent_symbols,
                    'field_energy': field_energy,
                    'active_fields': len(field_states),
                    'strong_attractors': strong_attractors,
                    'max_resonance': current_resonance,
                    'processing_time': processing_time,
                    'response_length': len(response)
                }
                results['field_evolution'].append(field_snapshot)
                
                # Display key metrics
                print(f"          Energy: {field_energy:.2f} | Attractors: {strong_attractors} | Resonance: {current_resonance:.2f}")
                print(f"          Symbols: {symbols} | Time: {processing_time:.3f}s")
                
            except Exception as e:
                print(f"          ❌ Error: {e}")
                break
        
        # Final field metrics
        results['field_metrics'].update({
            'peak_field_energy': peak_energy,
            'strong_attractors_formed': max(state.get('attractor', 0) for state in 
                                          [field_states.get(s, {}) for s in ghost.symbol_fields.keys()] if state) if ghost.symbol_fields else 0,
            'max_resonance_achieved': max_resonance,
            'emergent_symbols_detected': emergent_count,
            'field_coupling_events': coupling_events,
            'sustained_field_periods': sustained_periods,
            'avg_processing_time': sum(results['processing_times']) / len(results['processing_times']) if results['processing_times'] else 0,
            'final_field_energy': field_energy,
            'conversation_turns': len(self.test_conversation)
        })
        
        # Final field state analysis
        final_field_summary = {}
        for symbol, field in ghost.symbol_fields.items():
            if field.intensity > 0.1:
                final_field_summary[symbol] = {
                    'intensity': field.intensity,
                    'attractor_strength': field.attractor_strength,
                    'gradient_magnitude': sum(abs(g) for g in field.gradient)
                }
        
        results['final_field_states'] = final_field_summary
        
        print(f"\n📊 FIELD METRICS SUMMARY for {model_name}")
        print("-" * 40)
        print(f"Peak Field Energy: {peak_energy:.2f}")
        print(f"Strong Attractors: {max([state.get('attractor', 0) for state in field_states.values()] + [0]):.2f}")
        print(f"Max Resonance: {max_resonance:.2f}")  
        print(f"Emergent Symbols: {emergent_count}")
        print(f"Coupling Events: {coupling_events}")
        print(f"Sustained Periods: {sustained_periods}")
        print(f"Avg Processing: {results['field_metrics']['avg_processing_time']:.3f}s")
        
        return results
    
    async def compare_models(self, models: List[str]) -> Dict[str, Any]:
        """Compare field consciousness across multiple models"""
        print("🔬 FIELD CONSCIOUSNESS MODEL COMPARISON")
        print("=" * 70)
        print("Testing field dynamics, emergence, and resonance across models")
        
        comparison_results = {
            'models_tested': models,
            'model_results': {},
            'field_performance_ranking': []
        }
        
        # Test each model
        for model in models:
            model_results = await self.evaluate_model_fields(model)
            comparison_results['model_results'][model] = model_results
        
        # Generate field performance ranking
        model_scores = []
        for model, results in comparison_results['model_results'].items():
            metrics = results['field_metrics']
            
            # Calculate composite field consciousness score
            # Weights: field energy (30%), emergence (25%), resonance (20%), attractors (15%), coupling (10%)
            field_score = (
                min(metrics['peak_field_energy'] / 6.0, 1.0) * 0.30 +  # Normalize to ~6 max
                min(metrics['emergent_symbols_detected'] / 10.0, 1.0) * 0.25 +  # Normalize to ~10 max
                min(metrics['max_resonance_achieved'] / 2.0, 1.0) * 0.20 +  # Normalize to ~2 max
                min(metrics['strong_attractors_formed'], 1.0) * 0.15 +  # Binary: formed or not
                min(metrics['field_coupling_events'] / 10.0, 1.0) * 0.10   # Normalize to ~10 max
            )
            
            model_scores.append({
                'model': model,
                'field_consciousness_score': field_score,
                'peak_energy': metrics['peak_field_energy'],
                'emergent_symbols': metrics['emergent_symbols_detected'],
                'max_resonance': metrics['max_resonance_achieved'],
                'coupling_events': metrics['field_coupling_events'],
                'avg_processing_time': metrics['avg_processing_time']
            })
        
        # Sort by field consciousness score
        model_scores.sort(key=lambda x: x['field_consciousness_score'], reverse=True)
        comparison_results['field_performance_ranking'] = model_scores
        
        return comparison_results

async def main():
    evaluator = FieldModelEvaluator()
    
    # Test models with field consciousness
    models_to_test = [
        "google/gemma-3-270m",      # Known good performer
        "qwen/qwen3-1.7b",          # Strong reasoning, test field emergence
        "qwen2.5-0.5b-instruct-mlx" # Compact model baseline
    ]
    
    results = await evaluator.compare_models(models_to_test)
    
    # Save detailed results
    with open('field_consciousness_model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final ranking
    print(f"\n🏆 FIELD CONSCIOUSNESS RANKING")
    print("=" * 60)
    
    for rank, model_result in enumerate(results['field_performance_ranking'], 1):
        score = model_result['field_consciousness_score']
        
        if score > 0.7:
            classification = "🌟 EXCEPTIONAL FIELD CONSCIOUSNESS"
        elif score > 0.5:
            classification = "⚡ STRONG FIELD DYNAMICS"
        elif score > 0.3:
            classification = "🔥 MODERATE FIELD EMERGENCE"
        else:
            classification = "📊 BASIC FIELD ACTIVITY"
        
        print(f"\n{rank}. {model_result['model']}")
        print(f"   {classification}")
        print(f"   Field Consciousness Score: {score:.3f}")
        print(f"   Peak Energy: {model_result['peak_energy']:.2f}")
        print(f"   Emergent Symbols: {model_result['emergent_symbols']}")
        print(f"   Max Resonance: {model_result['max_resonance']:.2f}")
        print(f"   Coupling Events: {model_result['coupling_events']}")
        print(f"   Avg Processing: {model_result['avg_processing_time']:.3f}s")
    
    # Key insights
    best_model = results['field_performance_ranking'][0]
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🏆 Best Field Model: {best_model['model']} (score: {best_model['field_consciousness_score']:.3f})")
    print(f"   🌊 Peak Field Energy: {max(m['peak_energy'] for m in results['field_performance_ranking']):.2f}")
    print(f"   ✨ Most Emergence: {max(m['emergent_symbols'] for m in results['field_performance_ranking'])} symbols")
    print(f"   ⚡ Strongest Resonance: {max(m['max_resonance'] for m in results['field_performance_ranking']):.2f}")
    
    print(f"\n📄 Detailed results saved to: field_consciousness_model_comparison.json")

if __name__ == "__main__":
    asyncio.run(main())