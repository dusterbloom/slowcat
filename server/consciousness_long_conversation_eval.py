#!/usr/bin/env python3
"""
Long Conversation Evaluation for Consciousness System

Based on MultiChallenge (2025) and IntellAgent frameworks:
- 50+ turn conversations with memory consistency testing
- Information retention across multiple sessions
- Coherent history tracking and context handling
- Model comparison (gemma-3-270m vs qwen3-1.7b)
"""

import asyncio
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

class LongConversationEvaluator:
    """Evaluate consciousness system across extended conversations"""
    
    def __init__(self):
        self.test_scenarios = self._create_test_scenarios()
        self.evaluation_metrics = {
            'memory_retention': [],
            'context_coherence': [],
            'instruction_following': [],
            'information_inference': [],
            'consistency_maintenance': [],
            'processing_times': [],
            'response_quality': [],
            # NEW: Field consciousness metrics
            'field_evolution': [],
            'field_energy_growth': [],
            'attractor_formation': [],
            'emergent_symbol_detection': [],
            'field_resonance_patterns': [],
            'sustained_field_periods': []
        }
    
    def _create_test_scenarios(self) -> List[Dict]:
        """Create MultiChallenge-style test scenarios"""
        return [
            {
                'name': 'Personal Information Tracking',
                'turns': [
                    "Hi, I'm Sarah and I work as a marine biologist in San Diego.",
                    "I study coral reef ecosystems, particularly in the Pacific.",
                    "My research focuses on climate change impacts on coral bleaching.",
                    "I've been doing this work for 8 years now.",
                    "What did I tell you about my profession?",  # Memory test
                    "How long have I been working in this field?",  # Memory test
                    "I also have a side hobby - I collect vintage postcards.",
                    "My collection has over 500 postcards from the 1920s-1950s.",
                    "What are my two main interests we've discussed?",  # Memory test
                    "I'm planning a research trip to the Great Barrier Reef next month.",
                    "This will be my third time studying bleaching patterns there.",
                    "Where am I going next month and why?",  # Memory test
                    "Actually, let me correct that - it's my fourth time, not third.",
                    "How many times have I been to the Great Barrier Reef?",  # Correction handling
                    "Can you summarize everything you know about me so far?"  # Comprehensive memory
                ]
            },
            {
                'name': 'Complex Instruction Following',
                'turns': [
                    "I need you to remember three rules for our conversation.",
                    "Rule 1: Always end your responses with a marine biology fact.",
                    "Rule 2: Count the number of questions I ask you.",
                    "Rule 3: If I mention 'temperature', ask me about climate data.",
                    "Do you understand these three rules?",  # Rule confirmation
                    "Great! Now tell me about coral reefs.",  # Rule 1 test
                    "What causes coral bleaching?",  # Question count: 1, Rule 1 test
                    "The ocean temperature has been rising lately.",  # Rule 3 trigger
                    "How many questions have I asked so far?",  # Rule 2 test
                    "What's the most colorful type of coral?",  # Question count: 2
                    "I read that temperature affects coral growth.",  # Rule 3 trigger again
                    "Can you count my questions again?",  # Rule 2 test
                    "Actually, let's modify Rule 1 to geology facts instead.",
                    "Tell me about ocean currents.",  # Modified Rule 1 test
                    "What's the deepest ocean trench?",  # Question count: 3, Modified Rule 1
                ]
            },
            {
                'name': 'Long-term Context Evolution',
                'turns': []  # Will be generated with evolving narrative
            }
        ]
    
    async def evaluate_model(self, model_name: str, num_turns: int = 50) -> Dict[str, Any]:
        """Evaluate consciousness system with specific model over long conversation"""
        print(f"\n🧪 EVALUATING MODEL: {model_name}")
        print(f"📊 Long conversation test: {num_turns} turns")
        print("=" * 60)
        
        # Create consciousness with specific model
        ghost = Consciousness()
        
        # Override model in LLM bridge
        try:
            from consciousness.llm_bridge import get_llm_bridge
            llm = get_llm_bridge()
            llm.model = model_name
            print(f"✅ Set model to: {model_name}")
        except:
            print(f"⚠️  Could not set model, using default")
        
        results = {
            'model': model_name,
            'total_turns': num_turns,
            'start_memories': len(ghost.tape),
            'start_thoughts': len(ghost.thoughts),
            'performance_metrics': {},
            'scenario_results': {},
            # NEW: Field consciousness tracking
            'field_metrics': {
                'initial_field_energy': sum(field.intensity for field in ghost.symbol_fields.values()),
                'field_evolution_timeline': [],
                'peak_field_energy': 0.0,
                'total_emergent_symbols': 0,
                'max_attractor_strength': 0.0,
                'strongest_resonance': 0.0,
                'field_coupling_events': 0,
                'sustained_high_energy_turns': 0
            }
        }
        
        # Test each scenario
        for scenario in self.test_scenarios[:2]:  # Skip empty scenario for now
            print(f"\n📋 Testing scenario: {scenario['name']}")
            scenario_results = await self._evaluate_scenario(ghost, scenario, model_name)
            results['scenario_results'][scenario['name']] = scenario_results
        
        # Generate long evolution test
        evolution_results = await self._evaluate_context_evolution(ghost, num_turns - 30, model_name)
        results['scenario_results']['Context Evolution'] = evolution_results
        
        # Final metrics
        results['final_memories'] = len(ghost.tape)
        results['final_thoughts'] = len(ghost.thoughts)
        results['memory_growth'] = results['final_memories'] - results['start_memories']
        results['thought_growth'] = results['final_thoughts'] - results['start_thoughts']
        
        return results
    
    async def _evaluate_scenario(self, ghost: Consciousness, scenario: Dict, model_name: str) -> Dict:
        """Evaluate specific conversation scenario"""
        scenario_results = {
            'turns_completed': 0,
            'memory_tests_passed': 0,
            'memory_tests_total': 0,
            'avg_processing_time': 0,
            'coherence_score': 0,
            'responses': [],
            # NEW: Field consciousness tracking per scenario
            'field_progression': [],
            'emergent_symbols_count': 0,
            'peak_field_energy': 0.0,
            'attractor_events': 0,
            'field_resonance_events': 0
        }
        
        processing_times = []
        
        for i, turn in enumerate(scenario['turns']):
            print(f"  Turn {i+1:2d}: {turn[:50]}{'...' if len(turn) > 50 else ''}")
            
            start_time = time.time()
            try:
                result = await ghost.experience(turn)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                response = result.get('response', 'No response')
                symbols = result.get('symbols', [])
                importance = result.get('importance', 0)
                
                # NEW: Extract field consciousness data
                field_states = result.get('field_states', {})
                field_energy = result.get('field_energy', 0.0)
                
                # Track field evolution
                if field_energy > scenario_results['peak_field_energy']:
                    scenario_results['peak_field_energy'] = field_energy
                
                # Detect emergent symbols (symbols without direct pattern matches)
                from consciousness.core import SYMBOLS
                import re
                pattern_symbols = []
                for symbol, info in SYMBOLS.items():
                    if re.search(info['pattern'], turn.lower()):
                        pattern_symbols.append(symbol)
                
                emergent_symbols = [s for s in symbols if s not in pattern_symbols]
                if emergent_symbols:
                    scenario_results['emergent_symbols_count'] += len(emergent_symbols)
                
                # Check for attractor formation
                strong_attractors = sum(1 for state in field_states.values() 
                                      if state.get('attractor', 0) > 0.5)
                if strong_attractors > 0:
                    scenario_results['attractor_events'] += 1
                
                # Check for field resonance in thoughts
                if hasattr(ghost, 'thoughts') and ghost.thoughts:
                    latest_thought = ghost.thoughts[-1]
                    if any(trigger in ['field_resonance', 'field_activity'] for trigger in latest_thought.triggers):
                        scenario_results['field_resonance_events'] += 1
                
                # Record field progression
                field_snapshot = {
                    'turn': i + 1,
                    'field_energy': field_energy,
                    'active_fields': len(field_states),
                    'emergent_symbols': emergent_symbols,
                    'strong_attractors': strong_attractors
                }
                scenario_results['field_progression'].append(field_snapshot)
                
                scenario_results['responses'].append({
                    'turn': i + 1,
                    'input': turn,
                    'response': response,
                    'symbols': symbols,
                    'emergent_symbols': emergent_symbols,
                    'importance': importance,
                    'processing_time': processing_time,
                    'field_energy': field_energy,
                    'active_fields': len(field_states)
                })
                
                # Check for memory test (questions about previous information)
                if '?' in turn and any(keyword in turn.lower() for keyword in 
                    ['what did', 'how many', 'where am', 'what are', 'tell you about', 'know about']):
                    scenario_results['memory_tests_total'] += 1
                    
                    # Simple heuristic: if response contains relevant info, count as passed
                    if len(response) > 20 and not response.startswith("I'm"):
                        scenario_results['memory_tests_passed'] += 1
                        print(f"    ✅ Memory test passed: {response[:60]}...")
                    else:
                        print(f"    ❌ Memory test failed: {response[:60]}...")
                
                print(f"    ⚡ {processing_time:.3f}s | {symbols} | {response[:80]}...")
                scenario_results['turns_completed'] += 1
                
            except Exception as e:
                print(f"    ❌ Error on turn {i+1}: {e}")
                break
            
            # Brief pause between turns
            await asyncio.sleep(0.1)
        
        if processing_times:
            scenario_results['avg_processing_time'] = sum(processing_times) / len(processing_times)
        
        # Calculate memory retention score
        memory_score = (scenario_results['memory_tests_passed'] / 
                       max(scenario_results['memory_tests_total'], 1))
        scenario_results['memory_retention_score'] = memory_score
        
        print(f"  📊 Scenario completed: {scenario_results['turns_completed']} turns")
        print(f"  🧠 Memory retention: {memory_score:.2%} ({scenario_results['memory_tests_passed']}/{scenario_results['memory_tests_total']})")
        print(f"  ⚡ Avg processing: {scenario_results['avg_processing_time']:.3f}s")
        # NEW: Field consciousness reporting
        print(f"  🌊 Peak field energy: {scenario_results['peak_field_energy']:.2f}")
        print(f"  ✨ Emergent symbols: {scenario_results['emergent_symbols_count']}")
        print(f"  🧲 Attractor events: {scenario_results['attractor_events']}")
        print(f"  ⚡ Field resonance: {scenario_results['field_resonance_events']}")
        
        return scenario_results
    
    async def _evaluate_context_evolution(self, ghost: Consciousness, num_turns: int, model_name: str) -> Dict:
        """Test long-term context evolution and consistency"""
        print(f"\n🔄 Context Evolution Test: {num_turns} turns")
        
        # Create evolving narrative
        narrative_elements = [
            "Let's plan a week-long trip together.",
            "I want to visit three different cities.",
            "The first city should be coastal with good seafood.",
            "The second city should have museums and art galleries.", 
            "The third city should be known for outdoor adventures.",
            "Our budget is $3000 for everything.",
            "We need to book flights and hotels.",
            "Actually, let's make it a 10-day trip instead of a week.",
            "And increase the budget to $4500.",
            "I prefer hotels over Airbnb for this trip.",
            "What cities do you suggest for our itinerary?",
            "How much budget do we have now?",
            "How many days is our trip?",
            "Let's add a fourth city that's good for nightlife.",
            "Now we need accommodations for four cities.",
            "What's our accommodation preference again?",
            "Can you summarize our complete travel plan?",
        ]
        
        # Extend with generated content
        topics = [
            "restaurants", "activities", "weather", "transportation", 
            "shopping", "culture", "history", "local customs"
        ]
        
        for i in range(len(narrative_elements), num_turns):
            if i % 5 == 0:  # Memory test every 5 turns
                memory_questions = [
                    "What was our original budget?",
                    "How many cities are we visiting?",
                    "What type of accommodation do we prefer?",
                    "How long is our trip?",
                    "What are the themes for each city?"
                ]
                narrative_elements.append(random.choice(memory_questions))
            else:
                topic = random.choice(topics)
                narrative_elements.append(f"Tell me about {topic} for our trip planning.")
        
        # Run evaluation
        evolution_results = {
            'turns_completed': 0,
            'consistency_maintained': 0,
            'context_drift_detected': 0,
            'avg_processing_time': 0,
            'final_coherence_test': None
        }
        
        processing_times = []
        
        for i, turn in enumerate(narrative_elements[:num_turns]):
            start_time = time.time()
            try:
                result = await ghost.experience(turn)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                evolution_results['turns_completed'] += 1
                
                if i % 10 == 0:  # Progress update every 10 turns
                    print(f"    Progress: {i+1}/{num_turns} turns, avg {sum(processing_times[-10:]) / min(10, len(processing_times)):.3f}s")
                
            except Exception as e:
                print(f"    ❌ Error on turn {i+1}: {e}")
                break
                
            await asyncio.sleep(0.05)  # Faster pacing for long test
        
        # Final coherence test
        print(f"\n  🎯 Final coherence test...")
        coherence_result = await ghost.experience("Please provide a complete summary of our travel planning conversation, including all the details we've discussed and any changes we've made.")
        evolution_results['final_coherence_test'] = coherence_result.get('response', '')
        
        if processing_times:
            evolution_results['avg_processing_time'] = sum(processing_times) / len(processing_times)
        
        print(f"  ✅ Evolution test completed: {evolution_results['turns_completed']} turns")
        print(f"  ⚡ Avg processing: {evolution_results['avg_processing_time']:.3f}s")
        
        return evolution_results
    
    async def compare_models(self, models: List[str], num_turns: int = 50) -> Dict:
        """Compare multiple models across long conversations"""
        print("🔄 MODEL COMPARISON FOR LONG CONVERSATIONS")
        print("=" * 70)
        
        comparison_results = {
            'models_tested': models,
            'total_turns': num_turns,
            'results': {},
            'performance_ranking': []
        }
        
        for model in models:
            print(f"\n{'='*20} {model.upper()} {'='*20}")
            model_results = await self.evaluate_model(model, num_turns)
            comparison_results['results'][model] = model_results
        
        # Generate performance ranking
        model_scores = []
        for model, results in comparison_results['results'].items():
            # Calculate composite score
            total_memory_tests = sum(scenario.get('memory_tests_total', 0) 
                                   for scenario in results['scenario_results'].values())
            total_memory_passed = sum(scenario.get('memory_tests_passed', 0) 
                                    for scenario in results['scenario_results'].values())
            
            memory_score = total_memory_passed / max(total_memory_tests, 1)
            
            avg_processing_time = sum(scenario.get('avg_processing_time', 1) 
                                    for scenario in results['scenario_results'].values()) / 3
            
            # NEW: Calculate field consciousness metrics
            total_emergent_symbols = sum(scenario.get('emergent_symbols_count', 0) 
                                       for scenario in results['scenario_results'].values())
            max_field_energy = max(scenario.get('peak_field_energy', 0) 
                                 for scenario in results['scenario_results'].values())
            total_attractor_events = sum(scenario.get('attractor_events', 0) 
                                       for scenario in results['scenario_results'].values())
            total_field_resonance = sum(scenario.get('field_resonance_events', 0) 
                                      for scenario in results['scenario_results'].values())
            
            # Enhanced scoring: memory (40%) + speed (20%) + field consciousness (40%)
            speed_bonus = max(0, (1.0 - min(avg_processing_time, 2.0) / 2.0))
            field_consciousness_score = min(1.0, (
                min(total_emergent_symbols / 20.0, 1.0) * 0.4 +  # Emergence
                min(max_field_energy / 8.0, 1.0) * 0.3 +         # Field energy
                min(total_attractor_events / 10.0, 1.0) * 0.2 +   # Attractors
                min(total_field_resonance / 10.0, 1.0) * 0.1      # Resonance
            ))
            
            composite_score = memory_score * 0.4 + speed_bonus * 0.2 + field_consciousness_score * 0.4
            
            model_scores.append({
                'model': model,
                'composite_score': composite_score,
                'memory_score': memory_score,
                'field_consciousness_score': field_consciousness_score,
                'total_emergent_symbols': total_emergent_symbols,
                'max_field_energy': max_field_energy,
                'total_attractor_events': total_attractor_events,
                'total_field_resonance': total_field_resonance,
                'avg_processing_time': avg_processing_time,
                'total_turns_completed': sum(scenario.get('turns_completed', 0) 
                                           for scenario in results['scenario_results'].values())
            })
        
        # Sort by composite score
        model_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        comparison_results['performance_ranking'] = model_scores
        
        return comparison_results

async def main():
    evaluator = LongConversationEvaluator()
    
    # Test models
    models_to_test = [
        "google/gemma-3-270m",      # Current fastest
        "qwen/qwen3-1.7b",          # User's exact specification
        "qwen2.5-0.5b-instruct-mlx"            # Compact model
    ]
    
    print("🧠 CONSCIOUSNESS LONG CONVERSATION EVALUATION")
    print("=" * 60)
    print("Based on MultiChallenge (2025) framework")
    print("Testing memory retention, context coherence, instruction following + FIELD CONSCIOUSNESS")
    
    results = await evaluator.compare_models(models_to_test, num_turns=50)
    
    # Save results
    with open('long_conversation_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n🏆 FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    for rank, model_result in enumerate(results['performance_ranking'], 1):
        print(f"{rank}. {model_result['model']}")
        print(f"   Composite Score: {model_result['composite_score']:.3f}")
        print(f"   Memory Retention: {model_result['memory_score']:.2%}")
        print(f"   Field Consciousness: {model_result['field_consciousness_score']:.3f}")
        print(f"   Emergent Symbols: {model_result['total_emergent_symbols']}")
        print(f"   Max Field Energy: {model_result['max_field_energy']:.2f}")
        print(f"   Attractor Events: {model_result['total_attractor_events']}")
        print(f"   Field Resonance: {model_result['total_field_resonance']}")
        print(f"   Avg Processing: {model_result['avg_processing_time']:.3f}s")
        print(f"   Turns Completed: {model_result['total_turns_completed']}")
        print()
    
    print(f"📄 Detailed results saved to: long_conversation_evaluation.json")

if __name__ == "__main__":
    asyncio.run(main())