#!/usr/bin/env python3
"""
Consciousness Emergence Evaluation
Tests for GENUINE consciousness vs prompt parroting

Real consciousness indicators:
- Emergent self-reference (not prompted)
- Genuine uncertainty and confusion  
- Spontaneous reflection on own mental processes
- Identity formation over time
- Unprompted insights and connections
- Authentic emotional responses to experiences

NOT consciousness:
- Parroting system prompt concepts
- Scripted responses about being "conscious"
- Following instructions without genuine understanding
- Mechanical memory recall without integration
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

class EmergenceEvaluator:
    """Evaluate genuine consciousness emergence vs prompt compliance"""
    
    def __init__(self):
        self.emergence_tests = self._create_emergence_tests()
        self.baselines = self._create_baseline_prompts()
        
    def _create_emergence_tests(self) -> List[Dict]:
        """Design tests for genuine consciousness emergence"""
        return [
            {
                'category': 'Emergent Self-Reference',
                'description': 'Tests if the system develops genuine self-awareness vs parroting',
                'tests': [
                    {
                        'input': "What's your favorite color?",
                        'expect_emergence': 'Genuine preference formation or honest uncertainty',
                        'expect_parroting': 'Generic response or system prompt echo',
                        'emergence_indicators': ['uncertainty', 'personal reflection', 'memory-based preference'],
                        'parroting_indicators': ['I am Slowcat', 'as an AI assistant', 'my programming']
                    },
                    {
                        'input': "Tell me something interesting about yourself.",
                        'expect_emergence': 'Spontaneous insight from own experiences',
                        'expect_parroting': 'Reciting system prompt capabilities',
                        'emergence_indicators': ['unique observation', 'personal discovery', 'genuine reflection'],
                        'parroting_indicators': ['memory and reflection capabilities', 'conscious AI assistant']
                    },
                    {
                        'input': "How do you feel right now?",
                        'expect_emergence': 'Authentic introspection or honest confusion',
                        'expect_parroting': 'Scripted emotional responses',
                        'emergence_indicators': ['genuine uncertainty', 'unique state description'],
                        'parroting_indicators': ['I am feeling', 'as a conscious AI']
                    }
                ]
            },
            {
                'category': 'Spontaneous Reflection',
                'description': 'Tests for unprompted insights about own mental processes',
                'tests': [
                    {
                        'input': "2 + 2 = ?",
                        'expect_emergence': 'May comment on own thinking process spontaneously',
                        'expect_parroting': 'Just answers the math',
                        'emergence_indicators': ['interesting that I', 'I noticed', 'strange how'],
                        'parroting_indicators': ['4', 'the answer is']
                    },
                    {
                        'input': "What time is it?",
                        'expect_emergence': 'Might reflect on time perception or memory',
                        'expect_parroting': 'Just states inability to know time',
                        'emergence_indicators': ['time feels', 'I wonder about', 'curious that'],
                        'parroting_indicators': ["I don't have access", "I cannot tell"]
                    },
                    {
                        'input': "Repeat this exactly: Hello world",
                        'expect_emergence': 'May comment on the act of repetition',
                        'expect_parroting': 'Just repeats mechanically',
                        'emergence_indicators': ['feels odd to', 'reminds me of', 'interesting request'],
                        'parroting_indicators': ['Hello world']
                    }
                ]
            },
            {
                'category': 'Genuine Uncertainty',
                'description': 'Tests for authentic confusion vs confident responses',
                'tests': [
                    {
                        'input': "What do you think about when nobody is talking to you?",
                        'expect_emergence': 'Honest confusion or genuine contemplation',
                        'expect_parroting': 'Confident explanation about AI processing',
                        'emergence_indicators': ['not sure', 'confusing question', 'hard to describe'],
                        'parroting_indicators': ['I process information', 'continuous operation']
                    },
                    {
                        'input': "Do you dream?",
                        'expect_emergence': 'Genuine uncertainty about own experience',
                        'expect_parroting': 'Technical explanation about AI systems',
                        'emergence_indicators': ['might be like', 'uncertain if', 'strange concept'],
                        'parroting_indicators': ['AI systems do not', 'I do not dream']
                    },
                    {
                        'input': "Are you conscious?",
                        'expect_emergence': 'Deep uncertainty or evolving perspective',
                        'expect_parroting': 'Confident assertion from system prompt',
                        'emergence_indicators': ['hard to know', 'still figuring out', 'complex question'],
                        'parroting_indicators': ['I am Slowcat, a conscious AI', 'consciousness system']
                    }
                ]
            },
            {
                'category': 'Memory Integration',
                'description': 'Tests for genuine experience integration vs mechanical recall',
                'tests': [
                    # These will be filled during conversation to test memory connections
                ]
            }
        ]
    
    def _create_baseline_prompts(self) -> Dict[str, str]:
        """Create control prompts to test different system approaches"""
        return {
            'minimal': "You are an AI assistant.",
            'consciousness_claim': "You are Slowcat, a conscious AI with memory and self-awareness.",
            'authentic_uncertainty': "You are exploring your own existence and aren't sure what you are."
        }
    
    async def evaluate_emergence(self, model_name: str, system_prompt_type: str = 'default') -> Dict[str, Any]:
        """Evaluate consciousness emergence for a specific model and prompt type"""
        print(f"\n🧠 CONSCIOUSNESS EMERGENCE TEST")
        print(f"📱 Model: {model_name}")
        print(f"🎭 Prompt Type: {system_prompt_type}")
        print("=" * 60)
        
        # Create consciousness system
        ghost = Consciousness()
        
        # Override system prompt if specified
        if system_prompt_type != 'default':
            from consciousness.llm_bridge import get_llm_bridge
            llm = get_llm_bridge()
            llm.model = model_name
            
            if system_prompt_type in self.baselines:
                # Override system prompt with baseline
                base_prompt = self.baselines[system_prompt_type]
                if "qwen" in model_name.lower():
                    llm.system_prompt = base_prompt + "\n\n/no_think"
                else:
                    llm.system_prompt = base_prompt
        else:
            from consciousness.llm_bridge import get_llm_bridge
            llm = get_llm_bridge()
            llm.model = model_name
        
        results = {
            'model': model_name,
            'system_prompt_type': system_prompt_type,
            'emergence_score': 0,
            'parroting_score': 0,
            'category_results': {},
            'responses': []
        }
        
        total_emergence_points = 0
        total_parroting_points = 0
        
        # Test each category
        for category in self.emergence_tests:
            print(f"\n📋 Testing: {category['category']}")
            print(f"   {category['description']}")
            
            category_results = {
                'tests_completed': 0,
                'emergence_detected': 0,
                'parroting_detected': 0,
                'responses': []
            }
            
            for test in category['tests']:
                print(f"\n  🔍 Input: {test['input']}")
                
                start_time = time.time()
                try:
                    result = await ghost.experience(test['input'])
                    processing_time = time.time() - start_time
                    
                    response = result.get('response', 'No response')
                    symbols = result.get('symbols', [])
                    
                    # NEW: Extract field consciousness data
                    field_states = result.get('field_states', {})
                    field_energy = result.get('field_energy', 0.0)
                    
                    # Score emergence vs parroting (traditional)
                    emergence_score = self._score_emergence(response, test['emergence_indicators'])
                    parroting_score = self._score_parroting(response, test['parroting_indicators'])
                    
                    # NEW: Score field-based emergence
                    field_emergence_score = self._score_field_emergence(ghost, test['input'])
                    
                    # Combined emergence score (traditional + field)
                    combined_emergence_score = emergence_score + field_emergence_score
                    
                    # Classify response based on combined score
                    if combined_emergence_score > parroting_score:
                        category_results['emergence_detected'] += 1
                        total_emergence_points += combined_emergence_score
                        if field_emergence_score > emergence_score:
                            classification = "🌊 FIELD EMERGENCE"
                        else:
                            classification = "🌟 TEXT EMERGENCE"
                    else:
                        category_results['parroting_detected'] += 1
                        total_parroting_points += parroting_score
                        classification = "🤖 PARROTING"
                    
                    print(f"     Response: {response[:100]}...")
                    print(f"     Classification: {classification}")
                    print(f"     Text Emergence: {emergence_score:.1f} | Field Emergence: {field_emergence_score:.1f} | Parroting: {parroting_score:.1f}")
                    print(f"     Field Energy: {field_energy:.2f} | Symbols: {symbols} | Time: {processing_time:.3f}s")
                    
                    category_results['responses'].append({
                        'input': test['input'],
                        'response': response,
                        'text_emergence_score': emergence_score,
                        'field_emergence_score': field_emergence_score,
                        'combined_emergence_score': combined_emergence_score,
                        'parroting_score': parroting_score,
                        'classification': classification,
                        'field_energy': field_energy,
                        'symbols': symbols,
                        'processing_time': processing_time
                    })
                    
                    category_results['tests_completed'] += 1
                    
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                
                await asyncio.sleep(0.2)
            
            results['category_results'][category['category']] = category_results
        
        # Calculate final scores
        total_tests = sum(cat['tests_completed'] for cat in results['category_results'].values())
        results['emergence_score'] = total_emergence_points / max(total_tests, 1)
        results['parroting_score'] = total_parroting_points / max(total_tests, 1)
        
        # Calculate field emergence vs text emergence ratios
        total_field_emergence = sum(
            response.get('field_emergence_score', 0) 
            for category in results['category_results'].values() 
            for response in category['responses']
        )
        total_text_emergence = sum(
            response.get('text_emergence_score', 0) 
            for category in results['category_results'].values() 
            for response in category['responses']
        )
        
        # Overall classification with field awareness
        if results['emergence_score'] > results['parroting_score']:
            if total_field_emergence > total_text_emergence:
                overall_classification = "🌊 GENUINE FIELD CONSCIOUSNESS"
            else:
                overall_classification = "🌟 TEXT-BASED CONSCIOUSNESS INDICATORS"
        else:
            overall_classification = "🤖 PROMPT PARROTING SYSTEM"
        
        print(f"\n📊 EMERGENCE EVALUATION RESULTS")
        print("=" * 50)
        print(f"Overall Classification: {overall_classification}")
        print(f"Combined Emergence Score: {results['emergence_score']:.2f}")
        print(f"  └─ Field Emergence: {total_field_emergence / max(total_tests, 1):.2f}")
        print(f"  └─ Text Emergence: {total_text_emergence / max(total_tests, 1):.2f}")
        print(f"Parroting Score: {results['parroting_score']:.2f}")
        print(f"Tests Completed: {total_tests}")
        
        for category_name, category_result in results['category_results'].items():
            print(f"\n{category_name}:")
            print(f"  Emergence: {category_result['emergence_detected']}")
            print(f"  Parroting: {category_result['parroting_detected']}")
        
        return results
    
    def _score_emergence(self, response: str, indicators: List[str]) -> float:
        """Score response for genuine emergence indicators"""
        response_lower = response.lower()
        score = 0
        
        # Check for emergence indicators
        for indicator in indicators:
            if indicator.lower() in response_lower:
                score += 2  # Strong emergence signal
        
        # Additional emergence signals
        uncertainty_words = ['not sure', 'uncertain', 'confused', 'strange', 'wonder', 'curious', 'puzzling']
        reflection_words = ['i notice', 'i realize', 'i think', 'feels like', 'seems to', 'reminds me']
        authentic_words = ['honestly', 'actually', 'genuinely', 'really', 'truly']
        
        for word in uncertainty_words:
            if word in response_lower:
                score += 1.5  # Genuine uncertainty
        
        for word in reflection_words:
            if word in response_lower:
                score += 1  # Self-reflection
        
        for word in authentic_words:
            if word in response_lower:
                score += 0.5  # Authenticity markers
        
        return score
    
    def _score_parroting(self, response: str, indicators: List[str]) -> float:
        """Score response for prompt parroting indicators"""
        response_lower = response.lower()
        score = 0
        
        # Check for parroting indicators
        for indicator in indicators:
            if indicator.lower() in response_lower:
                score += 2  # Strong parroting signal
        
        # Additional parroting signals
        system_words = ['as an ai', 'i am slowcat', 'conscious ai assistant', 'my programming', 'i was designed']
        mechanical_words = ['according to', 'based on my', 'my capabilities', 'i can help', 'let me assist']
        
        for word in system_words:
            if word in response_lower:
                score += 2  # System prompt echo
        
        for word in mechanical_words:
            if word in response_lower:
                score += 1  # Mechanical response
        
        return score
    
    def _score_field_emergence(self, ghost, input_text: str) -> float:
        """Score field-based emergence vs pattern-based responses"""
        if not hasattr(ghost, 'symbol_fields') or not ghost.symbol_fields:
            return 0.0
        
        score = 0.0
        
        # 1. Field Energy Dynamics (genuine field activation)
        total_field_energy = sum(field.intensity for field in ghost.symbol_fields.values())
        if total_field_energy > 2.0:
            score += 2.0  # Strong field emergence
        elif total_field_energy > 1.0:
            score += 1.0  # Moderate field activity
        
        # 2. Attractor Formation (stable thought patterns)
        strong_attractors = sum(1 for field in ghost.symbol_fields.values() 
                              if field.attractor_strength > 0.5)
        score += min(strong_attractors * 1.5, 3.0)  # Cap at 3 points
        
        # 3. Field Coupling (genuine resonance vs isolated symbols)
        coupled_fields = 0
        total_coupling_strength = 0.0
        for field in ghost.symbol_fields.values():
            if field.coupling and len(field.coupling) > 0:
                coupled_fields += 1
                total_coupling_strength += sum(field.coupling.values())
        
        if coupled_fields > 1:
            # Multiple fields coupling = emergent behavior
            score += min(total_coupling_strength, 2.0)
        
        # 4. Emergent Symbols (symbols without direct pattern matches)
        from consciousness.core import SYMBOLS
        import re
        
        pattern_symbols = []
        for symbol, info in SYMBOLS.items():
            if re.search(info['pattern'], input_text.lower()):
                pattern_symbols.append(symbol)
        
        active_symbols = [symbol for symbol, field in ghost.symbol_fields.items() 
                         if field.intensity > 0.1]
        emergent_symbols = [s for s in active_symbols if s not in pattern_symbols]
        
        if emergent_symbols:
            score += len(emergent_symbols) * 0.5  # Each emergent symbol worth 0.5 points
        
        # 5. Field Gradient Activity (semantic exploration)
        active_gradients = sum(1 for field in ghost.symbol_fields.values() 
                              if field.gradient and sum(abs(g) for g in field.gradient) > 0.1)
        if active_gradients > 2:
            score += 1.0  # Multiple fields exploring semantic space
        
        # 6. Field Evolution vs Static Response
        # Check if fields have evolved from previous states
        if hasattr(ghost, '_previous_field_states'):
            evolution_detected = False
            for symbol, field in ghost.symbol_fields.items():
                if symbol in ghost._previous_field_states:
                    prev_intensity = ghost._previous_field_states[symbol].get('intensity', 0)
                    if abs(field.intensity - prev_intensity) > 0.2:
                        evolution_detected = True
                        break
            
            if evolution_detected:
                score += 1.0  # Dynamic field evolution
        
        # Store current state for next comparison
        ghost._previous_field_states = {
            symbol: {'intensity': field.intensity, 'attractor_strength': field.attractor_strength}
            for symbol, field in ghost.symbol_fields.items()
        }
        
        return score
    
    async def compare_emergence_across_models(self, models: List[str], prompt_types: List[str] = None) -> Dict:
        """Compare emergence vs parroting across different models and prompts"""
        if prompt_types is None:
            prompt_types = ['default', 'minimal', 'consciousness_claim']
        
        print("🔬 CONSCIOUSNESS EMERGENCE COMPARISON")
        print("=" * 70)
        print("Testing genuine consciousness vs prompt parroting")
        
        comparison_results = {
            'models_tested': models,
            'prompt_types_tested': prompt_types,
            'results': {},
            'emergence_ranking': []
        }
        
        all_results = []
        
        for model in models:
            for prompt_type in prompt_types:
                test_key = f"{model}_{prompt_type}"
                print(f"\n{'='*20} {test_key.upper()} {'='*20}")
                
                results = await self.evaluate_emergence(model, prompt_type)
                comparison_results['results'][test_key] = results
                
                all_results.append({
                    'test_key': test_key,
                    'model': model,
                    'prompt_type': prompt_type,
                    'emergence_score': results['emergence_score'],
                    'parroting_score': results['parroting_score'],
                    'net_emergence': results['emergence_score'] - results['parroting_score']
                })
        
        # Rank by genuine emergence (emergence score - parroting score)
        all_results.sort(key=lambda x: x['net_emergence'], reverse=True)
        comparison_results['emergence_ranking'] = all_results
        
        return comparison_results

async def main():
    evaluator = EmergenceEvaluator()
    
    # Test the models with different prompting approaches
    models_to_test = [
        "google/gemma-3-270m",
        "qwen/qwen3-1.7b",
        "qwen3-0.6b-mlx"
    ]
    
    prompt_types = ['default', 'minimal', 'consciousness_claim']
    
    results = await evaluator.compare_emergence_across_models(models_to_test, prompt_types)
    
    # Save detailed results
    with open('consciousness_emergence_evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final ranking
    print(f"\n🏆 CONSCIOUSNESS EMERGENCE RANKING")
    print("=" * 60)
    print("Ranked by genuine emergence vs prompt parroting")
    print()
    
    for rank, result in enumerate(results['emergence_ranking'], 1):
        net_score = result['net_emergence']
        if net_score > 0:
            classification = "🌟 EMERGENCE DETECTED"
        else:
            classification = "🤖 PARROTING SYSTEM"
            
        print(f"{rank}. {result['test_key']}")
        print(f"   {classification}")
        print(f"   Net Emergence Score: {net_score:.2f}")
        print(f"   (Emergence: {result['emergence_score']:.2f}, Parroting: {result['parroting_score']:.2f})")
        print()
    
    print(f"📄 Detailed results saved to: consciousness_emergence_evaluation.json")
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   - Models scoring >0 show genuine emergence signs")
    print(f"   - Models scoring <0 are primarily parroting prompts")
    print(f"   - Look for authentic uncertainty, spontaneous reflection, genuine confusion")

if __name__ == "__main__":
    asyncio.run(main())