#!/usr/bin/env python3
"""
Test fact extraction quality with different models
Compare Qwen 0.5B vs Qwen3 4B performance
"""

import os
import time
from memory.dspy_integration import extract_facts_from_text_dspy

def test_fact_extraction_scenarios():
    """Test realistic conversation scenarios for fact extraction quality"""
    
    test_scenarios = [
        {
            "text": "My dog's name is Potola and she's a golden retriever. She loves playing fetch in the park.",
            "expected_facts": ["user has_pet Potola", "Potola is_breed golden retriever", "Potola likes fetch"],
            "description": "Pet information"
        },
        {
            "text": "I work at Google as a software engineer in the Chrome team. I've been there for 3 years.",
            "expected_facts": ["user works_at Google", "user job_title software engineer", "user team Chrome"],
            "description": "Job information"
        },
        {
            "text": "I live in San Francisco with my wife Sarah. We have a 2-bedroom apartment downtown.",
            "expected_facts": ["user lives_in San Francisco", "user spouse Sarah", "user has_home apartment"],
            "description": "Living situation"
        },
        {
            "text": "My favorite food is sushi and I really enjoy Italian wine. I'm vegetarian.",
            "expected_facts": ["user likes sushi", "user prefers Italian wine", "user is_diet vegetarian"],
            "description": "Food preferences"
        },
        {
            "text": "I was born in 1990 in Chicago but moved to California when I was 18.",
            "expected_facts": ["user born_year 1990", "user birthplace Chicago", "user moved_to California"],
            "description": "Personal history"
        },
        {
            "text": "continue talking from where we left off in the last session",
            "expected_facts": [],
            "description": "Meta-query (should be filtered)"
        },
        {
            "text": "Hello, how are you today?",
            "expected_facts": [],
            "description": "Greeting (should be filtered)"
        }
    ]
    
    print(f"🧪 Testing fact extraction with models:")
    print(f"   REL_MODEL: {os.getenv('DSPY_REL_MODEL', 'qwen2.5-0.5b-instruct-mlx:2')}")
    print(f"   FACTS_MODEL: {os.getenv('DSPY_FACTS_MODEL', 'qwen2.5-0.5b-instruct-mlx')}")
    print()
    
    total_scenarios = len(test_scenarios)
    successful_extractions = 0
    total_facts_extracted = 0
    total_time = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"📝 Scenario {i}/{total_scenarios}: {scenario['description']}")
        print(f"   Input: \"{scenario['text']}\"")
        
        start_time = time.time()
        try:
            facts = extract_facts_from_text_dspy(scenario['text'])
            extraction_time = time.time() - start_time
            total_time += extraction_time
            
            print(f"   Extracted {len(facts)} facts in {extraction_time:.2f}s:")
            for fact in facts:
                subj = fact.get('subject', 'N/A')
                pred = fact.get('predicate', 'N/A') 
                val = fact.get('value', 'N/A')
                conf = fact.get('confidence', 0)
                print(f"     • {subj} {pred} {val} (confidence: {conf:.2f})")
            
            total_facts_extracted += len(facts)
            
            # Check if extraction matches expectations
            if len(scenario['expected_facts']) == 0:
                # Should be filtered (meta-queries)
                if len(facts) == 0:
                    print("   ✅ Correctly filtered meta-query")
                    successful_extractions += 1
                else:
                    print("   ❌ Should have filtered this meta-query")
            else:
                # Should extract meaningful facts
                if len(facts) > 0:
                    print("   ✅ Successfully extracted facts")
                    successful_extractions += 1
                else:
                    print("   ❌ Failed to extract expected facts")
            
        except Exception as e:
            extraction_time = time.time() - start_time
            total_time += extraction_time
            print(f"   ❌ Extraction failed: {e}")
        
        print()
    
    # Summary
    success_rate = (successful_extractions / total_scenarios) * 100
    avg_time = total_time / total_scenarios
    facts_per_scenario = total_facts_extracted / total_scenarios
    
    print("📊 Extraction Quality Results:")
    print(f"   Success Rate: {successful_extractions}/{total_scenarios} ({success_rate:.1f}%)")
    print(f"   Average Time: {avg_time:.2f}s per extraction")
    print(f"   Facts per Scenario: {facts_per_scenario:.1f}")
    print(f"   Total Facts Extracted: {total_facts_extracted}")
    
    return {
        'success_rate': success_rate,
        'avg_time': avg_time, 
        'facts_per_scenario': facts_per_scenario,
        'total_facts': total_facts_extracted
    }

def compare_models():
    """Compare extraction quality between different model sizes"""
    print("🚀 Fact Extraction Quality Test")
    print("=" * 50)
    
    results = test_fact_extraction_scenarios()
    
    # Model quality assessment
    if results['success_rate'] >= 85:
        quality = "Excellent"
    elif results['success_rate'] >= 70:
        quality = "Good" 
    elif results['success_rate'] >= 50:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"🎯 Overall Quality: {quality}")
    
    if results['avg_time'] < 1.0:
        speed = "Fast"
    elif results['avg_time'] < 3.0:
        speed = "Medium"
    else:
        speed = "Slow"
        
    print(f"⚡ Extraction Speed: {speed}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if results['success_rate'] < 70:
        print("   - Consider using larger models for better accuracy")
        print("   - Check if LM Studio is running with the correct models")
    if results['avg_time'] > 2.0:
        print("   - Models may be too large for real-time usage")
        print("   - Consider using smaller models for faster extraction")
    if results['facts_per_scenario'] > 5:
        print("   - May be over-extracting - consider stricter filtering")
    elif results['facts_per_scenario'] < 1:
        print("   - May be under-extracting - consider looser thresholds")
    
    return results

if __name__ == "__main__":
    compare_models()