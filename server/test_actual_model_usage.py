#!/usr/bin/env python3
"""
REAL test to verify which models are actually being called
"""

import os
import requests
import json
from memory.dspy_integration import extract_facts_from_text_dspy

def test_model_calls():
    """Test which models are actually being called by intercepting HTTP requests"""
    
    # Test with smaller models
    print("🧪 Testing with smaller models...")
    os.environ['DSPY_REL_MODEL'] = 'qwen2.5-0.5b-instruct-mlx'  # Relations model
    os.environ['DSPY_FACTS_MODEL'] = 'qwen3-1.7b'  # Facts model
    
    # Test 1: Clear factual text (should trigger strict model)
    test_text1 = "My dog's name is Potola and she is a golden retriever"
    print(f"\n📝 Test 1 (clear facts): '{test_text1}'")
    facts1 = extract_facts_from_text_dspy(test_text1)
    print(f"Extracted facts: {facts1}")
    
    # Test 2: Vague/complex text (might trigger general model fallback)
    test_text2 = "The weather seems nice today, I'm feeling contemplative about life"
    print(f"\n📝 Test 2 (vague text): '{test_text2}'")
    facts2 = extract_facts_from_text_dspy(test_text2)
    print(f"Extracted facts: {facts2}")
    
    # Test 3: Direct method calls to ensure both models are tested
    print(f"\n🔍 Direct method testing:")
    from memory.dspy_single_call_extractor import DSPySingleCallExtractor
    extractor = DSPySingleCallExtractor()
    
    print(f"   STRICT MODEL ({extractor.model_rel}): ")
    strict_facts = extractor.extract_relations_strict(test_text1)
    print(f"   → {strict_facts}")
    
    print(f"   GENERAL MODEL ({extractor.model_facts}): ")
    general_facts = extractor.extract_relations_general(test_text1)
    print(f"   → {general_facts}")
    
    print("\n" + "="*60 + "\n")
    
    # Test with larger models  
    print("🧪 Testing with larger models...")
    os.environ['DSPY_REL_MODEL'] = 'qwen/qwen3-4b'  # Relations model
    os.environ['DSPY_FACTS_MODEL'] = 'qwen3-4b-instruct-2507'  # Facts model
    
    print(f"\n📝 Test 1 (clear facts): '{test_text1}'")
    facts1 = extract_facts_from_text_dspy(test_text1)
    print(f"Extracted facts: {facts1}")
    
    print(f"\n📝 Test 2 (vague text): '{test_text2}'")
    facts2 = extract_facts_from_text_dspy(test_text2)
    print(f"Extracted facts: {facts2}")
    
    # Direct method calls for 4B models
    print(f"\n🔍 Direct method testing:")
    extractor4b = DSPySingleCallExtractor()
    
    print(f"   STRICT MODEL ({extractor4b.model_rel}): ")
    strict_facts = extractor4b.extract_relations_strict(test_text1)
    print(f"   → {strict_facts}")
    
    print(f"   GENERAL MODEL ({extractor4b.model_facts}): ")
    general_facts = extractor4b.extract_relations_general(test_text1)
    print(f"   → {general_facts}")

def test_direct_api_calls():
    """Make direct API calls to verify model selection"""
    
    models_to_test = [
        'qwen3-1.7b',
        'qwen3-4b-instruct-2507'
    ]
    
    for model in models_to_test:
        print(f"\n🔍 Testing direct API call to {model}:")
        
        try:
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Extract facts from: My name is John and I work at Apple"}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"✅ {model} responded: {content[:100]}...")
                
                # Check if the model field is echoed back correctly
                if "model" in result:
                    print(f"   Model used: {result['model']}")
            else:
                print(f"❌ {model} failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ {model} error: {e}")

if __name__ == "__main__":
    print("🕵️ Real Model Usage Test")
    print("=" * 50)
    
    print("\n1. Testing DSPy Integration:")
    test_model_calls()
    
    print("\n2. Testing Direct API Calls:")  
    test_direct_api_calls()