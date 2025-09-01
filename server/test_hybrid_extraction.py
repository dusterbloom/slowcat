#!/usr/bin/env python
"""Test hybrid fact extraction to find corruption source"""

from memory.hybrid_fact_extractor import HybridFactExtractor
import json

def test_hybrid():
    extractor = HybridFactExtractor()
    
    # Test problematic sentences
    test_cases = [
        'I lived in Sardinia',
        "My dog's name is Potola", 
        'Hello slowcat',
        'The birth of the space industry',
        'Sardinia is in Italy',
        "Sardinia's capital is Cagliari"
    ]
    
    results = []
    for sent in test_cases:
        print(f'\n{"="*60}')
        print(f'Input: "{sent}"')
        print("-"*60)
        
        facts = extractor.extract_facts(sent)
        print(f'Extracted {len(facts)} facts:')
        
        for fact in facts:
            f = fact if isinstance(fact, dict) else fact
            print(f'  Subject: "{f.get("subject", "")}"')
            print(f'  Predicate: "{f.get("predicate", "")}"')
            print(f'  Value: "{f.get("value", "")}"')
            print()
            
            results.append({
                'input': sent,
                'subject': f.get("subject", ""),
                'predicate': f.get("predicate", ""),
                'value': f.get("value", "")
            })
    
    # Save results for analysis
    with open('hybrid_extraction_debug.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to hybrid_extraction_debug.json")

if __name__ == "__main__":
    test_hybrid()