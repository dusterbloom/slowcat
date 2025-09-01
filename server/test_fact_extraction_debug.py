#!/usr/bin/env python
"""Debug fact extraction to understand the corruption"""

from memory.spacy_fact_extractor import HighAccuracyFactExtractor
import json

def test_extraction():
    extractor = HighAccuracyFactExtractor()
    
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
            f = fact if isinstance(fact, dict) else fact.to_dict()
            print(f'  Subject: "{f["subject"]}"')
            print(f'  Predicate: "{f["predicate"]}"')
            print(f'  Value: "{f["value"]}"')
            print()
            
            results.append({
                'input': sent,
                'subject': f["subject"],
                'predicate': f["predicate"],
                'value': f["value"]
            })
    
    # Save results for analysis
    with open('fact_extraction_debug.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to fact_extraction_debug.json")

if __name__ == "__main__":
    test_extraction()