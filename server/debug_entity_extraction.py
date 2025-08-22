#!/usr/bin/env python3
"""Debug entity extraction to understand what's happening"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.dynamic_tape_head import DynamicTapeHead
from unittest.mock import Mock

def debug_entity_extraction():
    """Debug the entity extraction to see what's happening"""
    
    # Create a DTH with mock memory
    mock_memory = Mock()
    dth = DynamicTapeHead(mock_memory, policy_path="config/test_tape_head_policy.json")
    
    print("=" * 60)
    print("DEBUGGING ENTITY EXTRACTION")
    print("=" * 60)
    
    # Check if NLP is available
    print(f"\nNLP available: {dth.nlp is not None}")
    print(f"Using fallback extraction: {dth.nlp is None}")
    
    # Test cases with detailed output
    test_texts = [
        "Recent relevant content about Python",
        "Old content about cooking recipes",
        "I need help with python code",
        "Working with Python and JavaScript",
        "Database SQL queries",
        "The Python programming language is great",
        "I love cooking Italian recipes",
        "Code review for my Python project"
    ]
    
    for text in test_texts:
        print(f"\nText: '{text}'")
        
        # Get entities
        entities = dth._extract_entities(text)
        print(f"  Extracted entities: {entities}")
        
        # Check what we expect to find
        text_lower = text.lower()
        expected = []
        
        # Check for specific terms
        if 'python' in text_lower:
            expected.append('Python')
        if 'code' in text_lower or 'coding' in text_lower:
            expected.append('Code')
        if 'cooking' in text_lower:
            expected.append('Cooking')
        if 'recipe' in text_lower or 'recipes' in text_lower:
            expected.append('Recipes' if 'recipes' in text_lower else 'Recipe')
        if 'javascript' in text_lower:
            expected.append('Javascript')
        if 'database' in text_lower:
            expected.append('Database')
        if 'sql' in text_lower:
            expected.append('Sql')
        
        print(f"  Expected to find: {expected}")
        
        # Check what was found
        found = [e for e in expected if e in entities]
        missing = [e for e in expected if e not in entities]
        
        if found:
            print(f"  ✓ Found: {found}")
        if missing:
            print(f"  ✗ Missing: {missing}")
    
    print("\n" + "=" * 60)
    print("TESTING ENTITY NORMALIZATION")
    print("=" * 60)
    
    # Test normalization
    test_entities = ["Python", "python", "PYTHON", "Code", "code", "JavaScript"]
    normalized = dth._normalize_entities(test_entities)
    print(f"\nOriginal: {test_entities}")
    print(f"Normalized: {normalized}")
    
    # Test overlap calculation
    query_entities = ["Python", "code"]
    text1_entities = ["Python", "Programming", "Code"]
    text2_entities = ["Cooking", "Recipes", "Food"]
    
    qn = set(dth._normalize_entities(query_entities))
    t1n = set(dth._normalize_entities(text1_entities))
    t2n = set(dth._normalize_entities(text2_entities))
    
    overlap1 = len(qn & t1n)
    overlap2 = len(qn & t2n)
    
    print(f"\nQuery entities (normalized): {qn}")
    print(f"Text1 entities (normalized): {t1n}")
    print(f"Text2 entities (normalized): {t2n}")
    print(f"Overlap with text1: {overlap1}")
    print(f"Overlap with text2: {overlap2}")
    
    if overlap1 > overlap2:
        print("✓ Overlap calculation working correctly")
    else:
        print("✗ Overlap calculation issue")

if __name__ == "__main__":
    debug_entity_extraction()
