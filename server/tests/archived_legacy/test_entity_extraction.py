#!/usr/bin/env python3
"""Test the entity extraction fix directly"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.dynamic_tape_head import DynamicTapeHead
from unittest.mock import Mock

def test_entity_extraction():
    """Test that entity extraction works correctly with the fallback"""
    
    # Create a DTH with mock memory
    mock_memory = Mock()
    dth = DynamicTapeHead(mock_memory, policy_path="config/test_tape_head_policy.json")
    
    # Test 1: Extract "Python" from the text
    text1 = "Recent relevant content about Python"
    entities1 = dth._extract_entities(text1)
    print(f"Test 1 - Text: '{text1}'")
    print(f"  Extracted entities: {entities1}")
    assert "Python" in entities1, f"Expected 'Python' in entities, got {entities1}"
    print("  ✓ PASSED: 'Python' found in entities")
    
    # Test 2: Extract from text with lowercase python
    text2 = "I need help with python code"
    entities2 = dth._extract_entities(text2)
    print(f"\nTest 2 - Text: '{text2}'")
    print(f"  Extracted entities: {entities2}")
    # Should extract "Python" (capitalized) and "Code"
    assert "Python" in entities2, f"Expected 'Python' in entities, got {entities2}"
    assert "Code" in entities2, f"Expected 'Code' in entities, got {entities2}"
    print("  ✓ PASSED: 'Python' and 'Code' found in entities")
    
    # Test 3: Extract from text about cooking
    text3 = "Old content about cooking recipes"
    entities3 = dth._extract_entities(text3)
    print(f"\nTest 3 - Text: '{text3}'")
    print(f"  Extracted entities: {entities3}")
    assert "Cooking" in entities3 or "Recipes" in entities3, f"Expected cooking-related entities, got {entities3}"
    print("  ✓ PASSED: Cooking-related entities found")
    
    # Test normalized entities matching
    query_entities = ["Python", "code"]
    entities_recent = dth._extract_entities("Recent relevant content about Python")
    entities_old = dth._extract_entities("Old content about cooking recipes")
    
    # Normalize and check overlap
    qn = set(dth._normalize_entities(query_entities))
    mn_recent = set(dth._normalize_entities(entities_recent))
    mn_old = set(dth._normalize_entities(entities_old))
    
    overlap_recent = len(qn & mn_recent)
    overlap_old = len(qn & mn_old)
    
    print(f"\nOverlap test:")
    print(f"  Query entities (normalized): {qn}")
    print(f"  Recent entities (normalized): {mn_recent}")
    print(f"  Old entities (normalized): {mn_old}")
    print(f"  Overlap with recent: {overlap_recent}")
    print(f"  Overlap with old: {overlap_old}")
    
    assert overlap_recent > overlap_old, f"Expected recent to have more overlap than old"
    print("  ✓ PASSED: Recent text has more entity overlap with query")
    
    print("\n✅ All entity extraction tests passed!")

if __name__ == "__main__":
    test_entity_extraction()
