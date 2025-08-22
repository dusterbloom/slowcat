#!/usr/bin/env python3
"""Verify the key fixes are working"""

import sys
import os
import asyncio
import time

# Add the server directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.dynamic_tape_head import DynamicTapeHead, MemorySpan
from unittest.mock import Mock, AsyncMock

async def verify_fixes():
    """Verify that our fixes are working"""
    print("=" * 60)
    print("VERIFYING SYMBOL SYSTEM FIXES")
    print("=" * 60)
    
    # Create a mock memory system
    mock_memory = Mock()
    
    # Test with the test policy (entities enabled, symbols disabled)
    dth = DynamicTapeHead(mock_memory, policy_path="config/test_tape_head_policy.json")
    
    print("\n1. Testing Entity Extraction Fix")
    print("-" * 40)
    
    # Test entity extraction with fallback
    # Note: Fallback only finds capitalized words or specific common terms
    test_cases = [
        ("Recent relevant content about Python", ["Python"]),  # Python is capitalized
        ("Old content about cooking recipes", ["Cooking", "Recipes"]),  # Should find if in common_terms
        ("I need help with python code", ["Python", "Code"]),  # python/code in common_terms
        ("Working with Python and JavaScript", ["Python", "JavaScript"]),  # Both capitalized
        ("Database SQL queries", ["Database", "SQL"])  # Should be found if capitalized
    ]
    
    all_passed = True
    for text, expected_entities in test_cases:
        entities = dth._extract_entities(text)
        print(f"  Text: '{text}'")
        print(f"  Extracted: {entities}")
        
        # Check if expected entities are found
        found = []
        for expected in expected_entities:
            if expected in entities:
                found.append(expected)
        
        if found:
            print(f"  ✓ Found expected entities: {found}")
        else:
            print(f"  ✗ Missing expected entities: {expected_entities}")
            all_passed = False
    
    print("\n2. Testing Scoring with Entity Matching")
    print("-" * 40)
    
    # Create test memories
    recent_memory = MemorySpan(
        content="Recent relevant content about Python",
        ts=time.time() - 60,  # 1 minute ago
        role="user",
        speaker_id="test",
        source_id="test_1",
        source_hash="hash1"
    )
    
    old_memory = MemorySpan(
        content="Old content about cooking recipes",
        ts=time.time() - 86400,  # 1 day ago
        role="user",
        speaker_id="test",
        source_id="test_2",
        source_hash="hash2"
    )
    
    # Score them with query entities
    query_entities = ["Python", "code"]
    score_recent, comp_recent = dth._score_memory(
        recent_memory, None, query_entities, []
    )
    score_old, comp_old = dth._score_memory(
        old_memory, None, query_entities, []
    )
    
    print(f"  Query entities: {query_entities}")
    print(f"  Recent memory score: {score_recent:.3f} (E={comp_recent['E']:.3f})")
    print(f"  Old memory score: {score_old:.3f} (E={comp_old['E']:.3f})")
    
    if comp_recent['E'] > comp_old['E']:
        print(f"  ✓ Entity matching working: Recent has higher E score")
    else:
        print(f"  ✗ Entity matching not working: E scores are {comp_recent['E']} vs {comp_old['E']}")
        all_passed = False
    
    if score_recent > score_old:
        print(f"  ✓ Overall scoring correct: Recent scores higher than old")
    else:
        print(f"  ✗ Overall scoring issue: Recent ({score_recent:.3f}) should score higher than old ({score_old:.3f})")
        all_passed = False
    
    print("\n3. Testing Memory Retrieval")
    print("-" * 40)
    
    # Create a fresh mock memory with proper structure
    mock_memory2 = Mock()
    # Don't give it a tape_store to force it to use direct get_recent
    mock_memory2.tape_store = None
    mock_memory2.knn_tape = None  # Disable knn_tape to avoid Mock await issues
    mock_memory2.search_tape = None
    
    # Mock memory with get_recent method returning proper data
    mock_memory2.get_recent = AsyncMock(return_value=[
        {
            'ts': time.time() - 100,
            'speaker_id': 'test_user',
            'role': 'user',
            'content': 'Test content from SurrealMemory',
            'embedding': None
        }
    ])
    
    # Create new DTH with this mock
    dth2 = DynamicTapeHead(mock_memory2, policy_path="config/test_tape_head_policy.json")
    
    # Try to get candidates
    candidates = await dth2._get_candidates("test query", None)
    
    if candidates:
        print(f"  ✓ Retrieved {len(candidates)} candidates from memory")
        print(f"    First candidate: '{candidates[0].content[:50]}...'")
    else:
        print(f"  ✗ Failed to retrieve candidates from memory")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED!")
        print("The fixes appear to be working correctly.")
    else:
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("Please review the output above for details.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(verify_fixes())
    sys.exit(0 if result else 1)
