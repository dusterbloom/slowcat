#!/usr/bin/env python3
"""Test that the buffer API error is fixed"""

import sys
import os
import asyncio
import hashlib

# Add the server directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.dynamic_tape_head import DynamicTapeHead, MemorySpan
from unittest.mock import Mock, AsyncMock

async def test_buffer_error_fix():
    """Test that non-string content doesn't cause buffer errors"""
    
    print("=" * 60)
    print("TESTING BUFFER ERROR FIX")
    print("=" * 60)
    
    # Test various problematic content types
    test_cases = [
        ("Normal string content", "Normal string content"),
        (None, ""),  # None should become empty string
        (123, "123"),  # Number should be converted to string
        ({"key": "value"}, "{'key': 'value'}"),  # Dict should be stringified
        ([1, 2, 3], "[1, 2, 3]"),  # List should be stringified
        (b"bytes content", "b'bytes content'"),  # Bytes should be stringified
    ]
    
    all_passed = True
    
    print("\n1. Testing hashlib.sha256 with various content types")
    print("-" * 40)
    
    for original, expected in test_cases:
        try:
            # Simulate what the fixed code does
            content_str = str(original) if original is not None else ''
            hash_result = hashlib.sha256(content_str.encode()).hexdigest()
            
            print(f"  ✓ {type(original).__name__}: {repr(original)[:30]}... -> hash OK")
            
            # Verify the string conversion
            if content_str != expected:
                print(f"    Warning: Expected '{expected}', got '{content_str}'")
                all_passed = False
        except Exception as e:
            print(f"  ✗ {type(original).__name__}: {repr(original)[:30]}... -> ERROR: {e}")
            all_passed = False
    
    print("\n2. Testing DTH with mock memory returning non-string content")
    print("-" * 40)
    
    # Create mock memory that returns problematic content
    mock_memory = Mock()
    mock_memory.get_recent = AsyncMock(return_value=[
        {
            'ts': 1234567890.0,
            'speaker_id': 'test_user',
            'role': 'user',
            'content': None  # This would cause the buffer error
        },
        {
            'ts': 1234567891.0,
            'speaker_id': 'test_user',
            'role': 'user',
            'content': 123  # This would also cause an error
        },
        {
            'ts': 1234567892.0,
            'speaker_id': 'test_user',
            'role': 'user',
            'content': {'type': 'dict'}  # And this
        }
    ])
    
    # Create DTH and try to get candidates
    dth = DynamicTapeHead(mock_memory, policy_path="config/test_tape_head_policy.json")
    
    try:
        candidates = await dth._get_candidates("test query", None)
        print(f"  ✓ Successfully retrieved {len(candidates)} candidates")
        
        # Check that all candidates have string content
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate.content, str):
                print(f"    ✗ Candidate {i} has non-string content: {type(candidate.content)}")
                all_passed = False
            else:
                print(f"    ✓ Candidate {i}: content is string, hash OK")
        
    except Exception as e:
        print(f"  ✗ Failed to get candidates: {e}")
        all_passed = False
    
    print("\n3. Testing embedding and entity extraction with non-string input")
    print("-" * 40)
    
    # Test _extract_entities with various inputs
    test_inputs = [None, 123, {"text": "Python"}, ["Python", "code"]]
    
    for input_val in test_inputs:
        try:
            entities = dth._extract_entities(input_val)
            print(f"  ✓ Extract entities from {type(input_val).__name__}: {entities}")
        except Exception as e:
            print(f"  ✗ Failed on {type(input_val).__name__}: {e}")
            all_passed = False
    
    # Test _text_similarity with non-string inputs
    try:
        sim1 = dth._text_similarity(None, "test")
        sim2 = dth._text_similarity(123, 456)
        sim3 = dth._text_similarity("test", None)
        print(f"  ✓ Text similarity with None/numbers: {sim1:.2f}, {sim2:.2f}, {sim3:.2f}")
    except Exception as e:
        print(f"  ✗ Text similarity failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL BUFFER ERROR TESTS PASSED!")
        print("The fix successfully handles non-string content.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please review the output above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    result = asyncio.run(test_buffer_error_fix())
    sys.exit(0 if result else 1)
