#!/usr/bin/env python3
"""
Test MemoBase UUID conversion for user IDs.
"""

try:
    from memobase.utils import string_to_uuid
    
    user_id = "default_user"
    uuid_for_user = string_to_uuid(user_id)
    
    print(f"Original user_id: {user_id}")
    print(f"Converted UUID: {uuid_for_user}")
    print(f"UUID type: {type(uuid_for_user)}")
    
    # Test with different strings
    test_cases = ["default_user", "peppi", "user123", "tundal"]
    for test_id in test_cases:
        uuid_result = string_to_uuid(test_id)
        print(f"{test_id} -> {uuid_result}")
        
except ImportError as e:
    print(f"❌ MemoBase not available: {e}")
except Exception as e:
    print(f"❌ Error: {e}")