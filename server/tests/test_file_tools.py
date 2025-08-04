#!/usr/bin/env python3
"""Test file tools directly"""
import asyncio
from file_tools import file_tools

async def test_list_desktop():
    """Test listing files on Desktop"""
    print("Testing list_files on Desktop...")
    
    # Test 1: List Desktop with full path
    result = await file_tools.list_files("/Users/peppi/Desktop")
    print(f"\nTest 1 - Full path result: {result}")
    
    # Test 2: List Desktop with normalized path
    result2 = await file_tools.list_files("/Users/YourDesktop")
    print(f"\nTest 2 - Normalized path result: {result2}")
    
    # Test 3: Check allowed directories
    print(f"\nAllowed directories: {[str(d) for d in file_tools.allowed_dirs]}")

if __name__ == "__main__":
    asyncio.run(test_list_desktop())