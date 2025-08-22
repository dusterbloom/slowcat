#!/usr/bin/env python3
"""Test the specific test cases that were failing"""

import subprocess
import sys
import os

def run_specific_tests():
    """Run just the two tests that were failing"""
    
    # Change to server directory
    os.chdir('/Users/peppi/Dev/macos-local-voice-agents/server')
    
    print("=" * 60)
    print("RUNNING SPECIFIC FAILING TESTS")
    print("=" * 60)
    
    # Test 1: Scoring algorithm
    print("\n1. Testing scoring algorithm...")
    cmd1 = [
        sys.executable, '-m', 'pytest',
        'tests/test_dynamic_tape_head.py::TestDynamicTapeHead::test_scoring_algorithm',
        '-xvs', '--tb=short'
    ]
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    if "PASSED" in result1.stdout:
        print("✅ test_scoring_algorithm PASSED")
    else:
        print("❌ test_scoring_algorithm FAILED")
        print("Output:", result1.stdout[-500:] if len(result1.stdout) > 500 else result1.stdout)
    
    # Test 2: Real surreal memory
    print("\n2. Testing with real surreal memory...")
    cmd2 = [
        sys.executable, '-m', 'pytest',
        'tests/test_dynamic_tape_head.py::TestIntegration::test_with_real_surreal_memory',
        '-xvs', '--tb=short'
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if "PASSED" in result2.stdout:
        print("✅ test_with_real_surreal_memory PASSED")
    else:
        print("❌ test_with_real_surreal_memory FAILED")
        print("Output:", result2.stdout[-500:] if len(result2.stdout) > 500 else result2.stdout)
    
    print("\n" + "=" * 60)
    
    # Return success if both passed
    return "PASSED" in result1.stdout and "PASSED" in result2.stdout

if __name__ == "__main__":
    success = run_specific_tests()
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS STILL FAILING")
        print("Run with pytest directly for full output:")
        print("  python -m pytest tests/test_dynamic_tape_head.py -xvs")
    
    sys.exit(0 if success else 1)
