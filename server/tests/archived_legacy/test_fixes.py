#!/usr/bin/env python3
"""Quick test script to verify the fixes"""

import subprocess
import sys

def run_tests():
    """Run the specific failing tests"""
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/test_dynamic_tape_head.py::TestDynamicTapeHead::test_scoring_algorithm',
        'tests/test_dynamic_tape_head.py::TestIntegration::test_with_real_surreal_memory',
        '-xvs'
    ]
    
    print("Running tests...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    return result.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    print(f"\nTests {'PASSED' if exit_code == 0 else 'FAILED'} with exit code: {exit_code}")
    sys.exit(exit_code)
