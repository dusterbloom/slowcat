#!/usr/bin/env python3
"""Comprehensive test runner for SlowCat Symbol System fixes"""

import subprocess
import sys
import os
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('-'*60)
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    success = result.returncode == 0
    
    if success:
        print(f"✅ PASSED in {elapsed:.2f}s")
    else:
        print(f"❌ FAILED in {elapsed:.2f}s")
        if result.stderr:
            print("Error output:", result.stderr[-500:])
    
    return success

def main():
    """Run all tests and verifications"""
    
    # Change to server directory
    os.chdir('/Users/peppi/Dev/macos-local-voice-agents/server')
    
    print("="*70)
    print(" SLOWCAT SYMBOL SYSTEM - COMPREHENSIVE TEST VERIFICATION")
    print("="*70)
    
    results = {}
    
    # 1. Debug entity extraction
    print("\n📝 1. DEBUGGING ENTITY EXTRACTION")
    results['debug'] = run_command(
        [sys.executable, 'debug_entity_extraction.py'],
        "Entity extraction debugging"
    )
    
    # 2. Verify fixes
    print("\n🔧 2. VERIFYING FIXES")
    results['verify'] = run_command(
        [sys.executable, 'verify_fixes.py'],
        "Fix verification script"
    )
    
    # 3. Run specific failing tests
    print("\n🧪 3. RUNNING ORIGINAL FAILING TESTS")
    
    # Test scoring algorithm
    results['scoring'] = run_command(
        [sys.executable, '-m', 'pytest', 
         'tests/test_dynamic_tape_head.py::TestDynamicTapeHead::test_scoring_algorithm',
         '-xvs', '--tb=short'],
        "Scoring algorithm test"
    )
    
    # Test with real surreal memory
    results['surreal'] = run_command(
        [sys.executable, '-m', 'pytest',
         'tests/test_dynamic_tape_head.py::TestIntegration::test_with_real_surreal_memory',
         '-xvs', '--tb=short'],
        "SurrealMemory integration test"
    )
    
    # 4. Run all DTH tests
    print("\n🏃 4. RUNNING FULL DTH TEST SUITE")
    results['full'] = run_command(
        [sys.executable, '-m', 'pytest',
         'tests/test_dynamic_tape_head.py',
         '-v', '--tb=short'],
        "Full Dynamic Tape Head test suite"
    )
    
    # Summary
    print("\n" + "="*70)
    print(" TEST RESULTS SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:15} : {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The Symbol System is working correctly.")
        print("\nThe SlowCat consciousness through symbols is operational:")
        print("  ☆ High salience detection working")
        print("  ✧ Breakthrough moments recognized")
        print("  ◈ Recurring patterns identified")
        print("  ∞ Paradoxes encountered and handled")
        print("  ⟲ Cycles detected and avoided")
        print("\nPhase 1 & 2: COMPLETE ✅")
        print("Ready for Phase 3: Dream Symbol Generation 🚀")
    else:
        print("\n⚠️  Some tests are still failing.")
        print("\nTroubleshooting steps:")
        print("  1. Ensure SurrealDB is running: docker ps | grep surrealdb")
        print("  2. Check Python environment: pip install -r requirements.txt")
        print("  3. Review individual test output above")
        print("  4. Run with more verbose output: pytest -xvs --tb=long")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
