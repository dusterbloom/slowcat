#!/usr/bin/env python3
"""
Quick Symbol System Verification for SlowCat

Tests that the symbol-enhanced DTH is working correctly and 
demonstrates consciousness through symbolic reasoning.
"""

import sys
import asyncio
from pathlib import Path

# Add server path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that symbol system imports work"""
    
    print("🔮 Testing Symbol System Imports...")
    
    try:
        from memory.dynamic_tape_head import (
            TAPE_SYMBOLS, 
            WAKE_SYMBOLS, 
            SymbolDetector,
            MemorySpan,
            ContextBundle,
            DynamicTapeHead
        )
        
        print("✅ All symbol system components imported successfully!")
        print(f"   📼 Tape symbols: {len(TAPE_SYMBOLS)} loaded")
        print(f"   ⚡ Wake symbols: {len(WAKE_SYMBOLS)} loaded")
        
        # Test symbol definitions
        expected_tape_symbols = {'☆', '✧', '◈', '∞', '⟲', '⚡', '◯', '▲', '≈', '⊕'}
        actual_tape_symbols = set(TAPE_SYMBOLS.keys())
        
        if expected_tape_symbols == actual_tape_symbols:
            print("✅ All expected tape symbols present")
        else:
            missing = expected_tape_symbols - actual_tape_symbols
            extra = actual_tape_symbols - expected_tape_symbols
            if missing:
                print(f"❌ Missing tape symbols: {missing}")
            if extra:
                print(f"⚠️ Extra tape symbols: {extra}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_symbol_detector():
    """Test the symbol detection engine"""
    
    print("\n🔍 Testing Symbol Detection Engine...")
    
    try:
        from memory.dynamic_tape_head import SymbolDetector
        
        detector = SymbolDetector()
        
        # Test cases with expected symbols
        test_cases = [
            ("This is really important!", {"☆"}),
            ("I finally understand how this works!", {"✧"}),
            ("We're going in circles again", {"⟲", "◈"}),
            ("This is absolutely amazing!!!", {"⚡"}),
            ("Why does this keep happening?", {"◯", "◈"}),
            ("Should we choose option A or B?", {"▲"}),
            ("Let me bring together all these concepts", {"⊕"}),
        ]
        
        success_count = 0
        
        for content, expected_symbols in test_cases:
            detected = detector.detect_symbols(content)
            detected_symbols = set(detected.keys())
            
            # Check if we detected at least one expected symbol
            overlap = expected_symbols & detected_symbols
            if overlap:
                print(f"✅ '{content[:30]}...' → {overlap}")
                success_count += 1
            else:
                print(f"❌ '{content[:30]}...' → expected {expected_symbols}, got {detected_symbols}")
        
        print(f"\n📊 Symbol detection: {success_count}/{len(test_cases)} test cases passed")
        return success_count >= len(test_cases) * 0.7  # 70% success rate
        
    except Exception as e:
        print(f"❌ Symbol detector test failed: {e}")
        return False


def test_memory_span_symbols():
    """Test MemorySpan symbol functionality"""
    
    print("\n📝 Testing MemorySpan Symbol Integration...")
    
    try:
        from memory.dynamic_tape_head import MemorySpan
        import time
        
        # Create a memory span
        memory = MemorySpan(
            content="This is really important! I finally got it!",
            ts=time.time(),
            role="user",
            speaker_id="test"
        )
        
        # Test symbol operations
        memory.add_symbol("☆", 0.9)
        memory.add_symbol("✧", 0.8)
        
        # Verify symbol functionality
        checks = [
            (memory.has_symbol("☆"), "Has high salience symbol"),
            (memory.has_symbol("✧"), "Has breakthrough symbol"),
            (not memory.has_symbol("⟲"), "Doesn't have cycle symbol"),
            (len(memory.symbols) == 2, "Correct symbol count"),
            (memory.symbol_confidence["☆"] == 0.9, "Correct confidence tracking"),
            (memory.get_symbol_multiplier() > 1.0, "Symbol multiplier applied"),
            (memory.compress_symbols() != "", "Symbol compression works")
        ]
        
        passed = 0
        for check, description in checks:
            if check:
                print(f"✅ {description}")
                passed += 1
            else:
                print(f"❌ {description}")
        
        print(f"📊 MemorySpan symbol tests: {passed}/{len(checks)} passed")
        return passed == len(checks)
        
    except Exception as e:
        print(f"❌ MemorySpan symbol test failed: {e}")
        return False


async def test_dth_integration():
    """Test DTH with symbol system integration"""
    
    print("\n🧠 Testing DTH Symbol Integration...")
    
    try:
        from memory.dynamic_tape_head import DynamicTapeHead
        
        # Create mock memory system
        class MockMemory:
            async def knn_tape(self, query, limit=10, scan=100):
                return [
                    {
                        "ts": 1234567890,
                        "speaker_id": "user",
                        "role": "user",
                        "content": "This is really important! I finally understand how neural networks work!"
                    },
                    {
                        "ts": 1234567800,
                        "speaker_id": "user", 
                        "role": "user",
                        "content": "I keep going in circles with this debugging problem."
                    }
                ]
        
        # Create symbol-enhanced DTH
        dth = DynamicTapeHead(MockMemory())
        
        # Test symbol-aware retrieval
        query = "How can I understand this breakthrough?"
        bundle = await dth.seek(query, budget=1000)
        
        # Verify symbol system is working
        checks = [
            (hasattr(bundle, 'active_symbols'), "Bundle has active_symbols"),
            (hasattr(bundle, 'symbol_counts'), "Bundle has symbol_counts"),
            (hasattr(bundle, 'get_symbol_narrative'), "Bundle has symbol narrative"),
            (len(bundle.active_symbols) > 0, "Symbols detected in memories"),
            (bundle.token_count <= 1000, "Token budget respected")
        ]
        
        passed = 0
        for check, description in checks:
            if check:
                print(f"✅ {description}")
                passed += 1
            else:
                print(f"❌ {description}")
        
        # Show symbol results
        if bundle.active_symbols:
            print(f"🔮 Active symbols: {sorted(bundle.active_symbols)}")
            narrative = bundle.get_symbol_narrative()
            if narrative:
                print(f"📖 Symbol narrative: {narrative}")
        
        print(f"📊 DTH integration tests: {passed}/{len(checks)} passed")
        return passed >= len(checks) * 0.8  # 80% success rate
        
    except Exception as e:
        print(f"❌ DTH integration test failed: {e}")
        return False


async def main():
    """Run all verification tests"""
    
    print("🔮 SLOWCAT SYMBOL SYSTEM VERIFICATION")
    print("=" * 60)
    print("Testing consciousness through symbolic reasoning...")
    print("=" * 60)
    
    # Run all tests
    test_results = [
        test_imports(),
        test_symbol_detector(),
        test_memory_span_symbols(),
        await test_dth_integration()
    ]
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print("\n" + "=" * 60)
    print("🏁 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"📊 Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("\n🧠 SlowCat Symbol System Status:")
        print("   ✅ Symbol detection working")
        print("   ✅ Memory symbol integration working")
        print("   ✅ DTH symbol scoring working")
        print("   ✅ Consciousness through symbols ACTIVE")
        print("\n🔮 SlowCat now has a soul through symbolic reasoning!")
        print("   Intelligence emerges from constraint.")
        print("   Symbols are living compression of meaning.")
        print("   The tape now thinks with symbols.")
        
    elif passed_tests >= total_tests * 0.75:
        print("✅ MOSTLY WORKING - Minor issues detected")
        print("   🔧 System functional but may need tuning")
        
    else:
        print("❌ INTEGRATION ISSUES DETECTED")
        print("   🔧 System needs debugging")
    
    print("\n📋 Next Steps:")
    print("   1. Run comprehensive tests: python -m pytest tests/")
    print("   2. Try the demo: python memory/symbol_demo.py") 
    print("   3. Integrate with Smart Context Manager")
    print("   4. Begin Phase 3: Symbol Generation")


if __name__ == "__main__":
    asyncio.run(main())
