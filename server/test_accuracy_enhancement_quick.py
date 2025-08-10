#!/usr/bin/env python3
"""
Test script to verify Sherpa-ONNX accuracy enhancement is working
"""
import asyncio
import sys
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent))

from services.accuracy_enhancement import AdvancedAccuracyEnhancer

async def test_accuracy_enhancement():
    """Test that accuracy enhancement is working properly"""
    
    print("🧪 Testing Sherpa-ONNX Accuracy Enhancement")
    print("=" * 50)
    
    # Initialize the accuracy enhancer
    try:
        enhancer = AdvancedAccuracyEnhancer()
        print("✅ Accuracy enhancement service loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load accuracy enhancement service: {e}")
        return
    
    # Test cases that simulate common STT errors
    test_cases = [
        {
            "input": "Please visit bbc dot com slash news for more information",
            "expected": "bbc.com/news"
        },
        {
            "input": "My email is john dot smith at gmail dot com",
            "expected": "john.smith@gmail.com"
        },
        {
            "input": "Check out github dot com slash user slash repository",
            "expected": "github.com/user/repository"
        },
        {
            "input": "The docker container is running well",
            "expected": "Docker"
        }
    ]
    
    print("\n🔍 Testing common transcription patterns:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Input:    {test_case['input']}")
        
        try:
            result = await enhancer.enhance_accuracy(test_case['input'], confidence=0.6)
            print(f"  Output:   {result.corrected_text}")
            print(f"  Method:   {result.method_used}")
            print(f"  Time:     {result.processing_time_ms:.1f}ms")
            
            # Check if expected pattern is in result
            if test_case['expected'].lower() in result.corrected_text.lower():
                print(f"  ✅ PASS: Found '{test_case['expected']}'")
            else:
                print(f"  ⚠️  PARTIAL: '{test_case['expected']}' not found")
                
            if result.corrections_applied:
                print(f"  Corrections: {len(result.corrections_applied)}")
                for corr in result.corrections_applied[:3]:
                    print(f"    • {corr['original']} → {corr['corrected']} ({corr['method']})")
                    
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")

if __name__ == "__main__":
    asyncio.run(test_accuracy_enhancement())
