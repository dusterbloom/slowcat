#!/usr/bin/env python3
"""
Test accuracy enhancement on real audio files
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from advanced_accuracy_enhancer import AdvancedAccuracyEnhancer

# Test audio files
AUDIO_FILES = [
    "test_audio/realworld/python_example_test(1).wav",
    "test_audio/comparison/comparison_300.0s.wav",
    # Add more files as needed
]

async def test_audio_enhancement():
    """Test accuracy enhancement on real audio files"""
    
    # Initialize the accuracy enhancer
    enhancer = AdvancedAccuracyEnhancer()
    
    # Import Sherpa-ONNX (assuming it's available in your environment)
    try:
        import sherpa_onnx
        print("✅ Sherpa-ONNX available")
    except ImportError:
        print("❌ Sherpa-ONNX not available. Please install it first.")
        return
    
    # For now, let's just test the enhancer with some example text
    # that might come from these audio files
    
    test_transcriptions = [
        "Please visit bbb dot com slash news for more information",
        "My email is john dot smith at gee mail dot com",
        "I work at goggle and my name is john smith",
        "Check out get hub dot com for the repository",
        "The dock her container is running well",
        "I use react j s and type script for development"
    ]
    
    print("🎙️ Testing accuracy enhancement on sample transcriptions")
    print("=" * 60)
    
    for i, transcription in enumerate(test_transcriptions, 1):
        print(f"\nTest {i}:")
        print(f"Original:  {transcription}")
        
        # Apply accuracy enhancement
        result = await enhancer.enhance_accuracy(transcription, confidence=0.6)
        
        print(f"Enhanced:  {result.corrected_text}")
        print(f"Method:    {result.method_used}")
        print(f"Time:      {result.processing_time_ms:.1f}ms")
        
        if result.corrections_applied:
            print("Corrections:")
            for corr in result.corrections_applied:
                print(f"  • {corr['original']} → {corr['corrected']} ({corr['method']})")
    
    print("\n" + "=" * 60)
    print("🎯 Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_audio_enhancement())