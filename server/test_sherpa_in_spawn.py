#!/usr/bin/env python3
"""
Test if sherpa-onnx works with multiprocessing spawn mode
"""

import multiprocessing
import sys
from pathlib import Path

def test_sherpa_in_subprocess():
    """Test sherpa in a spawned subprocess"""
    try:
        import sherpa_onnx as sherpa
        print("✅ sherpa_onnx imported in subprocess")
        
        # Test creating recognizer
        recognizer = sherpa.OnlineRecognizer.from_transducer(
            tokens='./models/sherpa-onnx-streaming-zipformer-en-2023-06-26/tokens.txt',
            encoder='./models/sherpa-onnx-streaming-zipformer-en-2023-06-26/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
            decoder='./models/sherpa-onnx-streaming-zipformer-en-2023-06-26/decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
            joiner='./models/sherpa-onnx-streaming-zipformer-en-2023-06-26/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            decoding_method='greedy_search',
            max_active_paths=4,
            provider='cpu',
        )
        print("✅ OnlineRecognizer created in subprocess")
        
        # Test stream creation
        stream = recognizer.create_stream()
        print("✅ Stream created in subprocess")
        
        del stream
        del recognizer
        print("✅ Cleanup successful in subprocess")
        return True
        
    except Exception as e:
        print(f"❌ Error in subprocess: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Testing sherpa-onnx with multiprocessing spawn...")
    
    # Test direct (no spawn)
    print("\n1. Testing direct (no spawn):")
    success_direct = test_sherpa_in_subprocess()
    
    # Test with spawn
    print("\n2. Testing with spawn mode:")
    multiprocessing.set_start_method('spawn', force=True)
    
    with multiprocessing.Pool(1) as pool:
        try:
            result = pool.apply_async(test_sherpa_in_subprocess)
            success_spawn = result.get(timeout=30)
        except Exception as e:
            print(f"❌ Spawn test failed: {e}")
            success_spawn = False
    
    print(f"\nResults:")
    print(f"  Direct: {'✅' if success_direct else '❌'}")
    print(f"  Spawn:  {'✅' if success_spawn else '❌'}")
    
    if not success_spawn:
        print("\n🚨 SPAWN MODE INCOMPATIBLE! This explains the segfault.")
        print("💡 Solution: Initialize sherpa BEFORE setting spawn mode")

if __name__ == "__main__":
    main()