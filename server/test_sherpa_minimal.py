#!/usr/bin/env python3
"""
🔍 Minimal test to isolate sherpa-onnx OnlineRecognizer segfault
Tests basic initialization without pipecat
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_minimal_sherpa():
    """Test minimal sherpa initialization"""
    
    print("🔍 Testing minimal Sherpa OnlineRecognizer...")
    
    try:
        import sherpa_onnx as sherpa
        print("✅ sherpa_onnx imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import sherpa_onnx: {e}")
        return False
    
    # Model directory from .env
    model_dir = Path("./models/sherpa-onnx-streaming-zipformer-en-2023-06-26")
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return False
    
    print(f"📁 Using model directory: {model_dir}")
    
    # Find model files
    tokens_file = model_dir / "tokens.txt"
    encoder_files = list(model_dir.glob("encoder*.onnx"))
    decoder_files = list(model_dir.glob("decoder*.onnx"))
    joiner_files = list(model_dir.glob("joiner*.onnx"))
    
    if not all([tokens_file.exists(), encoder_files, decoder_files, joiner_files]):
        print(f"❌ Missing required model files")
        print(f"   tokens.txt: {tokens_file.exists()}")
        print(f"   encoder files: {len(encoder_files)}")
        print(f"   decoder files: {len(decoder_files)}")
        print(f"   joiner files: {len(joiner_files)}")
        return False
    
    # Use int8 versions for stability
    encoder_file = next((f for f in encoder_files if 'int8' in f.name), encoder_files[0])
    decoder_file = next((f for f in decoder_files if 'int8' in f.name), decoder_files[0])
    joiner_file = next((f for f in joiner_files if 'int8' in f.name), joiner_files[0])
    
    print(f"🎯 Using files:")
    print(f"   encoder: {encoder_file.name}")
    print(f"   decoder: {decoder_file.name}")
    print(f"   joiner: {joiner_file.name}")
    
    try:
        print("🔄 Creating OnlineRecognizer...")
        
        recognizer = sherpa.OnlineRecognizer.from_transducer(
            tokens=str(tokens_file),
            encoder=str(encoder_file),
            decoder=str(decoder_file),
            joiner=str(joiner_file),
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            decoding_method="greedy_search",
            max_active_paths=4,
            provider="cpu",
        )
        print("✅ OnlineRecognizer created successfully!")
        
        print("🔄 Testing stream creation...")
        stream = recognizer.create_stream()
        print("✅ Stream created successfully!")
        
        print("🔄 Testing audio processing...")
        # Generate 1 second of silence
        sample_rate = 16000
        audio_samples = np.zeros(sample_rate, dtype=np.float32)
        
        # Process the audio
        stream.accept_waveform(sample_rate, audio_samples.tolist())
        
        if recognizer.is_ready(stream):
            recognizer.decode_streams([stream])
        
        result = recognizer.get_result(stream)
        print(f"✅ Audio processed, result: '{result.text if hasattr(result, 'text') else result}'")
        
        print("🔄 Testing cleanup...")
        del stream
        del recognizer
        print("✅ Cleanup successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 MINIMAL SHERPA TEST")
    print("=" * 50)
    
    success = test_minimal_sherpa()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("OnlineRecognizer works - issue might be in integration")
    else:
        print("\n❌ Test failed - fundamental OnlineRecognizer issue")
        print("Consider using OfflineRecognizer streaming instead")