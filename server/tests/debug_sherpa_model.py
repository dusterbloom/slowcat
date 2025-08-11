#!/usr/bin/env python3
"""Debug script to test sherpa model configuration"""

import sherpa_onnx
import numpy as np
from pathlib import Path

def test_sherpa_model():
    model_dir = Path("./models/sherpa-nemo-10lang")
    model_file = model_dir / "model.onnx"
    tokens_file = model_dir / "tokens.txt"
    
    print(f"Model file: {model_file}")
    print(f"Tokens file: {tokens_file}")
    print(f"Model file exists: {model_file.exists()}")
    print(f"Tokens file exists: {tokens_file.exists()}")
    
    # Try different configurations
    configs = [
        {"sample_rate": 16000, "feature_dim": 80},
        {"sample_rate": 8000, "feature_dim": 80}, 
        {"sample_rate": 16000, "feature_dim": 40},
        {"sample_rate": 22050, "feature_dim": 80},
    ]
    
    for config in configs:
        print(f"\n--- Testing config: {config} ---")
        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(model_file),
                tokens=str(tokens_file),
                sample_rate=config["sample_rate"],
                feature_dim=config["feature_dim"],
                num_threads=2,
                provider="cpu",
                debug=False  # Less verbose
            )
            print(f"✅ Successfully created recognizer with config: {config}")
            
            # Test with small audio data
            print("Testing with synthetic audio data...")
            # Create 1 second of silence at the specified sample rate
            audio_data = np.zeros(config["sample_rate"], dtype=np.float32)
            
            stream = recognizer.create_stream()
            stream.accept_waveform(config["sample_rate"], audio_data)
            recognizer.decode_stream(stream)
            result = stream.result.text
            print(f"Result: '{result}' (expected empty for silence)")
            
        except Exception as e:
            print(f"❌ Failed with config {config}: {e}")
            continue
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_sherpa_model()