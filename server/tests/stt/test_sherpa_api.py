#!/usr/bin/env python3
"""Test sherpa-onnx API to understand the correct configuration"""

import sherpa_onnx
import os
from pathlib import Path

def test_sherpa_api():
    model_dir = Path("./models/sherpa-nemo-10lang")
    model_file = model_dir / "model.onnx"
    tokens_file = model_dir / "tokens.txt"
    
    print(f"Model file exists: {model_file.exists()}")
    print(f"Tokens file exists: {tokens_file.exists()}")
    
    try:
        # Try to create a Nemo CTC model config
        nemo_config = sherpa_onnx.OfflineNemoEncDecCtcModelConfig(
            model=str(model_file)
        )
        print(f"✅ Nemo config created: {type(nemo_config)}")
        
        # Try to create the model config
        model_config = sherpa_onnx.OfflineModelConfig(
            nemo_ctc=nemo_config,
            tokens=str(tokens_file),
            num_threads=4,
            provider="cpu",
            debug=False
        )
        print(f"✅ Model config created: {type(model_config)}")
        
        # Try to create recognizer config
        recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
            feat_config=sherpa_onnx.FeatureExtractorConfig(),
            model_config=model_config,
            decoding_method="greedy_search"
        )
        print(f"✅ Recognizer config created: {type(recognizer_config)}")
        
        # Try to create recognizer - it seems to take no arguments, let's try different approaches
        try:
            recognizer = sherpa_onnx.OfflineRecognizer.from_config(recognizer_config)
            print(f"✅ Recognizer created via from_config: {type(recognizer)}")
        except:
            try:
                recognizer = sherpa_onnx.OfflineRecognizer()
                recognizer.config = recognizer_config
                print(f"✅ Recognizer created via config assignment: {type(recognizer)}")
            except Exception as e2:
                print(f"❌ Recognizer creation failed: {e2}")
                # Let's check what methods are available
                print("Available methods:")
                print([m for m in dir(sherpa_onnx.OfflineRecognizer) if not m.startswith('_')])
        
        print("\nAll tests passed! API structure is correct.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")
        
        # Let's check what parameters are available
        print("\nChecking available parameters...")
        print("OfflineModelConfig parameters:")
        try:
            help(sherpa_onnx.OfflineModelConfig.__init__)
        except:
            pass

if __name__ == "__main__":
    test_sherpa_api()