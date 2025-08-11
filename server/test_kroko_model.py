#!/usr/bin/env python3
"""
Test script for Kroko-ASR model - Ultra low latency streaming ASR
Tests accuracy, latency, and streaming performance
"""

import argparse
import time
import numpy as np
from pathlib import Path
import wave
import json
from typing import List, Tuple, Optional
import sys

try:
    import sherpa_onnx
except ImportError:
    print("❌ sherpa-onnx not installed. Please run: pip install sherpa-onnx")
    sys.exit(1)

from loguru import logger

# Test phrases covering various scenarios
TEST_PHRASES = [
    # Basic phrases
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    
    # Technical terms and URLs
    "Please visit github.com slash anthropics slash claude dash code",
    "Install the package using pip install sherpa dash onnx",
    "The API endpoint is localhost colon eight zero eight zero",
    
    # Numbers and special characters
    "The temperature is twenty three point five degrees celsius",
    "My phone number is five five five dash one two three four",
    "The price is ninety nine dollars and ninety nine cents",
    
    # Code-related phrases
    "Define a function called get underscore user underscore by underscore id",
    "Import numpy as np and pandas as pd",
    "The variable name is camel case user profile data",
    
    # Challenging pronunciations
    "The entrepreneur's initiative was unprecedented",
    "Pharmaceutical research requires meticulous methodology",
    "Massachusetts Institute of Technology",
    
    # Multi-language (if testing non-English models)
    "Bonjour comment allez vous",  # French
    "Hola como estas hoy",         # Spanish
]

class KrokoModelTester:
    """Test harness for Kroko-ASR models"""
    
    def __init__(self, model_dir: str, language: str = "en"):
        self.model_dir = Path(model_dir)
        self.language = language
        self.sample_rate = 16000
        self.recognizer = None
        
        # Performance metrics
        self.test_results = []
        self.latency_measurements = []
        
    def initialize_model(self) -> bool:
        """Initialize the Kroko-ASR model"""
        try:
            logger.info(f"🚀 Initializing Kroko-ASR model from: {self.model_dir}")
            
            # Check for required files
            required_files = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]
            lang_files = [f"{self.language}_encoder.onnx", f"{self.language}_decoder.onnx", 
                         f"{self.language}_joiner.onnx", f"{self.language}_tokens.txt"]
            
            # Check standard naming first
            has_standard = all((self.model_dir / f).exists() for f in required_files)
            has_lang_prefix = all((self.model_dir / f).exists() for f in lang_files)
            
            if not has_standard and not has_lang_prefix:
                logger.error(f"❌ Missing required model files in {self.model_dir}")
                logger.error(f"Expected: {required_files} OR {lang_files}")
                return False
            
            # Use appropriate file names
            if has_standard:
                encoder = self.model_dir / "encoder.onnx"
                decoder = self.model_dir / "decoder.onnx"  
                joiner = self.model_dir / "joiner.onnx"
                tokens = self.model_dir / "tokens.txt"
            else:
                encoder = self.model_dir / f"{self.language}_encoder.onnx"
                decoder = self.model_dir / f"{self.language}_decoder.onnx"
                joiner = self.model_dir / f"{self.language}_joiner.onnx"
                tokens = self.model_dir / f"{self.language}_tokens.txt"
            
            logger.info(f"📁 Using model files:")
            logger.info(f"  Encoder: {encoder.name} ({encoder.stat().st_size / 1024 / 1024:.1f}MB)")
            logger.info(f"  Decoder: {decoder.name} ({decoder.stat().st_size / 1024:.1f}KB)")
            logger.info(f"  Joiner: {joiner.name} ({joiner.stat().st_size / 1024:.1f}KB)")
            logger.info(f"  Tokens: {tokens.name}")
            
            # Create recognizer using transducer factory method
            start_time = time.time()
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens),
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                num_threads=1,
                sample_rate=self.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=1.2,  # Optimized for low latency
                rule2_min_trailing_silence=0.6,
                rule3_min_utterance_length=200,
                decoding_method="greedy_search",
                max_active_paths=4,
                provider="cpu",
                debug=False,
            )
            
            init_time = time.time() - start_time
            logger.info(f"✅ Model initialized in {init_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            return False
    
    def generate_test_audio(self, text: str, duration: float = 3.0) -> np.ndarray:
        """Generate synthetic audio for testing (sine wave pattern based on text)"""
        # Simple synthetic audio generation for testing
        # In real testing, you'd use TTS or recorded audio
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Create a unique frequency pattern based on text hash
        freq = 220 + (hash(text) % 440)  # 220-660 Hz range
        audio = 0.1 * np.sin(2 * np.pi * freq * t)
        
        # Add some noise to make it more realistic
        noise = 0.01 * np.random.randn(samples)
        audio = audio + noise
        
        return audio.astype(np.float32)
    
    def test_single_phrase(self, text: str, audio: np.ndarray) -> dict:
        """Test recognition of a single phrase"""
        if not self.recognizer:
            return {"error": "Model not initialized"}
        
        try:
            # Create stream
            stream = self.recognizer.create_stream()
            
            # Measure processing time
            start_time = time.time()
            
            # Process audio in chunks to simulate streaming
            chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
            results = []
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                stream.accept_waveform(self.sample_rate, chunk.tolist())
                
                if self.recognizer.is_ready(stream):
                    self.recognizer.decode_streams([stream])
                
                # Get intermediate result
                result = self.recognizer.get_result(stream)
                if hasattr(result, 'text') and result.text.strip():
                    results.append(result.text.strip())
                
                # Check for endpoint
                if self.recognizer.is_endpoint(stream):
                    final_result = self.recognizer.get_result(stream)
                    final_text = final_result.text if hasattr(final_result, 'text') else str(final_result)
                    self.recognizer.reset(stream)
                    break
            else:
                # If no endpoint detected, get final result
                final_result = self.recognizer.get_result(stream)
                final_text = final_result.text if hasattr(final_result, 'text') else str(final_result)
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            audio_duration = len(audio) / self.sample_rate
            rtf = processing_time / audio_duration  # Real-time factor
            
            result_dict = {
                "original_text": text,
                "recognized_text": final_text.strip(),
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "rtf": rtf,  # Lower is better for real-time
                "intermediate_results": results,
                "success": True
            }
            
            logger.info(f"🎯 '{text[:30]}...' -> '{final_text}' (RTF: {rtf:.3f})")
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ Error testing phrase '{text}': {e}")
            return {
                "original_text": text,
                "recognized_text": "",
                "error": str(e),
                "success": False
            }
    
    def calculate_word_error_rate(self, original: str, recognized: str) -> float:
        """Calculate Word Error Rate (WER)"""
        # Simple WER calculation
        original_words = original.lower().split()
        recognized_words = recognized.lower().split()
        
        if not original_words:
            return 0.0 if not recognized_words else 1.0
        
        # Simple edit distance approximation
        if original_words == recognized_words:
            return 0.0
        
        # For simplicity, just count word differences
        max_len = max(len(original_words), len(recognized_words))
        if max_len == 0:
            return 0.0
        
        different = sum(1 for i in range(min(len(original_words), len(recognized_words)))
                       if original_words[i] != recognized_words[i])
        different += abs(len(original_words) - len(recognized_words))
        
        return different / len(original_words)
    
    def run_comprehensive_test(self) -> dict:
        """Run comprehensive test suite"""
        if not self.initialize_model():
            return {"error": "Failed to initialize model"}
        
        logger.info(f"🧪 Starting comprehensive test with {len(TEST_PHRASES)} phrases")
        
        all_results = []
        total_wer = 0
        successful_tests = 0
        total_rtf = 0
        
        for i, phrase in enumerate(TEST_PHRASES):
            logger.info(f"📝 Test {i+1}/{len(TEST_PHRASES)}: Testing phrase...")
            
            # Generate test audio (in real scenario, use actual audio files)
            audio = self.generate_test_audio(phrase)
            
            # Test the phrase
            result = self.test_single_phrase(phrase, audio)
            all_results.append(result)
            
            if result.get("success", False):
                successful_tests += 1
                wer = self.calculate_word_error_rate(
                    result["original_text"], 
                    result["recognized_text"]
                )
                result["wer"] = wer
                total_wer += wer
                total_rtf += result["rtf"]
            
        # Calculate overall metrics
        avg_wer = total_wer / successful_tests if successful_tests > 0 else 1.0
        avg_rtf = total_rtf / successful_tests if successful_tests > 0 else float('inf')
        success_rate = successful_tests / len(TEST_PHRASES)
        
        summary = {
            "model_dir": str(self.model_dir),
            "language": self.language,
            "total_tests": len(TEST_PHRASES),
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_wer": avg_wer,
            "average_rtf": avg_rtf,
            "detailed_results": all_results
        }
        
        logger.info(f"📊 Test Summary:")
        logger.info(f"  Success Rate: {success_rate:.1%}")
        logger.info(f"  Average WER: {avg_wer:.3f}")
        logger.info(f"  Average RTF: {avg_rtf:.3f}")
        logger.info(f"  {'🚀 REAL-TIME' if avg_rtf < 1.0 else '🐌 SLOWER THAN REAL-TIME'}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Test Kroko-ASR model performance")
    parser.add_argument("--model-dir", 
                       default="./models/kroko-asr-en",
                       help="Path to Kroko-ASR model directory")
    parser.add_argument("--language", 
                       default="en",
                       choices=["en", "fr", "es"],
                       help="Language to test")
    parser.add_argument("--output", 
                       help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Create tester
    tester = KrokoModelTester(args.model_dir, args.language)
    
    # Run tests
    logger.info(f"🎯 Testing Kroko-ASR model: {args.language}")
    results = tester.run_comprehensive_test()
    
    if "error" in results:
        logger.error(f"❌ Test failed: {results['error']}")
        return 1
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {args.output}")
    
    # Final assessment
    if results["average_rtf"] < 0.5:
        logger.info("🚀 EXCELLENT: Ultra-low latency achieved!")
    elif results["average_rtf"] < 1.0:
        logger.info("✅ GOOD: Real-time processing achieved")
    else:
        logger.warning("⚠️ SLOW: Not suitable for real-time streaming")
    
    if results["average_wer"] < 0.05:
        logger.info("🎯 EXCELLENT: Very high accuracy")
    elif results["average_wer"] < 0.15:
        logger.info("👍 GOOD: Acceptable accuracy")
    else:
        logger.warning("⚠️ POOR: Accuracy needs improvement")
    
    return 0

if __name__ == "__main__":
    exit(main())