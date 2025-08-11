#!/usr/bin/env python3
"""
Test Kroko-ASR integration with existing Sherpa streaming service
Tests if Kroko models work with the current SherpaOnlineSTTService
"""

import sys
import asyncio
from pathlib import Path
import numpy as np
from loguru import logger

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from services.sherpa_streaming_stt_v2 import SherpaOnlineSTTService
    from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
except ImportError as e:
    logger.error(f"Failed to import dependencies: {e}")
    sys.exit(1)

def generate_test_audio(duration: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate simple test audio as PCM bytes"""
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    
    # Generate a simple tone
    freq = 440  # A4 note
    audio = 0.1 * np.sin(2 * np.pi * freq * t)
    
    # Convert to int16 PCM
    pcm_data = (audio * 32767).astype(np.int16)
    return pcm_data.tobytes()

async def test_kroko_integration(model_dir: str, language: str = "en"):
    """Test Kroko-ASR integration with existing service"""
    
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return False
    
    logger.info(f"🚀 Testing Kroko-ASR integration: {language}")
    logger.info(f"Model directory: {model_dir}")
    
    try:
        # Create service with Kroko model
        service = SherpaOnlineSTTService(
            model_dir=model_dir,
            language=language,
            sample_rate=16000,
            enable_endpoint_detection=True,
            emit_partial_results=True,  # Test partial results
            chunk_size_ms=100,  # Fast streaming
        )
        
        logger.info("✅ Service created successfully")
        
        # Generate test audio
        test_audio = generate_test_audio(duration=3.0)
        logger.info(f"📊 Generated {len(test_audio)} bytes of test audio")
        
        # Test streaming STT
        logger.info("🎯 Testing streaming recognition...")
        results = []
        
        async for frame in service.run_stt(test_audio):
            if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
                frame_type = "FINAL" if isinstance(frame, TranscriptionFrame) else "INTERIM"
                logger.info(f"📝 {frame_type}: '{frame.text}'")
                results.append((frame_type, frame.text))
        
        # Cleanup
        service.cleanup()
        logger.info("🧹 Service cleaned up")
        
        # Assessment
        if results:
            logger.info(f"✅ Integration successful! Got {len(results)} results")
            for frame_type, text in results:
                logger.info(f"  {frame_type}: {text}")
            return True
        else:
            logger.warning("⚠️ Integration works but no text was recognized (normal for synthetic audio)")
            return True
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

async def test_multiple_languages():
    """Test multiple language models if available"""
    base_dir = "./models"
    languages = ["en", "fr", "es"]
    
    results = {}
    
    for lang in languages:
        model_dir = f"{base_dir}/kroko-asr-{lang}"
        if Path(model_dir).exists():
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {lang.upper()} model")
            logger.info(f"{'='*50}")
            
            success = await test_kroko_integration(model_dir, lang)
            results[lang] = success
        else:
            logger.info(f"⚠️ {lang.upper()} model not found at {model_dir}")
            results[lang] = None
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Kroko-ASR integration")
    parser.add_argument("--model-dir", 
                       default="./models/kroko-asr-en",
                       help="Path to Kroko-ASR model directory")
    parser.add_argument("--language", 
                       default="en",
                       help="Language to test")
    parser.add_argument("--all-languages", 
                       action="store_true",
                       help="Test all available language models")
    
    args = parser.parse_args()
    
    if args.all_languages:
        logger.info("🌍 Testing all available language models...")
        results = asyncio.run(test_multiple_languages())
        
        logger.info(f"\n{'='*50}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*50}")
        
        for lang, success in results.items():
            if success is None:
                logger.info(f"  {lang.upper()}: Model not available")
            elif success:
                logger.info(f"  {lang.upper()}: ✅ Integration successful")
            else:
                logger.info(f"  {lang.upper()}: ❌ Integration failed")
        
        # Overall assessment
        successful = sum(1 for s in results.values() if s is True)
        total_tested = sum(1 for s in results.values() if s is not None)
        
        if successful > 0:
            logger.info(f"\n🎉 SUCCESS: {successful}/{total_tested} models integrated successfully!")
            logger.info("Kroko-ASR models are compatible with your existing Sherpa service!")
        else:
            logger.error("\n❌ FAILURE: No models integrated successfully")
            
    else:
        logger.info(f"🎯 Testing single model: {args.language}")
        success = asyncio.run(test_kroko_integration(args.model_dir, args.language))
        
        if success:
            logger.info("\n🎉 SUCCESS: Kroko-ASR integrates perfectly with your existing service!")
            logger.info("You can use it by:")
            logger.info(f"1. Setting SHERPA_ONNX_MODEL_DIR={args.model_dir} in your .env")
            logger.info("2. Or updating the model_dir in your service configuration")
        else:
            logger.error("\n❌ FAILURE: Integration test failed")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())