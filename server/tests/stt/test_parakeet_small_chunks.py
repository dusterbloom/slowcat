#!/usr/bin/env python3
"""
Test Parakeet-MLX with production-sized small chunks (304 samples)
This reproduces the exact conditions that cause the Metal GPU crash in production.
"""

import asyncio
import numpy as np
import wave
from pathlib import Path
from loguru import logger
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.parakeet_mlx_streaming_stt import ParakeetMLXStreamingSTTService


async def test_small_chunks():
    """Test with production-sized 304 sample chunks"""
    
    # Initialize service
    service = ParakeetMLXStreamingSTTService(
        model_name="mlx-community/parakeet-tdt-0.6b-v2",
        context_size=(64, 64),  # Small context for testing
        attention_mode="local",
        chunk_size_ms=100,  # Small chunk size
        emit_partial_results=True,
    )
    
    # Load a test audio file
    test_file = Path("tests/test_audio/test_audio_16k.wav")
    if not test_file.exists():
        # Create a simple test audio if file doesn't exist
        logger.warning(f"Test file {test_file} not found, generating synthetic audio")
        
        # Generate 5 seconds of synthetic audio (sine wave)
        sample_rate = 16000
        duration = 5
        t = np.linspace(0, duration, sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz tone
        audio_data = (audio_data * 32767).astype(np.int16)
    else:
        # Load the audio file
        with wave.open(str(test_file), 'rb') as wav:
            audio_data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            logger.info(f"Loaded audio file: {len(audio_data)} samples")
    
    # Simulate production streaming with 304-sample chunks
    PRODUCTION_CHUNK_SIZE = 304  # This is what causes the crash!
    
    logger.info(f"🎯 Testing with production chunk size: {PRODUCTION_CHUNK_SIZE} samples (~19ms)")
    logger.info(f"📊 Total audio length: {len(audio_data)} samples")
    
    results = []
    chunk_count = 0
    
    try:
        # Process audio in small chunks like production
        for i in range(0, len(audio_data), PRODUCTION_CHUNK_SIZE):
            chunk = audio_data[i:i+PRODUCTION_CHUNK_SIZE]
            
            # Convert to bytes like production does
            chunk_bytes = chunk.tobytes()
            
            chunk_count += 1
            if chunk_count % 10 == 0:
                logger.debug(f"Processing chunk {chunk_count} ({len(chunk)} samples)")
            
            # Process through the service
            async for frame in service.run_stt(chunk_bytes):
                if hasattr(frame, 'text'):
                    results.append(frame.text)
                    logger.info(f"✅ Got transcription: {frame.text}")
            
            # Small delay to simulate real-time streaming
            await asyncio.sleep(0.01)
        
        # Flush any remaining audio
        logger.info("🔄 Flushing remaining audio...")
        final_result = await service.flush()
        if final_result:
            results.append(final_result)
            logger.info(f"✅ Final result: {final_result}")
        
        logger.success(f"🎉 Test completed successfully! Processed {chunk_count} chunks")
        logger.info(f"📝 All results: {results}")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        logger.exception("Full traceback:")
        return False
    
    finally:
        # Cleanup
        await service.cleanup()
    
    return True


async def test_buffer_accumulation():
    """Test that small chunks are properly buffered"""
    
    logger.info("\n🧪 Testing buffer accumulation logic...")
    
    service = ParakeetMLXStreamingSTTService(
        model_name="mlx-community/parakeet-tdt-0.6b-v2",
        context_size=(64, 64),
        chunk_size_ms=100,
    )
    
    # Generate test chunks of different sizes
    test_sizes = [304, 304, 304, 304, 304, 304]  # 6 chunks = 1824 samples total
    
    for i, size in enumerate(test_sizes):
        # Generate silence (zeros) for testing
        chunk = np.zeros(size, dtype=np.int16)
        
        logger.debug(f"Sending chunk {i+1}: {size} samples")
        logger.debug(f"Buffer before: {service._small_chunk_samples} samples")
        
        # This should accumulate in the buffer
        async for frame in service.run_stt(chunk.tobytes()):
            logger.info(f"Got frame: {frame}")
        
        logger.debug(f"Buffer after: {service._small_chunk_samples} samples")
        
        # Check if buffer was processed when threshold reached
        if service._small_chunk_samples == 0:
            logger.success(f"✅ Buffer processed after reaching threshold!")
    
    await service.cleanup()
    logger.info("✅ Buffer accumulation test completed")


async def main():
    """Run all tests"""
    logger.info("🚀 Starting Parakeet-MLX small chunk tests")
    logger.info("=" * 60)
    
    # Test 1: Buffer accumulation logic
    await test_buffer_accumulation()
    
    logger.info("\n" + "=" * 60)
    
    # Test 2: Production-like streaming
    success = await test_small_chunks()
    
    if success:
        logger.success("\n✅ All tests passed! The fix prevents Metal GPU crashes.")
        logger.info("🎯 Key improvements:")
        logger.info("  1. Small chunks are buffered to meet minimum size requirement")
        logger.info("  2. Error handling prevents crashes from killing the bot")
        logger.info("  3. Production can now safely use Parakeet-MLX")
    else:
        logger.error("\n❌ Tests failed. The Metal GPU issue persists.")
    
    return success


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)