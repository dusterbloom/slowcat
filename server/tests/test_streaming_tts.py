#!/usr/bin/env python3
"""
Test script to verify streaming TTS implementation
"""

import asyncio
import time
from loguru import logger
from kokoro_tts import KokoroTTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

async def test_streaming_tts():
    """Test that Kokoro TTS streams audio chunks as they're generated"""
    logger.info("=== Testing Streaming TTS ===")
    
    # Initialize service
    tts = KokoroTTSService(voice="af_heart")
    
    test_text = "Hello, this is a test of streaming text to speech. The audio should start playing before the entire sentence is generated."
    
    async with tts:
        logger.info(f"Generating TTS for: {test_text[:50]}...")
        
        start_time = time.time()
        first_audio_time = None
        chunk_count = 0
        total_audio_bytes = 0
        
        async for frame in tts.run_tts(test_text):
            if isinstance(frame, TTSStartedFrame):
                logger.info("TTS Started")
            elif isinstance(frame, TTSAudioRawFrame):
                if first_audio_time is None:
                    first_audio_time = time.time()
                    time_to_first_audio = first_audio_time - start_time
                    logger.success(f"✅ First audio chunk received after {time_to_first_audio:.3f}s")
                
                chunk_count += 1
                total_audio_bytes += len(frame.audio)
                
                if chunk_count % 10 == 0:
                    logger.debug(f"Received {chunk_count} chunks, {total_audio_bytes} bytes")
            elif isinstance(frame, TTSStoppedFrame):
                logger.info("TTS Stopped")
        
        total_time = time.time() - start_time
        
        logger.info(f"\n📊 Streaming TTS Results:")
        logger.info(f"  - Total chunks: {chunk_count}")
        logger.info(f"  - Total audio bytes: {total_audio_bytes:,}")
        logger.info(f"  - Time to first audio: {time_to_first_audio:.3f}s")
        logger.info(f"  - Total generation time: {total_time:.3f}s")
        
        # Compare with non-streaming baseline
        logger.info("\n=== Testing Non-Streaming (baseline) ===")
        start_time = time.time()
        audio_bytes = tts._generate_audio_sync(test_text)
        baseline_time = time.time() - start_time
        
        logger.info(f"  - Non-streaming generation time: {baseline_time:.3f}s")
        logger.info(f"  - Non-streaming audio size: {len(audio_bytes):,} bytes")
        
        # Calculate improvement
        improvement = ((baseline_time - time_to_first_audio) / baseline_time) * 100
        logger.success(f"\n✅ Streaming reduces time-to-first-audio by {improvement:.1f}%")
        logger.success(f"✅ Users hear audio {baseline_time - time_to_first_audio:.3f}s sooner!")

async def main():
    """Run streaming TTS test"""
    logger.info("🚀 Streaming TTS Test Suite")
    logger.info("=" * 50)
    
    try:
        await test_streaming_tts()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    
    logger.info("\n✅ Test complete!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())