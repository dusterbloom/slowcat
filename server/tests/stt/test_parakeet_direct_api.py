#!/usr/bin/env python3
"""
Direct test of Parakeet-MLX streaming API to understand its behavior
"""

import numpy as np
import mlx.core as mx
from parakeet_mlx import from_pretrained
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_streaming():
    """Test the direct Parakeet-MLX streaming API"""
    
    # Load model
    logger.info("Loading Parakeet model...")
    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
    
    # Generate test audio (1 second of speech-like sound)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Speech-like pattern
    f0 = 150 + 50 * np.sin(2 * np.pi * 0.7 * t)
    audio = 0.3 * np.sin(2 * np.pi * f0 * t)
    audio = audio.astype(np.float32)
    
    logger.info(f"Generated {len(audio)} samples of test audio")
    logger.info(f"Audio stats - min: {audio.min():.3f}, max: {audio.max():.3f}, mean: {audio.mean():.3f}")
    
    # Test streaming
    logger.info("\n=== Testing Streaming Mode ===")
    with model.transcribe_stream(context_size=(256, 256)) as stream:
        # Process in chunks
        chunk_size = 1600  # 100ms chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk if needed
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            
            logger.info(f"\nProcessing chunk {i//chunk_size + 1}: {len(chunk)} samples")
            
            # Convert to MLX array
            mlx_chunk = mx.array(chunk)
            logger.info(f"MLX chunk shape: {mlx_chunk.shape}, dtype: {mlx_chunk.dtype}")
            
            # Add audio to stream
            stream.add_audio(mlx_chunk)
            
            # Check result
            result = stream.result
            logger.info(f"Current result: {result}")
            
            if hasattr(result, 'text'):
                logger.info(f"Text so far: '{result.text}'")
            
            # Check tokens
            logger.info(f"Finalized tokens: {len(stream.finalized_tokens)}")
            logger.info(f"Draft tokens: {len(stream.draft_tokens)}")
            
            if stream.finalized_tokens:
                logger.info(f"First finalized token: {stream.finalized_tokens[0]}")
            if stream.draft_tokens:
                logger.info(f"First draft token: {stream.draft_tokens[0]}")
    
    logger.info("\n=== Testing Non-Streaming Mode ===")
    # For comparison, test non-streaming
    result = model.transcribe("test_audio.wav")  # Would need to save audio first
    logger.info(f"Non-streaming result: {result}")

if __name__ == "__main__":
    test_streaming()