#!/usr/bin/env python3
"""
Test improved Parakeet streaming implementation
"""

import asyncio
import numpy as np
import time
from services.parakeet_mlx_streaming_stt import ParakeetMLXStreamingSTTService

async def test_improved_streaming():
    """Test our improved Parakeet streaming implementation"""
    
    print("Testing improved Parakeet streaming...")
    
    # Initialize service
    service = ParakeetMLXStreamingSTTService(
        model_name="mlx-community/parakeet-tdt-0.6b-v2",
        context_size=(64, 64),  # Smaller context for faster finalization
        chunk_size_ms=100,
        emit_partial_results=True,
        min_confidence=0.1,
        sample_rate=16000,
        language="en"
    )
    
    # Generate test audio (3 seconds of speech-like sound)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # More complex speech-like pattern
    f0 = 150 + 50 * np.sin(2 * np.pi * 0.7 * t)
    f1 = 800 + 300 * np.sin(2 * np.pi * 0.4 * t)
    audio = 0.2 * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f1 * t))
    
    # Add some variation
    for i in range(0, len(audio), sample_rate):
        if i // sample_rate % 2 == 0:
            audio[i:i+sample_rate] *= 0.7
    
    audio = audio.astype(np.float32)
    
    print(f"Generated {len(audio)} samples ({duration}s) of test audio")
    print(f"Audio stats - min: {audio.min():.3f}, max: {audio.max():.3f}, rms: {np.sqrt(np.mean(audio**2)):.3f}")
    
    # Process in small chunks like production (304 samples)
    chunk_size = 304
    results = []
    
    print("\n=== Processing audio in 304-sample chunks (production size) ===")
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        
        # Convert to 16-bit PCM bytes
        chunk_int16 = (chunk * 32767).astype(np.int16)
        chunk_bytes = chunk_int16.tobytes()
        
        # Process chunk
        async for frame in service.run_stt(chunk_bytes):
            if hasattr(frame, 'text'):
                result_type = "FINAL" if hasattr(frame, '__class__') and frame.__class__.__name__ == 'TranscriptionFrame' else "INTERIM"
                print(f"[{i//chunk_size:3d}] {result_type}: '{frame.text}'")
                results.append((result_type, frame.text))
        
        # Small delay to simulate real-time
        await asyncio.sleep(0.001)
    
    # Flush any remaining results
    print("\n=== Flushing remaining results ===")
    final = await service.flush()
    if final:
        print(f"FLUSH: '{final}'")
        results.append(("FLUSH", final))
    
    # Cleanup
    await service.cleanup()
    
    print("\n=== Summary ===")
    print(f"Total results: {len(results)}")
    final_results = [r[1] for r in results if r[0] == "FINAL"]
    interim_results = [r[1] for r in results if r[0] == "INTERIM"]
    print(f"Final results: {len(final_results)}")
    print(f"Interim results: {len(interim_results)}")
    
    if final_results:
        print(f"Last final transcription: '{final_results[-1]}'")
    elif interim_results:
        print(f"Last interim transcription: '{interim_results[-1]}'")
    else:
        print("No transcription produced!")

if __name__ == "__main__":
    asyncio.run(test_improved_streaming())