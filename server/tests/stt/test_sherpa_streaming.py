#!/usr/bin/env python3
"""
🔥 Test script for Sherpa Streaming STT Service
Tests the InterimTranscriptionFrame functionality
"""

import asyncio
import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.sherpa_streaming_stt import SherpaStreamingSTTService
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame

async def test_streaming_stt():
    """Test the streaming STT service with synthetic audio"""
    
    print("🔥 Testing Sherpa Streaming STT Service...")
    
    # Configuration
    model_dir = "models/sherpa-nemo-10lang"  # Adjust path as needed
    if not Path(model_dir).exists():
        print(f"❌ Model directory not found: {model_dir}")
        print("Please ensure your Sherpa model is available!")
        return
    
    # Create streaming STT service
    stt_service = SherpaStreamingSTTService(
        model_dir=model_dir,
        language="en",
        sample_rate=16000,
        chunk_duration_ms=500,    # Process every 500ms
        overlap_duration_ms=200,  # 200ms overlap
        min_confidence=0.3        # Low threshold for testing
    )
    
    print("✅ Sherpa Streaming STT Service created")
    
    # Generate some synthetic audio data (silence + random noise)
    sample_rate = 16000
    duration_seconds = 3.0
    num_samples = int(sample_rate * duration_seconds)
    
    # Create synthetic audio (white noise at low volume)
    audio_samples = np.random.normal(0, 0.1, num_samples).astype(np.float32)
    
    # Convert to int16 PCM bytes
    audio_int16 = (audio_samples * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    print(f"🎵 Generated {len(audio_bytes)} bytes of synthetic audio ({duration_seconds}s)")
    
    # Process audio in chunks
    chunk_size = int(sample_rate * 0.1 * 2)  # 100ms chunks in bytes
    interim_count = 0
    final_count = 0
    
    print("🚀 Processing audio chunks...")
    
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        print(f"📦 Processing chunk {i//chunk_size + 1}, size: {len(chunk)} bytes")
        
        try:
            # Process the chunk
            async for frame in stt_service.run_stt(chunk):
                if isinstance(frame, InterimTranscriptionFrame):
                    interim_count += 1
                    print(f"🔄 INTERIM[{interim_count}]: '{frame.text}'")
                elif isinstance(frame, TranscriptionFrame):
                    final_count += 1
                    print(f"✅ FINAL[{final_count}]: '{frame.text}'")
                else:
                    print(f"🔍 OTHER FRAME: {type(frame).__name__}")
                    
        except Exception as e:
            print(f"❌ Error processing chunk: {e}")
    
    print(f"\n📊 RESULTS:")
    print(f"   Interim results: {interim_count}")
    print(f"   Final results: {final_count}")
    print(f"   Total frames: {interim_count + final_count}")
    
    # Cleanup
    stt_service.cleanup()
    print("🧹 Cleanup complete")

if __name__ == "__main__":
    print("🔥 SHERPA STREAMING STT TEST")
    print("=" * 50)
    
    try:
        asyncio.run(test_streaming_stt())
        print("\n✅ Test completed successfully!")
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()