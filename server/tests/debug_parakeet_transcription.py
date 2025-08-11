#!/usr/bin/env python3
"""
Debug script to investigate why Parakeet-MLX isn't producing transcription output
"""

import asyncio
import wave
import numpy as np
from services.parakeet_mlx_streaming_stt import ParakeetMLXStreamingSTTService

async def debug_parakeet():
    print("🔍 Debugging Parakeet-MLX Transcription Output")
    
    # Initialize service with debug logging
    service = ParakeetMLXStreamingSTTService(
        model_name="mlx-community/parakeet-tdt-0.6b-v2",
        context_size=(64, 64),
        chunk_size_ms=1000,
        language="en"
    )
    
    # Load test audio
    audio_file = "test_audio/benchmark_15s.wav"
    print(f"📁 Loading audio: {audio_file}")
    
    try:
        with wave.open(audio_file, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            
        print(f"🎵 Audio loaded: {len(frames)} bytes, {sample_rate}Hz")
        
        # Process through STT with detailed logging
        print("🔄 Processing through Parakeet-MLX...")
        frame_count = 0
        text_results = []
        
        async for frame in service.run_stt(frames):
            frame_count += 1
            print(f"📦 Frame {frame_count}: {type(frame).__name__}")
            
            if hasattr(frame, 'text'):
                print(f"    📝 Text: '{frame.text}'")
                if frame.text and frame.text.strip():
                    text_results.append(frame.text.strip())
            
            if hasattr(frame, 'user_id'):
                print(f"    👤 User ID: {frame.user_id}")
            
            if hasattr(frame, 'timestamp'):
                print(f"    ⏰ Timestamp: {frame.timestamp}")
        
        print(f"\n📊 RESULTS:")
        print(f"   Frames received: {frame_count}")
        print(f"   Text results: {len(text_results)}")
        print(f"   Combined text: '{' '.join(text_results)}'")
        
        # Clean up
        await service.cleanup()
        
        return text_results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    asyncio.run(debug_parakeet())