#!/usr/bin/env python3
"""
Test script for voice recognition functionality
"""
import asyncio
import numpy as np
from voice_recognition import AutoEnrollVoiceRecognition

async def test_voice_recognition():
    config = {
        "enabled": True,
        "profile_dir": "data/test_speaker_profiles",
        "confidence_threshold": 0.75,
        "min_utterance_duration_seconds": 1.0,
        "auto_enroll": {
            "min_utterances": 3,
            "consistency_threshold": 0.85,
            "min_consistency_threshold": 0.70,
            "enrollment_window_minutes": 30,
            "new_speaker_grace_period_seconds": 60,
            "new_speaker_similarity_threshold": 0.65
        }
    }
    
    # Create voice recognition
    vr = AutoEnrollVoiceRecognition(config)
    
    # Set up callbacks
    def on_speaker_changed(data):
        print(f"Speaker changed: {data}")
    
    def on_speaker_enrolled(data):
        print(f"Speaker enrolled: {data}")
    
    vr.set_callbacks(
        on_speaker_changed=lambda data: asyncio.create_task(on_speaker_changed_async(data)),
        on_speaker_enrolled=lambda data: asyncio.create_task(on_speaker_enrolled_async(data))
    )
    
    async def on_speaker_changed_async(data):
        on_speaker_changed(data)
    
    async def on_speaker_enrolled_async(data):
        on_speaker_enrolled(data)
    
    # Initialize
    await vr.initialize()
    
    # Simulate some audio data (16kHz, 16-bit)
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    
    # Generate a simple sine wave as test audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    print("Testing voice recognition with simulated audio...")
    
    # Simulate user speaking
    await vr.on_user_started_speaking()
    
    # Process audio in chunks
    chunk_size = 1024
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i+chunk_size]
        vr.process_audio_frame(chunk)
    
    # Simulate user stopped speaking
    await vr.on_user_stopped_speaking()
    
    print("Test complete!")
    
    # Shutdown
    await vr.shutdown()

if __name__ == "__main__":
    asyncio.run(test_voice_recognition())