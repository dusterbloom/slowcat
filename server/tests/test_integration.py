#!/usr/bin/env python3
"""
Test the voice recognition integration with pipecat
"""
import asyncio
import logging
from pipecat.frames.frames import InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from voice_recognition import AutoEnrollVoiceRecognition
from processors import AudioTeeProcessor, VADEventBridge

logging.basicConfig(level=logging.DEBUG)

async def test_integration():
    # Initialize voice recognition
    config = {
        "enabled": True,
        "profile_dir": "data/test_profiles",
        "confidence_threshold": 0.75,
        "min_utterance_duration_seconds": 0.5,
        "auto_enroll": {
            "min_utterances": 2,
            "consistency_threshold": 0.85
        }
    }
    
    vr = AutoEnrollVoiceRecognition(config)
    await vr.initialize()
    
    # Create processors
    audio_tee = AudioTeeProcessor(enabled=True)
    audio_tee.register_audio_consumer(vr.process_audio_frame)
    
    vad_bridge = VADEventBridge()
    vad_bridge.set_callbacks(
        on_started=vr.on_user_started_speaking,
        on_stopped=vr.on_user_stopped_speaking
    )
    
    # Set up callbacks
    def on_speaker_changed(data):
        print(f"✅ Speaker changed: {data}")
    
    vr.set_callbacks(
        on_speaker_changed=lambda data: asyncio.create_task(on_speaker_changed_async(data))
    )
    
    async def on_speaker_changed_async(data):
        on_speaker_changed(data)
    
    print("Testing voice recognition pipeline integration...")
    
    # Simulate VAD start
    await vad_bridge.process_frame(UserStartedSpeakingFrame(), None)
    
    # Simulate audio frames
    sample_audio = b'\x00\x01' * 8000  # 1 second of dummy audio at 16kHz
    for i in range(10):
        frame = InputAudioRawFrame(
            audio=sample_audio,
            sample_rate=16000,
            num_channels=1
        )
        await audio_tee.process_frame(frame, None)
    
    # Simulate VAD stop
    await vad_bridge.process_frame(UserStoppedSpeakingFrame(), None)
    
    # Wait for processing
    await asyncio.sleep(1)
    
    print("✅ Integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_integration())