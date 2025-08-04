#!/usr/bin/env python3
"""
Live voice recognition test with real audio recording and public audio samples.
Tests speaker enrollment and differentiation between different voices.
"""
import asyncio
import numpy as np
import pyaudio
import wave
import os
import sys
import time
from datetime import datetime
import urllib.request
import tempfile
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import VoiceRecognitionConfig
from voice_recognition.auto_enroll import AutoEnrollVoiceRecognition

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Public domain audio samples
PUBLIC_AUDIO_SAMPLES = [
    {
        "name": "MLK Dream Speech",
        "url": "https://ia801605.us.archive.org/25/items/MLKDream/MLKDream.wav",
        "description": "Martin Luther King Jr. - I Have a Dream speech excerpt"
    }
]

class VoiceRecognitionTester:
    def __init__(self):
        self.config = VoiceRecognitionConfig()
        self.voice_recognition = AutoEnrollVoiceRecognition(self.config)
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
    async def initialize(self):
        """Initialize voice recognition system"""
        await self.voice_recognition.initialize()
        
        # Set up event handlers
        self.voice_recognition.set_callbacks(
            on_speaker_changed=self.on_speaker_changed,
            on_speaker_enrolled=self.on_speaker_enrolled
        )
        
    async def on_speaker_changed(self, data):
        """Handle speaker change events"""
        speaker = data['speaker_name']
        confidence = data['confidence']
        print(f"\n🎯 Speaker identified: {speaker} (confidence: {confidence:.2f})")
        
    async def on_speaker_enrolled(self, data):
        """Handle speaker enrollment events"""
        speaker_id = data['speaker_id']
        consistency = data.get('consistency', 0)
        print(f"\n✨ New speaker enrolled: {speaker_id} (consistency: {consistency:.2f})")
        print("This is a new voice that hasn't been heard before!")
        
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        
        p = pyaudio.PyAudio()
        
        # Find the default input device
        try:
            default_device = p.get_default_input_device_info()
            print(f"Using microphone: {default_device['name']}")
        except:
            print("No default input device found!")
            return None
            
        stream = p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            remaining = int(duration - (time.time() - start_time))
            print(f"\r⏱️  Recording... {remaining}s remaining", end='', flush=True)
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
            
        print("\n✅ Recording complete!")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Convert to numpy array
        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_array
    
    def download_public_audio(self, url, name):
        """Download and convert public audio sample"""
        print(f"\n📥 Downloading {name}...")
        
        try:
            # Download to temp file
            suffix = '.wav' if url.endswith('.wav') else '.ogg'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                urllib.request.urlretrieve(url, tmp_file.name)
                temp_path = tmp_file.name
                
            # If already WAV, try to read directly
            if suffix == '.wav':
                try:
                    with wave.open(temp_path, 'rb') as wf:
                        # Check if we need to resample
                        orig_rate = wf.getframerate()
                        orig_channels = wf.getnchannels()
                        
                        if orig_rate == self.sample_rate and orig_channels == 1:
                            # Direct read
                            frames = wf.readframes(wf.getnframes())
                            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        else:
                            # Need resampling - use scipy if available
                            print(f"   Original: {orig_rate}Hz, {orig_channels} channels")
                            print(f"   Converting to: {self.sample_rate}Hz, mono")
                            
                            frames = wf.readframes(wf.getnframes())
                            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            # If stereo, convert to mono
                            if orig_channels == 2:
                                audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                            
                            # Simple resampling if needed
                            if orig_rate != self.sample_rate:
                                try:
                                    from scipy import signal
                                    # Resample
                                    num_samples = int(len(audio_array) * self.sample_rate / orig_rate)
                                    audio_array = signal.resample(audio_array, num_samples)
                                except ImportError:
                                    # Basic resampling without scipy
                                    duration = len(audio_array) / orig_rate
                                    num_samples = int(duration * self.sample_rate)
                                    indices = np.linspace(0, len(audio_array)-1, num_samples).astype(int)
                                    audio_array = audio_array[indices]
                    
                    os.unlink(temp_path)
                    
                    # Take only first 5 seconds for testing
                    max_samples = 5 * self.sample_rate
                    if len(audio_array) > max_samples:
                        audio_array = audio_array[:max_samples]
                    
                    print(f"✅ Downloaded and converted {name}")
                    return audio_array
                    
                except Exception as e:
                    logger.error(f"Error reading WAV file: {e}")
                    os.unlink(temp_path)
                    return None
            else:
                # For non-WAV files, would need ffmpeg
                print(f"⚠️  Non-WAV format requires ffmpeg")
                os.unlink(temp_path)
                return None
                
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return None
    
    async def test_utterance(self, audio_array, description):
        """Test a single utterance through voice recognition"""
        print(f"\n🔊 Processing: {description}")
        print(f"   Audio shape: {audio_array.shape}, duration: {len(audio_array)/self.sample_rate:.2f}s")
        
        # Simulate VAD events
        await self.voice_recognition.on_user_started_speaking()
        
        # Process audio in chunks (simulate real-time)
        chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i:i+chunk_size]
            if len(chunk) > 0:
                # Create a mock frame object
                class MockFrame:
                    def __init__(self, audio):
                        self.audio = (chunk * 32768).astype(np.int16).tobytes()
                
                await self.voice_recognition.process_audio_frame(MockFrame(chunk), self.sample_rate)
                await asyncio.sleep(0.01)  # Small delay to simulate real-time
        
        # Simulate end of speech
        await self.voice_recognition.on_user_stopped_speaking()
        
        # Give time for processing
        await asyncio.sleep(0.5)
        
    async def run_test(self):
        """Run the complete test sequence"""
        print("=" * 60)
        print("🎙️  VOICE RECOGNITION LIVE TEST")
        print("=" * 60)
        
        await self.initialize()
        
        # Step 1: Record user's voice multiple times for enrollment
        print("\n📝 STEP 1: Enrolling your voice")
        print("We'll record 3 samples of your voice for enrollment.")
        print("Please say different things each time.\n")
        
        user_samples = []
        for i in range(3):
            input(f"Press ENTER to start recording {i+1}/3...")
            audio = self.record_audio(duration=4)
            if audio is not None:
                user_samples.append(audio)
                await self.test_utterance(audio, f"Your voice sample {i+1}")
            await asyncio.sleep(1)
        
        # Step 2: Test with the same voice (should recognize)
        print("\n📝 STEP 2: Testing recognition of your voice")
        input("\nPress ENTER to record a test sample (should recognize you)...")
        test_audio = self.record_audio(duration=4)
        if test_audio is not None:
            await self.test_utterance(test_audio, "Your test voice (should be recognized)")
        
        # Step 3: Test with different voices (should detect as different speakers)
        print("\n📝 STEP 3: Testing with different voices")
        print("We'll now test with public audio samples to verify speaker differentiation.\n")
        
        for sample in PUBLIC_AUDIO_SAMPLES:
            audio = self.download_public_audio(sample['url'], sample['name'])
            if audio is not None:
                await self.test_utterance(audio, sample['description'])
                await asyncio.sleep(2)
            else:
                print(f"⚠️  Skipping {sample['name']} - download failed")
        
        # Step 4: Final test with your voice again
        print("\n📝 STEP 4: Final recognition test")
        input("\nPress ENTER for final recording (should still recognize you)...")
        final_audio = self.record_audio(duration=4)
        if final_audio is not None:
            await self.test_utterance(final_audio, "Your final voice sample")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        if hasattr(self.voice_recognition, 'speakers'):
            print(f"\nTotal speakers enrolled: {len(self.voice_recognition.speakers)}")
            for speaker_name in self.voice_recognition.speakers:
                print(f"  - {speaker_name}")
        
        print("\n✅ Test complete!")
        
async def main():
    """Main test function"""
    tester = VoiceRecognitionTester()
    
    try:
        await tester.run_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import pyaudio
    except ImportError:
        print("ERROR: pyaudio not installed. Install with: pip install pyaudio")
        sys.exit(1)
        
    asyncio.run(main())