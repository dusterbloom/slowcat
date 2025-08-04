"""
Real audio player processor with PyAudio playback
Handles actual music playback, mixing, and queue management
"""

import asyncio
import threading
import queue
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from collections import deque
import numpy as np
import time
from loguru import logger

# Audio libraries
try:
    import pyaudio
    import soundfile as sf
    import librosa
    import resampy
    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Audio libraries not available: {e}")
    AUDIO_LIBS_AVAILABLE = False

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    OutputAudioRawFrame,
    InputAudioRawFrame,
    SystemFrame,
    TextFrame,
    ErrorFrame
)
from pipecat.processors.frame_processor import FrameProcessor


class MusicControlFrame(SystemFrame):
    """Frame for music control commands"""
    def __init__(self, command: str, data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.command = command  # play, pause, skip, queue, volume
        self.data = data or {}


class AudioPlayerRealProcessor(FrameProcessor):
    """
    Real audio player that plays music through PyAudio
    """
    
    def __init__(
        self,
        *,
        sample_rate: int = 16000,  # Pipeline sample rate
        channels: int = 1,
        initial_volume: float = 0.7,
        duck_volume: float = 0.3,
        crossfade_seconds: float = 2.0,
        buffer_size: int = 512,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not AUDIO_LIBS_AVAILABLE:
            logger.error("Audio libraries not available! Music playback disabled.")
            return
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.volume = initial_volume
        self.duck_volume = duck_volume
        self.normal_volume = initial_volume
        self.crossfade_duration = crossfade_seconds
        self.buffer_size = buffer_size
        
        # PyAudio setup
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Playback state
        self.is_playing = False
        self.is_paused = False
        self.is_voice_active = False
        self.current_song: Optional[Dict] = None
        self.current_audio_data: Optional[np.ndarray] = None
        self.playback_position = 0  # Sample position
        
        # Queue management
        self.play_queue: deque = deque()
        self.play_history: List[Dict] = []
        
        # Threading for playback
        self.playback_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.command_queue = queue.Queue()
        
        # Audio mixing buffer
        self.music_buffer = queue.Queue(maxsize=100)
        
        logger.info(f"🎵 Real AudioPlayer initialized: {sample_rate}Hz, volume={initial_volume}")
        
        # Start playback thread
        self._start_playback_thread()
    
    def _start_playback_thread(self):
        """Start the background playback thread"""
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
    
    def _playback_worker(self):
        """Background thread that handles audio playback"""
        logger.info("🎵 Playback thread started")
        
        while not self.stop_event.is_set():
            try:
                # Check for commands
                try:
                    command, data = self.command_queue.get(timeout=0.01)
                    self._handle_command_sync(command, data)
                except queue.Empty:
                    pass
                
                # Generate audio if playing
                if self.is_playing and not self.is_paused and self.current_audio_data is not None:
                    # Calculate how many samples we need
                    samples_needed = self.buffer_size
                    
                    # Get the next chunk of audio
                    start = self.playback_position
                    end = min(start + samples_needed, len(self.current_audio_data))
                    
                    if start < len(self.current_audio_data):
                        # Get audio chunk
                        chunk = self.current_audio_data[start:end]
                        
                        # Apply volume
                        chunk = chunk * self.volume
                        
                        # Update position
                        self.playback_position = end
                        
                        # Add to music buffer for mixing
                        try:
                            self.music_buffer.put(chunk, timeout=0.01)
                        except queue.Full:
                            pass
                        
                        # Check if song ended
                        if end >= len(self.current_audio_data):
                            logger.info("🎵 Song ended, moving to next")
                            self._next_song_sync()
                    else:
                        # Song ended
                        self._next_song_sync()
                else:
                    # Not playing, sleep a bit
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in playback thread: {e}")
                time.sleep(0.1)
        
        logger.info("🎵 Playback thread stopped")
    
    def _handle_command_sync(self, command: str, data: Dict):
        """Handle commands in the playback thread"""
        if command == "play":
            if "file_path" in data:
                self._load_and_play_sync(data)
            else:
                self.is_paused = False
        
        elif command == "pause":
            self.is_paused = True
        
        elif command == "skip":
            self._next_song_sync()
        
        elif command == "stop":
            self.is_playing = False
            self.current_audio_data = None
            self.playback_position = 0
        
        elif command == "volume":
            self.normal_volume = data.get("level", 0.7)
            if not self.is_voice_active:
                self.volume = self.normal_volume
    
    def _load_and_play_sync(self, song_info: Dict):
        """Load and start playing a song (sync version for thread)"""
        try:
            file_path = song_info.get("file_path")
            if not file_path or not Path(file_path).exists():
                logger.error(f"File not found: {file_path}")
                return
            
            logger.info(f"🎵 Loading: {file_path}")
            
            # Load audio file
            audio_data, file_sr = sf.read(file_path, dtype='float32')
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to pipeline sample rate if needed
            if file_sr != self.sample_rate:
                logger.info(f"Resampling from {file_sr}Hz to {self.sample_rate}Hz")
                audio_data = resampy.resample(audio_data, file_sr, self.sample_rate)
            
            # Normalize
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.8  # Leave some headroom
            
            # Set current song
            self.current_audio_data = audio_data
            self.current_song = song_info
            self.playback_position = 0
            self.is_playing = True
            self.is_paused = False
            
            # Add to history
            self.play_history.append(song_info)
            if len(self.play_history) > 100:
                self.play_history.pop(0)
            
            logger.info(f"🎵 Now playing: {song_info.get('title', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
    
    def _next_song_sync(self):
        """Move to next song in queue (sync version)"""
        if self.play_queue:
            next_song = self.play_queue.popleft()
            self._load_and_play_sync(next_song)
        else:
            self.is_playing = False
            self.current_audio_data = None
            logger.info("🎵 Queue empty, playback stopped")
    
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames - handle music control and mix audio"""
        
        # Let parent handle system frames
        await super().process_frame(frame, direction)
        
        # Handle music control commands
        if isinstance(frame, MusicControlFrame):
            await self._handle_control(frame)
            return  # Don't forward control frames
        
        # Detect voice activity for ducking
        if isinstance(frame, SystemFrame):
            if frame.__class__.__name__ == "UserStartedSpeakingFrame":
                await self._start_ducking()
            elif frame.__class__.__name__ == "UserStoppedSpeakingFrame":
                await self._stop_ducking()
        
        # Mix music with voice frames
        if isinstance(frame, (InputAudioRawFrame, AudioRawFrame)) and frame.audio:
            mixed_frame = await self._mix_audio(frame)
            await self.push_frame(mixed_frame, direction)
        else:
            # Forward non-audio frames
            await self.push_frame(frame, direction)
    
    async def _handle_control(self, control: MusicControlFrame):
        """Handle music control commands"""
        command = control.command
        data = control.data
        
        logger.info(f"🎵 Music control: {command}")
        
        # Send command to playback thread
        self.command_queue.put((command, data))
        
        # Send immediate feedback for some commands
        if command == "play" and "file_path" in data:
            await self.push_frame(TextFrame(
                f"🎵 Loading: {data.get('title', 'song')}..."
            ))
        elif command == "skip":
            await self.push_frame(TextFrame("⏭️ Skipping to next song"))
        elif command == "pause":
            await self.push_frame(TextFrame("⏸️ Music paused"))
    
    async def _mix_audio(self, voice_frame: AudioRawFrame) -> Frame:
        """Mix voice with music"""
        # Get voice audio
        voice_audio = np.frombuffer(voice_frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Try to get music audio
        music_audio = None
        try:
            music_chunk = self.music_buffer.get_nowait()
            # Ensure same length
            if len(music_chunk) < len(voice_audio):
                music_chunk = np.pad(music_chunk, (0, len(voice_audio) - len(music_chunk)))
            elif len(music_chunk) > len(voice_audio):
                music_chunk = music_chunk[:len(voice_audio)]
            music_audio = music_chunk
        except queue.Empty:
            pass
        
        # Mix if we have music
        if music_audio is not None:
            # Mix with current volume (ducking applied)
            mixed = voice_audio + music_audio * 0.5  # Reduce music in mix
            # Prevent clipping
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val
        else:
            mixed = voice_audio
        
        # Convert back to int16
        mixed_int16 = (mixed * 32768).astype(np.int16)
        
        # Return the same type of frame we received
        if isinstance(voice_frame, InputAudioRawFrame):
            # Keep it as input frame
            new_frame = InputAudioRawFrame(
                audio=mixed_int16.tobytes(),
                sample_rate=voice_frame.sample_rate,
                num_channels=voice_frame.num_channels
            )
            # Copy source info if available
            if hasattr(voice_frame, 'transport_source'):
                new_frame.transport_source = voice_frame.transport_source
            return new_frame
        else:
            # Generic AudioRawFrame
            return AudioRawFrame(
                audio=mixed_int16.tobytes(),
                sample_rate=voice_frame.sample_rate,
                num_channels=voice_frame.num_channels
            )
    
    async def _start_ducking(self):
        """Duck music volume when voice is active"""
        self.is_voice_active = True
        self.volume = self.duck_volume
        logger.debug(f"🔉 Ducking volume to {self.duck_volume}")
    
    async def _stop_ducking(self):
        """Restore music volume when voice stops"""
        self.is_voice_active = False
        self.volume = self.normal_volume
        logger.debug(f"🔊 Restoring volume to {self.normal_volume}")
    
    async def queue_song(self, song_info: Dict):
        """Add song to queue"""
        self.play_queue.append(song_info)
        logger.info(f"➕ Queued: {song_info.get('title', 'Unknown')}")
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get current queue information"""
        current_duration = 0
        if self.current_audio_data is not None:
            current_duration = len(self.current_audio_data) / self.sample_rate
            current_position = self.playback_position / self.sample_rate
        else:
            current_position = 0
        
        return {
            "current": self.current_song,
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "position_seconds": current_position,
            "duration_seconds": current_duration,
            "queue_length": len(self.play_queue),
            "queue": list(self.play_queue)[:5],
            "volume": int(self.normal_volume * 100)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🎵 Cleaning up audio player")
        
        # Stop playback thread
        self.stop_event.set()
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        
        # Close PyAudio
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
        
        await super().cleanup()