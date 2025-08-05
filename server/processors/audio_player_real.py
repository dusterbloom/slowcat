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
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.volume = initial_volume
        self.duck_volume = duck_volume
        self.normal_volume = initial_volume
        self.crossfade_duration = crossfade_seconds
        self.buffer_size = buffer_size
        
        # Use the simple player for actual audio output
        from .music_player_simple import get_player
        self.simple_player = get_player()
        
        # Playback state (synchronized with simple player)
        self.is_voice_active = False
        self.play_history: List[Dict] = []
        
        # Threading for monitoring
        self.playback_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.command_queue = queue.Queue()
        
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
                    command, data = self.command_queue.get(timeout=0.1)  # Increased timeout
                    logger.info(f"🎵 Playback thread got command: {command}")
                    self._handle_command_sync(command, data)
                except queue.Empty:
                    pass
                
                # Sync state with simple player
                if self.simple_player:
                    status = self.simple_player.get_status()
                    # No need to handle song ending - SimpleMusicPlayer handles queue automatically
                
                # Sleep a bit
                time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in playback thread: {e}")
                time.sleep(0.1)
        
        logger.info("🎵 Playback thread stopped")
    
    def _handle_command_sync(self, command: str, data: Dict):
        """Handle commands in the playback thread"""
        logger.info(f"🎵 Processing command in playback thread: {command}")
        
        if command == "play":
            if "file_path" in data:
                # Play immediately
                file_path = data.get("file_path")
                self.simple_player.play(file_path, data)
                
                # Add to history
                self.play_history.append(data)
                if len(self.play_history) > 100:
                    self.play_history.pop(0)
                
                logger.info(f"🎵 Now playing: {data.get('title', 'Unknown')}")
            else:
                # Resume playback
                self.simple_player.resume()
        
        elif command == "pause":
            logger.info("⏸️ Pausing playback")
            self.simple_player.pause()
        
        elif command == "skip":
            logger.info("⏭️ Skipping song")
            self.simple_player.skip_to_next()
        
        elif command == "stop":
            logger.info("⏹️ Stopping playback")
            self.simple_player.stop()
        
        elif command == "volume":
            self.normal_volume = data.get("level", 0.7)
            if not self.is_voice_active:
                self.volume = self.normal_volume
            self.simple_player.set_volume(self.volume)
        
        elif command == "queue":
            # Add to simple player queue
            file_path = data.get("file_path")
            self.simple_player.queue_song(file_path, data)
            logger.info(f"➕ Queued: {data.get('title', 'Unknown')}")
    
    
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames - handle music control and mix audio"""
        
        # Let parent handle system frames
        await super().process_frame(frame, direction)
        
        # Handle music control commands
        if isinstance(frame, MusicControlFrame):
            logger.info(f"🎵🎵🎵 AudioPlayerReal received MusicControlFrame: {frame.command}")
            await self._handle_control(frame)
            return  # Don't forward control frames
        
        # Detect TTS activity for ducking (DJ should duck music when speaking)
        if isinstance(frame, SystemFrame):
            if frame.__class__.__name__ == "TTSStartedFrame":
                await self._start_ducking()
            elif frame.__class__.__name__ == "TTSStoppedFrame":
                # Add small delay before restoring volume for clean transition
                await asyncio.sleep(0.2)
                await self._stop_ducking()
        
        # Forward all frames (no mixing needed - simple player handles audio)
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
        elif command == "stop":
            await self.push_frame(TextFrame("⏹️ Music stopped"))
    
    
    async def _start_ducking(self):
        """Duck music volume when TTS is active (smooth fade)"""
        self.is_voice_active = True
        
        # Smooth fade to duck volume
        start_volume = self.volume
        fade_steps = 10
        fade_duration = 0.3  # 300ms fade
        
        for i in range(fade_steps):
            progress = (i + 1) / fade_steps
            self.volume = start_volume + (self.duck_volume - start_volume) * progress
            self.simple_player.set_volume(self.volume)
            await asyncio.sleep(fade_duration / fade_steps)
        
        logger.debug(f"🔉 Ducked volume to {self.duck_volume}")
    
    async def _stop_ducking(self):
        """Restore music volume when TTS stops (smooth fade)"""
        self.is_voice_active = False
        
        # Smooth fade to normal volume
        start_volume = self.volume
        fade_steps = 10
        fade_duration = 0.5  # 500ms fade back
        
        for i in range(fade_steps):
            progress = (i + 1) / fade_steps
            self.volume = start_volume + (self.normal_volume - start_volume) * progress
            self.simple_player.set_volume(self.volume)
            await asyncio.sleep(fade_duration / fade_steps)
        
        logger.debug(f"🔊 Restored volume to {self.normal_volume}")
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get current queue information"""
        # Get status from simple player
        player_status = self.simple_player.get_status() if self.simple_player else {}
        
        return {
            "current": player_status.get("current_song"),
            "is_playing": player_status.get("is_playing", False),
            "is_paused": player_status.get("is_paused", False),
            "queue_length": player_status.get("queue_length", 0),
            "queue": player_status.get("queue", [])[:5],
            "volume": player_status.get("volume", int(self.normal_volume * 100))
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🎵 Cleaning up audio player")
        
        # Cleanup simple player
        if self.simple_player:
            self.simple_player.cleanup()
        
        await super().cleanup()