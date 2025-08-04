"""
Audio player processor for the DJ system
Handles music playback, mixing, and queue management
"""

import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque
import threading
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    SystemFrame,
    TextFrame
)
from pipecat.processors.frame_processor import FrameProcessor


class MusicControlFrame(SystemFrame):
    """Frame for music control commands"""
    def __init__(self, command: str, data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.command = command  # play, pause, skip, queue, volume
        self.data = data or {}


class NowPlayingFrame(SystemFrame):
    """Frame with current playing info"""
    def __init__(self, song_info: Dict[str, Any], status: str):
        super().__init__()
        self.song_info = song_info
        self.status = status  # playing, paused, stopped


class AudioPlayerProcessor(FrameProcessor):
    """
    Audio player that mixes music with voice in the pipeline
    """
    
    def __init__(
        self,
        *,
        initial_volume: float = 0.7,
        duck_volume: float = 0.3,  # Volume when voice is active
        crossfade_seconds: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.volume = initial_volume
        self.duck_volume = duck_volume
        self.normal_volume = initial_volume
        self.crossfade_duration = crossfade_seconds
        
        # Playback state
        self.is_playing = False
        self.is_voice_active = False
        self.current_song: Optional[Dict] = None
        self.playback_position = 0
        
        # Queue management
        self.play_queue: deque = deque()
        self.play_history: List[Dict] = []
        
        # Audio ducking
        self._ducking_task: Optional[asyncio.Task] = None
        
        logger.info(f"AudioPlayer initialized: volume={initial_volume}, duck={duck_volume}")
    
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
        
        # Mix audio if music is playing
        if isinstance(frame, AudioRawFrame) and self.is_playing:
            # This would mix music with voice
            # For now, just forward the voice
            pass
        
        # Forward all frames
        await self.push_frame(frame, direction)
    
    async def _handle_control(self, control: MusicControlFrame):
        """Handle music control commands"""
        command = control.command
        data = control.data
        
        logger.info(f"Music control: {command} with data: {data}")
        
        if command == "play":
            if "file_path" in data:
                await self._play_song(data)
            else:
                await self._resume()
        
        elif command == "pause":
            await self._pause()
        
        elif command == "skip":
            await self._skip()
        
        elif command == "queue":
            await self._queue_song(data)
        
        elif command == "volume":
            await self._set_volume(data.get("level", 0.7))
        
        elif command == "stop":
            await self._stop()
        
        # Send status update
        await self._send_now_playing()
    
    async def _play_song(self, song_info: Dict):
        """Start playing a song"""
        self.current_song = song_info
        self.is_playing = True
        self.playback_position = 0
        
        # Add to history
        self.play_history.append(song_info)
        if len(self.play_history) > 100:
            self.play_history.pop(0)
        
        logger.info(f"Now playing: {song_info.get('title', 'Unknown')}")
        
        # Notify about now playing
        await self.push_frame(TextFrame(
            f"🎵 Now playing: {song_info.get('title', 'Unknown')} by {song_info.get('artist', 'Unknown Artist')}"
        ))
    
    async def _pause(self):
        """Pause playback"""
        if self.is_playing:
            self.is_playing = False
            logger.info("Playback paused")
    
    async def _resume(self):
        """Resume playback"""
        if not self.is_playing and self.current_song:
            self.is_playing = True
            logger.info("Playback resumed")
    
    async def _skip(self):
        """Skip to next song in queue"""
        if self.play_queue:
            next_song = self.play_queue.popleft()
            await self._play_song(next_song)
        else:
            await self._stop()
            await self.push_frame(TextFrame("🎵 Queue is empty"))
    
    async def _queue_song(self, song_info: Dict):
        """Add song to queue"""
        self.play_queue.append(song_info)
        logger.info(f"Added to queue: {song_info.get('title', 'Unknown')}")
        await self.push_frame(TextFrame(
            f"✅ Added to queue: {song_info.get('title', 'Unknown')}"
        ))
    
    async def _stop(self):
        """Stop playback"""
        self.is_playing = False
        self.current_song = None
        self.playback_position = 0
        logger.info("Playback stopped")
    
    async def _set_volume(self, level: float):
        """Set playback volume"""
        self.volume = max(0.0, min(1.0, level))
        self.normal_volume = self.volume
        logger.info(f"Volume set to: {self.volume}")
    
    async def _start_ducking(self):
        """Duck music volume when voice is active"""
        if self._ducking_task:
            self._ducking_task.cancel()
        
        self.is_voice_active = True
        self._ducking_task = asyncio.create_task(self._fade_volume(self.duck_volume, 0.2))
    
    async def _stop_ducking(self):
        """Restore music volume when voice stops"""
        if self._ducking_task:
            self._ducking_task.cancel()
        
        self.is_voice_active = False
        self._ducking_task = asyncio.create_task(self._fade_volume(self.normal_volume, 0.5))
    
    async def _fade_volume(self, target: float, duration: float):
        """Smoothly fade volume to target level"""
        start_volume = self.volume
        steps = 20
        step_duration = duration / steps
        
        for i in range(steps):
            progress = (i + 1) / steps
            self.volume = start_volume + (target - start_volume) * progress
            await asyncio.sleep(step_duration)
    
    async def _send_now_playing(self):
        """Send current playing status"""
        if self.current_song:
            status = "playing" if self.is_playing else "paused"
        else:
            status = "stopped"
        
        frame = NowPlayingFrame(
            song_info=self.current_song or {},
            status=status
        )
        await self.push_frame(frame)
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get current queue information"""
        return {
            "current": self.current_song,
            "queue_length": len(self.play_queue),
            "queue": list(self.play_queue)[:5],  # Next 5 songs
            "is_playing": self.is_playing,
            "volume": self.volume,
            "history_count": len(self.play_history)
        }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recently played songs"""
        return self.play_history[-limit:]