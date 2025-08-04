"""
Simple music player that actually plays audio through speakers
Uses threading to play music in background
"""

import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from loguru import logger

# Audio libraries
try:
    import pyaudio
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError as e:
    logger.error(f"Audio libraries not available: {e}")
    AUDIO_AVAILABLE = False


class SimpleMusicPlayer:
    """Simple music player that actually plays audio"""
    
    def __init__(self, volume: float = 0.7):
        if not AUDIO_AVAILABLE:
            logger.error("Audio libraries not available!")
            return
            
        self.volume = volume
        self.is_playing = False
        self.is_paused = False
        self.current_song = None
        
        # PyAudio setup
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Threading
        self.play_thread = None
        self.stop_event = threading.Event()
        self.command_queue = queue.Queue()
        
        # Start command handler thread
        self.command_thread = threading.Thread(target=self._command_handler, daemon=True)
        self.command_thread.start()
        
        logger.info("🎵 SimpleMusicPlayer initialized")
    
    def play(self, file_path: str, song_info: Dict[str, Any] = None):
        """Play a music file"""
        self.command_queue.put(("play", {"file_path": file_path, "info": song_info}))
    
    def pause(self):
        """Pause playback"""
        self.is_paused = True
    
    def resume(self):
        """Resume playback"""
        self.is_paused = False
    
    def stop(self):
        """Stop playback"""
        self.command_queue.put(("stop", {}))
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
    
    def _command_handler(self):
        """Handle commands in background thread"""
        while not self.stop_event.is_set():
            try:
                command, data = self.command_queue.get(timeout=0.1)
                
                if command == "play":
                    self._stop_current()
                    file_path = data["file_path"]
                    song_info = data.get("info", {})
                    
                    # Start playback in new thread
                    self.play_thread = threading.Thread(
                        target=self._play_file,
                        args=(file_path, song_info),
                        daemon=True
                    )
                    self.play_thread.start()
                
                elif command == "stop":
                    self._stop_current()
                    
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in command handler: {e}")
    
    def _stop_current(self):
        """Stop current playback"""
        self.is_playing = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        # Wait for play thread to finish
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
    
    def _play_file(self, file_path: str, song_info: Dict[str, Any]):
        """Play audio file (runs in thread)"""
        try:
            if not Path(file_path).exists():
                logger.error(f"File not found: {file_path}")
                return
            
            logger.info(f"🎵 Playing: {file_path}")
            
            # Load audio file
            data, samplerate = sf.read(file_path, dtype='float32')
            
            # Convert to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Setup stream
            self.stream = self.pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=samplerate,
                output=True,
                frames_per_buffer=1024
            )
            
            self.current_song = song_info
            self.is_playing = True
            self.is_paused = False
            
            # Play audio in chunks
            chunk_size = 1024
            position = 0
            
            while position < len(data) and self.is_playing:
                if not self.is_paused:
                    # Get chunk
                    end = min(position + chunk_size, len(data))
                    chunk = data[position:end]
                    
                    # Apply volume
                    chunk = chunk * self.volume
                    
                    # Play chunk
                    self.stream.write(chunk.tobytes())
                    
                    position = end
                else:
                    # Paused - sleep a bit
                    time.sleep(0.1)
            
            # Cleanup
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            self.is_playing = False
            
            logger.info("🎵 Playback finished")
            
        except Exception as e:
            logger.error(f"Error playing file: {e}")
            self.is_playing = False
            if self.stream:
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get player status"""
        return {
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "current_song": self.current_song,
            "volume": int(self.volume * 100)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🎵 Cleaning up music player")
        self.stop_event.set()
        self._stop_current()
        
        if self.command_thread:
            self.command_thread.join(timeout=2.0)
        
        self.pyaudio.terminate()


# Global player instance
_player: Optional[SimpleMusicPlayer] = None


def get_player() -> SimpleMusicPlayer:
    """Get or create the global player instance"""
    global _player
    if _player is None:
        _player = SimpleMusicPlayer()
    return _player


def play_music_simple(file_path: str, song_info: Dict[str, Any] = None):
    """Simple interface to play music"""
    player = get_player()
    player.play(file_path, song_info)
    return player.get_status()