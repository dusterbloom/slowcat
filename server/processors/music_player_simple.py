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
        
        # Queue management
        self.play_queue = []
        self.queue_lock = threading.Lock()
        
        # PyAudio setup
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # Threading
        self.play_thread = None
        self.stop_event = threading.Event()  # Signals thread to exit
        self.force_stop = threading.Event()  # Signals playback to stop immediately
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
    
    def queue_song(self, file_path: str, song_info: Dict[str, Any] = None):
        """Add a song to the queue"""
        with self.queue_lock:
            self.play_queue.append({"file_path": file_path, "info": song_info or {}})
            logger.info(f"🎵 Queued: {song_info.get('title', file_path) if song_info else file_path}")
    
    def skip_to_next(self):
        """Skip to next song in queue"""
        self.command_queue.put(("skip", {}))
    
    def clear_queue(self):
        """Clear the play queue"""
        with self.queue_lock:
            self.play_queue.clear()
            logger.info("🎵 Queue cleared")
    
    def get_queue(self):
        """Get current queue"""
        with self.queue_lock:
            return list(self.play_queue)
    
    def _command_handler(self):
        """Handle commands in background thread"""
        while not self.stop_event.is_set():
            try:
                command, data = self.command_queue.get(timeout=0.1)
                
                if command == "play":
                    self._stop_current()
                    file_path = data["file_path"]
                    song_info = data.get("info", {})
                    self._start_playback_thread(file_path, song_info)
                
                elif command == "stop":
                    self._stop_current()
                    with self.queue_lock:
                        self.play_queue.clear()
                
                elif command == "skip":
                    self._stop_current()
                    self._play_next_in_queue()
                    
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in command handler: {e}")

    def _start_playback_thread(self, file_path: str, song_info: Dict[str, Any]):
        """Stops current playback and starts a new playback thread."""
        self._stop_current()  # Ensure any existing thread is stopped
        self.play_thread = threading.Thread(
            target=self._play_file,
            args=(file_path, song_info),
            daemon=True,
            name=f"PlaybackThread-{song_info.get('title', 'Unknown')[:10]}"
        )
        self.play_thread.start()

    def _play_next_in_queue(self):
        """Play next song in queue"""
        with self.queue_lock:
            if self.play_queue:
                next_song = self.play_queue.pop(0)
                logger.info(f"🎵 Playing next from queue: {next_song.get('info', {}).get('title', 'Unknown')}")
                self._start_playback_thread(next_song["file_path"], next_song["info"])
            else:
                logger.info("🎵 Queue empty, playback stopped")
    
    def _stop_current(self):
        """Stop current playback by signaling the thread."""
        if self.play_thread and self.play_thread.is_alive():
            logger.debug(f"Stopping playback thread {self.play_thread.name}...")
            self.force_stop.set()
            self.play_thread.join(timeout=1.0)  # Give it a moment to die
            if self.play_thread.is_alive():
                logger.warning(f"Playback thread {self.play_thread.name} did not terminate.")
        
        # Reset state regardless
        self.is_playing = False
        self.current_song = None
        self.play_thread = None
    
    def _play_file(self, file_path: str, song_info: Dict[str, Any]):
        """Play audio file (runs in thread)"""
        logger.debug(f"Playback thread {threading.current_thread().name} started for {file_path}")
        # Reset stop event for this playback
        self.force_stop.clear()
        
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
            
            logger.debug("Starting playback loop...")
            while position < len(data) and not self.force_stop.is_set():
                if not self.is_paused:
                    # Get chunk
                    end = min(position + chunk_size, len(data))
                    chunk = data[position:end]
                    
                    # Apply volume
                    chunk = chunk * self.volume
                    
                    # Play chunk
                    try:
                        self.stream.write(chunk.tobytes())
                    except Exception as e:
                        logger.error(f"Exception during stream.write: {e}")
                        break # Exit loop on write error
                    
                    position = end
                else:
                    # Paused - sleep a bit
                    time.sleep(0.1)
            logger.debug(f"Playback loop finished. force_stop.is_set() is {self.force_stop.is_set()}")
            
        except Exception as e:
            logger.error(f"Error playing file: {e}")
        
        finally:
            # Cleanup
            logger.debug("Entering finally block for playback thread.")
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Ignoring error during stream cleanup: {e}")
                finally:
                    self.stream = None
            
            self.is_playing = False
            
            # Check if we were forcibly stopped before auto-playing next
            logger.debug(f"In finally block, force_stop.is_set() is {self.force_stop.is_set()}")
            if not self.force_stop.is_set():
                logger.info("🎵 Playback finished naturally, starting next song.")
                # Auto-play next song in queue
                self._play_next_in_queue()
            else:
                logger.info("🎵 Playback was forcibly stopped. Not playing next song.")
    
    def get_status(self) -> Dict[str, Any]:
        """Get player status"""
        with self.queue_lock:
            queue_count = len(self.play_queue)
            next_songs = [song["info"] for song in self.play_queue[:3]]  # First 3 songs
        
        return {
            "is_playing": self.is_playing,
            "is_paused": self.is_paused,
            "current_song": self.current_song,
            "volume": int(self.volume * 100),
            "queue_length": queue_count,
            "queue": next_songs
        }
    
    async def cleanup(self):
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