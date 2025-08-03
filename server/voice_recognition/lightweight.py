"""
Base class for voice recognition modules.
This version uses an event-driven approach to process complete utterances,
which is more robust and suitable for libraries like Resemblyzer.
"""
import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import pickle
import os

logger = logging.getLogger(__name__)

try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    logger.warning("Resemblyzer not available. Voice recognition will be disabled.")


class LightweightVoiceRecognition:
    """
    Base class for voice recognition. It buffers audio when the user is
    speaking and processes the complete utterance when they stop.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Store config for subclasses
        self.config = config
        
        # Configuration
        self.enabled = config.get('enabled', True) and RESEMBLYZER_AVAILABLE
        self.sample_rate = 16000  # Fixed for consistency with Resemblyzer
        self.min_utterance_duration = config.get('min_utterance_duration_seconds', 1.0)
        self.similarity_threshold = config.get('confidence_threshold', 0.75)
        
        # Speaker database
        self.speakers = {}  # name -> fingerprints list
        self.current_speaker = None
        
        # Audio processing
        self.utterance_buffer = bytearray()
        self.is_speaking = False
        
        # Event callbacks
        self._on_speaker_changed: Optional[Callable] = None
        self._on_speaker_enrolled: Optional[Callable] = None
        
        # Profile storage
        self.profile_dir = config.get('profile_dir', 'data/speaker_profiles')
        self.profile_extension = config.get('profile_file_extension', '.pkl')
        
        # Initialize encoder if available
        if self.enabled:
            # Explicitly use CPU to avoid GPU detection overhead on macOS
            self.encoder = VoiceEncoder("cpu")
    
    async def initialize(self):
        """Initialize the module"""
        if not self.enabled:
            logger.warning("Voice recognition disabled (Resemblyzer not available or disabled in config)")
            return
            
        os.makedirs(self.profile_dir, exist_ok=True)
        self._load_profiles()
        logger.info(f"Lightweight voice recognition initialized with {len(self.speakers)} profiles")
    
    def set_callbacks(self, on_speaker_changed: Optional[Callable] = None, 
                     on_speaker_enrolled: Optional[Callable] = None):
        """Set event callbacks"""
        self._on_speaker_changed = on_speaker_changed
        self._on_speaker_enrolled = on_speaker_enrolled
    
    async def on_user_started_speaking(self):
        """Handle the start of a user utterance."""
        if not self.enabled:
            return
            
        self.is_speaking = True
        self.utterance_buffer.clear()
        logger.info("🎙️ Voice Recognition: User started speaking, clearing buffer")

    async def on_user_stopped_speaking(self):
        """Handle the end of a user utterance and process it."""
        if not self.enabled or not self.is_speaking:
            return
        
        self.is_speaking = False
        logger.info(f"🎙️ Voice Recognition: User stopped speaking. Processing {len(self.utterance_buffer)} bytes")
        
        utterance_duration = len(self.utterance_buffer) / (self.sample_rate * 2)
        if utterance_duration < self.min_utterance_duration:
            logger.info(f"Skipping speaker recognition for short utterance ({utterance_duration:.2f}s).")
            self.utterance_buffer.clear()
            return

        try:
            audio_array = np.frombuffer(self.utterance_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            logger.debug(f"Processing audio array: shape={audio_array.shape}, min={audio_array.min():.3f}, max={audio_array.max():.3f}")
            await self._process_speaker_identification(audio_array)
        except Exception as e:
            logger.error(f"Error processing utterance for speaker recognition: {e}")
        finally:
            self.utterance_buffer.clear()
    
    def process_audio_frame(self, audio_data: bytes):
        """Buffer audio frames when the user is speaking."""
        if self.enabled and self.is_speaking:
            self.utterance_buffer.extend(audio_data)

    async def _process_speaker_identification(self, audio_array: np.ndarray):
        """
        Placeholder for speaker identification.
        The actual implementation is in the AutoEnrollVoiceRecognition subclass.
        """
        logger.warning("Base class _process_speaker_identification called. Subclass should override this.")
        pass
    
    async def _emit_speaker_change(self, speaker_name: str, confidence: float):
        """Emit speaker change event"""
        if self._on_speaker_changed:
            await self._on_speaker_changed({
                'speaker_name': speaker_name,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        logger.info(f"Speaker changed to: {speaker_name} (confidence: {confidence:.2f})")

    def _save_profile(self, name: str, fingerprints: List[np.ndarray]):
        """Save speaker profile to disk"""
        filepath = os.path.join(self.profile_dir, f"{name}{self.profile_extension}")
        with open(filepath, 'wb') as f:
            pickle.dump(fingerprints, f)
    
    def _load_profiles(self):
        """Load all speaker profiles from disk"""
        if not os.path.exists(self.profile_dir):
            return
            
        for filename in os.listdir(self.profile_dir):
            if filename.endswith(self.profile_extension):
                name = filename[:-len(self.profile_extension)]
                filepath = os.path.join(self.profile_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        fingerprints = pickle.load(f)
                    self.speakers[name] = fingerprints
                    logger.info(f"Loaded profile: {name}")
                except Exception as e:
                    logger.error(f"Error loading profile {name}: {e}")
    
    async def shutdown(self):
        """Cleanup if necessary."""
        pass