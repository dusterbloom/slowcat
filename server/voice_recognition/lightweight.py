"""
Base class for voice recognition modules.
This version uses an event-driven approach to process complete utterances,
which is more robust and suitable for libraries like Resemblyzer.

Enhanced with M3-AudioGraph integration for advanced speaker identity resolution.
"""
import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import pickle
import os
from config import VoiceRecognitionConfig 

logger = logging.getLogger(__name__)

try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    logger.warning("Resemblyzer not available. Voice recognition will be disabled.")

# M3-AudioGraph integration
try:
    import sys
    import os
    
    # Add server directory to Python path for imports
    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    
    from voice_recognition.m3_voice_integration import create_m3_voice_integration, M3VoiceIntegration
    M3_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.debug(f"M3 voice integration not available: {e}")
    create_m3_voice_integration = None
    M3VoiceIntegration = None
    M3_INTEGRATION_AVAILABLE = False


class LightweightVoiceRecognition:
    """
    Base class for voice recognition. It buffers audio when the user is
    speaking and processes the complete utterance when they stop.
    """
    
    def __init__(self, config: VoiceRecognitionConfig, enable_m3_integration: bool = True):
        self.config = config
        self.enabled = config.enabled and RESEMBLYZER_AVAILABLE
        self.min_utterance_duration = config.min_utterance_duration_seconds
    

        # Configuration
        self.sample_rate = 16000  # Fixed for consistency with Resemblyzer
        self.similarity_threshold = config.confidence_threshold
        
        # Speaker database (legacy)
        self.speakers = {}  # name -> fingerprints list
        self.current_speaker = None
        
        # Audio processing
        self.utterance_buffer = bytearray()
        self.is_speaking = False
        
        # Event callbacks
        self._on_speaker_changed: Optional[Callable] = None
        self._on_speaker_enrolled: Optional[Callable] = None
        
        # Profile storage
        self.profile_dir = config.profile_dir
        self.profile_extension = config.profile_file_extension
        
        # Initialize encoder if available
        if self.enabled:
            # Explicitly use CPU to avoid GPU detection overhead on macOS
            self.encoder = VoiceEncoder("cpu")
        
        # M3-AudioGraph integration
        self.m3_integration: Optional[M3VoiceIntegration] = None
        self.use_m3_integration = enable_m3_integration and M3_INTEGRATION_AVAILABLE
        
        if self.use_m3_integration:
            try:
                m3_config = {
                    'voice_sample_rate': self.sample_rate,
                    'confidence_threshold': self.similarity_threshold
                }
                self.m3_integration = create_m3_voice_integration(config=m3_config)
                if self.m3_integration:
                    logger.info("🧠 M3-AudioGraph integration enabled for voice recognition")
                else:
                    self.use_m3_integration = False
                    logger.warning("Failed to create M3 integration - falling back to legacy mode")
            except Exception as e:
                self.use_m3_integration = False
                logger.error(f"M3 integration initialization failed: {e}")
        else:
            logger.debug("Using legacy voice recognition mode")
    
    async def initialize(self):
        """Initialize the module"""
        if not self.enabled:
            logger.warning("Voice recognition disabled (Resemblyzer not available or disabled in config)")
            return
            
        os.makedirs(self.profile_dir, exist_ok=True)
        
        # Load legacy profiles
        self._load_profiles()
        
        # Migrate to M3 if integration is enabled
        if self.use_m3_integration and self.m3_integration:
            try:
                migrated_count = self.m3_integration.migrate_all_profiles(
                    self.profile_dir, 
                    self.profile_extension
                )
                logger.info(f"📦 Migrated {migrated_count} legacy profiles to M3-AudioGraph")
            except Exception as e:
                logger.error(f"Error migrating profiles to M3: {e}")
        
        total_profiles = len(self.speakers)
        mode = "M3-AudioGraph" if self.use_m3_integration else "legacy"
        logger.info(f"Lightweight voice recognition initialized with {total_profiles} profiles ({mode} mode)")
    
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
    
    async def process_audio_frame(self, frame: Any, sample_rate: int):
        """Buffer audio frames when the user is speaking."""
        if self.enabled and self.is_speaking:
            # Extract audio bytes from frame
            if hasattr(frame, 'audio'):
                self.utterance_buffer.extend(frame.audio)
            else:
                # Fallback if frame is just bytes
                self.utterance_buffer.extend(frame)

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
    
    # ============================================================================
    # M3-AudioGraph Integration Methods
    # ============================================================================
    
    def create_voice_node(self, speaker_name: str, fingerprints: List[np.ndarray], 
                         transcript: str = "") -> bool:
        """
        Create voice node in M3-AudioGraph for speaker.
        
        Args:
            speaker_name: Speaker identifier
            fingerprints: Voice fingerprints from Resemblyzer
            transcript: Optional transcript content
            
        Returns:
            Success indicator
        """
        if not self.use_m3_integration or not self.m3_integration:
            logger.debug("M3 integration not available - skipping voice node creation")
            return False
        
        try:
            node_id = self.m3_integration.create_voice_node_from_embeddings(
                speaker_name, fingerprints, transcript
            )
            return node_id is not None
            
        except Exception as e:
            logger.error(f"Failed to create voice node for '{speaker_name}': {e}")
            return False
    
    def update_voice_node(self, speaker_name: str, fingerprints: List[np.ndarray],
                         transcript: str = "") -> bool:
        """
        Update existing voice node with new embeddings.
        
        Args:
            speaker_name: Speaker identifier
            fingerprints: New voice fingerprints
            transcript: Optional transcript content
            
        Returns:
            Success indicator
        """
        if not self.use_m3_integration or not self.m3_integration:
            return False
        
        try:
            return self.m3_integration.update_voice_node(
                speaker_name, fingerprints, transcript
            )
            
        except Exception as e:
            logger.error(f"Failed to update voice node for '{speaker_name}': {e}")
            return False
    
    def find_speaker_m3(self, fingerprint: np.ndarray) -> tuple[Optional[str], float]:
        """
        Find speaker using M3-AudioGraph voice node matching.
        
        Args:
            fingerprint: Voice fingerprint from Resemblyzer
            
        Returns:
            Tuple of (speaker_name, confidence) or (None, 0.0)
        """
        if not self.use_m3_integration or not self.m3_integration:
            return None, 0.0
        
        try:
            return self.m3_integration.find_speaker_voice_node(
                [fingerprint], self.similarity_threshold
            )
            
        except Exception as e:
            logger.error(f"Error in M3 speaker matching: {e}")
            return None, 0.0
    
    def resolve_speaker_identity(self, speaker_name: str) -> str:
        """
        Resolve canonical speaker identity using M3 equivalence detection.
        
        Args:
            speaker_name: Input speaker name
            
        Returns:
            Canonical speaker identifier
        """
        if not self.use_m3_integration or not self.m3_integration:
            return speaker_name
        
        try:
            return self.m3_integration.resolve_speaker_identity(speaker_name)
            
        except Exception as e:
            logger.error(f"Error resolving identity for '{speaker_name}': {e}")
            return speaker_name
    
    def get_m3_statistics(self) -> Dict[str, Any]:
        """Get M3-AudioGraph statistics."""
        if not self.use_m3_integration or not self.m3_integration:
            return {}
        
        try:
            return self.m3_integration.get_speaker_statistics()
            
        except Exception as e:
            logger.error(f"Error getting M3 statistics: {e}")
            return {}
    
    def cleanup_m3_connections(self, threshold: float = 0.1) -> int:
        """Clean up weak M3 voice node connections."""
        if not self.use_m3_integration or not self.m3_integration:
            return 0
        
        try:
            return self.m3_integration.cleanup_weak_connections(threshold)
            
        except Exception as e:
            logger.error(f"Error cleaning up M3 connections: {e}")
            return 0
    
    def refresh_m3_equivalences(self):
        """Refresh M3 speaker identity equivalences."""
        if self.use_m3_integration and self.m3_integration:
            try:
                self.m3_integration.refresh_equivalences()
                
            except Exception as e:
                logger.error(f"Error refreshing M3 equivalences: {e}")
    
    async def shutdown(self):
        """Cleanup if necessary."""
        # Refresh equivalences before shutdown
        if self.use_m3_integration:
            self.refresh_m3_equivalences()