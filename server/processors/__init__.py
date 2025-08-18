"""Processors module"""
from .audio_tee import AudioTeeProcessor
from .vad_event_bridge import VADEventBridge
from .speaker_context import SpeakerContextProcessor
from .video_sampler import VideoSamplerProcessor
from .speaker_name_manager import SpeakerNameManager
from .local_memory import LocalMemoryProcessor
from .memory_context_injector import MemoryContextInjector
from .greeting_filter import GreetingFilterProcessor
from .message_deduplicator import MessageDeduplicator
from .context_filter import ContextFilter
from .streaming_deduplicator import StreamingDeduplicator 
from .dj_mode_config_handler import DJModeConfigHandler
from .smart_turn_manager import SmartTurnManager

# Import StatelessMemoryProcessor with error handling
try:
    from .stateless_memory import StatelessMemoryProcessor
    STATELESS_MEMORY_AVAILABLE = True
except ImportError as e:
    # Log the error but don't break the import
    import logging
    logging.warning(f"StatelessMemoryProcessor not available: {e}")
    StatelessMemoryProcessor = None
    STATELESS_MEMORY_AVAILABLE = False


__all__ = [
    "AudioTeeProcessor", 
    "VADEventBridge", 
    "SpeakerContextProcessor", 
    "VideoSamplerProcessor", 
    "SpeakerNameManager", 
    "LocalMemoryProcessor", 
    "MemoryContextInjector", 
    "GreetingFilterProcessor", 
    "MessageDeduplicator",
    "ContextFilter", 
    "StreamingDeduplicator",
    "DJModeConfigHandler", 
    "SmartTurnManager"
]

# Add StatelessMemoryProcessor to exports if available
if STATELESS_MEMORY_AVAILABLE:
    __all__.append("StatelessMemoryProcessor")