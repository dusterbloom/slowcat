"""Processors module"""
from .audio_tee import AudioTeeProcessor
from .vad_event_bridge import VADEventBridge
from .speaker_context import SpeakerContextProcessor
from .video_sampler import VideoSamplerProcessor
from .speaker_name_manager import SpeakerNameManager

__all__ = ["AudioTeeProcessor", "VADEventBridge", "SpeakerContextProcessor", "VideoSamplerProcessor", "SpeakerNameManager"]