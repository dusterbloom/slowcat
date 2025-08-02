"""Voice recognition module for speaker identification"""
from .lightweight import LightweightVoiceRecognition
from .auto_enroll import AutoEnrollVoiceRecognition

__all__ = ["LightweightVoiceRecognition", "AutoEnrollVoiceRecognition"]