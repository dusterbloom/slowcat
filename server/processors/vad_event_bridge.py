"""
VAD Event Bridge - Converts VAD frames to events for voice recognition
"""
import asyncio
import logging
from typing import Optional, Callable
from pipecat.frames.frames import Frame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame, StartFrame, CancelFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class VADEventBridge(FrameProcessor):
    """
    Bridges VAD (Voice Activity Detection) frames to events that voice recognition can use.
    Detects when users start/stop speaking based on interruption frames.
    """
    
    def __init__(self):
        super().__init__()
        self._on_user_started_speaking: Optional[Callable] = None
        self._on_user_stopped_speaking: Optional[Callable] = None
        self._user_speaking = False
    
    def set_callbacks(self, on_started: Optional[Callable] = None, on_stopped: Optional[Callable] = None):
        """Set callbacks for speaking events"""
        self._on_user_started_speaking = on_started
        self._on_user_stopped_speaking = on_stopped
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames and detect speaking state changes"""
        # Handle system frames through parent
        await super().process_frame(frame, direction)
        
        # Detect VAD events and trigger callbacks in background
        if isinstance(frame, UserStartedSpeakingFrame):
            if not self._user_speaking:
                self._user_speaking = True
                logger.info("🎤 VAD: User started speaking")
                if self._on_user_started_speaking:
                    asyncio.create_task(self._on_user_started_speaking())
        
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._user_speaking:
                self._user_speaking = False
                logger.info("🔇 VAD: User stopped speaking")
                if self._on_user_stopped_speaking:
                    asyncio.create_task(self._on_user_stopped_speaking())
        
        # ALWAYS pass through ALL frames
        await self.push_frame(frame, direction)