"""
VAD Event Bridge - Enhanced with Smart Turn Management
Converts VAD frames to events and applies intelligent turn-taking logic.
"""
import asyncio
import logging
from typing import Optional, Callable
from pipecat.frames.frames import Frame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame, StartFrame, CancelFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class VADEventBridge(FrameProcessor):
    """
    Enhanced VAD Event Bridge with Smart Turn Management.
    Bridges VAD frames to events with intelligent turn-taking decisions.
    """
    
    def __init__(self, enable_smart_turn_management: bool = True):
        super().__init__()
        self._on_user_started_speaking: Optional[Callable] = None
        self._on_user_stopped_speaking: Optional[Callable] = None
        self._user_speaking = False
        
        # Smart turn management
        self._enable_smart_management = enable_smart_turn_management
        self._smart_turn_manager = None
        
        if self._enable_smart_management:
            try:
                from processors.smart_turn_manager import SmartTurnManager
                self._smart_turn_manager = SmartTurnManager()
                self._smart_turn_manager.set_turn_complete_callback(self._on_smart_turn_complete)
                logger.info("🧠 Smart Turn Manager enabled")
            except ImportError as e:
                logger.warning(f"Smart Turn Manager not available: {e}")
                self._enable_smart_management = False
    
    def set_callbacks(self, on_started: Optional[Callable] = None, on_stopped: Optional[Callable] = None):
        """Set callbacks for speaking events"""
        self._on_user_started_speaking = on_started
        self._on_user_stopped_speaking = on_stopped
    
    async def _on_smart_turn_complete(self):
        """Called by smart turn manager when turn is determined complete"""
        if self._on_user_stopped_speaking:
            await self._on_user_stopped_speaking()
    
    def update_conversation_context(self, user_text: str = "", assistant_text: str = ""):
        """Update conversation context for smart turn decisions"""
        if self._smart_turn_manager:
            self._smart_turn_manager.update_conversation_context(user_text, assistant_text)
    
    def report_interruption(self):
        """Report that an interruption occurred"""
        if self._smart_turn_manager:
            self._smart_turn_manager.report_interruption()
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames and detect speaking state changes"""
        # Handle system frames through parent
        await super().process_frame(frame, direction)
        
        # Process through smart turn manager if enabled
        if self._smart_turn_manager:
            await self._smart_turn_manager.handle_frame(frame)
        
        # Handle VAD events
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
                
                # If smart turn management is disabled, use immediate callback
                if not self._enable_smart_management and self._on_user_stopped_speaking:
                    asyncio.create_task(self._on_user_stopped_speaking())
        
        # ALWAYS pass through ALL frames
        await self.push_frame(frame, direction)