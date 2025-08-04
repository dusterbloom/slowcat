"""
DJ Voice Processor - Switches TTS voice for music/DJ mode
"""

from typing import Optional, Dict, Any
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    SystemFrame,
    TTSUpdateSettingsFrame
)
from pipecat.processors.frame_processor import FrameProcessor


class DJVoiceProcessor(FrameProcessor):
    """
    Switches TTS voice when entering/exiting music mode
    """
    
    def __init__(
        self,
        *,
        normal_voice: str = "af_heart",
        dj_voice: str = "am_echo",  # Cool male DJ voice
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.normal_voice = normal_voice
        self.dj_voice = dj_voice
        self.is_dj_mode = False
        
        logger.info(f"🎙️ DJ Voice: normal='{normal_voice}', dj='{dj_voice}'")
    
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames - switch voices based on mode"""
        
        # Let parent handle system frames
        await super().process_frame(frame, direction)
        
        # Check for music mode activation/deactivation
        if isinstance(frame, TextFrame):
            # Look for music mode notifications
            if "🎵 Music mode on" in frame.text:
                await self._switch_to_dj_voice()
            elif "Music mode off" in frame.text:
                await self._switch_to_normal_voice()
        
        # Forward the frame
        await self.push_frame(frame, direction)
    
    async def _switch_to_dj_voice(self):
        """Switch to DJ voice"""
        if not self.is_dj_mode:
            self.is_dj_mode = True
            # Send voice update frame
            voice_update = TTSUpdateSettingsFrame(settings={"voice": self.dj_voice})
            await self.push_frame(voice_update)
            logger.info(f"🎙️ Switched to DJ voice: {self.dj_voice}")
    
    async def _switch_to_normal_voice(self):
        """Switch back to normal voice"""
        if self.is_dj_mode:
            self.is_dj_mode = False
            # Send voice update frame
            voice_update = TTSUpdateSettingsFrame(settings={"voice": self.normal_voice})
            await self.push_frame(voice_update)
            logger.info(f"🎙️ Switched to normal voice: {self.normal_voice}")
    
    def set_dj_mode(self, enabled: bool):
        """Manually set DJ mode"""
        self.is_dj_mode = enabled