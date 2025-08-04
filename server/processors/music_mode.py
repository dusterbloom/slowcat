"""
Music mode processor - bot stays quiet and only responds to music commands
"""

from typing import Optional, List, Set
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TextFrame,
    SystemFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame
)
from pipecat.processors.frame_processor import FrameProcessor


class MusicModeProcessor(FrameProcessor):
    """
    Processor for music mode - minimal talking, just music control
    """
    
    # Music control keywords
    MUSIC_COMMANDS = {
        "play", "pause", "stop", "skip", "next", "volume", "louder", "quieter",
        "what's playing", "now playing", "current song", "music", "song",
        "turn up", "turn down", "resume", "quiet", "loud"
    }
    
    def __init__(
        self,
        *,
        mode_toggle_phrase: str = "music mode",
        exit_phrase: str = "stop music mode",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.mode_toggle_phrase = mode_toggle_phrase.lower()
        self.exit_phrase = exit_phrase.lower()
        self.music_mode_active = False
        
        logger.info(f"MusicMode initialized: toggle='{mode_toggle_phrase}'")
    
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames - in music mode, only allow music commands"""
        
        # Let parent handle system frames
        await super().process_frame(frame, direction)
        
        # Check for mode toggle
        if isinstance(frame, TranscriptionFrame):
            text_lower = frame.text.lower().strip()
            
            # Check for mode toggle
            if self.mode_toggle_phrase in text_lower and not self.music_mode_active:
                await self._enter_music_mode()
                return  # Don't forward toggle command
            
            # Check for exit
            if self.exit_phrase in text_lower and self.music_mode_active:
                await self._exit_music_mode()
                return  # Don't forward exit command
        
        # In music mode, filter what gets through
        if self.music_mode_active:
            # Check if it's a music command
            if isinstance(frame, TranscriptionFrame):
                if self._is_music_command(frame.text):
                    # Let music commands through
                    await self.push_frame(frame, direction)
                else:
                    # Block non-music commands
                    logger.debug(f"Blocking non-music command in music mode: {frame.text}")
                    return
            
            # Block ALL LLM responses except for music-related ones
            elif isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
                logger.debug(f"Blocking {type(frame).__name__} in music mode")
                return
            
            # Block TTS (text-to-speech) - stay quiet!
            elif isinstance(frame, (TTSStartedFrame, TTSStoppedFrame)):
                logger.debug(f"Blocking {type(frame).__name__} in music mode")
                return
            
            # Block assistant text responses
            elif isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame):
                # Only allow brief music confirmations
                if any(word in frame.text.lower() for word in ["playing", "paused", "volume", "skipping"]):
                    # Make it super brief
                    brief_text = self._make_brief(frame.text)
                    if brief_text:
                        await self.push_frame(TextFrame(brief_text), direction)
                    return
                else:
                    logger.debug("Blocking assistant text in music mode")
                    return
            
            # Let other frames through
            else:
                await self.push_frame(frame, direction)
        else:
            # Not in music mode, forward everything
            await self.push_frame(frame, direction)
    
    def _is_music_command(self, text: str) -> bool:
        """Check if text contains music-related commands"""
        text_lower = text.lower()
        
        # Check for any music command keywords
        for keyword in self.MUSIC_COMMANDS:
            if keyword in text_lower:
                return True
        
        # Check for artist/song names (basic heuristic)
        if any(word in text_lower for word in ["play", "queue", "find"]):
            return True
            
        return False
    
    def _make_brief(self, text: str) -> str:
        """Make responses super brief"""
        text_lower = text.lower()
        
        # Ultra-brief confirmations
        if "now playing" in text_lower or "playing:" in text_lower:
            # Extract just the song name
            if ":" in text:
                return "🎵 " + text.split(":", 1)[1].strip()
            return "🎵"
        elif "paused" in text_lower:
            return "⏸️"
        elif "resumed" in text_lower or "resuming" in text_lower:
            return "▶️"
        elif "volume" in text_lower:
            # Extract volume percentage if present
            import re
            match = re.search(r'(\d+)%?', text)
            if match:
                return f"🔊 {match.group(1)}%"
            return "🔊"
        elif "skip" in text_lower or "next" in text_lower:
            return "⏭️"
        elif "stop" in text_lower:
            return "⏹️"
        
        # Don't show other messages
        return ""
    
    async def _enter_music_mode(self):
        """Enter music mode"""
        self.music_mode_active = True
        
        # Brief notification
        notification = TextFrame(
            "🎵 Music mode on. I'll stay quiet and just control the music. "
            "Say 'stop music mode' to exit."
        )
        await self.push_frame(notification)
        
        logger.info("🎵 Entered music mode")
    
    async def _exit_music_mode(self):
        """Exit music mode"""
        self.music_mode_active = False
        
        # Brief notification
        notification = TextFrame("Music mode off. I can talk normally again!")
        await self.push_frame(notification)
        
        logger.info("🎵 Exited music mode")
    
    def is_music_mode_active(self) -> bool:
        """Check if music mode is active"""
        return self.music_mode_active