"""
DJ Mode Configuration Handler
Handles switching TTS voice and system prompt when entering/exiting DJ mode
"""

from typing import Optional, Dict, Any
from loguru import logger

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.tts_service import TTSService
from pipecat.services.llm_service import LLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext


class DJModeConfigHandler(FrameProcessor):
    """
    Handles configuration changes for DJ mode
    Intercepts DJModeConfigFrame and updates TTS/LLM services
    """
    
    def __init__(
        self,
        *,
        tts: Optional[TTSService] = None,
        llm_context: Optional[OpenAILLMContext] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tts = tts
        self.llm_context = llm_context
        self.original_voice = None
        self.original_system_prompt = None
        
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames and handle DJ mode configuration"""
        
        # CRITICAL: Let parent class handle system frames first (StartFrame, etc.)
        await super().process_frame(frame, direction)
        
        # Import here to avoid circular import
        from processors.music_mode import DJModeConfigFrame
        
        if isinstance(frame, DJModeConfigFrame):
            if frame.entering_mode:
                await self._enter_dj_mode(frame)
            else:
                await self._exit_dj_mode(frame)
            # Don't forward this frame
            return
        
        # Forward all other frames
        await self.push_frame(frame, direction)
    
    async def _enter_dj_mode(self, config_frame):
        """Apply DJ mode configuration"""
        logger.info("🎙️ Entering DJ mode - updating voice and prompt")
        
        # Store original settings
        if self.tts and hasattr(self.tts, '_voice'):
            self.original_voice = self.tts._voice
            
        if self.llm_context:
            # Get current system prompt
            messages = self.llm_context.get_messages()
            for msg in messages:
                if msg.get('role') == 'system':
                    self.original_system_prompt = msg.get('content', '')
                    break
        
        # Apply DJ settings
        if config_frame.voice and self.tts and hasattr(self.tts, '_voice'):
            self.tts._voice = config_frame.voice
            logger.info(f"🎤 Changed TTS voice to: {config_frame.voice}")
        
        if config_frame.system_prompt and self.llm_context:
            # Update system prompt by modifying the messages
            messages = self.llm_context.get_messages()
            
            # Find and update the system message
            system_message_found = False
            for msg in messages:
                if msg.get('role') == 'system':
                    msg['content'] = config_frame.system_prompt
                    system_message_found = True
                    break
            
            # If no system message found, add one at the beginning
            if not system_message_found:
                messages.insert(0, {"role": "system", "content": config_frame.system_prompt})
            
            # Update the context with modified messages
            self.llm_context.set_messages(messages)
            logger.info("📝 Updated system prompt for DJ mode")
    
    async def _exit_dj_mode(self, config_frame):
        """Restore original configuration"""
        logger.info("🎙️ Exiting DJ mode - restoring original settings")
        
        # Restore original voice
        if self.original_voice and self.tts and hasattr(self.tts, '_voice'):
            self.tts._voice = self.original_voice
            logger.info(f"🎤 Restored TTS voice to: {self.original_voice}")
        
        # Restore original system prompt
        if self.original_system_prompt and self.llm_context:
            messages = self.llm_context.get_messages()
            
            # Find and restore the system message
            for msg in messages:
                if msg.get('role') == 'system':
                    msg['content'] = self.original_system_prompt
                    break
            
            # Update the context with modified messages
            self.llm_context.set_messages(messages)
            logger.info("📝 Restored original system prompt")