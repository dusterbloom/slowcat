"""
Dictation mode processor for continuous, uninterrupted speech-to-text
Allows brain dump / stream of consciousness recording without AI responses
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    SystemFrame,
    CancelFrame
)
from pipecat.processors.frame_processor import FrameProcessor


class DictationModeProcessor(FrameProcessor):
    """
    Processor for continuous dictation mode - no interruptions, just transcription
    """
    
    def __init__(
        self,
        *,
        output_dir: str = "./data/dictation",
        file_prefix: str = "dictation",
        append_mode: bool = True,
        realtime_save: bool = True,
        save_interim: bool = False,
        language: str = "en",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.output_dir = Path(output_dir)
        self.file_prefix = file_prefix
        self.append_mode = append_mode
        self.realtime_save = realtime_save
        self.save_interim = save_interim
        
        # Get language-specific phrases
        self.translations = self.get_translations(language)
        self.mode_toggle_phrase = self.translations["mode_toggle_phrase"].lower()
        self.exit_phrase = self.translations["exit_phrase"].lower()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.dictation_active = False
        self.current_file: Optional[Path] = None
        self.session_start: Optional[datetime] = None
        self.buffer: List[str] = []
        self.file_handle = None
        
        logger.info(f"DictationMode initialized: output_dir={output_dir}, toggle='{self.mode_toggle_phrase}', exit='{self.exit_phrase}'")
    
    def get_translations(self, language: str) -> Dict[str, str]:
        """Returns a dictionary of translated strings for the given language."""
        translations = {
            "en": {
                "mode_toggle_phrase": "dictation mode",
                "exit_phrase": "stop dictation",
                "enter_notification": "📝 Dictation mode started. Speak freely - I'll just transcribe without responding.",
                "exit_notification": "📝 Dictation mode ended. Transcript saved to: {filename}"
            },
            "es": {
                "mode_toggle_phrase": "modo dictado", 
                "exit_phrase": "detener dictado",
                "enter_notification": "📝 Modo dictado iniciado. Habla libremente - solo transcribiré sin responder.",
                "exit_notification": "📝 Modo dictado terminado. Transcripción guardada en: {filename}"
            },
            "fr": {
                "mode_toggle_phrase": "mode dictée",
                "exit_phrase": "arrêter la dictée", 
                "enter_notification": "📝 Mode dictée activé. Parlez librement - je ne ferai que transcrire sans répondre.",
                "exit_notification": "📝 Mode dictée terminé. Transcription sauvegardée dans : {filename}"
            },
            "de": {
                "mode_toggle_phrase": "diktat-modus",
                "exit_phrase": "diktat stoppen",
                "enter_notification": "📝 Diktat-Modus gestartet. Sprechen Sie frei - ich werde nur transkribieren ohne zu antworten.",
                "exit_notification": "📝 Diktat-Modus beendet. Transkript gespeichert in: {filename}"
            },
            "it": {
                "mode_toggle_phrase": "modalità dettatura",
                "exit_phrase": "stop dettatura",
                "enter_notification": "📝 Modalità dettatura avviata. Parla liberamente - trascriverò solo senza rispondere.",
                "exit_notification": "📝 Modalità dettatura terminata. Trascrizione salvata in: {filename}"
            },
            "ja": {
                "mode_toggle_phrase": "ディクテーションモード",
                "exit_phrase": "ディクテーション停止",
                "enter_notification": "📝 ディクテーションモードが開始されました。自由に話してください - 応答せずに転写するだけです。",
                "exit_notification": "📝 ディクテーションモードが終了しました。転写が保存されました: {filename}"
            },
            "zh": {
                "mode_toggle_phrase": "听写模式",
                "exit_phrase": "停止听写",
                "enter_notification": "📝 听写模式已开始。请自由发言 - 我将只进行转录而不回应。",
                "exit_notification": "📝 听写模式已结束。转录已保存到：{filename}"
            },
            "pt": {
                "mode_toggle_phrase": "modo ditado",
                "exit_phrase": "parar ditado",
                "enter_notification": "📝 Modo ditado iniciado. Fale livremente - apenas transcreverei sem responder.",
                "exit_notification": "📝 Modo ditado terminado. Transcrição salva em: {filename}"
            }
        }
        
        # Return the translation for the specified language, fallback to English
        return translations.get(language, translations["en"])
    
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames - in dictation mode, block LLM responses"""
        
        # CRITICAL: Let parent class handle system frames first
        await super().process_frame(frame, direction)
        
        # Always forward the frame first (unless we handle it)
        should_forward = True
        
        # Check for mode toggle in transcriptions
        if isinstance(frame, TranscriptionFrame):
            if self._check_toggle_command(frame.text):
                await self._toggle_dictation_mode()
                # Don't forward the toggle command
                should_forward = False
        
        # In dictation mode, handle transcriptions specially
        if self.dictation_active and not should_forward:
            pass  # Already handled above
        elif self.dictation_active:
            # Block LLM processing and speech events
            if isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame, 
                                TTSStartedFrame, TTSStoppedFrame,
                                UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
                logger.debug(f"Blocking {type(frame).__name__} in dictation mode")
                should_forward = False
            
            # Block assistant text
            elif isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame):
                logger.debug("Blocking assistant TextFrame in dictation mode")
                should_forward = False
            
            # Save transcriptions but DON'T forward them to LLM
            elif isinstance(frame, TranscriptionFrame):
                await self._save_transcription(frame.text, final=True)
                # DON'T forward transcriptions in dictation mode - this prevents LLM from responding
                should_forward = False
                logger.debug(f"Saved but not forwarding transcription in dictation mode: {frame.text[:50]}...")
            
            # Optionally save interim transcriptions
            elif isinstance(frame, InterimTranscriptionFrame) and self.save_interim:
                await self._save_transcription(frame.text, final=False)
                # Also don't forward interim transcriptions
                should_forward = False
        
        # Forward frame if needed
        if should_forward:
            await self.push_frame(frame, direction)
    
    def _check_toggle_command(self, text: str) -> bool:
        """Check if text contains toggle command"""
        text_lower = text.lower().strip()
        
        # Check for language-specific toggle phrases
        toggle_phrases = [
            self.mode_toggle_phrase,  # Start dictation
            self.exit_phrase,         # Stop dictation
        ]
        
        return any(phrase in text_lower for phrase in toggle_phrases)
    
    async def _toggle_dictation_mode(self):
        """Toggle dictation mode on/off"""
        if not self.dictation_active:
            await self._start_dictation()
        else:
            await self._stop_dictation()
    
    async def _start_dictation(self):
        """Start dictation mode"""
        self.dictation_active = True
        self.session_start = datetime.now(timezone.utc)
        
        # Create new file
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.file_prefix}_{timestamp}.md"
        self.current_file = self.output_dir / filename
        
        # Open file handle for real-time writing
        if self.realtime_save:
            self.file_handle = open(self.current_file, 'a' if self.append_mode else 'w', encoding='utf-8')
            
            # Write header
            header = f"# Dictation Session\n\n"
            header += f"**Started**: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
            header += "---\n\n"
            self.file_handle.write(header)
            self.file_handle.flush()
        
        # Notify user
        notification = TextFrame(
            f"{self.translations['enter_notification']} Say '{self.exit_phrase}' to stop. Saving to: {filename}"
        )
        await self.push_frame(notification)
        
        logger.info(f"Started dictation mode: {self.current_file}")
    
    async def _stop_dictation(self):
        """Stop dictation mode"""
        self.dictation_active = False
        
        if self.file_handle:
            # Write footer
            now = datetime.now(timezone.utc)
            duration = int((now - self.session_start).total_seconds()) if self.session_start else 0
            
            footer = f"\n\n---\n\n"
            footer += f"**Ended**: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            footer += f"**Duration**: {duration // 60}m {duration % 60}s\n"
            footer += f"**Word Count**: ~{sum(len(text.split()) for text in self.buffer)} words\n"
            
            self.file_handle.write(footer)
            self.file_handle.close()
            self.file_handle = None
        
        # Notify user
        if self.current_file:
            notification = TextFrame(
                self.translations['exit_notification'].format(filename=self.current_file.name)
            )
            await self.push_frame(notification)
        
        # Reset state
        self.current_file = None
        self.session_start = None
        self.buffer.clear()
        
        logger.info("Stopped dictation mode")
    
    async def _save_transcription(self, text: str, final: bool = True):
        """Save transcription to file in real-time"""
        if not self.current_file:
            return
        
        # Add to buffer
        if final or not self.buffer or self.buffer[-1] != text:
            self.buffer.append(text)
        
        # Write to file
        if self.file_handle and (final or self.save_interim):
            # Calculate timestamp
            if self.session_start:
                elapsed = int((datetime.now(timezone.utc) - self.session_start).total_seconds())
                timestamp = f"[{elapsed // 60}:{elapsed % 60:02d}]"
            else:
                timestamp = "[0:00]"
            
            # Write with timestamp
            line = f"{timestamp} {text}\n\n"
            self.file_handle.write(line)
            self.file_handle.flush()  # Ensure it's written immediately
            
            logger.debug(f"Saved transcription: {text[:50]}...")
    
    async def cleanup(self):
        """Cleanup on shutdown"""
        try:
            if self.dictation_active:
                await self._stop_dictation()
        except Exception as e:
            logger.error(f"Error during dictation cleanup: {e}")
        
        await super().cleanup()
    
    def is_dictation_active(self) -> bool:
        """Check if dictation mode is active"""
        return self.dictation_active
    
    def get_current_file_path(self) -> Optional[str]:
        """Get current dictation file path"""
        return str(self.current_file) if self.current_file else None