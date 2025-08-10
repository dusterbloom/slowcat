"""
Wrapper service that adds accuracy enhancement to existing Sherpa services
"""
import asyncio
from typing import Optional
from loguru import logger

from .accuracy_enhancement import AdvancedAccuracyEnhancer
from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame
from pipecat.services.stt_service import STTService


class SherpaWithAccuracyEnhancement(STTService):
    """
    Wrapper service that adds accuracy enhancement to any Sherpa STT service.
    
    Usage:
        # Wrap existing Sherpa service
        base_service = SherpaStreamingSTTService(...)
        enhanced_service = SherpaWithAccuracyEnhancement(base_service)
    """
    
    def __init__(
        self,
        base_service: STTService,
        enable_accuracy_enhancement: bool = True,
        accuracy_model: str = "qwen3-1.7b:2",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_service = base_service
        self.enable_accuracy_enhancement = enable_accuracy_enhancement
        
        if self.enable_accuracy_enhancement:
            try:
                self.accuracy_enhancer = AdvancedAccuracyEnhancer()
                logger.info("Accuracy enhancement initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize accuracy enhancement: {e}")
                self.accuracy_enhancer = None
                self.enable_accuracy_enhancement = False
        else:
            self.accuracy_enhancer = None
    
    async def transcribe(self, audio_data: bytes, context: str = "") -> str:
        """Transcribe audio with optional accuracy enhancement"""
        # Get base transcription
        raw_transcription = await self.base_service.transcribe(audio_data, context)
        
        # Apply accuracy enhancement if enabled
        if self.enable_accuracy_enhancement and self.accuracy_enhancer:
            try:
                enhancement_result = await self.accuracy_enhancer.enhance_accuracy(
                    raw_transcription,
                    confidence=0.65,  # Default confidence, could be passed from base service
                    context=context
                )
                logger.debug(f"Accuracy enhancement applied: {len(enhancement_result.corrections_applied)} corrections")
                return enhancement_result.corrected_text
            except Exception as e:
                logger.warning(f"Accuracy enhancement failed, using raw transcription: {e}")
        
        return raw_transcription
    
    async def stream_transcribe(self, audio_stream, context: str = ""):
        """Stream transcription with optional accuracy enhancement for final results"""
        async for frame in self.base_service.stream_transcribe(audio_stream, context):
            # For interim results, pass through unchanged for low latency
            if isinstance(frame, InterimTranscriptionFrame):
                yield frame
            # For final results, apply accuracy enhancement
            elif isinstance(frame, TranscriptionFrame) and self.enable_accuracy_enhancement and self.accuracy_enhancer:
                try:
                    enhancement_result = await self.accuracy_enhancer.enhance_accuracy(
                        frame.text,
                        confidence=getattr(frame, 'confidence', 0.65),
                        context=context
                    )
                    # Create new frame with enhanced text
                    enhanced_frame = TranscriptionFrame(
                        text=enhancement_result.corrected_text,
                        timestamp=frame.timestamp,
                        confidence=frame.confidence if hasattr(frame, 'confidence') else 0.65
                    )
                    yield enhanced_frame
                except Exception as e:
                    logger.warning(f"Accuracy enhancement failed for stream, using raw: {e}")
                    yield frame
            else:
                # Pass through unchanged
                yield frame
    
    # Delegate other methods to base service
    def __getattr__(self, name):
        """Delegate unknown attributes to base service"""
        return getattr(self.base_service, name)