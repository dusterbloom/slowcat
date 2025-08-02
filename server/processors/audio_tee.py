"""
Audio Tee Processor for splitting audio stream to multiple consumers
"""
import asyncio
import logging
from typing import List, Callable
from pipecat.frames.frames import InputAudioRawFrame, Frame, StartFrame, CancelFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)

class AudioTeeProcessor(FrameProcessor):
    """
    Splits audio stream to multiple consumers while passing through the main pipeline.
    This allows voice recognition to process audio without interfering with the main flow.
    """
    
    def __init__(self, enabled: bool = True):
        super().__init__()
        self._enabled = enabled
        self._audio_consumers: List[Callable] = []
    
    def register_audio_consumer(self, consumer: Callable):
        """Register a consumer that will receive audio data"""
        if consumer not in self._audio_consumers:
            self._audio_consumers.append(consumer)
    
    def unregister_audio_consumer(self, consumer: Callable):
        """Unregister an audio consumer"""
        if consumer in self._audio_consumers:
            self._audio_consumers.remove(consumer)
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames, splitting audio to registered consumers"""
        # Call parent to handle system frames
        await super().process_frame(frame, direction)
        
        # For audio frames, send to consumers in background
        if self._enabled and isinstance(frame, InputAudioRawFrame):
            # Create tasks for consumers but don't await them
            for consumer in self._audio_consumers:
                try:
                    # Run consumer in background task
                    asyncio.create_task(self._run_consumer(consumer, frame.audio))
                except Exception as e:
                    logger.error(f"Error creating consumer task: {e}")
        
        # ALWAYS pass through the frame immediately
        await self.push_frame(frame, direction)
    
    async def _run_consumer(self, consumer: Callable, audio_data: bytes):
        """Run a consumer in the background"""
        try:
            consumer(audio_data)
        except Exception as e:
            logger.error(f"Error in audio consumer: {e}")