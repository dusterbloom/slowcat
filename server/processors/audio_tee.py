"""
Audio Tee Processor for splitting audio stream to multiple consumers
"""
import asyncio
import logging
from typing import List, Callable, Optional
from pipecat.frames.frames import InputAudioRawFrame, AudioRawFrame, UserAudioRawFrame, Frame, StartFrame, EndFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

logger = logging.getLogger(__name__)

class AudioTeeProcessor(FrameProcessor):
    """
    Splits audio stream to multiple consumers while passing through the main pipeline.
    This allows voice recognition to process audio without interfering with the main flow.
    """
    
    def __init__(self, *, enabled: bool = True, audio_consumers: Optional[List[Callable]] = None, **kwargs):
        super().__init__(**kwargs)
        self._enabled = enabled
        self._audio_consumers = audio_consumers or []
        self._consumer_tasks: List[asyncio.Task] = []
        
    def register_audio_consumer(self, consumer: Callable) -> None:
        """Register a consumer that will receive audio frames"""
        if consumer not in self._audio_consumers:
            self._audio_consumers.append(consumer)
            logger.debug(f"Registered audio consumer: {consumer}")
    
    def unregister_audio_consumer(self, consumer: Callable) -> None:
        """Unregister an audio consumer"""
        if consumer in self._audio_consumers:
            self._audio_consumers.remove(consumer)
            logger.debug(f"Unregistered audio consumer: {consumer}")
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable audio duplication."""
        self._enabled = enabled
        logger.info(f"Audio tee {'enabled' if enabled else 'disabled'}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, duplicating audio frames to registered consumers"""
        # Debug logging for first frame
        if not hasattr(self, '_first_frame_logged'):
            self._first_frame_logged = True
            logger.info(f"AudioTee.process_frame called! Frame type: {type(frame).__name__}, enabled: {self._enabled}")
            
        await super().process_frame(frame, direction)
        
        # Handle start/end frames for lifecycle management
        if isinstance(frame, StartFrame):
            logger.debug("AudioTeeProcessor started")
        elif isinstance(frame, EndFrame):
            # Cancel any pending consumer tasks
            for task in self._consumer_tasks:
                if not task.done():
                    task.cancel()
            self._consumer_tasks.clear()
        
        # Duplicate audio frames to consumers if enabled
        if isinstance(frame, (AudioRawFrame, InputAudioRawFrame, UserAudioRawFrame)):
            if not hasattr(self, '_logged_audio'):
                self._logged_audio = True
                logger.debug(f"AudioTee received audio frame, enabled={self._enabled}, consumers={len(self._audio_consumers)}")
                
        if (self._enabled and 
            isinstance(frame, (AudioRawFrame, InputAudioRawFrame, UserAudioRawFrame)) and
            self._audio_consumers):
            
            # Get sample rate from frame if available
            sample_rate = getattr(frame, 'sample_rate', 16000)
            
            # Create tasks for each consumer to process audio in parallel
            for consumer in self._audio_consumers:
                try:
                    # Create a copy of the audio data to avoid mutations
                    audio_copy = bytes(frame.audio)
                    frame_copy = type(frame)(
                        audio=audio_copy,
                        sample_rate=sample_rate,
                        num_channels=getattr(frame, 'num_channels', 1)
                    )
                    
                    # Schedule consumer processing without blocking
                    task = asyncio.create_task(
                        self._process_consumer(consumer, frame_copy, sample_rate)
                    )
                    self._consumer_tasks.append(task)
                    
                    # Clean up completed tasks
                    self._consumer_tasks = [
                        t for t in self._consumer_tasks if not t.done()
                    ]
                    
                except Exception as e:
                    logger.error(f"Error duplicating audio for consumer {consumer}: {e}")
        
        # Always pass the original frame downstream
        await self.push_frame(frame, direction)
    
    async def _process_consumer(self, consumer: Callable, frame: Frame, sample_rate: int):
        """Process audio frame with a consumer, handling errors gracefully"""
        try:
            await consumer(frame, sample_rate)
        except Exception as e:
            logger.error(f"Error in audio consumer {consumer}: {e}")