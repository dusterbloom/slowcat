"""
Video Sampler Processor - Samples video frames periodically for vision analysis
"""
import asyncio
import logging
import time
from typing import Optional
from pipecat.frames.frames import Frame, InputImageRawFrame, VisionImageRawFrame, StartFrame, CancelFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class VideoSamplerProcessor(FrameProcessor):
    """
    Samples video frames at intervals and converts them to VisionImageRawFrame
    for LLM vision analysis.
    """
    
    def __init__(self, sample_interval: float = 5.0, enabled: bool = True):
        """
        Initialize video sampler.
        
        Args:
            sample_interval: Seconds between frame samples (default 5.0)
            enabled: Whether sampling is enabled
        """
        super().__init__()
        self._sample_interval = sample_interval
        self._enabled = enabled
        self._last_sample_time = 0
        self._frame_count = 0
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames and sample video periodically"""
        # Handle system frames through parent
        await super().process_frame(frame, direction)
        
        # Sample video frames at intervals
        if self._enabled and isinstance(frame, InputImageRawFrame):
            self._frame_count += 1
            current_time = time.time()
            
            # Check if it's time to sample
            if current_time - self._last_sample_time >= self._sample_interval:
                self._last_sample_time = current_time
                
                # Create a vision frame without automatic description request
                # The LLM will have access to the image context for answering questions
                vision_frame = VisionImageRawFrame(
                    image=frame.image,
                    size=frame.size,
                    format=frame.format,
                    text=None  # Don't automatically describe every frame
                )
                
                logger.info(f"📸 Sampled video frame #{self._frame_count} for vision context")
                
                # Send the vision frame downstream
                await self.push_frame(vision_frame, direction)
        
        # Always pass through all frames
        await self.push_frame(frame, direction)