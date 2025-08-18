"""
Final Frame Filter Processor

Filters out partial/streaming frames and only passes final complete frames
to prevent duplication in context aggregators.
"""

import asyncio
from typing import Dict, Optional
from loguru import logger

from pipecat.frames.frames import (
    Frame, StartFrame, EndFrame, TextFrame, LLMFullResponseEndFrame,
    LLMFullResponseStartFrame, LLMTextFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


class FinalFrameFilter(FrameProcessor):
    """
    Filters out partial/streaming frames and only passes final complete frames.
    
    This prevents duplication issues in context aggregators by ensuring only
    complete, final responses are added to conversation history.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_response_parts = []
        self._is_collecting_response = False
        
        logger.info("🎯 FinalFrameFilter initialized - will filter partial frames")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, filtering out partials and passing only finals"""
        
        # Always call parent first
        await super().process_frame(frame, direction)
        
        # Handle StartFrame specially
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            logger.debug("FinalFrameFilter: StartFrame processed and forwarded")
            return
        
        # Handle EndFrame
        if isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)
            logger.debug("FinalFrameFilter: EndFrame processed and forwarded")
            return
        
        # Filter logic for text frames
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_text_frame(frame, direction)
            return
        
        # Handle LLM response frames
        if isinstance(frame, LLMFullResponseStartFrame):
            logger.debug("🎯 FinalFrameFilter: LLM response started - collecting parts")
            self._is_collecting_response = True
            self._current_response_parts = []
            # Don't forward start frame
            return
        
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._is_collecting_response:
                logger.debug("🎯 FinalFrameFilter: LLM response ended - forwarding complete response")
                await self._forward_complete_response(direction)
            self._is_collecting_response = False
            self._current_response_parts = []
            # Don't forward end frame
            return
        
        # Handle LLM text frames (streaming parts)
        if isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._is_collecting_response:
                if frame.text and frame.text.strip():
                    self._current_response_parts.append(frame.text)
                    logger.debug(f"🎯 FinalFrameFilter: Collected LLM part: '{frame.text[:30]}...'")
                # Don't forward partial
                return
        
        # Forward all other frames normally
        await self.push_frame(frame, direction)
    
    async def _handle_text_frame(self, frame: TextFrame, direction: FrameDirection):
        """Handle text frames - collect parts or forward finals"""
        
        if self._is_collecting_response:
            # Collect partial response
            if frame.text and frame.text.strip():
                self._current_response_parts.append(frame.text)
                logger.debug(f"🎯 FinalFrameFilter: Collected part: '{frame.text[:30]}...'")
            # Don't forward partial
            return
        else:
            # Forward standalone text frames (should be final)
            logger.debug(f"🎯 FinalFrameFilter: Forwarding standalone text: '{frame.text[:30] if frame.text else 'None'}...'")
            await self.push_frame(frame, direction)
    
    async def _forward_complete_response(self, direction: FrameDirection):
        """Forward the complete response as a single TextFrame"""
        
        if self._current_response_parts:
            # Join all parts into complete response
            complete_text = ''.join(self._current_response_parts).strip()
            
            if complete_text:
                # Create final text frame
                final_frame = TextFrame(complete_text)
                await self.push_frame(final_frame, direction)
                logger.info(f"🎯 FinalFrameFilter: ✅ Forwarded complete response: '{complete_text[:50]}...'")
            else:
                logger.debug("🎯 FinalFrameFilter: Empty response, not forwarding")
        else:
            logger.debug("🎯 FinalFrameFilter: No response parts collected")