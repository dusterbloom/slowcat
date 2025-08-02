"""
Speaker Context Processor - Adds speaker information to LLM context
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, StartFrame, CancelFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class SpeakerContextProcessor(FrameProcessor):
    """
    Adds speaker identification context to transcriptions and LLM messages.
    This helps the LLM know who is speaking.
    """
    
    def __init__(self, format_style: str = "natural", unknown_speaker_name: str = "User"):
        super().__init__()
        self.format_style = format_style
        self.unknown_speaker_name = unknown_speaker_name
        self.current_speaker = unknown_speaker_name
        self.speaker_confidence = 0.0
        self.speaker_history: Dict[str, datetime] = {}
    
    def update_speaker(self, speaker_data: Dict[str, Any]):
        """Update current speaker information"""
        self.current_speaker = speaker_data.get('speaker_name', self.unknown_speaker_name)
        self.speaker_confidence = speaker_data.get('confidence', 0.0)
        
        # Track speaker history
        if self.current_speaker != self.unknown_speaker_name:
            self.speaker_history[self.current_speaker] = datetime.now()
        
        logger.info(f"Speaker context updated: {self.current_speaker} (confidence: {self.speaker_confidence:.2f})")
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames and add speaker context where appropriate"""
        # Handle system frames through parent
        await super().process_frame(frame, direction)
        
        # Modify frames in-place without blocking
        if isinstance(frame, TranscriptionFrame):
            # Add speaker info to the transcription
            if self.current_speaker != self.unknown_speaker_name and self.speaker_confidence > 0.7:
                # Only add speaker context if we're confident
                frame.user = self.current_speaker
                logger.debug(f"Added speaker context to transcription: {self.current_speaker}")
        
        elif isinstance(frame, LLMMessagesFrame) and self.format_style == "system":
            # Optionally add a system message about the current speaker
            if self.current_speaker != self.unknown_speaker_name and self.speaker_confidence > 0.7:
                speaker_msg = {
                    "role": "system",
                    "content": f"The person speaking is {self.current_speaker}."
                }
                # Insert at the beginning of messages after the main system prompt
                if len(frame.messages) > 1:
                    frame.messages.insert(1, speaker_msg)
        
        # ALWAYS pass through ALL frames
        await self.push_frame(frame, direction)
    
    def get_speaker_summary(self) -> str:
        """Get a summary of speakers in the conversation"""
        if not self.speaker_history:
            return "No identified speakers yet."
        
        speakers = list(self.speaker_history.keys())
        if len(speakers) == 1:
            return f"Speaking with {speakers[0]}."
        else:
            return f"Speaking with {len(speakers)} people: {', '.join(speakers)}."