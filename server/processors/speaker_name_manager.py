"""
Speaker Name Manager - Handles updating speaker names after enrollment
"""
import logging
import re
from typing import Optional, Dict, Any, Callable
from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class SpeakerNameManager(FrameProcessor):
    """
    Manages the process of collecting speaker names after auto-enrollment.
    Monitors conversation for name responses and updates speaker profiles.
    """
    
    def __init__(self, voice_recognition):
        super().__init__()
        self.voice_recognition = voice_recognition
        self.waiting_for_name = False
        self.pending_speaker_id = None
        self.name_patterns = [
            # English patterns
            r"(?:my name is|i'm|i am|call me)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            # Italian patterns
            r"(?:mi chiamo|sono|il mio nome è)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            # Just the name
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\.?$",
            # Name with "It's" / "È"
            r"(?:it's|its|è)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            # "Name is X" / "Nome è X"
            r"(?:name is|nome è)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
        ]
    
    def start_name_collection(self, speaker_id: str):
        """Start waiting for a name for the given speaker ID"""
        self.waiting_for_name = True
        self.pending_speaker_id = speaker_id
        logger.info(f"Started name collection for {speaker_id}")
    
    def extract_name(self, text: str) -> Optional[str]:
        """Try to extract a name from the text"""
        text = text.strip()
        
        # Try each pattern
        for pattern in self.name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Capitalize properly
                name = ' '.join(word.capitalize() for word in name.split())
                return name
        
        # If it's a single word that looks like a name (capitalized)
        words = text.split()
        if len(words) <= 2 and words[0][0].isupper():
            return ' '.join(words).rstrip('.')
        
        return None
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames and look for name responses"""
        # Handle system frames through parent
        await super().process_frame(frame, direction)
        
        # Only process if we're waiting for a name
        if self.waiting_for_name and self.pending_speaker_id:
            if isinstance(frame, TranscriptionFrame):
                # Check if this transcription contains a name
                logger.debug(f"Checking for name in: '{frame.text}'")
                name = self.extract_name(frame.text)
                if name:
                    logger.info(f"Detected name '{name}' for {self.pending_speaker_id}")
                    
                    # Update the speaker name in voice recognition
                    self.voice_recognition.update_speaker_name(self.pending_speaker_id, name)
                    
                    # Reset state
                    self.waiting_for_name = False
                    self.pending_speaker_id = None
                    
                    # Log the successful name update
                    logger.info(f"Successfully updated speaker name: {self.pending_speaker_id} -> {name}")
        
        # Always pass through the frame
        await self.push_frame(frame, direction)