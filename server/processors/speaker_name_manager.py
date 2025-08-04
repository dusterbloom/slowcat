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
        self.asked_for_name_time = None
        self.name_patterns = [
            # English patterns - more specific
            r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            # Italian patterns
            r"(?:mi chiamo|sono|il mio nome è)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            # Just a proper name (must be capitalized and 4+ letters)
            r"^([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)?)\.?$",
            # Name with "It's" / "È" - must be capitalized
            r"(?:it's|its|è)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            # "Name is X" / "Nome è X" - must be capitalized
            r"(?:name is|nome è)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
    
    def start_name_collection(self, speaker_id: str):
        """Start waiting for a name for the given speaker ID"""
        # Check if we already have a name for this speaker
        if hasattr(self.voice_recognition, 'speaker_names') and speaker_id in self.voice_recognition.speaker_names:
            existing_name = self.voice_recognition.speaker_names[speaker_id]
            logger.info(f"Speaker {speaker_id} already has name: {existing_name}, skipping name collection")
            return
            
        self.waiting_for_name = True
        self.pending_speaker_id = speaker_id
        logger.info(f"Started name collection for {speaker_id}")
    
    def extract_name(self, text: str) -> Optional[str]:
        """Try to extract a name from the text"""
        text = text.strip()
        
        # Common words to reject as names
        common_words = {'so', 'what', 'yes', 'no', 'ok', 'okay', 'hi', 'hello', 'hey', 
                       'thanks', 'thank', 'you', 'please', 'sorry', 'the', 'and', 'but',
                       'it', 'is', 'was', 'are', 'been', 'have', 'has', 'had', 'do',
                       'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'}
        
        # Try each pattern
        for pattern in self.name_patterns:
            match = re.search(pattern, text)  # Case sensitive now
            if match:
                name = match.group(1).strip()
                # Validate it's not a common word
                if name.lower() not in common_words and len(name) >= 3:
                    # Capitalize properly
                    name = ' '.join(word.capitalize() for word in name.split())
                    return name
        
        # If it's a single capitalized word that looks like a name
        words = text.split()
        if len(words) == 1 and len(words[0]) >= 4 and words[0][0].isupper():
            name = words[0].rstrip('.')
            if name.lower() not in common_words:
                return name
        
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