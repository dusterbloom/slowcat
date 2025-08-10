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
        self._pending_system_message: Optional[str] = None
        self._first_speaker_change = True  # Track if this is the first speaker change
    
    def update_speaker(self, speaker_data: Dict[str, Any]):
        """Update current speaker information"""
        self.current_speaker = speaker_data.get('speaker_name', self.unknown_speaker_name)
        self.speaker_confidence = speaker_data.get('confidence', 0.0)
        
        # Track speaker history
        if self.current_speaker != self.unknown_speaker_name:
            is_new_speaker = self.current_speaker not in self.speaker_history
            self.speaker_history[self.current_speaker] = datetime.now()
            
            # If this is the first speaker change and we have a named speaker with high confidence,
            # it means they were recognized from a previous session
            if self._first_speaker_change and self.speaker_confidence > 0.7 and not self.current_speaker.startswith('Speaker_'):
                self._pending_system_message = f"[System: Recognized returning user {self.current_speaker} from previous sessions.]"
                logger.info(f"Queued system message for recognized returning speaker: {self.current_speaker}")
                self._first_speaker_change = False
            # If this is a returning speaker (already in history in this session), notify the LLM
            elif not is_new_speaker and self.speaker_confidence > 0.7:
                self._pending_system_message = f"[System: {self.current_speaker} has returned to the conversation.]"
                logger.info(f"Queued system message for returning speaker: {self.current_speaker}")
            else:
                self._first_speaker_change = False
        
        logger.info(f"Speaker context updated: {self.current_speaker} (confidence: {self.speaker_confidence:.2f})")
    
    def handle_speaker_enrolled(self, enrollment_data: Dict[str, Any]):
        """Handle when a new speaker is enrolled"""
        speaker_id = enrollment_data.get('speaker_id', '')
        speaker_name = enrollment_data.get('speaker_name', '')
        auto_enrolled = enrollment_data.get('auto_enrolled', False)
        needs_name = enrollment_data.get('needs_name', False)
        
        if auto_enrolled:
            if needs_name:
                logger.info(f"New speaker auto-enrolled: {speaker_id}")
                # Queue a system message to trigger name question
                self._pending_system_message = f"[System: New speaker detected and enrolled as {speaker_id}. This is their first time using the system.]"
                logger.info(f"Queued system message for new speaker")
            else:
                # Speaker was auto-enrolled but already has a name (returning speaker)
                logger.info(f"Returning speaker auto-enrolled: {speaker_name or speaker_id}")
                self._pending_system_message = f"[System: {speaker_name or speaker_id} has returned to the conversation.]"
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames and add speaker context where appropriate"""
        # Handle system frames through parent
        await super().process_frame(frame, direction)
        
        # Modify frames in-place without blocking
        if isinstance(frame, TranscriptionFrame):
            # Check if we have a pending system message to prepend
            if self._pending_system_message:
                # Prepend the system message to the transcription
                original_text = frame.text
                frame.text = f"{self._pending_system_message} {original_text}"
                logger.info(f"Injected system message into transcription: {self._pending_system_message}")
                self._pending_system_message = None  # Clear after using
            
            # Add speaker info to the transcription
            if self.current_speaker != self.unknown_speaker_name and self.speaker_confidence > 0.7:
                # Only add speaker context if we're confident
                frame.user_id = self.current_speaker
                logger.debug(f"Added speaker context to transcription: {self.current_speaker}")
        
        elif isinstance(frame, LLMMessagesFrame):
            # Always check for pending system messages to inject
            if self._pending_system_message:
                # Find the last user message to inject our system message before it
                last_user_idx = -1
                for i in range(len(frame.messages) - 1, -1, -1):
                    if frame.messages[i].get("role") == "user":
                        last_user_idx = i
                        break
                
                if last_user_idx >= 0:
                    # Inject our system message right before the last user message
                    system_msg = {
                        "role": "system",
                        "content": self._pending_system_message
                    }
                    frame.messages.insert(last_user_idx, system_msg)
                    logger.info(f"Injected system message into LLM context: {self._pending_system_message}")
                    self._pending_system_message = None  # Clear after using
            
            # Optionally add current speaker info if format_style is "system"
            if self.format_style == "system" and self.current_speaker != self.unknown_speaker_name and self.speaker_confidence > 0.7:
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
    
    async def cleanup(self):
        """Clean up speaker context manager resources"""
        logger.debug("SpeakerContextManagerProcessor cleanup completed")