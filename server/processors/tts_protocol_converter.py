"""
TTS Protocol Converter Processor

Converts TTSProtocolFrame to TTSTextFrame for WebRTC transport.
This should be placed AFTER the transcript processor in the pipeline
to prevent protocol messages from being added to conversation history.
"""

from pipecat.frames.frames import Frame, TTSTextFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from services.tts_message_protocol import TTSProtocolFrame


class TTSProtocolConverter(FrameProcessor):
    """
    Converts TTSProtocolFrame to TTSTextFrame for client transport.
    
    This processor should be placed near the end of the pipeline,
    AFTER the transcript processor but BEFORE the WebRTC transport.
    This ensures protocol messages don't get added to conversation
    history but still reach the client.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("TTSProtocolConverter initialized")
    
    async def process_frame(self, frame: Frame, direction=None):
        """Convert TTSProtocolFrame to TTSTextFrame, pass others through."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TTSProtocolFrame):
            # Convert protocol frame to text frame for client
            text_frame = TTSTextFrame(text=frame.protocol_data)
            logger.trace(f"Converting TTSProtocolFrame to TTSTextFrame: {frame.protocol_data[:50]}...")
            await self.push_frame(text_frame, direction)
        else:
            # Pass all other frames through unchanged
            await self.push_frame(frame, direction)