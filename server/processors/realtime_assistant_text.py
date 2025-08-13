from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame, StartFrame, EndFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger


class RealtimeAssistantText(FrameProcessor):
    """
    Processor that intercepts assistant TextFrames and routes them to multiple destinations:
    1. TTS service for audio generation
    2. WebSocket output for real-time client display  
    3. Assistant context aggregator for conversation history
    
    This enables smooth incremental transcript updates during TTS playback.
    """

    def __init__(self, assistant_aggregator=None):
        super().__init__()
        self.assistant_aggregator = assistant_aggregator
        self._started = False

    async def process_frame(self, frame: Frame, direction=None):
        # Always call super().process_frame first
        await super().process_frame(frame, direction)

        # Handle StartFrame
        if isinstance(frame, StartFrame):
            self._started = True
            # Forward to assistant aggregator
            if self.assistant_aggregator:
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                except Exception as e:
                    logger.error(f"Failed to forward StartFrame to aggregator: {e}")
            # Continue pipeline flow
            await self.push_frame(frame, direction)
            return

        # Handle EndFrame
        if isinstance(frame, EndFrame):
            if self.assistant_aggregator:
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                except Exception as e:
                    logger.error(f"Failed to forward EndFrame to aggregator: {e}")
            await self.push_frame(frame, direction)
            return

        # Handle assistant TextFrames (not user transcriptions)
        if isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame):
            # 1. Forward to assistant aggregator for conversation history
            if self.assistant_aggregator and self._started:
                try:
                    await self.assistant_aggregator.process_frame(frame, direction)
                    # Successfully forwarded to aggregator
                except Exception as e:
                    logger.error(f"Failed to forward to assistant aggregator: {e}")

            # 2. Continue pipeline flow - TextFrame goes to TTS which will:
            #    - Generate audio frames
            #    - Generate TTSProtocolFrames with the text chunks
            # The TTSProtocolFrames will be converted by TTSProtocolConverter
            # and sent to the client for real-time transcription display
            await self.push_frame(frame, direction)
            return

        # For all other frames, pass through the pipeline
        await self.push_frame(frame, direction)