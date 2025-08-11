from pipecat.frames.frames import Frame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger

class RealtimeAssistantText(FrameProcessor):
    """
    Processor that intercepts assistant TextFrames in real time
    and immediately forwards them to both TTS and WebSocket output.
    This enables smooth incremental transcript updates during TTS playback.
    """

    def __init__(self, tts_processor=None, websocket_output=None):
        super().__init__()
        self.tts_processor = tts_processor
        self.websocket_output = websocket_output
        # Patch compatibility for older Pipecat code paths expecting _FrameProcessor__input_queue
        if not hasattr(self, "_FrameProcessor__input_queue"):
            logger.warning("[RealtimeAssistantText] Patching missing _FrameProcessor__input_queue for backward compatibility")
            setattr(self, "_FrameProcessor__input_queue", None)

    async def process_frame(self, frame: Frame, direction=None):
        # IMPORTANT: Always call super().process_frame first to ensure base class handling
        # This is critical for StartFrame processing and the _check_started mechanism
        await super().process_frame(frame, direction)

        # Pass through everything unchanged unless it's a TextFrame that we want to intercept
        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        # logger.debug(f"[RealtimeAssistantText] Intercepted TextFrame: {frame.text[:100]!r}")

        # Forward to TTS processor immediately if available and not part of circular loop
        if self.tts_processor and self.tts_processor is not self:
            try:
                await self.tts_processor.push_frame(frame)
                logger.trace("[RealtimeAssistantText] Forwarded TextFrame to TTS processor")
            except Exception as e:
                logger.error(f"[RealtimeAssistantText] Failed to forward to TTS processor: {e}")

        # Forward to WebSocket output immediately if available and not part of circular loop
        if self.websocket_output and self.websocket_output is not self:
            try:
                await self.websocket_output.push_frame(frame)
                logger.trace("[RealtimeAssistantText] Forwarded TextFrame to WebSocket output")
            except Exception as e:
                logger.error(f"[RealtimeAssistantText] Failed to forward to WebSocket output: {e}")

        # Continue normal pipeline flow
        await self.push_frame(frame, direction)