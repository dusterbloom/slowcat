"""
KeepAliveProcessor - injects periodic MetricsFrame to keep the pipeline active.

Prevents idle-timeout cancellations during long LLM/TTS activity or quiet user pauses.
"""

import asyncio
from typing import Optional
from loguru import logger

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import MetricsFrame, Frame, StartFrame


class KeepAliveProcessor(FrameProcessor):
    """Periodically emits a MetricsFrame to reset idle timers."""

    def __init__(self, interval_seconds: float = 12.0, **kwargs):
        super().__init__(**kwargs)
        self._interval = max(3.0, float(interval_seconds))
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)  # REQUIRED - handles initialization state
        
        if isinstance(frame, StartFrame):
            # Push StartFrame downstream IMMEDIATELY
            await self.push_frame(frame, direction)
            # Then start keepalive on initialization
            if not self._running:
                self._running = True
                self._task = asyncio.create_task(self._run())
                logger.info(f"💓 Server keepalive enabled ({self._interval}s interval)")
            return
        
        # Forward all other frames
        await self.push_frame(frame, direction)

    async def _run(self):
        try:
            while self._running:
                await asyncio.sleep(self._interval)
                try:
                    # Inject a lightweight MetricsFrame downstream; ignored by TTS/LLM but
                    # sufficient to update pipeline activity and prevent idle cancellation.
                    await self.push_frame(MetricsFrame(), FrameDirection.DOWNSTREAM)
                except Exception:
                    # Avoid crashing on transient pipeline states
                    pass
        except asyncio.CancelledError:
            pass

    async def cleanup(self):
        try:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except Exception:
                    pass
                self._task = None
        finally:
            await super().cleanup()

