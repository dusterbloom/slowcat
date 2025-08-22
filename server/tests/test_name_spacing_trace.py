"""
Trace name spacing artifacts across a minimal pipeline to identify root cause.

This test uses PipelineTracer to record TextFrame content as it flows through
processors, verifying whether any processor introduces or fixes the spaced-name
artifact (e.g., "Pe p py").
"""

import asyncio
import time
import pytest
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import Frame, StartFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline

from debug.pipeline_tracer import PipelineTracer
from processors.response_formatter import ResponseFormatterProcessor
from processors.streaming_deduplicator import StreamingDeduplicator


class TracedProcessor(FrameProcessor):
    """FrameProcessor that reports to PipelineTracer."""
    def __init__(self, tracer: PipelineTracer, name: str):
        super().__init__()
        self._tracer = tracer
        self._name = name

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        evt = self._tracer.trace_frame_received(self, frame, direction)
        start = time.time()
        try:
            await self.push_frame(frame, direction)
            self._tracer.trace_frame_forwarded(self, frame, direction, evt, start)
        except Exception as e:
            self._tracer.trace_error(self, frame, direction, e)
            raise


class Source(FrameProcessor):
    """Push a StartFrame and a single TextFrame with provided content."""
    def __init__(self, tracer: PipelineTracer, text: str):
        super().__init__()
        self._tracer = tracer
        self._text = text

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # On first StartFrame propagate our text downstream
        if isinstance(frame, StartFrame):
            # Emit the text as a downstream TextFrame
            tf = TextFrame(self._text)
            evt = self._tracer.trace_frame_received(self, tf, FrameDirection.DOWNSTREAM)
            start = time.time()
            await self.push_frame(tf, FrameDirection.DOWNSTREAM)
            self._tracer.trace_frame_forwarded(self, tf, FrameDirection.DOWNSTREAM, evt, start)
        await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    """Capture final TextFrames for assertions."""
    def __init__(self, tracer: PipelineTracer):
        super().__init__()
        self._tracer = tracer
        self.received_texts = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        evt = self._tracer.trace_frame_received(self, frame, direction)
        start = time.time()
        if isinstance(frame, TextFrame):
            self.received_texts.append(frame.text)
        await self.push_frame(frame, direction)
        self._tracer.trace_frame_forwarded(self, frame, direction, evt, start)


async def run_pipeline_with_text(text: str, formatter_mode: str = 'full'):
    tracer = PipelineTracer(output_dir="server/debug/traces")
    tracer.start_tracing()

    src = Source(tracer, text)
    fmt = ResponseFormatterProcessor(mode=formatter_mode)
    dedup = StreamingDeduplicator()
    thru = TracedProcessor(tracer, "Thru")
    sink = Sink(tracer)

    pipeline = Pipeline([src, fmt, dedup, thru, sink])

    # Kick off with a StartFrame
    await pipeline.push_frame(StartFrame(), FrameDirection.UPSTREAM)
    # Let the pipeline settle
    await asyncio.sleep(0.05)
    report = tracer.stop_tracing()
    return sink.received_texts, report


@pytest.mark.asyncio
async def test_name_spacing_trace_preserves_correct_name(tmp_path):
    text = "Hello, Peppy!"
    outputs, report = await run_pipeline_with_text(text)
    # Ensure output contains correct name
    assert any("Peppy" in s for s in outputs), f"Expected 'Peppy' in outputs, got: {outputs}"
    # Report file should exist
    assert Path(report).exists(), f"Trace report not found: {report}"


@pytest.mark.asyncio
async def test_name_spacing_trace_identifies_spaced_name_unfixed(tmp_path):
    text = "Hello, Pe p py !"
    outputs, report = await run_pipeline_with_text(text)
    # Current formatter collapses only simple 'P eppy' patterns; 'Pe p py' may persist
    persisted = any("Pe p py" in s for s in outputs)
    assert persisted, (
        "Expected 'Pe p py' to persist (identifying model output as root cause or missing formatter rule)."
    )
    assert Path(report).exists(), f"Trace report not found: {report}"
    logger.info(f"Trace report: {report}")

