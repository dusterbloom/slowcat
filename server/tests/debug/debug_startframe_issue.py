#!/usr/bin/env python3
"""
Debug script to identify why StartFrame is not reaching RTVIProcessor
"""

import asyncio
import os
import sys
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{time} | {level} | {message}")

async def debug_startframe_flow():
    """Debug the StartFrame flow through the pipeline"""
    
    # Import core classes
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.processors.frameworks.rtvi import RTVIProcessor
    from pipecat.frames.frames import StartFrame, TextFrame
    from pipecat.processors.frame_processor import FrameProcessor
    
    class DebugProcessor(FrameProcessor):
        def __init__(self, debug_name: str):
            super().__init__()
            self.debug_name = debug_name
            
        async def process_frame(self, frame, direction):
            logger.debug(f"{self.debug_name}: received {type(frame).__name__} (direction: {direction})")
            logger.debug(f"{self.debug_name}: __started = {getattr(self, '_FrameProcessor__started', 'N/A')}")
            await super().process_frame(frame, direction)
    
    # Create components
    debug1 = DebugProcessor("Debug1")
    rtvi = RTVIProcessor()
    debug2 = DebugProcessor("Debug2")
    
    # Create simple pipeline: Debug1 -> RTVI -> Debug2
    pipeline = Pipeline([debug1, rtvi, debug2])
    
    # Create task
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False))
    
    logger.info("🔍 Starting pipeline task...")
    
    # Create a minimal task that just sends StartFrame
    async def minimal_task():
        logger.info("⏳ Waiting 1 second before sending StartFrame...")
        await asyncio.sleep(1)
        
        logger.info("📤 Manually sending StartFrame...")
        start_frame = StartFrame()
        await task.queue_frame(start_frame)
        
        await asyncio.sleep(1)
        
        logger.info("📤 Sending test TextFrame...")
        text_frame = TextFrame("test")
        await task.queue_frame(text_frame)
        
        await asyncio.sleep(1)
        
        logger.info("🛑 Stopping task...")
        await task.stop_when_done()
    
    # Create proper params
    from pipecat.pipeline.task import PipelineTaskParams
    params = PipelineTaskParams(loop=asyncio.get_event_loop())
    
    # Run both tasks
    await asyncio.gather(
        task.run(params),
        minimal_task()
    )
    
    logger.info("✅ Debug completed")

if __name__ == "__main__":
    # Setup environment
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    
    # Run debug
    asyncio.run(debug_startframe_flow())