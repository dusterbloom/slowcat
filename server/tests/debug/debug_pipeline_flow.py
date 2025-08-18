#!/usr/bin/env python3
"""
Debug script to test pipeline flow with integrated tracing

This script helps identify frame flow issues and tests the stateless memory integration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add server to path
sys.path.append(str(Path(__file__).parent))

import time
from typing import Dict, Any
from loguru import logger

# Import our debugging tools
from debug.frame_monitor import (
    monitor, MonitoringSession, continuous_monitoring,
    monitor_frame_flow, monitor_push_frame, full_monitor
)

# Import core components
from core.pipeline_builder import PipelineBuilder
from core.service_factory import ServiceFactory
from config import config

# Pipecat imports
from pipecat.frames.frames import (
    Frame, StartFrame, EndFrame, TextFrame, TranscriptionFrame,
    LLMMessagesFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.base_task import PipelineTaskParams


class DebugFrameGenerator(FrameProcessor):
    """Generate test frames to simulate conversation flow"""
    
    def __init__(self, test_scenario: str = "basic"):
        super().__init__()
        self.test_scenario = test_scenario
        self.frames_sent = 0
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartFrame):
            # Start generating test frames after pipeline starts
            asyncio.create_task(self._generate_test_frames())
        
        await self.push_frame(frame, direction)
    
    async def _generate_test_frames(self):
        """Generate test conversation frames"""
        await asyncio.sleep(1)  # Wait for pipeline to settle
        
        logger.info(f"🎬 Starting {self.test_scenario} test scenario")
        
        if self.test_scenario == "basic":
            await self._basic_conversation_test()
        elif self.test_scenario == "memory_intensive":
            await self._memory_intensive_test()
        elif self.test_scenario == "stress":
            await self._stress_test()
    
    async def _basic_conversation_test(self):
        """Basic conversation flow test"""
        test_messages = [
            "Hello, my name is David",
            "I have a cat named Muffin",
            "What do you remember about me?",
            "Tell me about my cat",
            "What's my name again?"
        ]
        
        for i, message in enumerate(test_messages):
            logger.info(f"📝 Sending test message {i+1}: {message}")
            
            # Simulate user speaking
            await self.push_frame(
                UserStartedSpeakingFrame(), 
                FrameDirection.UPSTREAM
            )
            
            await asyncio.sleep(0.1)
            
            # Send transcription
            await self.push_frame(
                TranscriptionFrame(text=message, user_id="test_user", timestamp=str(time.time())),
                FrameDirection.UPSTREAM
            )
            
            await asyncio.sleep(0.1)
            
            # Send LLM messages
            llm_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
            
            await self.push_frame(
                LLMMessagesFrame(messages=llm_messages),
                FrameDirection.UPSTREAM
            )
            
            # Simulate user stopping
            await self.push_frame(
                UserStoppedSpeakingFrame(),
                FrameDirection.UPSTREAM
            )
            
            # Wait between messages
            await asyncio.sleep(2)
            self.frames_sent += 1
        
        logger.info(f"✅ Basic test completed, sent {self.frames_sent} message sets")
    
    async def _memory_intensive_test(self):
        """Memory-intensive test with lots of context"""
        base_messages = [
            "I'm a data scientist working on fraud detection",
            "My dog is named Rex and he's 5 years old",
            "I live in San Francisco",
            "My favorite food is sushi",
            "I graduated from Stanford in 2015",
        ]
        
        # Send base context
        for msg in base_messages:
            await self._send_test_message(msg)
            await asyncio.sleep(1)
        
        # Test memory retrieval
        memory_queries = [
            "What do you know about my work?",
            "Tell me about Rex",
            "Where do I live?",
            "What's my educational background?",
            "What do I like to eat?"
        ]
        
        for query in memory_queries:
            await self._send_test_message(query)
            await asyncio.sleep(1.5)
        
        logger.info("✅ Memory intensive test completed")
    
    async def _stress_test(self):
        """Stress test with rapid frame generation"""
        logger.info("🚨 Starting stress test - rapid frame generation")
        
        for i in range(20):
            await self._send_test_message(f"Stress test message {i+1}")
            await asyncio.sleep(0.2)  # Rapid fire
        
        logger.info("✅ Stress test completed")
    
    async def _send_test_message(self, message: str):
        """Helper to send a complete test message"""
        await self.push_frame(
            UserStartedSpeakingFrame(),
            FrameDirection.UPSTREAM
        )
        
        await self.push_frame(
            TranscriptionFrame(text=message, user_id="test_user", timestamp=str(time.time())),
            FrameDirection.UPSTREAM
        )
        
        llm_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
        
        await self.push_frame(
            LLMMessagesFrame(messages=llm_messages),
            FrameDirection.UPSTREAM
        )
        
        await self.push_frame(
            UserStoppedSpeakingFrame(),
            FrameDirection.UPSTREAM
        )
        
        self.frames_sent += 1


class MockWebRTCConnection:
    """Mock WebRTC connection for testing"""
    def __init__(self):
        self.id = "test_connection"


@full_monitor
class MockSTTService(FrameProcessor):
    """Mock STT service for testing"""
    
    def __init__(self):
        super().__init__()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            logger.debug(f"STT processed: {frame.text}")
        
        await self.push_frame(frame, direction)


@full_monitor  
class MockLLMService(FrameProcessor):
    """Mock LLM service for testing"""
    
    def __init__(self):
        super().__init__()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMMessagesFrame):
            # Simulate LLM processing
            user_message = ""
            for msg in frame.messages:
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                    break
            
            if user_message:
                # Generate mock response
                response = f"I understand you said: '{user_message}'. How can I help?"
                
                # Create response frame
                response_frame = TextFrame(text=response)
                await self.push_frame(response_frame, FrameDirection.DOWNSTREAM)
                
                logger.debug(f"LLM generated response: {response}")
        
        await self.push_frame(frame, direction)


@full_monitor
class MockTTSService(FrameProcessor):
    """Mock TTS service for testing"""
    
    def __init__(self):
        super().__init__()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            logger.debug(f"TTS processed: {frame.text}")
        
        await self.push_frame(frame, direction)


class DebugTransport:
    """Mock transport for testing"""
    
    def __init__(self):
        self.input_processor = DebugFrameGenerator("basic")
        self.output_processor = FrameProcessor()
    
    def input(self):
        return self.input_processor
    
    def output(self):
        return self.output_processor


async def create_debug_pipeline(test_scenario: str = "basic") -> Pipeline:
    """Create a simplified pipeline for debugging"""
    
    logger.info("🔧 Creating debug pipeline...")
    
    # Import stateless memory
    from processors.stateless_memory import StatelessMemoryProcessor
    
    # Create components
    transport = DebugTransport()
    transport.input_processor.test_scenario = test_scenario
    
    # Add monitoring to stateless memory
    @full_monitor
    class MonitoredStatelessMemory(StatelessMemoryProcessor):
        pass
    
    # Create services
    memory_processor = MonitoredStatelessMemory(
        db_path="data/debug_memory",
        max_context_tokens=512,
        perfect_recall_window=5
    )
    
    stt_service = MockSTTService()
    llm_service = MockLLMService() 
    tts_service = MockTTSService()
    
    # Create pipeline
    components = [
        transport.input(),
        stt_service,
        memory_processor,
        llm_service,
        tts_service,
        transport.output()
    ]
    
    pipeline = Pipeline(components)
    logger.info(f"✅ Debug pipeline created with {len(components)} components")
    
    return pipeline


async def run_debug_session(test_scenario: str = "basic", duration: int = 30):
    """Run a debug session with monitoring"""
    
    logger.info("🚀 Starting pipeline debug session")
    
    with MonitoringSession(f"debug_{test_scenario}") as monitor_instance:
        try:
            # Create pipeline
            pipeline = await create_debug_pipeline(test_scenario)
            
            # Create task
            task = PipelineTask(pipeline)
            
            # Start continuous monitoring
            monitoring_task = asyncio.create_task(continuous_monitoring(interval=2.0))
            
            # Run pipeline
            logger.info(f"▶️  Running {test_scenario} scenario for {duration}s")
            
            # Get current event loop and create task params
            loop = asyncio.get_event_loop()
            task_params = PipelineTaskParams(loop=loop)
            
            pipeline_task = asyncio.create_task(task.run(task_params))
            
            # Wait for specified duration or pipeline completion
            try:
                await asyncio.wait_for(pipeline_task, timeout=duration)
            except asyncio.TimeoutError:
                logger.info("⏰ Test duration reached, stopping pipeline")
                task.stop()
            
            # Stop monitoring
            monitor.active = False
            monitoring_task.cancel()
            
            # Wait a moment for cleanup
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ Debug session failed: {e}")
            raise
        
        finally:
            # Final status report
            logger.info("📊 Final monitoring report:")
            monitor_instance.print_status()
    
    logger.info("✅ Debug session completed")


async def run_stateless_memory_test():
    """Test stateless memory processor in isolation"""
    
    logger.info("🧠 Testing stateless memory processor in isolation")
    
    from processors.stateless_memory import StatelessMemoryProcessor
    
    # Create processor
    memory = StatelessMemoryProcessor(
        db_path="data/test_memory",
        max_context_tokens=512,
        perfect_recall_window=5
    )
    
    # Test messages
    test_exchanges = [
        ("Hello, my name is Alice", "Nice to meet you, Alice!"),
        ("I have a dog named Buddy", "That's wonderful! Tell me more about Buddy."),
        ("What do you remember about me?", "You're Alice and you have a dog named Buddy."),
        ("What's my dog's name?", "Your dog's name is Buddy."),
        ("I also like reading books", "That's great! Reading is a wonderful hobby.")
    ]
    
    # Process exchanges
    for i, (user_msg, assistant_msg) in enumerate(test_exchanges):
        logger.info(f"🔄 Processing exchange {i+1}")
        
        # Simulate the flow
        memory.current_speaker = "alice"
        memory.current_user_message = user_msg
        
        # Create LLM messages frame
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_msg}
        ]
        
        # Test memory injection
        enhanced_messages = await memory._inject_memory_context(messages, "alice")
        
        logger.info(f"Original messages: {len(messages)}")
        logger.info(f"Enhanced messages: {len(enhanced_messages)}")
        
        # Check for memory context
        has_memory = any("[Memory Context" in str(msg.get('content', '')) for msg in enhanced_messages)
        logger.info(f"Memory injected: {has_memory}")
        
        # Store the exchange
        await memory._store_exchange(user_msg, assistant_msg)
        
        await asyncio.sleep(0.5)
    
    # Get stats
    stats = memory.get_performance_stats()
    logger.info("📊 Stateless Memory Stats:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("✅ Stateless memory test completed")


def main():
    """Main debug function"""
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>"
    )
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Debug pipeline flow")
    parser.add_argument("--scenario", choices=["basic", "memory_intensive", "stress", "memory_only"], 
                       default="basic", help="Test scenario to run")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    if args.scenario == "memory_only":
        asyncio.run(run_stateless_memory_test())
    else:
        asyncio.run(run_debug_session(args.scenario, args.duration))


if __name__ == "__main__":
    main()