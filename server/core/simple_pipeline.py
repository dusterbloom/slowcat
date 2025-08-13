"""
Simple, reliable pipeline: STT -> LLM -> TTS
No tools, no fancy processors, maximum reliability
"""

import asyncio
import os
from typing import Dict, Any, Tuple
from loguru import logger

from config import config
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

from .service_factory import ServiceFactory


class SimplePipeline:
    """Ultra-reliable STT -> LLM -> TTS pipeline"""
    
    def __init__(self, service_factory: ServiceFactory):
        self.service_factory = service_factory
    
    async def build(self, webrtc_connection, language: str = "en") -> Tuple[Pipeline, Any]:
        """
        Build the simplest possible voice pipeline
        
        STT -> LLM -> TTS (no tools, no processors, no complexity)
        """
        logger.info(f"🚜 Building simple pipeline for {language}")
        
        # Get language config
        lang_config = config.get_language_config(language)
        logger.info(f"📝 System prompt: {lang_config.system_instruction[:100]}...")
        
        # Create transport with metrics enabled
        transport = SmallWebRTCTransport(
            connection=webrtc_connection,
            params=TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                add_wav_header=False,
                vad_enabled=True,
                vad_analyzer="silero",
                vad_audio_passthrough=True,
                # Enable metrics for debug UI
                enable_metrics=True,
                enable_usage_metrics=True,
            )
        )
        
        # Create services
        stt_service = await self.service_factory.create_stt_service(language)
        llm_service = await self.service_factory.create_llm_service()
        tts_service = await self.service_factory.create_tts_service(language)
        
        logger.info(f"✅ STT: {type(stt_service).__name__} (has metrics: {hasattr(stt_service, 'start_processing_metrics')})")
        logger.info(f"✅ LLM: {type(llm_service).__name__} (has metrics: {hasattr(llm_service, 'start_processing_metrics')})")
        logger.info(f"✅ TTS: {type(tts_service).__name__} (has metrics: {hasattr(tts_service, 'start_processing_metrics')})")
        
        # Check if STT service has the proper service name for metrics
        if hasattr(stt_service, 'name'):
            logger.info(f"🏷️ STT service name: {stt_service.name}")
        else:
            logger.warning("⚠️ STT service missing 'name' attribute for metrics")
        
        # Create context aggregator (minimal)
        context = OpenAILLMContext(
            messages=[
                {
                    "role": "system",
                    "content": lang_config.system_instruction
                }
            ]
        )
        
        # Build the simplest possible pipeline
        pipeline = Pipeline([
            transport.input_proc(),      # Audio input from WebRTC
            stt_service,                 # Speech to text
            context.user(),              # Add user message to context
            llm_service,                 # Language model (NO TOOLS)
            tts_service,                 # Text to speech
            transport.output_proc(),     # Audio output to WebRTC
            context.assistant(),         # Add assistant response to context
        ])
        
        logger.info("🚜 Simple pipeline built - John Deere reliability!")
        
        # Create task
        task = asyncio.create_task(pipeline.run())
        
        return pipeline, task


async def create_simple_pipeline(webrtc_connection, language: str = "en") -> Tuple[Pipeline, Any]:
    """Factory function for creating simple pipeline"""
    service_factory = ServiceFactory()
    simple_pipeline = SimplePipeline(service_factory)
    return await simple_pipeline.build(webrtc_connection, language)