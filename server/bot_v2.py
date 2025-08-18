"""
Refactored bot.py using the new service factory and pipeline builder approach.
This version maintains the same CLI interface and behavior as the original bot.py
while using the new modular architecture.

This file serves as:
1. A migration target - showing how the new architecture works
2. A compatibility layer - can be used as a drop-in replacement for bot.py
3. A demonstration - showcasing the cleaner separation of concerns
"""

import argparse
import asyncio
import os
import sys
import multiprocessing
import threading
from loguru import logger

# Set multiprocessing start method to 'spawn' for macOS Metal GPU safety
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from dotenv import load_dotenv

# Load environment variables BEFORE importing config
load_dotenv(override=True)

# RTVIProcessor fix is now applied in pipeline_builder.py

# Enable offline mode for HuggingFace transformers (conditional)
if os.getenv("HF_HUB_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    logger.info("📱 HuggingFace Hub offline mode enabled")
else:
    logger.info("🌐 HuggingFace Hub online mode (can download models)")

if os.getenv("TRANSFORMERS_OFFLINE", "0") == "1": 
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("🤖 Transformers offline mode enabled")
else:
    logger.info("🌐 Transformers online mode (can download models)")

from loguru import logger
from config import config

# 🧪 A/B TEST: Minimal vs Full system prompts (NO GLOBAL SIDE EFFECTS)
minimal_config = None
if os.getenv("USE_MINIMAL_PROMPTS", "false").lower() == "true":
    logger.info("🧪 A/B TEST: Using MINIMAL system prompts")
    from config_minimal import MinimalConfig
    minimal_config = MinimalConfig().apply_to_config(config)
else:
    logger.info("📝 Using FULL system prompts (default)")

# Import the new modular components
from core.service_factory import service_factory
from server import create_app, run_server


async def run_bot(webrtc_connection, language="en", llm_model=None):
    """
    ROCK SOLID bot function using simple STT -> LLM -> TTS pipeline.
    No tools, no fancy processors, maximum reliability.
    
    Args:
        webrtc_connection: WebRTC connection instance
        language: Language code (default: "en")
        llm_model: Optional LLM model override (ignored for simplicity)
    """
    from core.simple_pipeline import create_simple_pipeline
    
    try:
        logger.info("🚜 Starting rock-solid simple pipeline...")
        
        # Create simple pipeline - STT -> LLM -> TTS only
        pipeline, task = await create_simple_pipeline(webrtc_connection, language)
        
        # Run the pipeline
        from pipecat.pipeline.runner import PipelineRunner
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
        
    except Exception as e:
        logger.error(f"❌ Error in simple bot pipeline: {e}")
        raise



def main():
    """
    Main entry point with the same CLI interface as original bot.py
    """
    parser = argparse.ArgumentParser(description="Slowcat Bot Runner - Refactored Version")
    parser.add_argument("--host", default=config.network.server_host)
    parser.add_argument("--port", type=int, default=config.network.server_port)
    parser.add_argument("--language", default=config.default_language, 
                       choices=list(config.language_configs.keys()))
    parser.add_argument("--llm", dest="llm_model", default=None)
    parser.add_argument("--stt", dest="stt_model", default=None, 
                       choices=["TINY", "MEDIUM", "LARGE_V3", "LARGE_V3_TURBO", "LARGE_V3_TURBO_Q4", "DISTIL_LARGE_V3"],
                       help="STT model to use")
    parser.add_argument("--mode", choices=["server", "standalone"], default="server",
                       help="Run as server (default) or standalone pipeline")
    args = parser.parse_args()

    logger.info(f"🚀 Starting Slowcat Bot v2 - Refactored Architecture")
    logger.info(f"🌍 Language: {args.language}")
    logger.info(f"🏃 Mode: {args.mode}")
    
    if args.llm_model:
        logger.info(f"🤖 LLM Model: {args.llm_model}")
    if args.stt_model:
        logger.info(f"🎤 STT Model: {args.stt_model}")
    
    # MCP tools are handled natively by LM Studio via mcp.json
    if config.mcp.enabled:
        logger.info("🔧 MCP integration enabled - tools handled by LM Studio")
    
    if args.mode == "server":
        # Run as FastAPI server (default behavior)
        logger.info("🚀 Server starting while ML modules load in background...")
        run_server(
            host=args.host,
            port=args.port,
            language=args.language,
            llm_model=args.llm_model,
            stt_model=args.stt_model
        )
    else:
        # Standalone mode for testing/development
        logger.info("🔧 Running in standalone mode...")
        asyncio.run(run_standalone_bot(args.language, args.llm_model))


async def run_standalone_bot(language: str, llm_model: str = None):
    """
    Run bot in standalone mode for testing/development
    This creates a mock WebRTC connection for testing purposes
    """
    logger.info("🔧 Creating mock WebRTC connection for standalone mode...")
    
    # For standalone testing, we'd need to create a mock connection
    # This is mainly for development/testing purposes
    class MockWebRTCConnection:
        def __init__(self):
            self.pc_id = "mock-connection"
        
        def get_answer(self):
            return {"pc_id": self.pc_id, "sdp": "mock-sdp", "type": "answer"}
    
    mock_connection = MockWebRTCConnection()
    
    try:
        await run_bot(mock_connection, language, llm_model)
    except KeyboardInterrupt:
        logger.info("🛑 Standalone bot stopped")
    except Exception as e:
        logger.error(f"❌ Error in standalone bot: {e}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup minimal config if applied
        if minimal_config:
            minimal_config.restore_original()
            logger.info("✅ Minimal config restored on exit")


# ================================
# MIGRATION COMPATIBILITY LAYER
# ================================

# Export the original functions for gradual migration
# This allows existing code to import from this module during transition

# Original global variables (now managed by service factory)
def get_global_vad_analyzer():
    """Get global VAD analyzer (compatibility function)"""
    try:
        analyzers = service_factory.registry.get_instance("global_analyzers")
        return analyzers['vad_analyzer'] if analyzers else None
    except:
        return None

def get_global_turn_analyzer():
    """Get global turn analyzer (compatibility function)"""
    try:
        analyzers = service_factory.registry.get_instance("global_analyzers")
        return analyzers['turn_analyzer'] if analyzers else None
    except:
        return None

# For any code that might import these directly
GLOBAL_VAD_ANALYZER = None  # Deprecated: use get_global_vad_analyzer()
GLOBAL_TURN_ANALYZER = None  # Deprecated: use get_global_turn_analyzer()

# Helper functions that might be imported elsewhere
def _get_language_config(language: str) -> dict:
    """Compatibility function - use PipelineBuilder instead"""
    from core.pipeline_builder import PipelineBuilder
    builder = PipelineBuilder(service_factory)
    return builder._get_language_config(language)

def _initialize_services(lang_config: dict, language: str, llm_model: str):
    """Compatibility function - use ServiceFactory instead"""
    async def async_wrapper():
        services = await service_factory.create_services_for_language(language, llm_model)
        return services['stt'], services['tts'], services['llm']
    
    return asyncio.run(async_wrapper())

# Export for compatibility
__all__ = [
    'run_bot',  # Main function - same interface as original
    'get_global_vad_analyzer',  # Compatibility helper
    'get_global_turn_analyzer',  # Compatibility helper
    '_get_language_config',  # Compatibility helper
    '_initialize_services',  # Compatibility helper
]