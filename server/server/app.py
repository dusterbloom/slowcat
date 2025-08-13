"""
FastAPI application setup and configuration
"""

import asyncio
import os
import signal
import sys
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from loguru import logger

from config import config
from core.service_factory import service_factory
from core.pipeline_builder import PipelineBuilder
from .webrtc import WebRTCManager


class OfferRequest(BaseModel):
    sdp: str
    type: Literal["offer"]
    pc_id: Optional[str] = None
    restart_pc: bool = False

def create_app(language: str = None, llm_model: str = None, stt_model: str = None) -> FastAPI:
    """
    Create FastAPI application with proper configuration
    
    Args:
        language: Default language for the application
        llm_model: Default LLM model
        
    Returns:
        Configured FastAPI application
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("🚀 Slowcat server starting up (lifespan)...")

        # Start background services without threads
        ml_loader_task = asyncio.create_task(service_factory.get_service("ml_loader"))
        analyzer_task = asyncio.create_task(service_factory.get_service("global_analyzers"))

        # Only discover MCP tools if MCP is enabled and tools are not disabled
        enable_mcp = os.getenv("ENABLE_MCP", "false").lower() == "true"
        disable_all_tools = os.getenv("DISABLE_ALL_TOOLS", "false").lower() == "true"
        
        if enable_mcp and not disable_all_tools:
            from services.simple_mcp_tool_manager import discover_mcp_tools_background
            mcp_task = asyncio.create_task(discover_mcp_tools_background(language or config.default_language))
            logger.info("🔧 MCP tool discovery started")
        else:
            logger.info("🚫 MCP tool discovery skipped (tools disabled)")

        async def prewarm_llm():
            try:
                logger.info("🔄 Pre-warming LLM service...")
                await service_factory.wait_for_ml_modules()
                llm_service = await service_factory._create_llm_service_for_language(
                    app.state.language, app.state.llm_model
                )
                from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
                context = OpenAILLMContext([{"role": "user", "content": "Hi"}])
                stream = llm_service._stream_chat_completions(context)
                async for _ in stream:
                    break
                logger.info("✅ LLM service pre-warmed")
            except Exception as e:
                logger.warning(f"Failed to pre-warm LLM: {e}")

        llm_prewarm_task = asyncio.create_task(prewarm_llm())

        try:
            yield
        finally:
            await app.state.webrtc_manager.cleanup_all_connections()
            logger.info("✅ Lifespan shutdown complete")

    app = FastAPI(title="Slowcat Voice Agent", version="1.0.0", lifespan=lifespan)
    
    # Store configuration in app state
    app.state.language = language or config.default_language
    app.state.llm_model = llm_model
    app.state.stt_model = stt_model
    
    # Initialize managers
    webrtc_manager = WebRTCManager()
    pipeline_builder = PipelineBuilder(service_factory)
    
    # Store managers in app state
    app.state.webrtc_manager = webrtc_manager
    app.state.pipeline_builder = pipeline_builder
    
    
    @app.post("/api/offer")
    async def offer(req: OfferRequest, background_tasks: BackgroundTasks, x_slowcat_token: Optional[str] = Header(None)):
        """Handle WebRTC offer"""
        offer_token = getattr(config.network, "offer_token", None)
        if offer_token and x_slowcat_token != offer_token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        # Simple in-memory per-IP rate limiting
        from fastapi import Request
        import time
        if not hasattr(app.state, "_offer_rate_limit"):
            app.state._offer_rate_limit = {}
        client_ip = None
        try:
            from starlette.requests import Request as StarletteRequest
            request_instance: StarletteRequest = Request  # type: ignore
        except:
            request_instance = None
        # Note: for real-world use, replace with slowapi or redis-based limiter
        now = time.time()
        window = 10  # seconds
        max_requests = 5
        key = client_ip or "unknown"
        history = app.state._offer_rate_limit.get(key, [])
        history = [t for t in history if now - t < window]
        if len(history) >= max_requests:
            raise HTTPException(status_code=429, detail="Too Many Requests")
        history.append(now)
        app.state._offer_rate_limit[key] = history

        try:
            # Handle WebRTC offer
            answer, connection = await webrtc_manager.handle_offer(req.dict())
            
            # Start bot pipeline in background
            language = getattr(app.state, 'language', config.default_language)
            llm_model = getattr(app.state, 'llm_model', None)
            
            background_tasks.add_task(
                run_bot_pipeline,
                pipeline_builder,
                connection,
                language,
                llm_model,
                getattr(app.state, 'stt_model', None)
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error handling offer: {e}")
            raise
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "language": app.state.language,
            "llm_model": app.state.llm_model,
            "stt_model": app.state.stt_model,
            "active_connections": len(webrtc_manager.get_active_connections())
        }
    
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown"""
        logger.info("🛑 Slowcat server shutting down...")
        await webrtc_manager.cleanup_all_connections()
        logger.info("✅ Shutdown complete")
    
    return app


async def run_bot_pipeline(pipeline_builder: PipelineBuilder, webrtc_connection, 
                          language: str = "en", llm_model: str = None, stt_model: str = None):
    """
    Run bot pipeline for a WebRTC connection
    
    Args:
        pipeline_builder: Pipeline builder instance
        webrtc_connection: WebRTC connection
        language: Language for the bot
        llm_model: Optional LLM model
    """
    try:
        logger.info(f"🤖 Starting bot pipeline for language: {language}")
        
        # Build pipeline
        pipeline, task = await pipeline_builder.build_pipeline(
            webrtc_connection, language, llm_model, stt_model
        )
        
        # Run pipeline
        from pipecat.pipeline.runner import PipelineRunner
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
        
        logger.info("✅ Bot pipeline completed")
        
    except Exception as e:
        logger.error(f"❌ Error in bot pipeline: {e}")
        raise


def setup_signal_handlers(webrtc_manager: WebRTCManager):
    """Setup graceful shutdown signal handlers"""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        
        # Start async cleanup
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(webrtc_manager.cleanup_all_connections())
            else:
                asyncio.run(webrtc_manager.cleanup_all_connections())
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        # Give services time to cleanup
        threading.Event().wait(0.5)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def run_server(host: str = None, port: int = None, language: str = None, 
               llm_model: str = None, stt_model: str = None):
    """
    Run the Slowcat server
    
    Args:
        host: Server host (default from config)
        port: Server port (default from config)  
        language: Default language (default from config)
        llm_model: Default LLM model
        stt_model: STT model to use
    """
    # Use config defaults if not specified
    host = host or config.network.server_host
    port = port or config.network.server_port
    language = language or config.default_language
    
    # Create application
    app = create_app(language, llm_model, stt_model)
    
    # Setup signal handlers
    setup_signal_handlers(app.state.webrtc_manager)
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    logger.info(f"🌍 Default language: {language}")
    if llm_model:
        logger.info(f"🤖 LLM model: {llm_model}")
    if stt_model:
        logger.info(f"🎤 STT model: {stt_model}")
    
    # Run server
    uvicorn.run(app, host=host, port=port)


# For compatibility, expose the core run_bot function
# This allows the existing bot.py to import and use it during migration
async def run_bot(webrtc_connection, language="en", llm_model=None):
    """
    Compatibility function for existing bot.py
    
    This function maintains the same interface as the original run_bot
    but uses the new pipeline builder approach
    """
    pipeline_builder = PipelineBuilder(service_factory)
    await run_bot_pipeline(pipeline_builder, webrtc_connection, language, llm_model)