"""
FastAPI application setup and configuration
"""

import asyncio
import signal
import sys
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from loguru import logger

from config import config
from core.service_factory import service_factory
from core.pipeline_builder import PipelineBuilder
from .webrtc import WebRTCManager


def create_app(language: str = None, llm_model: str = None) -> FastAPI:
    """
    Create FastAPI application with proper configuration
    
    Args:
        language: Default language for the application
        llm_model: Default LLM model
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="Slowcat Voice Agent", version="1.0.0")
    
    # Store configuration in app state
    app.state.language = language or config.default_language
    app.state.llm_model = llm_model
    
    # Initialize managers
    webrtc_manager = WebRTCManager()
    pipeline_builder = PipelineBuilder(service_factory)
    
    # Store managers in app state
    app.state.webrtc_manager = webrtc_manager
    app.state.pipeline_builder = pipeline_builder
    
    
    @app.post("/api/offer")
    async def offer(request: dict, background_tasks: BackgroundTasks):
        """Handle WebRTC offer"""
        try:
            # Handle WebRTC offer
            answer, connection = await webrtc_manager.handle_offer(request)
            
            # Start bot pipeline in background
            language = getattr(app.state, 'language', config.default_language)
            llm_model = getattr(app.state, 'llm_model', None)
            
            background_tasks.add_task(
                run_bot_pipeline, 
                pipeline_builder, 
                connection, 
                language, 
                llm_model
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
            "active_connections": len(webrtc_manager.get_active_connections())
        }
    
    @app.on_event("startup")
    async def startup_event():
        """Application startup"""
        logger.info("🚀 Slowcat server starting up...")
        
        # Start background ML loading
        ml_loader_thread = threading.Thread(
            target=lambda: service_factory.registry.get_instance("ml_loader") or 
                           asyncio.run(service_factory.get_service("ml_loader")),
            daemon=True
        )
        ml_loader_thread.start()
        
        # Start analyzer initialization
        analyzer_thread = threading.Thread(
            target=lambda: asyncio.run(service_factory.get_service("global_analyzers")),
            daemon=True
        )
        analyzer_thread.start()
        
        logger.info("✅ Background services started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Application shutdown"""
        logger.info("🛑 Slowcat server shutting down...")
        await webrtc_manager.cleanup_all_connections()
        logger.info("✅ Shutdown complete")
    
    return app


async def run_bot_pipeline(pipeline_builder: PipelineBuilder, webrtc_connection, 
                          language: str = "en", llm_model: str = None):
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
            webrtc_connection, language, llm_model
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
               llm_model: str = None):
    """
    Run the Slowcat server
    
    Args:
        host: Server host (default from config)
        port: Server port (default from config)  
        language: Default language (default from config)
        llm_model: Default LLM model
    """
    # Use config defaults if not specified
    host = host or config.network.server_host
    port = port or config.network.server_port
    language = language or config.default_language
    
    # Create application
    app = create_app(language, llm_model)
    
    # Setup signal handlers
    setup_signal_handlers(app.state.webrtc_manager)
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    logger.info(f"🌍 Default language: {language}")
    if llm_model:
        logger.info(f"🤖 LLM model: {llm_model}")
    
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