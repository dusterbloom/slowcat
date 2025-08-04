import argparse
import asyncio
import os
import sys
import multiprocessing
import importlib
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Any, Optional

# Set multiprocessing start method to 'spawn' for macOS Metal GPU safety
# This must be done before any other multiprocessing operations
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv

# Load environment variables BEFORE importing config
load_dotenv(override=True)

# Enable offline mode for HuggingFace transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Import centralized config AFTER loading .env
from config import config, VoiceRecognitionConfig
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

# Light-weight imports that are always needed
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from openai import NOT_GIVEN
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

# Import processors (lightweight)
from processors import (AudioTeeProcessor, VADEventBridge, SpeakerContextProcessor, 
                        VideoSamplerProcessor, SpeakerNameManager, LocalMemoryProcessor, 
                        MemoryContextInjector, GreetingFilterProcessor)
from tools import get_tools, set_memory_processor

# Lazy-loaded heavy ML modules (will be populated by _lazy_load_ml_modules)
WhisperSTTServiceMLX: Optional[type] = None
MLXModel: Optional[type] = None
KokoroTTSService: Optional[type] = None
LLMWithToolsService: Optional[type] = None
OpenAILLMService: Optional[type] = None
AutoEnrollVoiceRecognition: Optional[type] = None
SileroVADAnalyzer: Optional[type] = None
LocalSmartTurnAnalyzerV2: Optional[type] = None

# Flag to track if ML modules are loaded
_ml_modules_loaded = threading.Event()

def _lazy_load_ml_modules():
    """Load heavy ML modules in the background"""
    global WhisperSTTServiceMLX, MLXModel, KokoroTTSService, LLMWithToolsService
    global OpenAILLMService, AutoEnrollVoiceRecognition, SileroVADAnalyzer, LocalSmartTurnAnalyzerV2
    
    try:
        logger.info("🔄 Starting lazy load of ML modules...")
        
        # Load STT modules
        whisper_module = importlib.import_module("services.whisper_stt_with_lock")
        WhisperSTTServiceMLX = whisper_module.WhisperSTTServiceMLX
        
        stt_module = importlib.import_module("pipecat.services.whisper.stt")
        MLXModel = stt_module.MLXModel
        
        # Load TTS module
        tts_module = importlib.import_module("kokoro_tts")
        KokoroTTSService = tts_module.KokoroTTSService
        
        # Load LLM modules
        llm_tools_module = importlib.import_module("services.llm_with_tools")
        LLMWithToolsService = llm_tools_module.LLMWithToolsService
        
        openai_module = importlib.import_module("pipecat.services.openai.llm")
        OpenAILLMService = openai_module.OpenAILLMService
        
        # Load voice recognition
        voice_module = importlib.import_module("voice_recognition")
        AutoEnrollVoiceRecognition = voice_module.AutoEnrollVoiceRecognition
        
        # Load audio analyzers
        vad_module = importlib.import_module("pipecat.audio.vad.silero")
        SileroVADAnalyzer = vad_module.SileroVADAnalyzer
        
        turn_module = importlib.import_module("pipecat.audio.turn.smart_turn.local_smart_turn_v2")
        LocalSmartTurnAnalyzerV2 = turn_module.LocalSmartTurnAnalyzerV2
        
        logger.info("✅ ML modules loaded successfully")
        _ml_modules_loaded.set()
        
    except Exception as e:
        logger.error(f"❌ Failed to load ML modules: {e}")
        raise

# Global singleton instances for analyzers (initialized after ML modules load)
GLOBAL_VAD_ANALYZER: Optional[Any] = None
GLOBAL_TURN_ANALYZER: Optional[Any] = None

def _initialize_global_analyzers():
    """Initialize global analyzer instances after ML modules are loaded"""
    global GLOBAL_VAD_ANALYZER, GLOBAL_TURN_ANALYZER
    
    # Wait for ML modules to be loaded
    _ml_modules_loaded.wait()
    
    logger.info("🔄 Initializing global VAD and Smart Turn analyzers...")
    
    GLOBAL_VAD_ANALYZER = SileroVADAnalyzer(params=VADParams(
        stop_secs=config.audio.vad_stop_secs,
        start_secs=config.audio.vad_start_secs,
    ))
    
    GLOBAL_TURN_ANALYZER = LocalSmartTurnAnalyzerV2(
        smart_turn_model_path=config.models.smart_turn_model_path
    )
    
    logger.info("✅ Global analyzers initialized")
    
    # Pre-warm Kokoro TTS model
    try:
        logger.info("🔄 Pre-warming Kokoro TTS model...")
        kokoro = KokoroTTSService(voice="af_heart")
        # The model will load on first access
        logger.info("✅ Kokoro TTS model ready")
    except Exception as e:
        logger.warning(f"Failed to pre-warm Kokoro TTS: {e}")

# --- FastAPI and WebRTC Setup ---
app = FastAPI()
pcs_map: Dict[str, SmallWebRTCConnection] = {}
ice_servers = [IceServer(urls=config.network.stun_server)]

# --- Language Configuration ---
LANGUAGE_CONFIG = {
    lang: {
        "voice": cfg.voice,
        "whisper_language": getattr(Language, cfg.whisper_language),
        "system_instruction": cfg.system_instruction
    } for lang, cfg in config.language_configs.items()
}
DEFAULT_LANGUAGE = config.default_language

#
# --- Refactored Helper Functions ---
#

def _get_language_config(language: str) -> dict:
    """Gets language-specific configuration and injects language consistency notes."""
    lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG[DEFAULT_LANGUAGE])
    logger.info(f"🌍 Using language: {language}")
    
    language_consistency_note = {
        "en": "\n\nIMPORTANT: Always respond in English only.",
        "es": "\n\nIMPORTANTE: Siempre responde solo en español.",
        "fr": "\n\nIMPORTANT: Répondez toujours en français uniquement.",
        "ja": "\n\n重要：常に日本語のみで応答してください。",
        "it": "\n\nIMPORTANTE: Rispondi sempre solo in italiano.",
        "zh": "\n\n重要提示：始终只用中文回复。",
        "pt": "\n\nIMPORTANTE: Sempre responda apenas em português.",
        "de": "\n\nWICHTIG: Antworten Sie immer nur auf Deutsch."
    }
    
    modified_instruction = lang_config["system_instruction"]
    if language in language_consistency_note:
        modified_instruction += language_consistency_note[language]
    
    final_config = dict(lang_config)
    final_config["system_instruction"] = modified_instruction
    return final_config

def _initialize_services(lang_config: dict, language: str, llm_model: str) -> Tuple:
    """Initializes and returns core services (STT, TTS, LLM)."""
    # Ensure ML modules are loaded
    if not _ml_modules_loaded.is_set():
        logger.info("Waiting for ML modules to load...")
        _ml_modules_loaded.wait()
    stt_model = MLXModel.DISTIL_LARGE_V3 if language == "en" else MLXModel.MEDIUM
    logger.info(f"Using STT model: {stt_model.name} for {language}")
    stt = WhisperSTTServiceMLX(model=stt_model, language=lang_config["whisper_language"])

    tts = KokoroTTSService(
        model=config.models.tts_model, 
        voice=lang_config["voice"], 
        language=lang_config["whisper_language"],
        sample_rate=config.audio.tts_sample_rate,
        max_workers=config.audio.tts_max_workers
    )

    selected_model = llm_model or config.models.default_llm_model
    logger.info(f"🤖 Using LLM model: {selected_model}")
    
    llm_params = {
        "api_key": None, 
        "model": selected_model, 
        "base_url": config.network.llm_base_url, 
        "max_tokens": config.models.llm_max_tokens
    }
    if config.mcp.enabled:
        logger.info("🔧 Tool-enabled LLM service initialized")
        llm = LLMWithToolsService(**llm_params)
    else:
        logger.info("🤖 Standard LLM service initialized")
        llm = OpenAILLMService(**llm_params)
        
    return stt, tts, llm

async def _setup_processors(vr_config: VoiceRecognitionConfig) -> Tuple:
    """Initializes and wires up all frame processors."""
    # Memory
    if config.memory.enabled:
        logger.info("🧠 Memory is ENABLED")
        memory_processor = LocalMemoryProcessor(
            data_dir=config.memory.data_dir,
            user_id=config.memory.default_user_id,
            max_history_items=config.memory.max_history_items,
            include_in_context=config.memory.include_in_context
        )
        # Set memory processor for tool handlers
        set_memory_processor(memory_processor)
        
        memory_injector = MemoryContextInjector(
            memory_processor=memory_processor,
            system_prompt=config.memory.context_system_prompt,
            inject_as_system=True
        )
    else:
        logger.info("🚫 Memory is DISABLED")
        memory_processor, memory_injector = None, None

    # Video
    video_sampler = VideoSamplerProcessor() if config.video.enabled else None
    if config.video.enabled: logger.info("📹 Video is ENABLED")
    else: logger.info("📷 Video is DISABLED")

    # Voice Recognition
    voice_recognition, audio_tee, vad_bridge, speaker_context, speaker_name_manager = (None,) * 5
    if vr_config.enabled:
        logger.info("🎙️ Voice recognition is ENABLED")
        voice_recognition = AutoEnrollVoiceRecognition(vr_config)
        await voice_recognition.initialize()
        
        audio_tee = AudioTeeProcessor()
        audio_tee.register_audio_consumer(voice_recognition.process_audio_frame)
        
        vad_bridge = VADEventBridge()
        vad_bridge.set_callbacks(voice_recognition.on_user_started_speaking, voice_recognition.on_user_stopped_speaking)
        
        speaker_context = SpeakerContextProcessor()
        speaker_name_manager = SpeakerNameManager(voice_recognition)
        
        async def on_speaker_changed(data: Dict[str, Any]):
            speaker_context.update_speaker(data)
            if memory_processor:
                # Use speaker_name if available, otherwise speaker_id
                user_id = data.get('speaker_name', data.get('speaker_id', 'unknown'))
                await memory_processor.update_user_id(user_id)
                logger.info(f"Updated memory processor to use user_id: {user_id}")
                logger.info(f"📝 Memory switched to user: {user_id}")
        
        async def on_speaker_enrolled(data: Dict[str, Any]):
            if data.get('needs_name'):
                speaker_name_manager.start_name_collection(data['speaker_id'])
            speaker_context.handle_speaker_enrolled(data)

        voice_recognition.set_callbacks(on_speaker_changed, on_speaker_enrolled)
    else:
        logger.info("🔇 Voice recognition is DISABLED")

    return (memory_processor, memory_injector, video_sampler, voice_recognition, 
            audio_tee, vad_bridge, speaker_context, speaker_name_manager)

def _build_pipeline(components: List[Any]) -> Pipeline:
    """Builds the pipeline from a list of non-None components."""
    return Pipeline([comp for comp in components if comp is not None])

#
# --- Main Bot Logic ---
#

async def run_bot(webrtc_connection, language="en", llm_model=None):
    # 1. Get Configs
    lang_config = _get_language_config(language)
    
    # 2. Initialize Core Services
    stt, tts, llm = _initialize_services(lang_config, language, llm_model)

    # 3. Setup Processors
    processors = await _setup_processors(config.voice_recognition)
    (memory_processor, memory_injector, video_sampler, voice_recognition, 
     audio_tee, vad_bridge, speaker_context, speaker_name_manager) = processors

    # 4. Setup Transport
    # Wait for global analyzers if not ready
    retry_count = 0
    while (GLOBAL_VAD_ANALYZER is None or GLOBAL_TURN_ANALYZER is None) and retry_count < 50:
        if retry_count == 0:
            logger.info("Waiting for global analyzers to initialize...")
        await asyncio.sleep(0.1)
        retry_count += 1
    
    if GLOBAL_VAD_ANALYZER is None or GLOBAL_TURN_ANALYZER is None:
        logger.error("Global analyzers failed to initialize after 5 seconds!")
        return
        
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True, audio_out_enabled=True, video_in_enabled=config.video.enabled,
            vad_analyzer=GLOBAL_VAD_ANALYZER,
            turn_analyzer=GLOBAL_TURN_ANALYZER,
        )
    )

    # 5. Build Context
    tools = get_tools() if config.mcp.enabled else NOT_GIVEN
    context = OpenAILLMContext([{"role": "system", "content": lang_config["system_instruction"]}], tools=tools)
    context_aggregator = llm.create_context_aggregator(context)

    # 6. Build Pipeline
    rtvi = RTVIProcessor()


    # NEW: Instantiate the GreetingFilterProcessor with the greeting text
    # We need to get the greeting text from the original, unmodified system prompt.
    greeting_text = "Hello, I'm Slowcat!" # This should be kept in sync with the prompt.
    greeting_filter = GreetingFilterProcessor(greeting_text=greeting_text)

    pipeline_components = [
        transport.input(),
        video_sampler,
        audio_tee,
        vad_bridge,
        stt,
        memory_processor,
        speaker_context,
        rtvi,
        speaker_name_manager,
        context_aggregator.user(),
        memory_injector,
        llm,
        tts,
        transport.output(),
        greeting_filter,
        context_aggregator.assistant(),
    ]
    pipeline = _build_pipeline(pipeline_components)

    # 7. Create and Run Task
    task = PipelineTask(
        pipeline, 
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True), 
        observers=[RTVIObserver(rtvi)]
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi_proc):
        await rtvi_proc.set_bot_ready()
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

#
# --- FastAPI Server Endpoints and Main Execution ---
#

@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")
    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(sdp=request["sdp"], type=request["type"], restart_pc=request.get("restart_pc", False))
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])
        
        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(conn: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {conn.pc_id}")
            pcs_map.pop(conn.pc_id, None)
        
        language = getattr(app.state, 'language', DEFAULT_LANGUAGE)
        llm_model = getattr(app.state, 'llm_model', None)
        background_tasks.add_task(run_bot, pipecat_connection, language, llm_model)

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument("--host", default=config.network.server_host)
    parser.add_argument("--port", type=int, default=config.network.server_port)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, choices=list(LANGUAGE_CONFIG.keys()))
    parser.add_argument("--llm", dest="llm_model", default=None)
    args = parser.parse_args()

    app.state.language = args.language
    app.state.llm_model = args.llm_model
    
    # Start loading ML modules in background thread immediately
    ml_loader_thread = threading.Thread(target=_lazy_load_ml_modules, daemon=True)
    ml_loader_thread.start()
    
    # Initialize global analyzers in background after ML modules load
    analyzer_thread = threading.Thread(target=_initialize_global_analyzers, daemon=True)
    analyzer_thread.start()
    
    # Add signal handling for graceful shutdown
    import signal
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        # Cleanup any active connections
        for pc_id, pc in pcs_map.items():
            try:
                logger.info(f"Closing connection {pc_id}")
                # Note: We can't use asyncio.create_task here as we're not in an async context
                # The connections will be cleaned up by uvicorn shutdown
            except Exception as e:
                logger.error(f"Error handling connection {pc_id}: {e}")
        
        # Give services time to cleanup
        threading.Event().wait(0.5)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting bot with language: {args.language}")
    logger.info("🚀 Server starting while ML modules load in background...")
    uvicorn.run(app, host=args.host, port=args.port)