import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

# Import centralized config
from config import config, get_voice_recognition_config

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from openai import NOT_GIVEN
from pipecat.services.openai.llm import OpenAILLMService
from services.tool_enabled_llm import ToolEnabledLLMService
from kokoro_tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

# Import voice recognition components
from voice_recognition import AutoEnrollVoiceRecognition
from processors import AudioTeeProcessor, VADEventBridge, SpeakerContextProcessor, VideoSamplerProcessor, SpeakerNameManager
from processors.local_memory import LocalMemoryProcessor
from processors.memory_context_injector import MemoryContextInjector

load_dotenv(override=True)

app = FastAPI()

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    """Debug endpoint to check memory contents"""
    import json
    from pathlib import Path
    
    memory_file = Path(config.memory.data_dir) / f"{user_id}_memory.json"
    if memory_file.exists():
        with open(memory_file, 'r') as f:
            return json.load(f)
    return {"error": "No memory found for user"}

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls=config.network.stun_server,
    )
]


# Build language config from centralized config
LANGUAGE_CONFIG = {}
for lang, cfg in config.language_configs.items():
    LANGUAGE_CONFIG[lang] = {
        "voice": cfg.voice,
        "whisper_language": getattr(Language, cfg.whisper_language),
        "greeting": cfg.greeting,
        "system_instruction": cfg.system_instruction
    }

# Use config defaults
DEFAULT_LANGUAGE = config.default_language
VOICE_RECOGNITION_CONFIG = get_voice_recognition_config()


async def run_bot(webrtc_connection, language="en", llm_model=None):
    # Log voice recognition status
    if VOICE_RECOGNITION_CONFIG["enabled"]:
        logger.info("🎙️ Voice recognition is ENABLED")
    else:
        logger.info("🔇 Voice recognition is DISABLED (set ENABLE_VOICE_RECOGNITION=true to enable)")
    
    # Log video status
    video_enabled = config.video.enabled
    if video_enabled:
        logger.info("📹 Video is ENABLED")
    else:
        logger.info("📷 Video is DISABLED (set ENABLE_VIDEO=true to enable)")
    
    # Get language-specific configuration
    lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG[config.default_language])
    
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=config.video.enabled,
            video_in_width=config.video.video_width,
            video_in_height=config.video.video_height,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                stop_secs=config.audio.vad_stop_secs,
                start_secs=config.audio.vad_start_secs,
            )),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path=config.models.smart_turn_model_path,  # Download from HuggingFace
                params=SmartTurnParams(),
            ),
        ),
    )

    # Choose faster model based on language
    if language == "en":
        # Use faster distil model for English
        stt_model = MLXModel.DISTIL_LARGE_V3
        logger.info("Using DISTIL_LARGE_V3 for faster English STT")
    else:
        # Use turbo model for other languages (MEDIUM or LARGE_V3_TURBO_Q4 for multilingual support)
        stt_model = MLXModel.MEDIUM
        logger.info(f"Using MEDIUM for {language} language")
    
    stt = WhisperSTTServiceMLX(model=stt_model, language=lang_config["whisper_language"])

    tts = KokoroTTSService(
        model=config.models.tts_model, 
        voice=lang_config["voice"], 
        language=lang_config["whisper_language"],
        sample_rate=config.audio.tts_sample_rate
    )

    # Use provided LLM model or default
    if llm_model:
        logger.info(f"🤖 Using LLM model: {llm_model}")
        selected_model = llm_model
    else:
        selected_model = config.models.default_llm_model
        logger.info(f"🤖 Using default LLM model: {selected_model}")
    
    # Use tool-enabled LLM if MCP is enabled
    if config.mcp.enabled:
        llm = ToolEnabledLLMService(
            api_key=None,
            model=selected_model,
            base_url=config.network.llm_base_url,
            max_tokens=config.models.llm_max_tokens,
        )
        logger.info(f"🔧 Tool-enabled LLM service initialized")
        logger.info(f"📏 Context length: {config.models.llm_context_length} (set LLM_CONTEXT_LENGTH in .env)")
    else:
        llm = OpenAILLMService(
            api_key=None,
            model=selected_model,
            base_url=config.network.llm_base_url,
            max_tokens=config.models.llm_max_tokens,
        )
        logger.info(f"🤖 LLM service initialized")

    # Import tools if MCP is enabled
    tools = NOT_GIVEN
    tool_choice = NOT_GIVEN
    if config.mcp.enabled:
        from tools_config import AVAILABLE_TOOLS, TOOL_CHOICE_AUTO
        tools = AVAILABLE_TOOLS
        tool_choice = TOOL_CHOICE_AUTO
        logger.info(f"🛠️ Loaded {len(tools)} tools for function calling")
        # Debug: log tool names
        tool_names = [t["function"]["name"] for t in tools]
        logger.info(f"📋 Available tools: {', '.join(tool_names)}")
    
    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": lang_config["system_instruction"],
            }
        ],
        tools=tools,
        tool_choice=tool_choice,
    )
    
    # Debug: verify context has tools
    logger.info(f"🔍 Context tools type: {type(context.tools)}")
    logger.info(f"🔍 Context tool_choice: {context.tool_choice}")
    context_aggregator = llm.create_context_aggregator(context)
    
    #
    # Memory components
    #
    memory_processor = None
    memory_injector = None
    
    # Check if memory is enabled
    memory_enabled = config.memory.enabled
    if memory_enabled:
        logger.info("🧠 Memory is ENABLED - conversations will be persisted locally")
        # Create local memory processor
        memory_processor = LocalMemoryProcessor(
            data_dir=config.memory.data_dir,
            user_id=config.memory.default_user_id,  # Will be updated dynamically when speaker is identified
            max_history_items=config.memory.max_history_items,
            include_in_context=config.memory.include_in_context
        )
        
        # Create memory context injector
        memory_injector = MemoryContextInjector(
            memory_processor=memory_processor,
            system_prompt=config.memory.context_system_prompt,
            inject_as_system=True
        )
    else:
        logger.info("🚫 Memory is DISABLED (set ENABLE_MEMORY=true to enable)")

    #
    # Voice recognition components
    #
    voice_recognition = None
    audio_tee = None
    vad_bridge = None
    speaker_context = None
    speaker_name_manager = None
    
    #
    # Video components
    #
    video_sampler = None
    if transport._params.video_in_enabled:
        logger.info("📹 Video input enabled, creating video sampler")
        video_sampler = VideoSamplerProcessor(
            sample_interval=config.video.sample_interval_seconds,  # Sample every 30 seconds to avoid overload
            enabled=True
        )
    
    # Initialize callback refs for voice recognition
    callback_refs = {}
    
    if VOICE_RECOGNITION_CONFIG["enabled"]:
        logger.info("🎤 Initializing voice recognition...")
        logger.info(f"   Profile directory: {VOICE_RECOGNITION_CONFIG['profile_dir']}")
        logger.info(f"   Auto-enrollment after {VOICE_RECOGNITION_CONFIG['auto_enroll']['min_utterances']} utterances")
        # Create voice recognition
        voice_recognition = AutoEnrollVoiceRecognition(VOICE_RECOGNITION_CONFIG)
        await voice_recognition.initialize()
        logger.info("✅ Voice recognition initialized and ready!")
        
        # Create audio tee to split audio stream
        audio_tee = AudioTeeProcessor(enabled=True)
        audio_tee.register_audio_consumer(voice_recognition.process_audio_frame)
        
        # Create VAD event bridge
        vad_bridge = VADEventBridge()
        vad_bridge.set_callbacks(
            on_started=voice_recognition.on_user_started_speaking,
            on_stopped=voice_recognition.on_user_stopped_speaking
        )
        
        # Create speaker context processor
        speaker_context = SpeakerContextProcessor(
            format_style="natural",
            unknown_speaker_name="User"
        )
        
        # Create speaker name manager
        speaker_name_manager = SpeakerNameManager(voice_recognition)
        
        # Store references for callbacks
        callback_refs = {
            'speaker_context': speaker_context,
            'memory_processor': memory_processor,
            'speaker_name_manager': speaker_name_manager,
            'task': None,  # Will be set after task creation
            'context_aggregator': context_aggregator,
            'llm': llm
        }
        
        # Connect voice recognition to speaker context and memory
        async def on_speaker_changed(data):
            callback_refs['speaker_context'].update_speaker(data)
            # Update memory processor with speaker ID if available
            if callback_refs['memory_processor'] and data.get('speaker_id'):
                speaker_id = data['speaker_id']
                # Use speaker name if available, otherwise use speaker ID
                user_id = data.get('speaker_name', speaker_id)
                callback_refs['memory_processor'].set_user_id(user_id)
                logger.info(f"📝 Memory switched to user: {user_id}")
        
        async def on_speaker_enrolled(data):
            """Handle when a new speaker is auto-enrolled"""
            logger.info(f"New speaker enrolled: {data}")
            if data.get('needs_name', False):
                speaker_id = data['speaker_id']
                # Start name collection in the manager
                callback_refs['speaker_name_manager'].start_name_collection(speaker_id)
                
                # Update speaker context to inject system message
                if callback_refs['speaker_context']:
                    callback_refs['speaker_context'].handle_speaker_enrolled(data)
                    logger.info("Speaker context updated with enrollment data")
        
        voice_recognition.set_callbacks(
            on_speaker_changed=on_speaker_changed,
            on_speaker_enrolled=on_speaker_enrolled
        )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Build pipeline with optional voice recognition and video
    pipeline_components = [transport.input()]
    
    # Add video sampler early in pipeline if video is enabled
    if video_sampler:
        pipeline_components.append(video_sampler)
    
    if audio_tee:
        pipeline_components.append(audio_tee)
    
    if vad_bridge:
        pipeline_components.append(vad_bridge)
    
    # STT outputs TranscriptionFrame
    pipeline_components.append(stt)
    
    # Memory processor should capture STT output immediately
    if memory_processor:
        pipeline_components.append(memory_processor)
    
    # Speaker context modifies TranscriptionFrame.user_id
    if speaker_context:
        pipeline_components.append(speaker_context)
    
    # RTVI processor
    pipeline_components.append(rtvi)
    
    # Speaker name manager
    if speaker_name_manager:
        pipeline_components.append(speaker_name_manager)
    
    # Context aggregator processes user messages
    pipeline_components.append(context_aggregator.user())
    
    # Memory injector adds context before LLM
    if memory_injector:
        pipeline_components.append(memory_injector)
    
    # LLM, TTS, and output
    pipeline_components.extend([
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    pipeline = Pipeline(pipeline_components)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    
    # Set the task reference for voice recognition callbacks
    if VOICE_RECOGNITION_CONFIG["enabled"] and 'callback_refs' in locals():
        callback_refs['task'] = task

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        print(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        # Get language and llm_model from app state if available
        language = getattr(app.state, 'language', DEFAULT_LANGUAGE)
        llm_model = getattr(app.state, 'llm_model', None)
        background_tasks.add_task(run_bot, pipecat_connection, language, llm_model)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default=config.network.server_host, help=f"Host for HTTP server (default: {config.network.server_host})"
    )
    parser.add_argument(
        "--port", type=int, default=config.network.server_port, help=f"Port for HTTP server (default: {config.network.server_port})"
    )
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE, 
        choices=list(LANGUAGE_CONFIG.keys()),
        help=f"Language for the bot (default: {DEFAULT_LANGUAGE})"
    )
    parser.add_argument(
        "--llm", dest="llm_model", default=None,
        help=f"LLM model to use (e.g., mistral:7b, llama2:13b, gemma:2b). Default: {config.models.default_llm_model}"
    )
    args = parser.parse_args()

    # Set language and llm_model in app state
    app.state.language = args.language
    app.state.llm_model = args.llm_model
    
    logger.info(f"Starting bot with language: {args.language}")
    if args.llm_model:
        logger.info(f"Using custom LLM model: {args.llm_model}")
    else:
        logger.info(f"Using default LLM model: {config.models.default_llm_model}")
    
    uvicorn.run(app, host=args.host, port=args.port)
