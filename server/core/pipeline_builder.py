"""
Pipeline builder for creating and configuring processing pipelines with dependency injection
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from config import config, VoiceRecognitionConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from openai import NOT_GIVEN

from .service_factory import ServiceFactory


class PipelineBuilder:
    """Builder for creating processing pipelines with dependency injection"""
    
    def __init__(self, service_factory: ServiceFactory):
        self.service_factory = service_factory
    
    async def build_pipeline(self, webrtc_connection, language: str = "en", 
                           llm_model: str = None) -> Tuple[Pipeline, Any]:
        """
        Build complete processing pipeline with all components
        
        Args:
            webrtc_connection: WebRTC connection instance
            language: Language code for services
            llm_model: Optional specific LLM model
            
        Returns:
            Tuple of (pipeline, task) ready for execution
        """
        logger.info(f"🔨 Building pipeline for language: {language}")
        
        # 1. Get language configuration
        lang_config = self._get_language_config(language)
        
        # 2. Create core services
        services = await self._create_core_services(language, llm_model)
        
        # 3. Setup processors
        processors = await self._setup_processors(language)
        
        # 4. Setup transport
        transport = await self._setup_transport(webrtc_connection)
        
        # 5. Build context and aggregator
        context, context_aggregator = await self._build_context(lang_config, services['llm'], language)
        
        # 5a. Configure DJ mode handler with actual services
        if processors.get('music_mode'): # Check if music mode is enabled
            from processors.dj_mode_config_handler import DJModeConfigHandler
            processors['dj_config_handler'] = DJModeConfigHandler(
                tts=services['tts'],
                llm_context=context
            )
            logger.info("🎧 DJ Mode Config Handler configured.")
        
        # 5b. Connect audio player to LLM service for music control
        if processors.get('audio_player') and hasattr(services['llm'], 'set_audio_player'):
            services['llm'].set_audio_player(processors['audio_player'])
            logger.info("🎵 Connected audio player to LLM service")
        
        # 6. Build pipeline components
        pipeline_components = await self._build_pipeline_components(
            transport, services, processors, context_aggregator
        )
        
        # 7. Create pipeline
        pipeline = self._create_pipeline(pipeline_components)
        
        # 8. Create task
        task = await self._create_task(pipeline, context_aggregator)
        
        logger.info("✅ Pipeline built successfully")
        return pipeline, task
    
    def _get_language_config(self, language: str) -> dict:
        """Get language-specific configuration with consistency notes"""
        lang_config = config.get_language_config(language)
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
        
        # Create modified config with language consistency
        final_config = {
            "voice": lang_config.voice,
            "whisper_language": lang_config.whisper_language,
            "system_instruction": lang_config.system_instruction,
            "dj_voice": lang_config.dj_voice,
            "dj_system_prompt": lang_config.dj_system_prompt
        }
        
        if language in language_consistency_note:
            final_config["system_instruction"] += language_consistency_note[language]
        
        return final_config
    
    async def _create_core_services(self, language: str, llm_model: str = None) -> Dict[str, Any]:
        """Create core services (STT, TTS, LLM)"""
        logger.info("🔧 Creating core services...")
        
        # Wait for ML modules to be ready
        await self.service_factory.wait_for_ml_modules()
        
        # Create services
        services = await self.service_factory.create_services_for_language(language, llm_model)
        
        logger.info("✅ Core services created")
        return services
    
    async def _setup_processors(self, language: str) -> Dict[str, Any]:
        """Setup all pipeline processors"""
        logger.info("🔧 Setting up processors...")
        
        processors = {}
        
        # Memory processor
        memory_processor = await self.service_factory.get_service("memory_service")
        if memory_processor:
            # Set memory processor for tool handlers
            from tools import set_memory_processor
            set_memory_processor(memory_processor)
            
            from processors import MemoryContextInjector
            memory_injector = MemoryContextInjector(
                memory_processor=memory_processor,
                system_prompt=config.memory.context_system_prompt,
                inject_as_system=True
            )
            processors['memory_processor'] = memory_processor
            processors['memory_injector'] = memory_injector
        else:
            processors['memory_processor'] = None
            processors['memory_injector'] = None
        
        # Video processor
        if config.video.enabled:
            logger.info("📹 Video is ENABLED")
            from processors import VideoSamplerProcessor
            processors['video_sampler'] = VideoSamplerProcessor()
        else:
            logger.info("📷 Video is DISABLED")
            processors['video_sampler'] = None
        
        # Voice recognition processors
        voice_recognition = await self.service_factory.get_service("voice_recognition")
        if voice_recognition:
            processors.update(await self._setup_voice_recognition_processors(
                voice_recognition, memory_processor
            ))
        else:
            logger.info("🔇 Voice recognition is DISABLED")
            processors.update({
                'voice_recognition': None,
                'audio_tee': None,
                'vad_bridge': None,
                'speaker_context': None,
                'speaker_name_manager': None
            })
        
        # Other processors
        from processors import GreetingFilterProcessor, MessageDeduplicator
        processors['greeting_filter'] = GreetingFilterProcessor(greeting_text="Hello, I'm Slowcat!")
        processors['message_deduplicator'] = MessageDeduplicator()
        
        # Response formatter - fix markdown links from stubborn Qwen2.5  
        from processors.response_formatter import ResponseFormatterProcessor
        processors['response_formatter'] = ResponseFormatterProcessor()
        
        # Time-aware executor processor
        from processors.time_aware_executor import TimeAwareExecutor
        processors['time_executor'] = TimeAwareExecutor(
            base_output_dir='./data',
            enable_auto_save=True
        )
        logger.info("⏰ Time-aware executor enabled")
        
        # Set executor for tools
        from tools.time_tools import set_time_executor
        set_time_executor(processors['time_executor'])
        
        # Dictation mode processor
        if config.dictation_mode.enabled:
            from processors.dictation_mode import DictationModeProcessor
            processors['dictation_mode'] = DictationModeProcessor(
                output_dir=config.dictation_mode.output_dir,
                file_prefix=config.dictation_mode.file_prefix,
                append_mode=config.dictation_mode.append_mode,
                realtime_save=config.dictation_mode.realtime_save,
                save_interim=config.dictation_mode.save_interim,
                language=language
            )
            logger.info("📝 Dictation mode enabled")
        else:
            processors['dictation_mode'] = None
        
        # DJ/Music mode processor
        if config.dj_mode.enabled:
            from processors.audio_player_real import AudioPlayerRealProcessor
            from music.library_scanner import MusicLibraryScanner
            from processors.music_mode import MusicModeProcessor
            from processors.dj_mode_config_handler import DJModeConfigHandler
            
            # Initialize music scanner
            scanner = MusicLibraryScanner(config.dj_mode.index_file)
            
            # Scan music folders on startup
            for folder in config.dj_mode.music_folders:
                folder_path = Path(folder).expanduser()
                if folder_path.exists():
                    logger.info(f"🎵 Scanning music folder: {folder_path}")
                    scanner.scan_directory(str(folder_path))
            
            # Initialize REAL audio player
            processors['audio_player'] = AudioPlayerRealProcessor(
                sample_rate=config.audio.stt_sample_rate,
                channels=1,  # Mono audio for voice pipeline
                initial_volume=config.dj_mode.default_volume,
                duck_volume=config.dj_mode.duck_volume,
                crossfade_seconds=config.dj_mode.crossfade_seconds
            )
            
            # Initialize music mode processor
            processors['music_mode'] = MusicModeProcessor(
                language=language,
            )
            
            # Initialize DJ mode config handler as a placeholder
            processors['dj_config_handler'] = None
            
            # Set up music tools
            from tools.music_tools import set_music_scanner
            set_music_scanner(scanner)  
            
            # Log music library stats
            stats = scanner.get_stats()
            logger.info(f"🎵 Music library: {stats['total_songs']} songs, {stats['unique_artists']} artists")
        else:
            processors['audio_player'] = None
            processors['music_mode'] = None
            processors['dj_config_handler'] = None
        
        logger.info("✅ Processors setup complete")
        return processors
    
    async def _setup_voice_recognition_processors(self, voice_recognition, memory_processor) -> Dict[str, Any]:
        """Setup voice recognition related processors"""
        logger.info("🎙️ Setting up voice recognition processors...")
        
        from processors import AudioTeeProcessor, VADEventBridge, SpeakerContextProcessor, SpeakerNameManager
        
        processors = {}
        
        # Audio tee for voice recognition
        audio_tee = AudioTeeProcessor()
        audio_tee.register_audio_consumer(voice_recognition.process_audio_frame)
        processors['audio_tee'] = audio_tee
        
        # VAD bridge
        vad_bridge = VADEventBridge()
        vad_bridge.set_callbacks(
            voice_recognition.on_user_started_speaking,
            voice_recognition.on_user_stopped_speaking
        )
        processors['vad_bridge'] = vad_bridge
        
        # Speaker context
        speaker_context = SpeakerContextProcessor()
        processors['speaker_context'] = speaker_context
        
        # Speaker name manager
        speaker_name_manager = SpeakerNameManager(voice_recognition)
        processors['speaker_name_manager'] = speaker_name_manager
        
        # Setup callbacks
        async def on_speaker_changed(data: Dict[str, Any]):
            speaker_context.update_speaker(data)
            if memory_processor:
                user_id = data.get('speaker_name', data.get('speaker_id', 'unknown'))
                await memory_processor.update_user_id(user_id)
                logger.info(f"📝 Memory switched to user: {user_id}")
        
        async def on_speaker_enrolled(data: Dict[str, Any]):
            speaker_id = data.get('speaker_id')
            logger.info(f"🎓 Speaker enrolled event: {speaker_id}")
            
            # Only ask for name if they don't already have one
            if speaker_id in getattr(voice_recognition, 'speaker_names', {}):
                existing_name = voice_recognition.speaker_names[speaker_id]
                logger.info(f"✅ Speaker {speaker_id} already has name: {existing_name}")
            else:
                if data.get('needs_name'):
                    logger.info(f"🤔 Asking for name for new speaker: {speaker_id}")
                    speaker_name_manager.start_name_collection(speaker_id)
            
            speaker_context.handle_speaker_enrolled(data)
        
        voice_recognition.set_callbacks(on_speaker_changed, on_speaker_enrolled)
        processors['voice_recognition'] = voice_recognition
        
        return processors
    
    async def _setup_transport(self, webrtc_connection) -> SmallWebRTCTransport:
        """Setup WebRTC transport with analyzers"""
        logger.info("🔧 Setting up transport...")
        
        # Wait for global analyzers
        await self.service_factory.wait_for_global_analyzers()
        global_analyzers = self.service_factory.registry.get_instance("global_analyzers")
        
        if not global_analyzers:
            logger.error("Global analyzers not available!")
            raise RuntimeError("Global analyzers failed to initialize")
        
        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_in_enabled=config.video.enabled,
                vad_analyzer=global_analyzers['vad_analyzer'],
                turn_analyzer=global_analyzers['turn_analyzer'],
            )
        )
        
        logger.info("✅ Transport setup complete")
        return transport
    
    async def _build_context(self, lang_config: dict, llm_service: Any, language: str) -> Tuple[Any, Any]:
        """Build LLM context and aggregator"""
        logger.info("🔧 Building context...")

        from services.llm_with_tools import LLMWithToolsService
        from tools.definitions import ALL_FUNCTION_SCHEMAS
        from core.prompt_builder import generate_final_system_prompt
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        from services.simple_mcp_tool_manager import SimpleMCPToolManager
        from collections import namedtuple

        MockFunctionSchema = namedtuple("MockFunctionSchema", ["name"])

        base_system_prompt = lang_config["system_instruction"]
        local_tools = list(ALL_FUNCTION_SCHEMAS)

        # 🚀 USE CACHED MCP TOOLS: Get pre-discovered tools from global cache
        from services.simple_mcp_tool_manager import get_global_mcp_manager
        mcp_tool_manager = get_global_mcp_manager(language)
        
        # Get OpenAI-compatible tools (uses cached tools, no HTTP calls)
        mcp_tools_array = mcp_tool_manager.get_cached_tools_for_llm()
        
        # Convert to FunctionSchema for Pipecat integration
        mcp_function_schemas = []
        for tool_def in mcp_tools_array:
            function_info = tool_def["function"]
            from pipecat.adapters.schemas.function_schema import FunctionSchema
            
            mcp_schema = FunctionSchema(
                name=function_info["name"],
                description=function_info["description"],
                properties=function_info.get("parameters", {"type": "object", "properties": {}}),
                required=function_info.get("parameters", {}).get("required", [])
            )
            mcp_function_schemas.append(mcp_schema)
        
        # Create unified tools schema with BOTH local and MCP tools
        unified_tools = local_tools + mcp_function_schemas
        tools_schema = ToolsSchema(standard_tools=unified_tools)
        
        # Store MCP manager reference and WAIT for tool registration
        if hasattr(llm_service, 'set_mcp_tool_manager'):
            llm_service.set_mcp_tool_manager(mcp_tool_manager)
            
            # CRITICAL FIX: Wait for MCP tools to be registered before proceeding
            logger.info("⏳ Waiting for MCP tool registration to complete...")
            await llm_service._register_mcp_tools()  # Ensure tools are registered synchronously
            logger.info("✅ MCP tool registration completed")
        
        # 🚀 MCPO INTEGRATION: MCP tools are now guaranteed to be registered
        logger.info(f"🔧 MCP tool registration completed synchronously")
        
        logger.info(f"🔧 Unified tools schema created:")
        logger.info(f"   Local tools: {len(local_tools)} ({[t.name for t in local_tools]})")
        logger.info(f"   MCP tools: {len(mcp_function_schemas)} ({[t.name for t in mcp_function_schemas]})")
        logger.info(f"   Total tools: {len(unified_tools)}")

        # Generate the full system prompt with local and MCP tools
        final_system_prompt = generate_final_system_prompt(
            base_prompt=base_system_prompt,
            local_tools=local_tools,
            mcp_tools=mcp_function_schemas
        )

        logger.debug(f"Final System Prompt:\n{final_system_prompt}")

        context = OpenAILLMContext(
            [{"role": "system", "content": final_system_prompt}],
            tools=tools_schema
        )
        
        context_aggregator = llm_service.create_context_aggregator(context)
        
        logger.info("✅ Context built")
        return context, context_aggregator
    
    async def _build_pipeline_components(self, transport, services: Dict[str, Any], 
                                       processors: Dict[str, Any], context_aggregator) -> List[Any]:
        """Build ordered list of pipeline components"""
        logger.info("🔧 Building pipeline components...")
        
        from pipecat.processors.frameworks.rtvi import RTVIProcessor
        rtvi = RTVIProcessor()
        
        components = [
            transport.input(),
            processors['video_sampler'],
            processors['audio_tee'],
            processors['vad_bridge'],
            services['stt'],
            processors['dictation_mode'],  # Must be after STT but before LLM
            processors['music_mode'],  # Music mode filtering (after STT, before LLM)
            processors['dj_config_handler'],  # Handle DJ mode voice/prompt changes
            processors['audio_player'],  # Music player with ducking
            processors['time_executor'],  # Time-aware task execution
            processors['memory_processor'],
            processors['speaker_context'],
            rtvi,
            processors['speaker_name_manager'],
            context_aggregator.user(),
            processors['memory_injector'],
            processors['message_deduplicator'],
            services['llm'],
            services['tts'],
            transport.output(),
            processors['greeting_filter'],
            context_aggregator.assistant(),
        ]
        
        # Filter out None components
        filtered_components = [comp for comp in components if comp is not None]
        
        logger.info(f"✅ Built pipeline with {len(filtered_components)} components")
        return filtered_components
    
    def _create_pipeline(self, components: List[Any]) -> Pipeline:
        """Create pipeline from components"""
        return Pipeline(components)
    
    async def _create_task(self, pipeline: Pipeline, context_aggregator) -> Any:
        """Create pipeline task with observers"""
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
        
        # Find RTVI processor
        rtvi_processor = None
        for component in pipeline._processors:
            if isinstance(component, RTVIProcessor):
                rtvi_processor = component
                break
        
        if not rtvi_processor:
            raise RuntimeError("RTVI processor not found in pipeline")
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True, 
                enable_usage_metrics=True,
                idle_timeout_secs=600  # 10-minute timeout
            ),
            observers=[RTVIObserver(rtvi_processor)]
        )
        
        # Setup RTVI event handler
        @rtvi_processor.event_handler("on_client_ready")
        async def on_client_ready(rtvi_proc):
            await rtvi_proc.set_bot_ready()
            await task.queue_frames([context_aggregator.user().get_context_frame()])
        
        return task