"""
Service factory for creating and managing service instances with dependency injection
"""

import asyncio
import os
import threading
import importlib
from typing import Dict, Any, Optional, Type, TypeVar, Generic, Callable
from dataclasses import dataclass
from loguru import logger
from config import config, VoiceRecognitionConfig

T = TypeVar('T')


@dataclass
class ServiceDefinition:
    """Definition of a service including its factory function and dependencies"""
    factory: Callable[..., Any]
    dependencies: list[str]
    singleton: bool = True
    lazy: bool = False
    initialized: bool = False


class ServiceRegistry:
    """Registry for service definitions and instances"""
    
    def __init__(self):
        self._definitions: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, factory: Callable[..., Any], 
                 dependencies: list[str] = None, singleton: bool = True, lazy: bool = False):
        """Register a service definition"""
        with self._lock:
            self._definitions[name] = ServiceDefinition(
                factory=factory,
                dependencies=dependencies or [],
                singleton=singleton,
                lazy=lazy
            )
            logger.debug(f"Registered service: {name}")
    
    def get_definition(self, name: str) -> Optional[ServiceDefinition]:
        """Get service definition by name"""
        return self._definitions.get(name)
    
    def set_instance(self, name: str, instance: Any):
        """Set service instance"""
        with self._lock:
            self._instances[name] = instance
    
    def get_instance(self, name: str) -> Optional[Any]:
        """Get service instance by name"""
        return self._instances.get(name)
    
    def has_instance(self, name: str) -> bool:
        """Check if service instance exists"""
        return name in self._instances
    
    def list_services(self) -> list[str]:
        """List all registered service names"""
        return list(self._definitions.keys())


class ServiceFactory:
    """Factory for creating services with dependency injection"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self._ml_modules_loaded = threading.Event()
        self._global_analyzers_ready = threading.Event()
        self._setup_core_services()
    
    def _setup_core_services(self):
        """Register core service factories"""
        
        # ML Module Loader (special service)
        self.registry.register(
            "ml_loader",
            self._create_ml_loader,
            dependencies=[],
            singleton=True,
            lazy=False
        )
        
        # Global Analyzers
        self.registry.register(
            "global_analyzers", 
            self._create_global_analyzers,
            dependencies=["ml_loader"],
            singleton=True,
            lazy=False
        )
        
        # STT Service
        self.registry.register(
            "stt_service",
            self._create_stt_service,
            dependencies=["ml_loader"],
            singleton=True,
            lazy=True
        )
        
        # TTS Service  
        self.registry.register(
            "tts_service",
            self._create_tts_service,
            dependencies=["ml_loader"],
            singleton=True,
            lazy=True
        )
        
        # LLM Service
        self.registry.register(
            "llm_service",
            self._create_llm_service,
            dependencies=["ml_loader"],
            singleton=True,
            lazy=True
        )
        
        # Voice Recognition
        self.registry.register(
            "voice_recognition",
            self._create_voice_recognition,
            dependencies=["ml_loader"],
            singleton=True,
            lazy=True
        )
        
        # Memory Service
        self.registry.register(
            "memory_service",
            self._create_memory_service,
            dependencies=[],
            singleton=True,
            lazy=True
        )
    
    async def get_service(self, name: str) -> Any:
        """Get service instance, creating if necessary"""
        # Check if we already have an instance
        if self.registry.has_instance(name):
            return self.registry.get_instance(name)
        
        # Get service definition
        definition = self.registry.get_definition(name)
        if not definition:
            raise ValueError(f"Service '{name}' not registered")
        
        # Wait for dependencies
        dependency_instances = []
        for dep_name in definition.dependencies:
            dep_instance = await self.get_service(dep_name)
            dependency_instances.append(dep_instance)
        
        # Create instance
        logger.info(f"Creating service: {name}")
        try:
            if asyncio.iscoroutinefunction(definition.factory):
                instance = await definition.factory(*dependency_instances)
            else:
                instance = definition.factory(*dependency_instances)
            
            # Store instance if singleton
            if definition.singleton:
                self.registry.set_instance(name, instance)
            
            definition.initialized = True
            logger.info(f"✅ Service created: {name}")
            return instance
            
        except Exception as e:
            logger.error(f"❌ Failed to create service '{name}': {e}")
            raise
    
    def _create_ml_loader(self) -> Dict[str, Any]:
        """Create ML module loader"""
        logger.info("🔄 Loading ML modules...")
        
        modules = {}
        try:
            # Load STT modules
            whisper_module = importlib.import_module("services.whisper_stt_with_lock")
            modules['WhisperSTTServiceMLX'] = whisper_module.WhisperSTTServiceMLX
            
            sherpa_module = importlib.import_module("services.sherpa_stt")
            modules['SherpaONNXSTTService'] = sherpa_module.SherpaONNXSTTService
            
            # 🔥 NEW: Load streaming Sherpa service
            sherpa_streaming_module = importlib.import_module("services.sherpa_streaming_stt")
            modules['SherpaStreamingSTTService'] = sherpa_streaming_module.SherpaStreamingSTTService
            
            # 🚀 REAL: Load PROPER OnlineRecognizer streaming service
            sherpa_online_module = importlib.import_module("services.sherpa_streaming_stt_v2")
            modules['SherpaOnlineSTTService'] = sherpa_online_module.SherpaOnlineSTTService
            
            stt_module = importlib.import_module("pipecat.services.whisper.stt")
            modules['MLXModel'] = stt_module.MLXModel
            
            # Load TTS module
            tts_module = importlib.import_module("kokoro_tts")
            modules['KokoroTTSService'] = tts_module.KokoroTTSService
            
            # Load LLM modules
            llm_tools_module = importlib.import_module("services.llm_with_tools")
            modules['LLMWithToolsService'] = llm_tools_module.LLMWithToolsService
            
            openai_module = importlib.import_module("pipecat.services.openai.llm")
            modules['OpenAILLMService'] = openai_module.OpenAILLMService
            
            # Load voice recognition
            voice_module = importlib.import_module("voice_recognition")
            modules['AutoEnrollVoiceRecognition'] = voice_module.AutoEnrollVoiceRecognition
            
            # Load audio analyzers
            vad_module = importlib.import_module("pipecat.audio.vad.silero")
            modules['SileroVADAnalyzer'] = vad_module.SileroVADAnalyzer
            
            turn_module = importlib.import_module("pipecat.audio.turn.smart_turn.local_smart_turn_v2")
            modules['LocalSmartTurnAnalyzerV2'] = turn_module.LocalSmartTurnAnalyzerV2
            
            logger.info("✅ ML modules loaded successfully")
            self._ml_modules_loaded.set()
            return modules
            
        except Exception as e:
            logger.error(f"❌ Failed to load ML modules: {e}")
            raise
    
    def _create_global_analyzers(self, ml_modules: Dict[str, Any]) -> Dict[str, Any]:
        """Create global analyzer instances"""
        logger.info("🔄 Initializing global analyzers...")
        
        # Initialize audio buffer pool for latency optimization
        from utils.audio_pool import initialize_audio_pool
        initialize_audio_pool(pool_size=15)  # Larger pool for voice app
        
        from pipecat.audio.vad.vad_analyzer import VADParams
        
        vad_analyzer = ml_modules['SileroVADAnalyzer'](params=VADParams(
            stop_secs=config.audio.vad_stop_secs,
            start_secs=config.audio.vad_start_secs,
        ))
        
        turn_analyzer = ml_modules['LocalSmartTurnAnalyzerV2'](
            smart_turn_model_path=config.models.smart_turn_model_path
        )
        
        analyzers = {
            'vad_analyzer': vad_analyzer,
            'turn_analyzer': turn_analyzer
        }
        
        # Pre-warm Kokoro TTS model
        try:
            logger.info("🔄 Pre-warming Kokoro TTS model...")
            kokoro = ml_modules['KokoroTTSService'](voice="af_heart")
            logger.info("✅ Kokoro TTS model ready")
        except Exception as e:
            logger.warning(f"Failed to pre-warm Kokoro TTS: {e}")
        
        # Pre-warm STT model (just create instance to load model)
        try:
            logger.info("🔄 Pre-warming STT model...")
            
            MLXModel = ml_modules['MLXModel']
            WhisperSTTServiceMLX = ml_modules['WhisperSTTServiceMLX']
            
            # Import here to avoid circular imports
            from pipecat.transcriptions.language import Language
            
            # Temporarily allow downloads for turbo models
            import os
            original_offline = os.environ.get("HF_HUB_OFFLINE")
            original_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
            logger.info("🔄 Temporarily enabling downloads for STT model pre-warming")
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            
            try:
                # Create STT service instance - this loads the model
                stt_model = MLXModel.LARGE_V3_TURBO_Q4
                whisper_language = Language.EN
                stt_service = WhisperSTTServiceMLX(model=stt_model, language=whisper_language)
            finally:
                # Restore offline mode
                if original_offline:
                    os.environ["HF_HUB_OFFLINE"] = original_offline
                if original_transformers_offline:
                    os.environ["TRANSFORMERS_OFFLINE"] = original_transformers_offline
            
            logger.info("✅ STT model pre-warmed")
        except Exception as e:
            logger.warning(f"Failed to pre-warm STT: {e}")
        
        logger.info("✅ Global analyzers initialized")
        self._global_analyzers_ready.set()
        return analyzers
    
    def _create_stt_service(self, ml_modules: Dict[str, Any], language: str = "en", stt_model: str = None):
        """Create STT service"""
        backend = os.getenv("STT_BACKEND", "whisper-mlx").lower()
        
        if backend == "sherpa-onnx":
            # 🔥 CHECK FOR STREAMING MODE
            streaming_mode = os.getenv("SHERPA_STREAMING", "true").lower() == "true"
            
            # 🔥 CHECK FOR ONLINE STREAMING MODE (NEW!)
            online_streaming = os.getenv("SHERPA_ONLINE_STREAMING", "true").lower() == "true"
            
            if online_streaming:
                logger.info("🚀 Using ONLINE Sherpa-ONNX STT Service (OnlineRecognizer)!")
                SherpaSTT = ml_modules["SherpaOnlineSTTService"]
            elif streaming_mode:
                logger.info("🔄 Using STREAMING Sherpa-ONNX STT Service (OfflineRecognizer hack)")
                SherpaSTT = ml_modules["SherpaStreamingSTTService"]
            else:
                logger.info("📦 Using SEGMENTED Sherpa-ONNX STT Service")
                SherpaSTT = ml_modules["SherpaONNXSTTService"]
            
            model_dir = os.getenv("SHERPA_ONNX_MODEL_DIR", "").strip()
            if not model_dir:
                raise RuntimeError(
                    "STT_BACKEND is 'sherpa-onnx' but SHERPA_ONNX_MODEL_DIR is not set"
                )
            decoding_method = os.getenv("SHERPA_DECODING_METHOD", "greedy_search")
            provider = os.getenv("SHERPA_PROVIDER", "cpu")
            
            # Base configuration
            sherpa_config = {
                "model_dir": model_dir,
                "language": language,
                "decoding_method": decoding_method,
                "provider": provider,
            }
            
            if online_streaming:
                # 🚀 ONLINE STREAMING CONFIG (OnlineRecognizer)
                hotwords_file = os.getenv("SHERPA_HOTWORDS_FILE", "").strip() or ""
                sherpa_config.update({
                    "enable_endpoint_detection": os.getenv("SHERPA_ENDPOINT_DETECTION", "true").lower() == "true",
                    "chunk_size_ms": int(os.getenv("SHERPA_CHUNK_MS", "200")),
                    "emit_partial_results": os.getenv("SHERPA_PARTIAL_RESULTS", "false").lower() == "true",
                    "max_active_paths": int(os.getenv("SHERPA_MAX_PATHS", "8")),
                    "num_threads": int(os.getenv("SHERPA_THREADS", "1")),
                    "hotwords_file": hotwords_file,
                    "hotwords_score": float(os.getenv("SHERPA_HOTWORDS_SCORE", "1.5")),
                })
                logger.info(f"🚀 Online streaming config: chunk={sherpa_config['chunk_size_ms']}ms, endpoint_detection={sherpa_config['enable_endpoint_detection']}")
            elif streaming_mode:
                # 🔥 STREAMING CONFIG (OfflineRecognizer hack)
                sherpa_config.update({
                    "chunk_duration_ms": int(os.getenv("SHERPA_CHUNK_MS", "500")),
                    "overlap_duration_ms": int(os.getenv("SHERPA_OVERLAP_MS", "200")),
                    "min_confidence": float(os.getenv("SHERPA_MIN_CONFIDENCE", "0.3")),
                })
                logger.info(f"🔥 Streaming config: chunk={sherpa_config['chunk_duration_ms']}ms, overlap={sherpa_config['overlap_duration_ms']}ms")
            else:
                # Legacy segmented config
                hotwords_file = os.getenv("SHERPA_HOTWORDS_FILE", "").strip() or None
                hotwords_score = float(os.getenv("SHERPA_HOTWORDS_SCORE", "1.5"))
                language_lock = os.getenv("SHERPA_LANGUAGE_LOCK", "").strip() or None
                language_lock_mode = os.getenv("SHERPA_LANGUAGE_LOCK_MODE", "strict")
                
                if not language_lock and language != "auto" and language in ['be', 'de', 'en', 'es', 'fr', 'hr', 'it', 'pl', 'ru', 'uk']:
                    language_lock = language
                    logger.info(f"🔐 Auto-enabling language lock for --language {language}")
                
                sherpa_config.update({
                    "hotwords_file": hotwords_file,
                    "hotwords_score": hotwords_score,
                    "language_lock": language_lock,
                    "language_lock_mode": language_lock_mode,
                })
            
            return SherpaSTT(**sherpa_config)
        
        # default: Whisper MLX
        MLXModel = ml_modules['MLXModel']
        WhisperSTTServiceMLX = ml_modules['WhisperSTTServiceMLX']
        
        # Import here to avoid circular imports
        from pipecat.transcriptions.language import Language
        
        # Determine STT model to use
        if stt_model:
            # Use specified model from CLI argument
            selected_model = getattr(MLXModel, stt_model)
            logger.info(f"Using CLI-specified STT model: {stt_model} for {language}")
        else:
            # Use default logic
            selected_model = MLXModel.LARGE_V3_TURBO_Q4 if language == "en" else MLXModel.MEDIUM
            logger.info(f"Using default STT model: {selected_model.name} for {language}")
        
        whisper_language = getattr(Language, config.get_language_config(language).whisper_language)
        return WhisperSTTServiceMLX(model=selected_model, language=whisper_language)
    
    def _create_tts_service(self, ml_modules: Dict[str, Any], language: str = "en"):
        """Create TTS service"""
        KokoroTTSService = ml_modules['KokoroTTSService']
        lang_config = config.get_language_config(language)
        
        # Import here to avoid circular imports  
        from pipecat.transcriptions.language import Language
        whisper_language = getattr(Language, lang_config.whisper_language)
        
        return KokoroTTSService(
            model=config.models.tts_model,
            voice=lang_config.voice,
            language=whisper_language,
            sample_rate=config.audio.tts_sample_rate,
            max_workers=config.audio.tts_max_workers
        )
    
    async def _create_llm_service(self, ml_modules: Dict[str, Any], language: str = "en", llm_model: str = None):
        """Create LLM service"""
        selected_model = llm_model or config.models.default_llm_model
        logger.info(f"🤖 Using LLM model: {selected_model}")
        
        # Check if streaming is enabled
        streaming_enabled = os.getenv("LLM_STREAMING", "true").lower() == "true"
        
        llm_params = {
            "api_key": None,
            "model": selected_model,
            "base_url": config.network.llm_base_url,
            "max_tokens": config.models.llm_max_tokens
        }
        
        if streaming_enabled:
            logger.info("🌊 LLM streaming mode ENABLED")
        else:
            logger.info("🔒 LLM streaming mode DISABLED")
        
        # MCP tools are handled natively by LM Studio via mcp.json
        if config.mcp.enabled:
            logger.info("🔧 Tool-enabled LLM service initialized (MCP via LM Studio)")
            return ml_modules['LLMWithToolsService'](**llm_params)
        else:
            logger.info("🤖 Standard LLM service initialized")
            return ml_modules['OpenAILLMService'](**llm_params)
    
    async def _create_voice_recognition(self, ml_modules: Dict[str, Any], vr_config: VoiceRecognitionConfig = None):
        """Create voice recognition service"""
        if not vr_config:
            vr_config = config.voice_recognition
            
        if not vr_config.enabled:
            return None
            
        logger.info("🎙️ Voice recognition is ENABLED")
        voice_recognition = ml_modules['AutoEnrollVoiceRecognition'](vr_config)
        await voice_recognition.initialize()
        return voice_recognition
    
    def _create_memory_service(self):
        """Create memory service"""
        if not config.memory.enabled:
            return None
            
        logger.info("🧠 Memory is ENABLED")
        from processors import LocalMemoryProcessor
        return LocalMemoryProcessor(
            data_dir=config.memory.data_dir,
            user_id=config.memory.default_user_id,
            max_history_items=config.memory.max_history_items,
            include_in_context=config.memory.include_in_context
        )
    
    async def create_services_for_language(self, language: str, llm_model: str = None, stt_model: str = None) -> Dict[str, Any]:
        """Create core services for a specific language"""
        # Ensure ML modules are loaded
        await self.get_service("ml_loader")
        
        # Create language-specific services
        services = {}
        services['stt'] = await self._create_stt_service_for_language(language, stt_model)
        services['tts'] = await self._create_tts_service_for_language(language) 
        services['llm'] = await self._create_llm_service_for_language(language, llm_model)
        
        return services
    
    async def _create_stt_service_for_language(self, language: str, stt_model: str = None):
        """Create STT service for specific language"""
        ml_modules = self.registry.get_instance("ml_loader")
        return self._create_stt_service(ml_modules, language, stt_model)
    
    async def _create_tts_service_for_language(self, language: str):
        """Create TTS service for specific language"""
        ml_modules = self.registry.get_instance("ml_loader")
        return self._create_tts_service(ml_modules, language)
    
    async def _create_llm_service_for_language(self, language: str, llm_model: str = None):
        """Create LLM service for specific language and model"""
        ml_modules = self.registry.get_instance("ml_loader")
        return await self._create_llm_service(ml_modules, language, llm_model)
    
    async def wait_for_ml_modules(self):
        """Wait for ML modules to be loaded"""
        if not self._ml_modules_loaded.is_set():
            logger.info("Waiting for ML modules to load...")
            await asyncio.get_event_loop().run_in_executor(None, self._ml_modules_loaded.wait)
    
    async def wait_for_global_analyzers(self):
        """Wait for global analyzers to be ready"""
        if not self._global_analyzers_ready.is_set():
            logger.info("Waiting for global analyzers...")
            await asyncio.get_event_loop().run_in_executor(None, self._global_analyzers_ready.wait)


# Global service factory instance
service_factory = ServiceFactory()