"""
Service factory for creating and managing service instances with dependency injection
"""

import asyncio
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
        
        logger.info("✅ Global analyzers initialized")
        self._global_analyzers_ready.set()
        return analyzers
    
    def _create_stt_service(self, ml_modules: Dict[str, Any], language: str = "en"):
        """Create STT service"""
        MLXModel = ml_modules['MLXModel']
        WhisperSTTServiceMLX = ml_modules['WhisperSTTServiceMLX']
        
        # Import here to avoid circular imports
        from pipecat.transcriptions.language import Language
        
        stt_model = MLXModel.DISTIL_LARGE_V3 if language == "en" else MLXModel.MEDIUM
        whisper_language = getattr(Language, config.get_language_config(language).whisper_language)
        
        logger.info(f"Using STT model: {stt_model.name} for {language}")
        return WhisperSTTServiceMLX(model=stt_model, language=whisper_language)
    
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
        
        llm_params = {
            "api_key": None,
            "model": selected_model,
            "base_url": config.network.llm_base_url,
            "max_tokens": config.models.llm_max_tokens
        }
        
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
    
    async def create_services_for_language(self, language: str, llm_model: str = None) -> Dict[str, Any]:
        """Create core services for a specific language"""
        # Ensure ML modules are loaded
        await self.get_service("ml_loader")
        
        # Create language-specific services
        services = {}
        services['stt'] = await self._create_stt_service_for_language(language)
        services['tts'] = await self._create_tts_service_for_language(language) 
        services['llm'] = await self._create_llm_service_for_language(language, llm_model)
        
        return services
    
    async def _create_stt_service_for_language(self, language: str):
        """Create STT service for specific language"""
        ml_modules = self.registry.get_instance("ml_loader")
        return self._create_stt_service(ml_modules, language)
    
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