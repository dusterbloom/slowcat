"""
M3 Integration Module - Complete M3-AudioGraph integration for Slowcat

Provides factory functions and integration utilities for connecting
M3-inspired AudioGraph system with existing Slowcat components:

- AudioGraph initialization with M3 parameters
- Voice recognition integration
- SurrealDB persistence setup  
- Pipeline builder integration
- Configuration management

This is the main entry point for adopting M3 patterns in Slowcat.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import os
import time

logger = logging.getLogger(__name__)

# M3 components
from .audio_graph import AudioGraph
from .voice_processing import VoiceProcessor
from .memory_processing import MemoryProcessor  
from .m3_surreal_schema import M3SurrealIntegration, connect_audio_graph_to_surreal

# Slowcat integration
try:
    from processors.m3_smart_context_manager import M3SmartContextManager
    from voice_recognition.lightweight import LightweightVoiceRecognition
    M3_PROCESSORS_AVAILABLE = True
except ImportError:
    M3SmartContextManager = None
    LightweightVoiceRecognition = None
    M3_PROCESSORS_AVAILABLE = False
    logger.warning("M3 processors not fully available - some integrations may fail")


class M3AudioGraphFactory:
    """
    Factory for creating M3-configured AudioGraph systems.
    
    Handles:
    - AudioGraph initialization with optimal M3 parameters
    - Component integration and wiring
    - Configuration from environment/config files
    - Performance optimization for Apple Silicon
    """
    
    @staticmethod
    def create_audio_graph(config: Optional[Dict] = None) -> AudioGraph:
        """
        Create AudioGraph with M3 optimal configuration.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Configured AudioGraph instance
        """
        # M3 optimal parameters for voice agents
        default_config = {
            'max_voice_embeddings': 20,        # M3 default for audio nodes
            'max_text_embeddings': 10,         # M3 default for text nodes
            'voice_matching_threshold': 0.6,   # M3's proven threshold
            'text_matching_threshold': 0.3     # M3's proven threshold
        }
        
        # Override with provided config
        if config:
            default_config.update(config)
        
        # Create AudioGraph with M3 parameters
        audio_graph = AudioGraph(
            max_voice_embeddings=default_config['max_voice_embeddings'],
            max_text_embeddings=default_config['max_text_embeddings'],
            voice_matching_threshold=default_config['voice_matching_threshold'],
            text_matching_threshold=default_config['text_matching_threshold']
        )
        
        logger.info(f"🧠 Created AudioGraph with M3 configuration")
        logger.debug(f"   Voice embeddings: {default_config['max_voice_embeddings']}")
        logger.debug(f"   Text embeddings: {default_config['max_text_embeddings']}")
        logger.debug(f"   Voice threshold: {default_config['voice_matching_threshold']}")
        logger.debug(f"   Text threshold: {default_config['text_matching_threshold']}")
        
        return audio_graph
    
    @staticmethod
    def create_voice_processor(audio_graph: AudioGraph, config: Optional[Dict] = None) -> VoiceProcessor:
        """
        Create VoiceProcessor integrated with AudioGraph.
        
        Args:
            audio_graph: AudioGraph instance for integration
            config: Optional voice processor configuration
            
        Returns:
            Configured VoiceProcessor instance
        """
        voice_config = {
            'voice_sample_rate': 16000,      # Resemblyzer standard
            'min_voice_duration': 1.0,       # Minimum processing duration
            'confidence_threshold': 0.7      # M3's matching threshold
        }
        
        if config:
            voice_config.update(config)
        
        processor = VoiceProcessor(audio_graph, voice_config)
        
        logger.info(f"🎤 Created VoiceProcessor for AudioGraph")
        
        return processor
    
    @staticmethod
    def create_memory_processor(audio_graph: AudioGraph, llm_service=None, 
                               config: Optional[Dict] = None) -> MemoryProcessor:
        """
        Create MemoryProcessor integrated with AudioGraph.
        
        Args:
            audio_graph: AudioGraph instance for integration
            llm_service: Optional LLM service for memory generation
            config: Optional memory processor configuration
            
        Returns:
            Configured MemoryProcessor instance
        """
        memory_config = {
            'max_context_length': 2000,                # Context for memory generation
            'embedding_model_name': 'all-MiniLM-L6-v2' # Compact, fast model
        }
        
        if config:
            memory_config.update(config)
        
        processor = MemoryProcessor(audio_graph, llm_service, memory_config)
        
        logger.info(f"🧠 Created MemoryProcessor for AudioGraph")
        
        return processor
    
    @staticmethod
    async def create_surreal_integration(audio_graph: AudioGraph, 
                                        connection_config: Optional[Dict] = None) -> Optional[M3SurrealIntegration]:
        """
        Create SurrealDB integration for AudioGraph persistence.
        
        Args:
            audio_graph: AudioGraph instance for persistence
            connection_config: Optional SurrealDB connection config
            
        Returns:
            M3SurrealIntegration instance if successful, None otherwise
        """
        # Default SurrealDB configuration
        default_config = {
            'url': os.getenv('SURREALDB_URL', 'ws://localhost:8000/rpc'),
            'namespace': os.getenv('SURREALDB_NAMESPACE', 'slowcat'),
            'database': os.getenv('SURREALDB_DATABASE', 'm3_memory')
        }
        
        if connection_config:
            default_config.update(connection_config)
        
        integration = await connect_audio_graph_to_surreal(audio_graph, default_config)
        
        if integration:
            logger.info(f"🗄️  Created SurrealDB integration for AudioGraph")
        else:
            logger.warning("Failed to create SurrealDB integration")
        
        return integration


class M3ProcessorFactory:
    """
    Factory for creating M3-integrated Slowcat processors.
    
    Handles:
    - M3SmartContextManager creation
    - Voice recognition integration
    - Pipeline component wiring
    """
    
    @staticmethod
    def create_m3_context_manager(context, audio_graph: AudioGraph, 
                                 config: Optional[Dict] = None) -> Optional[M3SmartContextManager]:
        """
        Create M3SmartContextManager with AudioGraph integration.
        
        Args:
            context: LLMContext instance
            audio_graph: AudioGraph instance for memory
            config: Optional configuration
            
        Returns:
            M3SmartContextManager instance if available, None otherwise
        """
        if not M3_PROCESSORS_AVAILABLE or not M3SmartContextManager:
            logger.error("M3SmartContextManager not available")
            return None
        
        # M3 context manager configuration
        context_config = {
            'max_context_tokens': 4096,  # Fixed context size
            'memory_tokens': 2000,       # Tokens for memory context
        }
        
        if config:
            context_config.update(config)
        
        manager = M3SmartContextManager(
            context=context,
            audio_graph=audio_graph,
            max_context_tokens=context_config['max_context_tokens'],
            memory_tokens=context_config['memory_tokens']
        )
        
        logger.info(f"🎯 Created M3SmartContextManager with AudioGraph")
        
        return manager
    
    @staticmethod
    def integrate_voice_recognition(audio_graph: AudioGraph, voice_recognition) -> bool:
        """
        Integrate existing voice recognition with AudioGraph.
        
        Args:
            audio_graph: AudioGraph instance
            voice_recognition: LightweightVoiceRecognition instance
            
        Returns:
            Success indicator
        """
        try:
            if hasattr(voice_recognition, 'set_audio_graph'):
                voice_recognition.set_audio_graph(audio_graph)
                logger.info(f"🔗 Integrated voice recognition with AudioGraph")
                return True
            else:
                logger.warning("Voice recognition doesn't support AudioGraph integration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to integrate voice recognition: {e}")
            return False


class M3SystemBuilder:
    """
    Builder for complete M3-integrated Slowcat systems.
    
    Provides high-level assembly of M3 components with existing Slowcat infrastructure.
    """
    
    def __init__(self):
        """Initialize M3 system builder."""
        self.audio_graph = None
        self.voice_processor = None
        self.memory_processor = None
        self.surreal_integration = None
        self.context_manager = None
        
        self.config = {}
        
        logger.info(f"🏗️  M3SystemBuilder initialized")
    
    def with_config(self, config: Dict[str, Any]) -> 'M3SystemBuilder':
        """Set configuration for M3 system."""
        self.config.update(config)
        return self
    
    def with_audio_graph(self, config: Optional[Dict] = None) -> 'M3SystemBuilder':
        """Add AudioGraph with M3 configuration."""
        graph_config = self.config.get('audio_graph', {})
        if config:
            graph_config.update(config)
            
        self.audio_graph = M3AudioGraphFactory.create_audio_graph(graph_config)
        return self
    
    def with_voice_processing(self, config: Optional[Dict] = None) -> 'M3SystemBuilder':
        """Add voice processing with M3 patterns."""
        if not self.audio_graph:
            raise ValueError("AudioGraph must be created first")
        
        voice_config = self.config.get('voice_processing', {})
        if config:
            voice_config.update(config)
            
        self.voice_processor = M3AudioGraphFactory.create_voice_processor(
            self.audio_graph, voice_config
        )
        return self
    
    def with_memory_processing(self, llm_service=None, config: Optional[Dict] = None) -> 'M3SystemBuilder':
        """Add memory processing with M3 patterns."""
        if not self.audio_graph:
            raise ValueError("AudioGraph must be created first")
        
        memory_config = self.config.get('memory_processing', {})
        if config:
            memory_config.update(config)
            
        self.memory_processor = M3AudioGraphFactory.create_memory_processor(
            self.audio_graph, llm_service, memory_config
        )
        return self
    
    async def with_persistence(self, config: Optional[Dict] = None) -> 'M3SystemBuilder':
        """Add SurrealDB persistence with M3 schema."""
        if not self.audio_graph:
            raise ValueError("AudioGraph must be created first")
        
        persistence_config = self.config.get('persistence', {})
        if config:
            persistence_config.update(config)
            
        self.surreal_integration = await M3AudioGraphFactory.create_surreal_integration(
            self.audio_graph, persistence_config
        )
        return self
    
    def with_context_management(self, context, config: Optional[Dict] = None) -> 'M3SystemBuilder':
        """Add M3SmartContextManager."""
        if not self.audio_graph:
            raise ValueError("AudioGraph must be created first")
        
        context_config = self.config.get('context_management', {})
        if config:
            context_config.update(config)
            
        self.context_manager = M3ProcessorFactory.create_m3_context_manager(
            context, self.audio_graph, context_config
        )
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build complete M3-integrated system.
        
        Returns:
            Dictionary of M3 system components
        """
        if not self.audio_graph:
            raise ValueError("AudioGraph is required")
        
        system = {
            'audio_graph': self.audio_graph,
            'voice_processor': self.voice_processor,
            'memory_processor': self.memory_processor,
            'surreal_integration': self.surreal_integration,
            'context_manager': self.context_manager
        }
        
        # Filter None values
        system = {k: v for k, v in system.items() if v is not None}
        
        logger.info(f"✅ M3 system built with components: {list(system.keys())}")
        
        return system
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'build_time': time.time(),
            'components_built': [],
            'audio_graph_stats': None,
            'voice_processor_stats': None,
            'memory_processor_stats': None
        }
        
        if self.audio_graph:
            stats['components_built'].append('audio_graph')
            stats['audio_graph_stats'] = self.audio_graph.get_stats()
        
        if self.voice_processor:
            stats['components_built'].append('voice_processor')
            stats['voice_processor_stats'] = self.voice_processor.get_processing_stats()
        
        if self.memory_processor:
            stats['components_built'].append('memory_processor')
            stats['memory_processor_stats'] = getattr(self.memory_processor, 'stats', {})
        
        if self.surreal_integration:
            stats['components_built'].append('surreal_integration')
        
        if self.context_manager:
            stats['components_built'].append('context_manager')
        
        return stats


# Convenience functions for easy integration

def create_m3_system(config: Optional[Dict] = None, llm_service=None, 
                    context=None) -> Dict[str, Any]:
    """
    Create complete M3 system with default configuration.
    
    Args:
        config: Optional configuration overrides
        llm_service: Optional LLM service for memory generation
        context: Optional LLM context for context manager
        
    Returns:
        Complete M3 system components
    """
    builder = M3SystemBuilder()
    
    if config:
        builder.with_config(config)
    
    # Build system with all components
    builder.with_audio_graph()
    builder.with_voice_processing()
    builder.with_memory_processing(llm_service)
    
    if context:
        builder.with_context_management(context)
    
    return builder.build()


async def create_m3_system_with_persistence(config: Optional[Dict] = None, 
                                           llm_service=None, context=None) -> Dict[str, Any]:
    """
    Create complete M3 system with SurrealDB persistence.
    
    Args:
        config: Optional configuration overrides
        llm_service: Optional LLM service for memory generation
        context: Optional LLM context for context manager
        
    Returns:
        Complete M3 system components with persistence
    """
    builder = M3SystemBuilder()
    
    if config:
        builder.with_config(config)
    
    # Build system with all components including persistence
    builder.with_audio_graph()
    builder.with_voice_processing() 
    builder.with_memory_processing(llm_service)
    
    await builder.with_persistence()
    
    if context:
        builder.with_context_management(context)
    
    return builder.build()


def get_m3_config_template() -> Dict[str, Any]:
    """
    Get M3 configuration template with all options.
    
    Returns:
        Configuration template with M3 optimal defaults
    """
    return {
        'audio_graph': {
            'max_voice_embeddings': 20,
            'max_text_embeddings': 10,
            'voice_matching_threshold': 0.6,
            'text_matching_threshold': 0.3
        },
        'voice_processing': {
            'voice_sample_rate': 16000,
            'min_voice_duration': 1.0,
            'confidence_threshold': 0.7
        },
        'memory_processing': {
            'max_context_length': 2000,
            'embedding_model_name': 'all-MiniLM-L6-v2'
        },
        'context_management': {
            'max_context_tokens': 4096,
            'memory_tokens': 2000
        },
        'persistence': {
            'url': 'ws://localhost:8000/rpc',
            'namespace': 'slowcat',
            'database': 'm3_memory'
        }
    }


# Export main classes and functions
__all__ = [
    'M3AudioGraphFactory',
    'M3ProcessorFactory', 
    'M3SystemBuilder',
    'create_m3_system',
    'create_m3_system_with_persistence',
    'get_m3_config_template'
]