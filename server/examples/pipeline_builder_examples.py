"""
Examples demonstrating PipelineBuilder usage patterns
"""

import asyncio
from unittest.mock import Mock
from core.pipeline_builder import PipelineBuilder
from core.service_factory import service_factory
from loguru import logger


class MockWebRTCConnection:
    """Mock WebRTC connection for examples"""
    
    def __init__(self, connection_id="example-connection"):
        self.pc_id = connection_id
        self.closed = False
    
    def get_answer(self):
        return {
            "pc_id": self.pc_id,
            "sdp": "mock-sdp-offer",
            "type": "answer"
        }
    
    def close(self):
        self.closed = True


async def basic_pipeline_building():
    """Basic pipeline building example"""
    print("=== Basic Pipeline Building ===")
    
    # Create mock connection
    connection = MockWebRTCConnection()
    
    # Create pipeline builder
    builder = PipelineBuilder(service_factory)
    
    try:
        # Build pipeline for English
        pipeline, task = await builder.build_pipeline(
            webrtc_connection=connection,
            language="en"
        )
        
        print(f"Pipeline created with {len(pipeline._processors)} processors")
        print(f"Task created: {type(task).__name__}")
        
        # Inspect pipeline components
        processor_types = [type(p).__name__ for p in pipeline._processors]
        print(f"Pipeline processors: {processor_types}")
        
    except Exception as e:
        logger.warning(f"Pipeline building failed (expected in examples): {e}")


async def language_specific_pipelines():
    """Build pipelines for different languages"""
    print("\n=== Language-Specific Pipelines ===")
    
    languages = ["en", "es", "fr", "ja"]
    
    for lang in languages:
        try:
            connection = MockWebRTCConnection(f"{lang}-connection")
            builder = PipelineBuilder(service_factory)
            
            # Get language config to show differences
            lang_config = builder._get_language_config(lang)
            print(f"\n{lang.upper()} Configuration:")
            print(f"  Voice: {lang_config['voice']}")
            print(f"  Whisper Language: {lang_config['whisper_language']}")
            print(f"  System prompt preview: {lang_config['system_instruction'][:100]}...")
            
        except Exception as e:
            logger.warning(f"Language {lang} config failed: {e}")


async def custom_pipeline_components():
    """Example of customizing pipeline components"""
    print("\n=== Custom Pipeline Components ===")
    
    class CustomPipelineBuilder(PipelineBuilder):
        """Custom pipeline builder with additional components"""
        
        async def _build_pipeline_components(self, transport, services, processors, context_aggregator):
            # Get standard components
            components = await super()._build_pipeline_components(
                transport, services, processors, context_aggregator
            )
            
            # Add custom processor
            custom_processor = CustomDebugProcessor()
            
            # Insert before TTS (find TTS position)
            tts_index = None
            for i, comp in enumerate(components):
                if hasattr(comp, '__class__') and 'TTS' in comp.__class__.__name__:
                    tts_index = i
                    break
            
            if tts_index:
                components.insert(tts_index, custom_processor)
                print(f"Added custom processor at position {tts_index}")
            
            return components
    
    # Use custom builder
    connection = MockWebRTCConnection("custom-connection")
    custom_builder = CustomPipelineBuilder(service_factory)
    
    try:
        pipeline, task = await custom_builder.build_pipeline(connection, "en")
        print(f"Custom pipeline has {len(pipeline._processors)} processors")
    except Exception as e:
        logger.warning(f"Custom pipeline failed: {e}")


class CustomDebugProcessor:
    """Example custom processor for debugging"""
    
    def __init__(self):
        self.processed_frames = 0
    
    async def process_frame(self, frame):
        self.processed_frames += 1
        if self.processed_frames % 100 == 0:
            print(f"Debug: Processed {self.processed_frames} frames")
        return frame


async def processor_configuration_examples():
    """Examples of processor configuration"""
    print("\n=== Processor Configuration ===")
    
    builder = PipelineBuilder(service_factory)
    
    # Mock the heavy dependencies to show configuration
    try:
        # Show memory processor configuration
        memory_enabled = True  # config.memory.enabled
        if memory_enabled:
            print("Memory Configuration:")
            print(f"  Data dir: data/memory")
            print(f"  Max history: 200")
            print(f"  Context items: 10")
        
        # Show voice recognition configuration  
        vr_enabled = True  # config.voice_recognition.enabled
        if vr_enabled:
            print("\nVoice Recognition Configuration:")
            print(f"  Profile dir: data/speaker_profiles")
            print(f"  Confidence threshold: 0.70")
            print(f"  Min utterance duration: 1.0s")
        
        # Show video configuration
        video_enabled = False  # config.video.enabled
        print(f"\nVideo Processing: {'Enabled' if video_enabled else 'Disabled'}")
        
    except Exception as e:
        logger.warning(f"Configuration example failed: {e}")


async def pipeline_lifecycle_management():
    """Example of pipeline lifecycle management"""
    print("\n=== Pipeline Lifecycle Management ===")
    
    class PipelineManager:
        """Manages multiple pipeline instances"""
        
        def __init__(self):
            self.pipelines = {}
            self.builder = PipelineBuilder(service_factory)
        
        async def create_pipeline(self, connection_id, language="en", llm_model=None):
            """Create and store a pipeline"""
            connection = MockWebRTCConnection(connection_id)
            
            pipeline, task = await self.builder.build_pipeline(
                connection, language, llm_model
            )
            
            self.pipelines[connection_id] = {
                'pipeline': pipeline,
                'task': task,
                'connection': connection,
                'language': language,
                'created_at': asyncio.get_event_loop().time()
            }
            
            print(f"Created pipeline for {connection_id} ({language})")
            return pipeline, task
        
        async def cleanup_pipeline(self, connection_id):
            """Clean up a specific pipeline"""
            if connection_id in self.pipelines:
                pipeline_info = self.pipelines.pop(connection_id)
                pipeline_info['connection'].close()
                print(f"Cleaned up pipeline for {connection_id}")
        
        async def cleanup_all(self):
            """Clean up all pipelines"""
            for connection_id in list(self.pipelines.keys()):
                await self.cleanup_pipeline(connection_id)
            print("All pipelines cleaned up")
        
        def get_stats(self):
            """Get pipeline statistics"""
            return {
                'active_pipelines': len(self.pipelines),
                'languages': list(set(p['language'] for p in self.pipelines.values())),
                'connections': list(self.pipelines.keys())
            }
    
    # Use pipeline manager
    manager = PipelineManager()
    
    try:
        # Create multiple pipelines
        await manager.create_pipeline("user1", "en")
        await manager.create_pipeline("user2", "es", "custom-model")
        await manager.create_pipeline("user3", "fr")
        
        # Show stats
        stats = manager.get_stats()
        print(f"Pipeline stats: {stats}")
        
        # Cleanup
        await manager.cleanup_pipeline("user2")
        await manager.cleanup_all()
        
    except Exception as e:
        logger.warning(f"Pipeline management failed: {e}")


async def transport_configuration_examples():
    """Examples of transport configuration"""
    print("\n=== Transport Configuration ===")
    
    builder = PipelineBuilder(service_factory)
    
    # Mock transport setup to show configuration
    print("WebRTC Transport Configuration:")
    print("  Audio input: Enabled")
    print("  Audio output: Enabled") 
    print("  Video input: Based on config.video.enabled")
    print("  VAD analyzer: Silero VAD")
    print("  Turn analyzer: Smart Turn v2")
    print("  ICE servers: Google STUN server")
    
    # Show analyzer configuration
    try:
        await service_factory.wait_for_global_analyzers()
        analyzers = service_factory.registry.get_instance("global_analyzers")
        
        if analyzers:
            print("\nAnalyzer Status:")
            print(f"  VAD analyzer ready: {analyzers.get('vad_analyzer') is not None}")
            print(f"  Turn analyzer ready: {analyzers.get('turn_analyzer') is not None}")
        
    except Exception as e:
        logger.warning(f"Analyzer status check failed: {e}")


async def context_and_tools_examples():
    """Examples of context and tools configuration"""
    print("\n=== Context and Tools Configuration ===")
    
    builder = PipelineBuilder(service_factory)
    
    # Mock LLM service to show context building
    class MockLLMService:
        def __init__(self, has_tools=False):
            self.has_tools = has_tools
            if has_tools:
                self.__class__.__name__ = "LLMWithToolsService"
            else:
                self.__class__.__name__ = "OpenAILLMService"
        
        def create_context_aggregator(self, context):
            return Mock()
    
    # Test with tools enabled
    llm_with_tools = MockLLMService(has_tools=True)
    lang_config = {"system_instruction": "You are a helpful assistant."}
    
    try:
        context, aggregator = await builder._build_context(lang_config, llm_with_tools)
        print("Tools enabled context created")
    except Exception as e:
        logger.warning(f"Tools context creation failed: {e}")
    
    # Test without tools
    llm_standard = MockLLMService(has_tools=False)
    
    try:
        context, aggregator = await builder._build_context(lang_config, llm_standard)
        print("Standard context created")
    except Exception as e:
        logger.warning(f"Standard context creation failed: {e}")


async def error_handling_and_recovery():
    """Examples of error handling in pipeline building"""
    print("\n=== Error Handling and Recovery ===")
    
    builder = PipelineBuilder(service_factory)
    
    # Test with invalid connection
    class BrokenConnection:
        def __init__(self):
            self.pc_id = "broken"
        
        def get_answer(self):
            raise RuntimeError("Connection failed")
    
    broken_connection = BrokenConnection()
    
    try:
        await builder.build_pipeline(broken_connection, "en")
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}: {e}")
    
    # Test with unsupported language
    try:
        connection = MockWebRTCConnection()
        await builder.build_pipeline(connection, "unsupported_lang")
    except Exception as e:
        print(f"Language error: {type(e).__name__}: {e}")
    
    # Test recovery strategies
    print("Recovery strategies:")
    print("1. Fallback to default language")
    print("2. Use cached services when available")
    print("3. Graceful degradation (disable optional components)")


async def main():
    """Run all pipeline builder examples"""
    try:
        await basic_pipeline_building()
        await language_specific_pipelines()
        await custom_pipeline_components()
        await processor_configuration_examples()
        await pipeline_lifecycle_management()
        await transport_configuration_examples()
        await context_and_tools_examples()
        await error_handling_and_recovery()
        
        print("\n✅ All pipeline builder examples completed!")
        
    except Exception as e:
        logger.error(f"Pipeline examples failed: {e}")
        raise


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())