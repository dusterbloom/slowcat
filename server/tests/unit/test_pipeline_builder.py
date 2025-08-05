"""
Unit tests for the PipelineBuilder
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from core.pipeline_builder import PipelineBuilder
from core.service_factory import ServiceFactory


class TestPipelineBuilder:
    """Test PipelineBuilder functionality"""
    
    @pytest.fixture
    def mock_service_factory(self):
        """Create a mock ServiceFactory"""
        factory = Mock(spec=ServiceFactory)
        factory.wait_for_ml_modules = AsyncMock()
        factory.wait_for_global_analyzers = AsyncMock()
        factory.create_services_for_language = AsyncMock()
        factory.get_service = AsyncMock()
        factory.registry = Mock()
        factory.registry.get_instance = Mock()
        return factory
    
    @pytest.fixture
    def pipeline_builder(self, mock_service_factory):
        """Create PipelineBuilder with mock factory"""
        return PipelineBuilder(mock_service_factory)
    
    @pytest.fixture
    def mock_webrtc_connection(self):
        """Create a mock WebRTC connection"""
        connection = Mock()
        connection.pc_id = "test-connection"
        return connection
    
    def test_get_language_config(self, pipeline_builder):
        """Test language configuration retrieval"""
        with patch('core.pipeline_builder.config') as mock_config:
            mock_lang_config = Mock()
            mock_lang_config.voice = "test_voice"
            mock_lang_config.whisper_language = "EN"
            mock_lang_config.system_instruction = "Test instruction"
            mock_lang_config.dj_voice = "test_dj_voice"
            mock_lang_config.dj_system_prompt = "Test DJ prompt"
            
            mock_config.get_language_config.return_value = mock_lang_config
            
            result = pipeline_builder._get_language_config("en")
            
            assert result["voice"] == "test_voice"
            assert result["whisper_language"] == "EN"
            assert "Test instruction" in result["system_instruction"]
            assert "Always respond in English only" in result["system_instruction"]
            assert result["dj_voice"] == "test_dj_voice"
            assert result["dj_system_prompt"] == "Test DJ prompt"
            assert result["dj_voice"] == "test_dj_voice"
            assert result["dj_system_prompt"] == "Test DJ prompt"
    
    @pytest.mark.asyncio
    async def test_create_core_services(self, pipeline_builder, mock_service_factory):
        """Test core services creation"""
        mock_services = {
            'stt': Mock(),
            'tts': Mock(), 
            'llm': Mock()
        }
        mock_service_factory.create_services_for_language.return_value = mock_services
        
        result = await pipeline_builder._create_core_services("en", "test-model")
        
        assert result == mock_services
        mock_service_factory.wait_for_ml_modules.assert_called_once()
        mock_service_factory.create_services_for_language.assert_called_once_with("en", "test-model")
    
    @pytest.mark.asyncio
    async def test_setup_processors_memory_enabled(self, pipeline_builder, mock_service_factory):
        """Test processor setup with memory enabled"""
        mock_memory_processor = Mock()
        mock_service_factory.get_service.return_value = mock_memory_processor
        
        with patch('tools.set_memory_processor') as mock_set_memory, \
             patch('processors.MemoryContextInjector') as mock_injector, \
             patch('processors.VideoSamplerProcessor') as mock_video, \
             patch('processors.GreetingFilterProcessor') as mock_greeting, \
             patch('processors.MessageDeduplicator') as mock_dedup, \
             patch('config.config') as mock_config:
            
            # Configure mocks
            mock_config.video.enabled = True
            mock_injector_instance = Mock()
            mock_injector.return_value = mock_injector_instance
            
            result = await pipeline_builder._setup_processors("en")
            
            # Verify memory processor was set up
            mock_set_memory.assert_called_once_with(mock_memory_processor)
            assert result['memory_processor'] == mock_memory_processor
            assert result['memory_injector'] == mock_injector_instance
    
    @pytest.mark.asyncio
    async def test_setup_processors_memory_disabled(self, pipeline_builder, mock_service_factory):
        """Test processor setup with memory disabled"""
        mock_service_factory.get_service.return_value = None
        
        with patch('processors.VideoSamplerProcessor') as mock_video, \
             patch('processors.GreetingFilterProcessor') as mock_greeting, \
             patch('processors.MessageDeduplicator') as mock_dedup, \
             patch('config.config') as mock_config:
            
            mock_config.video.enabled = False
            
            result = await pipeline_builder._setup_processors("en")
            
            assert result['memory_processor'] is None
            assert result['memory_injector'] is None
            assert result['video_sampler'] is None
    
    @pytest.mark.asyncio
    async def test_setup_voice_recognition_processors(self, pipeline_builder):
        """Test voice recognition processor setup"""
        mock_voice_recognition = Mock()
        mock_memory_processor = Mock()
        
        with patch('processors.AudioTeeProcessor') as mock_audio_tee, \
             patch('processors.VADEventBridge') as mock_vad_bridge, \
             patch('processors.SpeakerContextProcessor') as mock_speaker_context, \
             patch('processors.SpeakerNameManager') as mock_name_manager:
            
            # Create mock instances
            mock_audio_tee_instance = Mock()
            mock_vad_bridge_instance = Mock()
            mock_speaker_context_instance = Mock()
            mock_name_manager_instance = Mock()
            
            mock_audio_tee.return_value = mock_audio_tee_instance
            mock_vad_bridge.return_value = mock_vad_bridge_instance
            mock_speaker_context.return_value = mock_speaker_context_instance
            mock_name_manager.return_value = mock_name_manager_instance
            
            result = await pipeline_builder._setup_voice_recognition_processors(
                mock_voice_recognition, mock_memory_processor
            )
            
            # Verify processors were created
            assert result['audio_tee'] == mock_audio_tee_instance
            assert result['vad_bridge'] == mock_vad_bridge_instance
            assert result['speaker_context'] == mock_speaker_context_instance
            assert result['speaker_name_manager'] == mock_name_manager_instance
            assert result['voice_recognition'] == mock_voice_recognition
            
            # Verify registration and callbacks were set up
            mock_audio_tee_instance.register_audio_consumer.assert_called_once()
            mock_vad_bridge_instance.set_callbacks.assert_called_once()
            mock_voice_recognition.set_callbacks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_transport(self, pipeline_builder, mock_service_factory, mock_webrtc_connection):
        """Test transport setup"""
        with patch('pipecat.transports.network.small_webrtc.SmallWebRTCTransport') as mock_transport, \
             patch('config.config') as mock_config, \
             patch('pipecat.vad.webrtc_vad.VADAnalyzer', create=True) as mock_vad_analyzer, \
             patch('pipecat.turn.analyzer.BaseTurnAnalyzer', create=True) as mock_turn_analyzer:

            mock_analyzers = {
                'vad_analyzer': mock_vad_analyzer,
                'turn_analyzer': mock_turn_analyzer
            }
            mock_service_factory.registry.get_instance.return_value = mock_analyzers
            
            mock_config.video.enabled = True
            mock_transport_instance = Mock()
            mock_transport.return_value = mock_transport_instance
            
            result = await pipeline_builder._setup_transport(mock_webrtc_connection)
            
            assert result == mock_transport_instance
            mock_service_factory.wait_for_global_analyzers.assert_called_once()
            mock_transport.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_build_context_with_tools(self, pipeline_builder):
        """Test context building with tools enabled"""
        mock_llm_service = Mock()
        mock_llm_service.__class__.__name__ = "LLMWithToolsService"
        
        lang_config = {"system_instruction": "Test instruction"}
        
        with patch('config.config') as mock_config, \
             patch('tools.get_tools') as mock_get_tools, \
             patch('core.pipeline_builder.OpenAILLMContext') as mock_context:
            
            mock_config.mcp.enabled = True
            mock_tools = Mock()
            mock_tools.standard_tools = [Mock()]
            mock_get_tools.return_value = mock_tools
            
            mock_context_instance = Mock()
            mock_context.return_value = mock_context_instance
            mock_context_aggregator = Mock()
            mock_llm_service.create_context_aggregator.return_value = mock_context_aggregator
            
            context, aggregator = await pipeline_builder._build_context(lang_config, mock_llm_service)
            
            mock_context.assert_called_once_with(
                [{"role": "system", "content": "Test instruction"}],
                tools=mock_tools
            )
            assert aggregator == mock_context_aggregator
    
    @pytest.mark.asyncio
    async def test_build_context_without_tools(self, pipeline_builder):
        """Test context building without tools"""
        mock_llm_service = Mock()
        mock_llm_service.__class__.__name__ = "OpenAILLMService"
        
        lang_config = {"system_instruction": "Test instruction"}
        
        with patch('config.config') as mock_config, \
             patch('core.pipeline_builder.OpenAILLMContext') as mock_context:
            
            mock_config.mcp.enabled = False
            
            mock_context_instance = Mock()
            mock_context.return_value = mock_context_instance
            mock_context_aggregator = Mock()
            mock_llm_service.create_context_aggregator.return_value = mock_context_aggregator
            
            context, aggregator = await pipeline_builder._build_context(lang_config, mock_llm_service)
            
            # Verify context was created without tools
            mock_context.assert_called_once()
            args, kwargs = mock_context.call_args
            from openai import NOT_GIVEN
            assert kwargs.get('tools') is NOT_GIVEN
    
    @pytest.mark.asyncio
    async def test_build_pipeline_components(self, pipeline_builder):
        """Test pipeline components building"""
        mock_transport = Mock()
        mock_transport.input.return_value = Mock()
        mock_transport.output.return_value = Mock()
        
        services = {
            'stt': Mock(),
            'tts': Mock(),
            'llm': Mock()
        }
        
        processors = {
            'video_sampler': Mock(),
            'audio_tee': Mock(),
            'vad_bridge': Mock(),
            'memory_processor': Mock(),
            'speaker_context': Mock(),
            'speaker_name_manager': Mock(),
            'memory_injector': Mock(),
            'message_deduplicator': Mock(),
            'greeting_filter': Mock(),
            'dictation_mode': Mock(),
            'music_mode': Mock(),
            'dj_config_handler': Mock(),
            'audio_player': Mock(),
            'time_executor': Mock()
        }
        
        mock_context_aggregator = Mock()
        mock_context_aggregator.user.return_value = Mock()
        mock_context_aggregator.assistant.return_value = Mock()
        
        with patch('pipecat.processors.frameworks.rtvi.RTVIProcessor') as mock_rtvi:
            mock_rtvi_instance = Mock()
            mock_rtvi.return_value = mock_rtvi_instance
            
            components = await pipeline_builder._build_pipeline_components(
                mock_transport, services, processors, mock_context_aggregator
            )
            
            # Verify all components are included (non-None ones)
            assert len(components) == 12  # All components should be present
            assert mock_rtvi_instance in components
    
    @pytest.mark.asyncio
    async def test_build_pipeline_full_integration(self, pipeline_builder, mock_service_factory, mock_webrtc_connection):
        """Test full pipeline building integration"""
        # Setup comprehensive mocks
        mock_services = {
            'stt': Mock(),
            'tts': Mock(),
            'llm': Mock()
        }
        mock_service_factory.create_services_for_language.return_value = mock_services
        mock_service_factory.get_service.return_value = None  # No memory/voice recognition
        
        mock_analyzers = {
            'vad_analyzer': Mock(),
            'turn_analyzer': Mock()
        }
        mock_service_factory.registry.get_instance.return_value = mock_analyzers
        
        with patch.multiple(
            'processors',
            VideoSamplerProcessor=Mock(),
            GreetingFilterProcessor=Mock(),
            MessageDeduplicator=Mock()
        ), patch.multiple(
            'pipecat.transports.network.small_webrtc',
            SmallWebRTCTransport=Mock()
        ), patch.multiple(
            'pipecat.processors.aggregators.openai_llm_context',
            OpenAILLMContext=Mock()
        ), patch.multiple(
            'pipecat.processors.frameworks.rtvi',
            RTVIProcessor=Mock(),
            RTVIObserver=Mock()
        ), patch.multiple(
            'pipecat.pipeline.task',
            PipelineTask=Mock(),
            PipelineParams=Mock()
        ), patch('pipecat.pipeline.pipeline.Pipeline') as mock_pipeline, \
           patch('config.config') as mock_config:
            
            mock_config.video.enabled = False
            mock_config.mcp.enabled = False
            mock_config.get_language_config.return_value = Mock(
                voice="test_voice",
                whisper_language="EN", 
                system_instruction="Test instruction"
            )
            
            # Mock pipeline and task creation
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            
            mock_task = Mock()
            
            # Test the full build process
            pipeline, task = await pipeline_builder.build_pipeline(
                mock_webrtc_connection, "en", "test-model"
            )
            
            # Verify services were created
            mock_service_factory.create_services_for_language.assert_called_once_with("en", "test-model")
            
            # Verify pipeline was created
            assert pipeline == mock_pipeline_instance


if __name__ == "__main__":
    pytest.main([__file__])