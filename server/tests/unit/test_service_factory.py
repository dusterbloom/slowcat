"""
Unit tests for the ServiceFactory and ServiceRegistry
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.service_factory import ServiceFactory, ServiceRegistry, ServiceDefinition


class TestServiceRegistry:
    """Test ServiceRegistry functionality"""
    
    def test_register_service(self):
        """Test service registration"""
        registry = ServiceRegistry()
        
        def mock_factory():
            return "test_service"
        
        registry.register("test", mock_factory, ["dep1"], singleton=True, lazy=False)
        
        definition = registry.get_definition("test")
        assert definition is not None
        assert definition.factory == mock_factory
        assert definition.dependencies == ["dep1"]
        assert definition.singleton is True
        assert definition.lazy is False
    
    def test_instance_management(self):
        """Test instance storage and retrieval"""
        registry = ServiceRegistry()
        test_instance = "test_instance"
        
        assert not registry.has_instance("test")
        
        registry.set_instance("test", test_instance)
        assert registry.has_instance("test")
        assert registry.get_instance("test") == test_instance
    
    def test_list_services(self):
        """Test service listing"""
        registry = ServiceRegistry()
        
        registry.register("service1", lambda: None)
        registry.register("service2", lambda: None)
        
        services = registry.list_services()
        assert "service1" in services
        assert "service2" in services
        assert len(services) == 2


class TestServiceFactory:
    """Test ServiceFactory functionality"""
    
    @pytest.fixture
    def factory(self):
        """Create a ServiceFactory instance for testing"""
        return ServiceFactory()
    
    def test_factory_initialization(self, factory):
        """Test that factory initializes with core services"""
        services = factory.registry.list_services()
        
        expected_services = [
            "ml_loader", "global_analyzers", "stt_service", 
            "tts_service", "llm_service", "voice_recognition", "memory_service"
        ]
        
        for service in expected_services:
            assert service in services
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_service(self, factory):
        """Test getting a service that doesn't exist"""
        with pytest.raises(ValueError, match="Service 'nonexistent' not registered"):
            await factory.get_service("nonexistent")
    
    @pytest.mark.asyncio
    async def test_singleton_behavior(self, factory):
        """Test that singleton services return the same instance"""
        # Mock a simple service
        factory.registry.register("test_singleton", lambda: "singleton_instance")
        
        instance1 = await factory.get_service("test_singleton")
        instance2 = await factory.get_service("test_singleton")
        
        assert instance1 == instance2
        assert instance1 is instance2  # Same object reference
    
    @pytest.mark.asyncio
    async def test_dependency_injection(self, factory):
        """Test dependency injection works correctly"""
        # Register a dependency
        factory.registry.register("dependency", lambda: "dep_value")
        
        # Register a service that depends on it
        def dependent_factory(dep):
            return f"service_with_{dep}"
        
        factory.registry.register("dependent", dependent_factory, ["dependency"])
        
        result = await factory.get_service("dependent")
        assert result == "service_with_dep_value"
    
    @pytest.mark.asyncio
    async def test_async_factory_function(self, factory):
        """Test async factory functions work correctly"""
        async def async_factory():
            await asyncio.sleep(0.01)  # Simulate async work
            return "async_result"
        
        factory.registry.register("async_service", async_factory)
        
        result = await factory.get_service("async_service")
        assert result == "async_result"
    
    @pytest.mark.asyncio
    @patch('core.service_factory.importlib.import_module')
    async def test_ml_loader_creation(self, mock_import, factory):
        """Test ML loader service creation"""
        # Mock the ML modules
        mock_whisper = Mock()
        mock_whisper.WhisperSTTServiceMLX = Mock()
        
        mock_stt = Mock() 
        mock_stt.MLXModel = Mock()
        
        mock_tts = Mock()
        mock_tts.KokoroTTSService = Mock()
        
        mock_llm_tools = Mock()
        mock_llm_tools.LLMWithToolsService = Mock()
        
        mock_openai = Mock()
        mock_openai.OpenAILLMService = Mock()
        
        mock_voice = Mock()
        mock_voice.AutoEnrollVoiceRecognition = Mock()
        
        mock_vad = Mock()
        mock_vad.SileroVADAnalyzer = Mock()
        
        mock_turn = Mock()
        mock_turn.LocalSmartTurnAnalyzerV2 = Mock()
        
        # Setup import_module mock to return appropriate modules
        def import_side_effect(module_name):
            if module_name == "services.whisper_stt_with_lock":
                return mock_whisper
            elif module_name == "pipecat.services.whisper.stt":
                return mock_stt
            elif module_name == "kokoro_tts":
                return mock_tts
            elif module_name == "services.llm_with_tools":
                return mock_llm_tools
            elif module_name == "pipecat.services.openai.llm":
                return mock_openai
            elif module_name == "voice_recognition":
                return mock_voice
            elif module_name == "pipecat.audio.vad.silero":
                return mock_vad
            elif module_name == "pipecat.audio.turn.smart_turn.local_smart_turn_v2":
                return mock_turn
            else:
                raise ImportError(f"No module named '{module_name}'")
        
        mock_import.side_effect = import_side_effect
        
        # Test ML loader creation
        ml_modules = await factory.get_service("ml_loader")
        
        assert 'WhisperSTTServiceMLX' in ml_modules
        assert 'MLXModel' in ml_modules
        assert 'KokoroTTSService' in ml_modules
        assert 'LLMWithToolsService' in ml_modules
        assert 'OpenAILLMService' in ml_modules
        assert 'AutoEnrollVoiceRecognition' in ml_modules
        assert 'SileroVADAnalyzer' in ml_modules
        assert 'LocalSmartTurnAnalyzerV2' in ml_modules
    
    @pytest.mark.asyncio
    async def test_memory_service_creation_disabled(self, factory):
        """Test memory service returns None when disabled"""
        with patch('config.config.memory.enabled', False):
            memory_service = await factory.get_service("memory_service")
            assert memory_service is None
    
    @pytest.mark.asyncio
    async def test_memory_service_creation_enabled(self, factory):
        """Test memory service creation when enabled"""
        with patch('config.config.memory.enabled', True), \
             patch('processors.LocalMemoryProcessor') as mock_processor:
            
            mock_instance = Mock()
            mock_processor.return_value = mock_instance
            
            memory_service = await factory.get_service("memory_service")
            assert memory_service is mock_instance
            
            # Verify it was called with correct parameters
            mock_processor.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_language_specific_service_creation(self, factory):
        """Test creating services for specific languages"""
        # Mock ML modules
        factory.registry.set_instance("ml_loader", {
            'WhisperSTTServiceMLX': Mock(),
            'MLXModel': Mock(),
            'KokoroTTSService': Mock(),
            'LLMWithToolsService': Mock(),
            'OpenAILLMService': Mock()
        })
        
        with patch.object(factory, '_create_stt_service') as mock_stt, \
             patch.object(factory, '_create_tts_service') as mock_tts, \
             patch.object(factory, '_create_llm_service') as mock_llm:
            
            mock_stt.return_value = "stt_service"
            mock_tts.return_value = "tts_service" 
            mock_llm.return_value = "llm_service"
            
            services = await factory.create_services_for_language("es", "custom-model")
            
            assert services['stt'] == "stt_service"
            assert services['tts'] == "tts_service"
            assert services['llm'] == "llm_service"
            
            # Verify methods were called with correct parameters
            mock_stt.assert_called_once()
            mock_tts.assert_called_once()
            mock_llm.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])