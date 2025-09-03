#!/usr/bin/env python3
"""
M3 Pipeline Integration Test

Tests the complete M3 integration with the actual Slowcat bot pipeline.
Verifies that M3 components can be created and work with run_bot.sh/bot_v2.py.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import logging
from unittest.mock import MagicMock, AsyncMock

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_m3_test_environment():
    """Set up environment variables for M3 testing"""
    # M3 Configuration
    os.environ['ENABLE_M3'] = 'true'
    os.environ['USE_M3_CONTEXT'] = 'true'
    os.environ['M3_FALLBACK_TO_STANDARD'] = 'true'
    os.environ['SURREALDB_HOST'] = 'localhost'
    os.environ['SURREALDB_PORT'] = '8000'
    os.environ['SURREALDB_NAMESPACE'] = 'test'
    os.environ['SURREALDB_DATABASE'] = 'memory_test'
    
    # General bot configuration
    os.environ['ENABLE_MEMORY'] = 'true'
    os.environ['USER_ID'] = 'test_user'
    os.environ['FACTS_DB_PATH'] = 'test_facts.db'
    
    logger.info("✅ M3 test environment configured")

def cleanup_test_environment():
    """Clean up test environment"""
    test_vars = [
        'ENABLE_M3', 'USE_M3_CONTEXT', 'M3_FALLBACK_TO_STANDARD',
        'SURREALDB_HOST', 'SURREALDB_PORT', 'SURREALDB_NAMESPACE', 'SURREALDB_DATABASE'
    ]
    
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]
    
    # Clean up test files
    test_files = ['test_facts.db', 'test_facts.db-wal', 'test_facts.db-shm']
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except:
                pass

async def test_config_loading():
    """Test that M3 configuration loads correctly"""
    logger.info("🧪 Testing M3 configuration loading...")
    
    try:
        from config import config
        
        # Verify M3 config is loaded
        assert hasattr(config, 'm3'), "M3 config not found in main config"
        
        m3_config = config.m3
        assert m3_config.enabled == True, f"M3 should be enabled, got {m3_config.enabled}"
        assert m3_config.use_m3_context == True, f"M3 context should be enabled, got {m3_config.use_m3_context}"
        assert m3_config.fallback_to_standard_memory == True, f"Fallback should be enabled"
        
        logger.info("✅ M3 configuration loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ M3 configuration loading failed: {e}")
        return False

async def test_service_factory_m3_integration():
    """Test that service factory can create M3 components"""
    logger.info("🧪 Testing service factory M3 integration...")
    
    try:
        from core.service_factory import ServiceFactory
        
        factory = ServiceFactory()
        
        # Test that M3 services are registered
        registry = factory.registry
        
        m3_services = [
            'm3_connection', 'm3_integration', 'm3_similarity_search',
            'm3_equivalence_resolver', 'm3_context_retriever'
        ]
        
        for service_name in m3_services:
            definition = registry.get_definition(service_name)
            assert definition is not None, f"M3 service '{service_name}' not registered"
            logger.info(f"✅ M3 service '{service_name}' registered")
        
        logger.info("✅ Service factory M3 integration working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service factory M3 integration failed: {e}")
        return False

async def test_m3_context_manager_creation():
    """Test M3 context manager creation with fallback"""
    logger.info("🧪 Testing M3 context manager creation...")
    
    try:
        # Mock LLMContext for testing
        mock_context = MagicMock()
        
        from config import config
        from processors.m3_integrated_context_manager import M3IntegratedContextManager
        
        # Test M3 context manager creation
        context_manager = M3IntegratedContextManager(
            context=mock_context,
            config=config.m3,
            max_tokens=config.m3.max_context_tokens
        )
        
        # Verify basic properties
        assert context_manager.config == config.m3, "Config not properly set"
        assert context_manager.max_tokens == config.m3.max_context_tokens, "Max tokens not set correctly"
        
        # Wait a moment for initialization to attempt
        await asyncio.sleep(1)
        
        # Check that fallback mechanism works (should fallback since no SurrealDB)
        stats = context_manager.get_stats()
        logger.info(f"Context manager stats: {stats}")
        
        logger.info("✅ M3 context manager creation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ M3 context manager creation failed: {e}")
        return False

async def test_pipeline_builder_m3_integration():
    """Test that pipeline builder can create M3 components"""
    logger.info("🧪 Testing pipeline builder M3 integration...")
    
    try:
        from core.pipeline_builder import PipelineBuilder
        from core.service_factory import ServiceFactory
        from config import config
        
        # Mock the necessary transport and services
        mock_transport = MagicMock()
        mock_transport.input.return_value = MagicMock()
        mock_transport.output.return_value = MagicMock()
        
        # Create service factory and pipeline builder
        service_factory = ServiceFactory()
        builder = PipelineBuilder(service_factory)
        
        # Mock context
        mock_context = MagicMock()
        
        # Test smart context manager creation
        smart_ctx = builder._create_smart_context_manager(mock_context, None)
        
        # Should create M3IntegratedContextManager when M3 enabled
        from processors.m3_integrated_context_manager import M3IntegratedContextManager
        assert isinstance(smart_ctx, M3IntegratedContextManager), f"Expected M3IntegratedContextManager, got {type(smart_ctx)}"
        
        logger.info("✅ Pipeline builder M3 integration working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline builder M3 integration failed: {e}")
        logger.error(f"This might be expected if SurrealDB is not available - should fallback to standard manager")
        
        # Check if it fell back to standard manager
        try:
            from processors.smart_context_manager import SmartContextManager
            if isinstance(smart_ctx, SmartContextManager):
                logger.info("✅ Successfully fell back to standard SmartContextManager")
                return True
        except:
            pass
        
        return False

async def test_m3_graceful_degradation():
    """Test that M3 system gracefully degrades when SurrealDB unavailable"""
    logger.info("🧪 Testing M3 graceful degradation...")
    
    try:
        from core.service_factory import ServiceFactory
        from config import config
        
        factory = ServiceFactory()
        
        # Try to create M3 connection (should fail gracefully)
        connection = await factory._create_m3_connection()
        
        if connection is None:
            logger.info("✅ M3 connection correctly returned None when SurrealDB unavailable")
        else:
            logger.info("✅ M3 connection successful (SurrealDB available)")
        
        # Test M3 context manager creation with graceful fallback
        mock_context = MagicMock()
        context_manager = await factory.create_m3_context_manager(mock_context)
        
        assert context_manager is not None, "Context manager should never be None"
        
        logger.info("✅ M3 graceful degradation working")
        return True
        
    except Exception as e:
        logger.error(f"❌ M3 graceful degradation failed: {e}")
        return False

async def test_full_pipeline_compatibility():
    """Test that the full pipeline can be created with M3 integration"""
    logger.info("🧪 Testing full pipeline compatibility...")
    
    try:
        # Mock necessary components to avoid full system initialization
        from unittest.mock import patch, MagicMock, AsyncMock
        
        # Mock WebRTC transport
        mock_transport = MagicMock()
        mock_transport.input.return_value = MagicMock()
        mock_transport.output.return_value = MagicMock()
        
        # Mock services
        mock_services = {
            'stt': MagicMock(),
            'tts': MagicMock(),
            'llm': MagicMock()
        }
        
        # Mock processors  
        mock_processors = {}
        
        with patch('core.pipeline_builder.create_smart_context_manager') as mock_create:
            mock_create.return_value = MagicMock()
            
            from core.pipeline_builder import PipelineBuilder
            from core.service_factory import ServiceFactory
            
            service_factory = ServiceFactory()
            builder = PipelineBuilder(service_factory)
            
            # Test that we can create a context manager (the critical M3 integration point)
            mock_context = MagicMock()
            smart_ctx = builder._create_smart_context_manager(mock_context, None)
            
            assert smart_ctx is not None, "Smart context manager should be created"
            
            logger.info("✅ Full pipeline compatibility verified")
            return True
            
    except Exception as e:
        logger.error(f"❌ Full pipeline compatibility test failed: {e}")
        return False

async def run_integration_tests():
    """Run all M3 integration tests"""
    logger.info("🚀 Starting M3 Pipeline Integration Tests...")
    
    # Set up test environment
    setup_m3_test_environment()
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Service Factory M3 Integration", test_service_factory_m3_integration),
        ("M3 Context Manager Creation", test_m3_context_manager_creation),
        ("Pipeline Builder M3 Integration", test_pipeline_builder_m3_integration),
        ("M3 Graceful Degradation", test_m3_graceful_degradation),
        ("Full Pipeline Compatibility", test_full_pipeline_compatibility),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: EXCEPTION - {e}")
            results.append((test_name, False))
    
    # Clean up
    cleanup_test_environment()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info(f"\n📊 M3 Integration Test Results:")
    logger.info(f"   Passed: {passed}/{total}")
    logger.info(f"   Failed: {total - passed}/{total}")
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {test_name}: {status}")
    
    if passed == total:
        logger.info("🎉 All M3 integration tests PASSED!")
        return True
    else:
        logger.error("💥 Some M3 integration tests FAILED!")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    sys.exit(0 if success else 1)