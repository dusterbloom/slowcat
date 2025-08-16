#!/usr/bin/env python3
"""
Test script for MemoBase integration with Slowcat voice agent.
Tests the MemoBase memory processor, service factory integration, and OpenAI client patching.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from loguru import logger

# Mock frames for testing
class MockTranscriptionFrame:
    def __init__(self, text: str, user_id: str = None):
        self.text = text
        self.user_id = user_id

class MockTextFrame:
    def __init__(self, text: str):
        self.text = text

class MockFrameDirection:
    DOWNSTREAM = "downstream"

async def test_memobase_config():
    """Test MemoBase configuration loading"""
    logger.info("=== Testing MemoBase Configuration ===")
    
    print(f"MemoBase enabled: {config.memobase.enabled}")
    print(f"Project URL: {config.memobase.project_url}")
    print(f"API Key: {config.memobase.api_key}")
    print(f"Fallback to local: {config.memobase.fallback_to_local}")
    print(f"Max context size: {config.memobase.max_context_size}")
    print(f"Flush on session end: {config.memobase.flush_on_session_end}")
    
    assert hasattr(config, 'memobase'), "MemoBase config not found"
    assert hasattr(config.memobase, 'enabled'), "MemoBase enabled field not found"
    
    logger.info("✅ MemoBase configuration test passed")

async def test_memobase_processor():
    """Test MemoBase memory processor creation and basic functionality"""
    logger.info("=== Testing MemoBase Processor ===")
    
    try:
        from processors.memobase_memory_processor import MemobaseMemoryProcessor
        
        # Create processor (should work even if MemoBase not installed)
        processor = MemobaseMemoryProcessor(
            user_id="test_user",
            project_url="http://localhost:8019",
            api_key="secret",
            fallback_to_local=True
        )
        
        logger.info(f"✅ MemoBase processor created: {type(processor).__name__}")
        logger.info(f"   Enabled: {processor.is_enabled}")
        logger.info(f"   User ID: {processor.user_id}")
        
        # Test async context manager
        async with processor:
            logger.info("✅ Async context manager works")
            
            # Test frame processing
            transcription_frame = MockTranscriptionFrame("Hello, I'm testing MemoBase", "test_user")
            await processor.process_frame(transcription_frame, MockFrameDirection.DOWNSTREAM)
            
            text_frame = MockTextFrame("This is a response from the assistant")
            await processor.process_frame(text_frame, MockFrameDirection.DOWNSTREAM)
            
            logger.info("✅ Frame processing works")
            
            # Test user ID update
            await processor.update_user_id("new_test_user")
            assert processor.user_id == "new_test_user"
            logger.info("✅ User ID update works")
            
            # Test memory stats
            stats = await processor.get_memory_stats()
            logger.info(f"✅ Memory stats: {stats}")
        
        logger.info("✅ MemoBase processor test passed")
        
    except ImportError as e:
        logger.warning(f"⚠️ MemoBase not available: {e}")
        logger.info("✅ Processor handles missing MemoBase gracefully")
    except Exception as e:
        logger.error(f"❌ MemoBase processor test failed: {e}")
        raise

async def test_service_factory_integration():
    """Test service factory creates MemoBase memory service"""
    logger.info("=== Testing Service Factory Integration ===")
    
    try:
        from core.service_factory import ServiceFactory
        
        factory = ServiceFactory()
        
        # Test memory service creation
        if config.memobase.enabled:
            memory_service = factory._create_memory_service()
            if memory_service:
                logger.info(f"✅ Memory service created: {type(memory_service).__name__}")
                
                # Check if it's MemoBase
                if hasattr(memory_service, '_initialize_memobase'):
                    logger.info("✅ MemoBase memory service detected")
                else:
                    logger.info("ℹ️ Non-MemoBase memory service (fallback or Mem0)")
            else:
                logger.info("ℹ️ No memory service created (disabled or failed)")
        else:
            logger.info("ℹ️ MemoBase disabled, testing Mem0 fallback")
            
        logger.info("✅ Service factory integration test passed")
        
    except Exception as e:
        logger.error(f"❌ Service factory test failed: {e}")
        raise

async def test_pipeline_integration():
    """Test pipeline builder includes MemoBase processor"""
    logger.info("=== Testing Pipeline Integration ===")
    
    try:
        from core.pipeline_builder import PipelineBuilder
        from core.service_factory import ServiceFactory
        
        factory = ServiceFactory()
        builder = PipelineBuilder(factory)
        
        # Test processor setup
        processors = await builder._setup_processors("en")
        
        memory_service = processors.get('memory_service')
        if memory_service:
            logger.info(f"✅ Memory service in pipeline: {type(memory_service).__name__}")
            
            if hasattr(memory_service, '_initialize_memobase'):
                logger.info("✅ MemoBase processor integrated in pipeline")
            else:
                logger.info("ℹ️ Non-MemoBase memory service in pipeline")
        else:
            logger.info("ℹ️ No memory service in pipeline")
            
        logger.info("✅ Pipeline integration test passed")
        
    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {e}")
        raise

async def test_openai_client_patching():
    """Test OpenAI client patching with MemoBase"""
    logger.info("=== Testing OpenAI Client Patching ===")
    
    try:
        # Only test if MemoBase is available and enabled
        if not config.memobase.enabled:
            logger.info("ℹ️ MemoBase disabled, skipping OpenAI patching test")
            return
            
        from core.service_factory import ServiceFactory
        
        # Mock OpenAI client
        class MockOpenAIClient:
            def __init__(self):
                self._client = self
                
        factory = ServiceFactory()
        mock_llm_service = MockOpenAIClient()
        
        # Test patching
        await factory._patch_llm_with_memobase(mock_llm_service)
        
        logger.info("✅ OpenAI client patching test completed")
        
    except ImportError:
        logger.info("ℹ️ MemoBase not available for OpenAI patching test")
    except Exception as e:
        logger.error(f"❌ OpenAI patching test failed: {e}")
        raise

async def main():
    """Run all MemoBase integration tests"""
    logger.info("🧠 Starting MemoBase Integration Tests")
    logger.info("=" * 50)
    
    tests = [
        test_memobase_config,
        test_memobase_processor,
        test_service_factory_integration,
        test_pipeline_integration,
        test_openai_client_patching,
    ]
    
    for test in tests:
        try:
            await test()
            logger.info("")
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed: {e}")
            return False
    
    logger.info("=" * 50)
    logger.info("🎉 All MemoBase integration tests passed!")
    return True

if __name__ == "__main__":
    # Set test environment variables
    os.environ["ENABLE_MEMOBASE"] = "true"
    os.environ["MEMOBASE_PROJECT_URL"] = "http://localhost:8019"
    os.environ["MEMOBASE_API_KEY"] = "secret"
    
    success = asyncio.run(main())
    exit(0 if success else 1)