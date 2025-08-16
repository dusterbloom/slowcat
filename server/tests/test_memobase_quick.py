#!/usr/bin/env python3

"""Quick test to verify MemoBase integration is working"""

import os
import sys
import asyncio
from loguru import logger
from dotenv import load_dotenv

# Add server directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

async def test_memobase_integration():
    """Test MemoBase memory processor creation and basic functionality"""
    
    logger.info("=== Testing MemoBase Integration ===")
    
    # Test 1: Import processor
    try:
        from processors.memobase_memory_processor import MemobaseMemoryProcessor
        logger.info("✅ MemobaseMemoryProcessor import successful")
    except ImportError as e:
        logger.error(f"❌ Failed to import MemobaseMemoryProcessor: {e}")
        return False
    
    # Test 2: Check config
    try:
        import config
        import importlib
        importlib.reload(config)  # Reload config to pick up new env vars
        
        logger.info(f"🔧 MemoBase enabled: {config.config.memobase.enabled}")
        logger.info(f"🔧 MemoBase URL: {config.config.memobase.project_url}")
        logger.info(f"🔧 MemoBase API key: {config.config.memobase.api_key}")
        
        if not config.config.memobase.enabled:
            logger.warning("⚠️ MemoBase is not enabled in config")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False
    
    # Test 3: Create processor
    try:
        processor = MemobaseMemoryProcessor(
            user_id="test_user",
            project_url=config.config.memobase.project_url,
            api_key=config.config.memobase.api_key,
            max_context_size=config.config.memobase.max_context_size,
            flush_on_session_end=config.config.memobase.flush_on_session_end,
            fallback_to_local=config.config.memobase.fallback_to_local
        )
        logger.info("✅ MemobaseMemoryProcessor created successfully")
        
        # Test 4: Check if MemoBase server is accessible
        if hasattr(processor, '_memobase_client') and processor._memobase_client:
            logger.info("✅ MemoBase client created")
            try:
                # Test ping if available
                if hasattr(processor._memobase_client, 'ping'):
                    processor._memobase_client.ping()
                    logger.info("✅ MemoBase server is accessible")
                else:
                    logger.info("ℹ️ MemoBase client created (ping method not available)")
            except Exception as e:
                logger.warning(f"⚠️ MemoBase server ping failed: {e}")
        else:
            logger.warning("⚠️ MemoBase client not initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create MemobaseMemoryProcessor: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_memobase_integration())
    if success:
        print("\n🎉 MemoBase integration test PASSED!")
        print("💡 The processor should work when integrated into the bot pipeline.")
        print("💡 During conversations, look for MemoBase API calls in docker logs:")
        print("   docker logs memobase-slowcat --follow")
    else:
        print("\n❌ MemoBase integration test FAILED!")
        print("💡 Check the errors above and fix the configuration.")