#!/usr/bin/env python3
"""
Quick test to verify memory system activation and basic functionality
"""

import os
import asyncio

# CRITICAL: Set environment variable BEFORE importing config
os.environ["USE_STATELESS_MEMORY"] = "true"

from core.service_factory import ServiceFactory
from config import config

async def test_memory_activation():
    """Test if memory system is properly activated"""
    
    print("🧠 Testing Memory System Activation")
    print("=" * 40)
    
    # Check configuration
    print(f"Environment variable USE_STATELESS_MEMORY: {os.getenv('USE_STATELESS_MEMORY')}")
    print(f"Config stateless_memory.enabled: {config.stateless_memory.enabled}")
    print(f"Config stateless_memory.use_enhanced: {getattr(config.stateless_memory, 'use_enhanced', False)}")
    print(f"Config stateless_memory.db_path: {config.stateless_memory.db_path}")
    
    # Initialize service factory
    service_factory = ServiceFactory()
    
    try:
        # Get memory service
        memory_service = service_factory._create_memory_service()
        
        if memory_service is None:
            print("❌ Memory service is None - not activated")
            return False
        
        print(f"✅ Memory service created: {type(memory_service).__name__}")
        print(f"   Class: {memory_service.__class__.__name__}")
        
        # Test basic attributes
        if hasattr(memory_service, 'current_speaker'):
            print(f"   Current speaker: {memory_service.current_speaker}")
        
        if hasattr(memory_service, 'max_context_tokens'):
            print(f"   Max context tokens: {memory_service.max_context_tokens}")
        
        if hasattr(memory_service, 'hot_tier_size'):
            print(f"   Hot tier size: {memory_service.hot_tier_size}")
        
        # Test storage path
        if hasattr(memory_service, 'db_path'):
            print(f"   Database path: {memory_service.db_path}")
            
        # Test basic functionality
        if hasattr(memory_service, '_store_exchange'):
            print("   ✅ Has _store_exchange method")
        
        if hasattr(memory_service, '_get_relevant_memories'):
            print("   ✅ Has _get_relevant_memories method")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating memory service: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_memory_activation())
    exit(0 if success else 1)