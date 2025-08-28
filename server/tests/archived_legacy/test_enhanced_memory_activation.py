#!/usr/bin/env python3
"""
Test enhanced memory system specifically
"""

import os
import asyncio

# CRITICAL: Set environment variables BEFORE importing config
os.environ["USE_STATELESS_MEMORY"] = "true"
os.environ["USE_ENHANCED_MEMORY"] = "true"

from core.service_factory import ServiceFactory
from config import config

async def test_enhanced_memory():
    """Test enhanced memory system activation"""
    
    print("🧠 Enhanced Memory System Test")
    print("=" * 40)
    
    # Check configuration
    print(f"USE_STATELESS_MEMORY: {os.getenv('USE_STATELESS_MEMORY')}")
    print(f"USE_ENHANCED_MEMORY: {os.getenv('USE_ENHANCED_MEMORY')}")
    print(f"Config stateless_memory.enabled: {config.stateless_memory.enabled}")
    print(f"Config stateless_memory.use_enhanced: {getattr(config.stateless_memory, 'use_enhanced', False)}")
    
    # Create service factory
    service_factory = ServiceFactory()
    
    try:
        # Get memory service
        memory_service = service_factory._create_memory_service()
        
        if memory_service is None:
            print("❌ Memory service is None")
            return False
        
        memory_class = memory_service.__class__.__name__
        print(f"✅ Memory service: {memory_class}")
        
        if memory_class == "EnhancedStatelessMemoryProcessor":
            print("🎉 Enhanced memory system is ACTIVE!")
            
            # Test enhanced features
            if hasattr(memory_service, 'hot_tier'):
                print(f"   Hot tier: {len(memory_service.hot_tier)} items")
            if hasattr(memory_service, 'hot_tier_size'):
                print(f"   Hot tier size: {memory_service.hot_tier_size}")
            if hasattr(memory_service, 'warm_tier_size'):
                print(f"   Warm tier size: {memory_service.warm_tier_size}")
            if hasattr(memory_service, 'cold_tier_size'):
                print(f"   Cold tier size: {memory_service.cold_tier_size}")
                
            return True
            
        elif memory_class == "StatelessMemoryProcessor":
            print("⚠️ Standard stateless memory (not enhanced)")
            return True
            
        else:
            print(f"⚠️ Traditional memory system: {memory_class}")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_memory())
    exit(0 if success else 1)