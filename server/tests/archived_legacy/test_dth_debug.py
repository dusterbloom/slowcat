#!/usr/bin/env python3
"""
Debug DTH initialization and memory retrieval
"""

import asyncio
import os
import sys
from pathlib import Path

# Add server path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

async def test_dth_initialization():
    """Test DTH and SurrealDB connectivity step by step"""
    
    print("🧪 DTH Debug Test")
    print("=" * 50)
    
    dth_candidates = []
    
    # 1. Check environment variables
    print("📋 Environment Configuration:")
    print(f"   ENABLE_DTH: {os.getenv('ENABLE_DTH', 'false')}")
    print(f"   USE_SURREALDB: {os.getenv('USE_SURREALDB', 'false')}")
    print(f"   SURREALDB_URL: {os.getenv('SURREALDB_URL', 'not set')}")
    print(f"   DSPY_OPTIMIZATION_ENABLED: {os.getenv('DSPY_OPTIMIZATION_ENABLED', 'false')}")
    print()
    
    # 2. Test SurrealDB connection
    try:
        from memory.surreal_memory import SurrealMemory
        print("✅ SurrealMemory import successful")
        
        surreal_mem = SurrealMemory()
        try:
            await surreal_mem.connect()
            print(f"📡 SurrealDB Connection: ✅ Connected")
            
            # Test basic query
            result = await surreal_mem.search_tape("test query", limit=1)
            print(f"🔍 Basic search test: {len(result) if result else 0} results")
        except Exception as conn_e:
            print(f"📡 SurrealDB Connection: ❌ Failed - {conn_e}")
            return False
        
    except Exception as e:
        print(f"❌ SurrealMemory failed: {e}")
        return False
    
    # 3. Test DTH import and initialization
    try:
        from memory.dynamic_tape_head import DynamicTapeHead
        from memory import create_smart_memory_system
        print("✅ DynamicTapeHead import successful")
        
        # Create memory system
        memory_system = create_smart_memory_system()
        print("✅ Smart memory system created")
        
        # Initialize DTH
        tape_head = DynamicTapeHead(memory_system)
        print("✅ DTH initialized successfully")
        
        # Test memory seek
        dth_bundle = await tape_head.seek(
            query="test query", 
            budget=1000,
            speaker_id="test_user"
        )
        
        verbatim_count = len(dth_bundle.verbatim) if hasattr(dth_bundle, 'verbatim') and dth_bundle.verbatim else 0
        shadows_count = len(dth_bundle.shadows) if hasattr(dth_bundle, 'shadows') and dth_bundle.shadows else 0
        recents_count = len(dth_bundle.recents) if hasattr(dth_bundle, 'recents') and dth_bundle.recents else 0
        
        print(f"🎯 DTH Seek Test Results:")
        print(f"   Verbatim: {verbatim_count}")
        print(f"   Shadows: {shadows_count}")  
        print(f"   Recents: {recents_count}")
        
        # Prepare candidates for DSPy (same format as SmartContextManager)
        dth_candidates = []
        if hasattr(dth_bundle, 'verbatim') and dth_bundle.verbatim:
            for item in dth_bundle.verbatim:
                dth_candidates.append(str(item.content))
        
        if hasattr(dth_bundle, 'shadows') and dth_bundle.shadows:
            for item in dth_bundle.shadows:
                dth_candidates.append(str(item.content))
        
        if hasattr(dth_bundle, 'recents') and dth_bundle.recents:
            for item in dth_bundle.recents:
                dth_candidates.append(str(item.content))
        
        print(f"   📚 Total candidates for DSPy: {len(dth_candidates)}")
        
    except Exception as e:
        print(f"❌ DTH initialization failed: {e}")
        import traceback
        traceback.print_exc()
        dth_candidates = []
    
    # 4. Test DSPy integration with DTH candidates
    try:
        if not dth_candidates:
            print("⚠️ No DTH candidates available, skipping DSPy test")
            return
            
        from slowcat_dspy import DSPY_AVAILABLE, create_unified_memory_optimizer
        print(f"🧠 DSPy Available: {'✅' if DSPY_AVAILABLE else '❌'}")
        
        if DSPY_AVAILABLE and dth_candidates:
            print("\n🚀 TESTING COMPLETE DTH + DSPy INTEGRATION")
            print("=" * 50)
            
            optimizer = create_unified_memory_optimizer()
            print("✅ DSPy optimizer created successfully")
            
            # Test DSPy optimization with real DTH candidates
            result = optimizer(
                query="test query about memory",
                dth_candidates=dth_candidates,
                target_tokens=2800,
                mode="chat"
            )
            
            print(f"✨ DSPy Optimization Results:")
            print(f"   📝 Selected Memory: {len(result['selected_memory'])} chars")
            print(f"   🎯 Token Efficiency: {result['token_efficiency']:.2f}")
            print(f"   💭 Reasoning: {result['selection_reasoning'][:100]}...")
            print(f"   📊 Optimization Stats: {result['optimization_stats']}")
            
            # Performance summary
            perf = optimizer.get_performance_summary()
            print(f"\n📊 DSPy Performance Summary:")
            print(f"   Total Optimizations: {perf['total_optimizations']}")
            print(f"   Avg Token Efficiency: {perf['avg_token_efficiency']:.3f}")
            print(f"   DSPy Available: {perf['dspy_available']}")
            
            print(f"\n🎉 COMPLETE INTEGRATION TEST SUCCESSFUL!")
            print(f"   ✅ DTH retrieved {len(dth_candidates)} memory candidates")
            print(f"   ✅ DSPy optimized selection with {result['token_efficiency']:.2f} efficiency")
            print(f"   ✅ Ready for production use in SmartContextManager!")
        
    except Exception as e:
        print(f"❌ DSPy integration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dth_initialization())