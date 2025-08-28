#!/usr/bin/env python3
"""
Test script for consolidated SurrealDB memory system + DSPy foundation

This test validates:
1. Legacy memory processors have been removed
2. SurrealDB memory system is working
3. Smart context manager integration
4. DSPy foundation is properly set up
5. Query router is functional

Run: python test_consolidation.py
"""

import sys
import os
import asyncio
from pathlib import Path
from loguru import logger

# Add server to path
sys.path.insert(0, str(Path(__file__).parent))

def test_legacy_removal():
    """Test that legacy memory processors are removed"""
    logger.info("🧹 Testing legacy processor removal...")
    
    processors_dir = Path(__file__).parent / "processors"
    
    # These should be gone
    legacy_files = [
        "local_memory.py",
        "stateless_memory.py", 
        "enhanced_stateless_memory.py",
        "memory_context_aggregator.py",
        "memory_context_injector.py"
    ]
    
    for file in legacy_files:
        file_path = processors_dir / file
        if file_path.exists():
            logger.error(f"❌ Legacy file still exists: {file}")
            return False
        else:
            logger.info(f"✅ Legacy file removed: {file}")
    
    # Check processors __init__.py doesn't import legacy classes
    init_file = processors_dir / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        legacy_imports = [
            "LocalMemoryProcessor",
            "StatelessMemoryProcessor", 
            "MemoryContextAggregator",
            "MemoryContextInjector"
        ]
        
        for legacy_import in legacy_imports:
            if legacy_import in content:
                logger.error(f"❌ Legacy import still in __init__.py: {legacy_import}")
                return False
    
    logger.info("✅ Legacy processor removal test passed")
    return True


def test_surrealdb_memory_system():
    """Test SurrealDB memory system availability"""
    logger.info("🗃️ Testing SurrealDB memory system...")
    
    try:
        from memory import create_smart_memory_system
        logger.info("✅ Smart memory system import successful")
        
        # Test creation (should default to SurrealDB)
        memory_system = create_smart_memory_system()
        logger.info("✅ SurrealDB memory system created")
        
        # Check if it's the SurrealDB adapter
        if hasattr(memory_system, 'surreal_memory'):
            logger.info("✅ SurrealDB adapter detected")
        else:
            logger.warning("⚠️ Not using SurrealDB adapter (fallback to SQLite?)")
        
        # Test query router
        if hasattr(memory_system, 'query_router') and memory_system.query_router:
            logger.info("✅ SurrealDB query router available")
        else:
            logger.warning("⚠️ SurrealDB query router not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SurrealDB memory system test failed: {e}")
        return False


def test_smart_context_manager():
    """Test SmartContextManager integration"""
    logger.info("🧠 Testing SmartContextManager...")
    
    try:
        from processors.smart_context_manager import SmartContextManager
        logger.info("✅ SmartContextManager import successful")
        
        # Test that it can create memory system
        from memory import create_smart_memory_system
        memory_system = create_smart_memory_system()
        
        # Mock context object
        class MockContext:
            def __init__(self):
                self.messages = []
            def set_messages(self, messages):
                self.messages = messages
        
        context = MockContext()
        manager = SmartContextManager(
            context=context,
            facts_db_path="data/test_facts.db",
            max_tokens=4096
        )
        
        logger.info("✅ SmartContextManager created successfully")
        
        # Test async compatibility
        if hasattr(manager, '_maybe_await'):
            logger.info("✅ SmartContextManager has SurrealDB async compatibility")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SmartContextManager test failed: {e}")
        return False


def test_dspy_foundation():
    """Test DSPy foundation setup"""
    logger.info("🔬 Testing DSPy foundation...")
    
    try:
        # Test DSPy module import
        import dspy as dspy_module
        logger.info("✅ DSPy module available")
        
        # Test Slowcat DSPy integration
        from dspy import (
            DSPY_AVAILABLE, 
            OPTIMIZATION_ENABLED,
            MODEL_PATH,
            TRAINING_DAYS
        )
        
        logger.info(f"✅ DSPy integration imported - Available: {DSPY_AVAILABLE}")
        logger.info(f"   Optimization enabled: {OPTIMIZATION_ENABLED}")
        logger.info(f"   Training window: {TRAINING_DAYS} days")
        
        # Test optimizers import
        try:
            from dspy.surreal_optimizers import (
                SurrealContextOptimizer,
                SurrealResponseGenerator,
                create_surreal_context_optimizer
            )
            logger.info("✅ SurrealDB optimizers available")
        except ImportError as e:
            logger.warning(f"⚠️ SurrealDB optimizers not fully available: {e}")
        
        # Test metrics import
        try:
            from dspy.metrics.surreal_metrics import (
                context_relevance_metric,
                response_quality_metric
            )
            logger.info("✅ SurrealDB metrics available")
        except ImportError as e:
            logger.warning(f"⚠️ SurrealDB metrics not available: {e}")
        
        return True
        
    except ImportError as e:
        logger.info(f"📦 DSPy not installed: {e}")
        logger.info("   Install with: pip install dspy-ai")
        return True  # Not a failure - DSPy is optional
    except Exception as e:
        logger.error(f"❌ DSPy foundation test failed: {e}")
        return False


def test_query_router():
    """Test SurrealDB query router"""
    logger.info("🔀 Testing SurrealDB query router...")
    
    try:
        from memory.surreal_query_router import SurrealQueryRouter, create_surreal_query_router
        logger.info("✅ SurrealDB query router import successful")
        
        # Mock SurrealDB memory for testing
        class MockSurrealMemory:
            async def search_facts(self, query, limit=10):
                return []
            async def search_tape(self, query, limit=10):
                return []
        
        mock_memory = MockSurrealMemory()
        router = create_surreal_query_router(mock_memory)
        
        logger.info("✅ SurrealDB query router created")
        
        # Test router has expected methods
        expected_methods = ['route_query', 'get_performance_stats']
        for method in expected_methods:
            if hasattr(router, method):
                logger.info(f"✅ Router has method: {method}")
            else:
                logger.error(f"❌ Router missing method: {method}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Query router test failed: {e}")
        return False


async def test_integration():
    """Test integration between components"""
    logger.info("🔗 Testing component integration...")
    
    try:
        # Create memory system
        from memory import create_smart_memory_system
        memory_system = create_smart_memory_system()
        
        # Test that adapter has router
        if hasattr(memory_system, 'query_router') and memory_system.query_router:
            logger.info("✅ Memory system has query router")
            
            # Test a simple query routing
            response = await memory_system.process_query("What's my name?", {"speaker_id": "test_user"})
            
            if hasattr(response, 'results') and hasattr(response, 'strategy_used'):
                logger.info(f"✅ Query routing successful - strategy: {response.strategy_used}")
            else:
                logger.warning("⚠️ Query response format may need adjustment")
        
        # Test SmartContextManager with memory system
        from processors.smart_context_manager import SmartContextManager
        
        class MockContext:
            def set_messages(self, messages): pass
        
        context_manager = SmartContextManager(
            context=MockContext(),
            facts_db_path="data/test_facts.db"
        )
        
        # Test that context manager can access memory
        if hasattr(context_manager, 'memory_system'):
            logger.info("✅ Context manager has memory system")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all consolidation tests"""
    logger.info("🧪 Starting SurrealDB consolidation tests...\n")
    
    tests = [
        ("Legacy Removal", test_legacy_removal),
        ("SurrealDB Memory System", test_surrealdb_memory_system), 
        ("SmartContextManager", test_smart_context_manager),
        ("DSPy Foundation", test_dspy_foundation),
        ("Query Router", test_query_router),
        ("Integration", lambda: asyncio.run(test_integration()))
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} Test ---")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} test PASSED\n")
            else:
                logger.error(f"❌ {test_name} test FAILED\n")
        except Exception as e:
            logger.error(f"❌ {test_name} test ERRORED: {e}\n")
            results.append((test_name, False))
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All consolidation tests passed! System ready for DSPy.")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)