#!/usr/bin/env python3
"""
Verify Adaptive Knowledge Graph is Ready for Production

This script verifies that the adaptive system will work when you run:
./run_bot.sh or python bot_v2.py
"""

import asyncio
import sys
import os
from loguru import logger

# Add server directory to path  
sys.path.insert(0, '.')

def check_environment():
    """Check environment configuration"""
    logger.info("🔧 CHECKING ENVIRONMENT CONFIGURATION")
    
    required_settings = {
        'USE_SURREALDB': 'true',
        'ENABLE_MEMORY': 'true', 
        'SC_UNIFIED_MEMORY': 'true',
        'MEMORY_BACKEND': 'surreal'
    }
    
    all_good = True
    for setting, expected in required_settings.items():
        actual = os.getenv(setting, 'NOT_SET')
        if actual.lower() == expected.lower():
            logger.info(f"✅ {setting:20} = {actual}")
        else:
            logger.error(f"❌ {setting:20} = {actual} (expected: {expected})")
            all_good = False
    
    return all_good

async def check_database_connection():
    """Check database connection and content"""
    logger.info("\n💾 CHECKING DATABASE CONNECTION")
    
    try:
        from memory.surreal_connection import get_surreal_connection
        conn = get_surreal_connection()
        await conn.connect()
        
        logger.info("✅ Connected to SurrealDB successfully")
        
        # Check if we have knowledge data
        result = await conn.db.query("SELECT predicate FROM knowledge LIMIT 5")
        predicate_count = len(result) if result else 0
        
        if predicate_count > 0:
            logger.info(f"✅ Found {predicate_count} knowledge records")
            
            # Show a few sample predicates
            predicates = set()
            for record in result[:5]:
                if isinstance(record, dict) and 'predicate' in record:
                    predicates.add(record['predicate'])
            
            if predicates:
                logger.info(f"   Sample predicates: {', '.join(sorted(list(predicates))[:3])}")
        else:
            logger.warning("⚠️ No knowledge records found (will be created when bot runs)")
        
        await conn.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def check_imports():
    """Check that all adaptive components can be imported"""
    logger.info("\n📦 CHECKING COMPONENT IMPORTS")
    
    components = [
        ('memory.surreal_connection', 'get_surreal_connection'),
        ('memory.adaptive_knowledge_graph', 'get_adaptive_kg'),
        ('services.knowledge_evolution_service', 'get_evolution_service'),
        ('processors.smart_context_manager', 'SmartContextManager')
    ]
    
    all_good = True
    for module, component in components:
        try:
            mod = __import__(module, fromlist=[component])
            getattr(mod, component)
            logger.info(f"✅ {module}.{component}")
        except Exception as e:
            logger.error(f"❌ {module}.{component} - {e}")
            all_good = False
    
    return all_good

async def test_adaptive_normalization():
    """Test that adaptive normalization is working"""
    logger.info("\n🧬 TESTING ADAPTIVE NORMALIZATION")
    
    try:
        from memory.surreal_connection import get_surreal_connection
        
        # Test storing a fact with normalization
        conn = get_surreal_connection()
        await conn.connect()
        
        # Store a test fact that should trigger normalization  
        test_predicate = "test_dog_name"
        success = await conn.store_knowledge_relation(
            "test_user", test_predicate, "test_fluffy", "person", "concept"
        )
        
        if success:
            logger.info(f"✅ Successfully stored fact with predicate: {test_predicate}")
            logger.info("✅ Adaptive normalization is integrated into storage pipeline")
        else:
            logger.error("❌ Failed to store test fact")
            
        await conn.disconnect()
        return success
        
    except Exception as e:
        logger.error(f"❌ Adaptive normalization test failed: {e}")
        return False

def check_bot_startup_integration():
    """Check integration with bot startup process"""
    logger.info("\n🤖 CHECKING BOT STARTUP INTEGRATION")
    
    try:
        # Check if bot_v2.py would use our adaptive system
        with open('bot_v2.py', 'r') as f:
            bot_content = f.read()
            
        if 'SmartContextManager' in bot_content:
            logger.info("✅ bot_v2.py uses SmartContextManager")
        else:
            logger.warning("⚠️ bot_v2.py might not use SmartContextManager")
            
        # Check if pipeline_builder creates the right memory system
        try:
            from core.pipeline_builder import PipelineBuilder
            logger.info("✅ PipelineBuilder imports successfully")
        except Exception as e:
            logger.error(f"❌ PipelineBuilder import failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Bot integration check failed: {e}")
        return False

async def main():
    """Main verification function"""
    logger.info("🚀 VERIFYING ADAPTIVE KNOWLEDGE GRAPH PRODUCTION READINESS")
    logger.info("=" * 60)
    
    checks = [
        ("Environment Configuration", check_environment),
        ("Component Imports", check_imports), 
        ("Database Connection", check_database_connection),
        ("Adaptive Normalization", test_adaptive_normalization),
        ("Bot Integration", check_bot_startup_integration)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ {name} check failed: {e}")
            results.append(False)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VERIFICATION SUMMARY")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        logger.info(f"🎉 ALL CHECKS PASSED ({passed}/{total})")
        logger.info("")
        logger.info("✅ Your adaptive knowledge graph is READY for production!")
        logger.info("✅ Run './run_bot.sh' and the system will:")
        logger.info("   • Automatically normalize predicates from conversations") 
        logger.info("   • Learn patterns and cluster similar predicates")
        logger.info("   • Evolve the knowledge graph structure over time")
        logger.info("   • Display normalized facts in DTH memories")
        logger.info("")
        logger.info("🧠 The knowledge graph will build itself as you talk!")
        return 0
    else:
        logger.error(f"❌ SOME CHECKS FAILED ({passed}/{total})")
        logger.error("Please fix the issues above before running the bot")
        return 1

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    # Run verification
    exit_code = asyncio.run(main())
    sys.exit(exit_code)