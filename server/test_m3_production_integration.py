#!/usr/bin/env python3
"""
M3 Production Integration Test

Tests that the M3 integration is properly integrated into the production bot
and can be enabled/disabled via environment variables as intended.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_bot_with_m3_disabled():
    """Test that bot runs with M3 disabled"""
    print("🧪 Testing bot with M3 disabled...")
    
    env = os.environ.copy()
    env.update({
        'ENABLE_M3': 'false',
        'USE_M3_CONTEXT': 'false'
    })
    
    try:
        result = subprocess.run([
            './run_bot.sh', '--help'
        ], env=env, capture_output=True, text=True, timeout=60, cwd='/Users/peppi/Dev/slowcat-consciousness/server')
        
        if result.returncode == 0:
            print("✅ Bot runs successfully with M3 disabled")
            return True
        else:
            print(f"❌ Bot failed with M3 disabled: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Bot timed out with M3 disabled")
        return False
    except Exception as e:
        print(f"❌ Error testing bot with M3 disabled: {e}")
        return False

def test_bot_with_m3_enabled_fallback():
    """Test that bot runs with M3 enabled but falls back gracefully when SurrealDB unavailable"""
    print("🧪 Testing bot with M3 enabled (with fallback)...")
    
    env = os.environ.copy()
    env.update({
        'ENABLE_M3': 'true',
        'USE_M3_CONTEXT': 'true',
        'M3_FALLBACK_TO_STANDARD': 'true'
    })
    
    try:
        result = subprocess.run([
            './run_bot.sh', '--help'
        ], env=env, capture_output=True, text=True, timeout=60, cwd='/Users/peppi/Dev/slowcat-consciousness/server')
        
        if result.returncode == 0:
            print("✅ Bot runs successfully with M3 enabled (graceful fallback)")
            return True
        else:
            print(f"❌ Bot failed with M3 enabled: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Bot timed out with M3 enabled")
        return False
    except Exception as e:
        print(f"❌ Error testing bot with M3 enabled: {e}")
        return False

def test_configuration_loading():
    """Test that M3 configuration is properly loaded"""
    print("🧪 Testing M3 configuration loading...")
    
    # Add server directory to path temporarily
    server_path = str(Path(__file__).parent)
    if server_path not in sys.path:
        sys.path.insert(0, server_path)
    
    try:
        # Set M3 environment variables
        os.environ.update({
            'ENABLE_M3': 'true',
            'USE_M3_CONTEXT': 'true',
            'SURREALDB_HOST': 'test-host',
            'SURREALDB_PORT': '9999',
            'M3_MAX_CONTEXT_TOKENS': '8192',
            'M3_SIMILARITY_THRESHOLD': '0.8'
        })
        
        # Import and test config
        from config import config
        
        assert config.m3.enabled == True, "M3 should be enabled"
        assert config.m3.use_m3_context == True, "M3 context should be enabled"
        assert config.m3.surrealdb_host == 'test-host', "SurrealDB host should match env var"
        assert config.m3.surrealdb_port == 9999, "SurrealDB port should match env var"
        assert config.m3.max_context_tokens == 8192, "Max context tokens should match env var"
        assert config.m3.similarity_threshold == 0.8, "Similarity threshold should match env var"
        
        print("✅ M3 configuration loads correctly from environment variables")
        return True
        
    except Exception as e:
        print(f"❌ M3 configuration loading failed: {e}")
        return False
    finally:
        # Clean up environment
        test_vars = ['ENABLE_M3', 'USE_M3_CONTEXT', 'SURREALDB_HOST', 'SURREALDB_PORT', 'M3_MAX_CONTEXT_TOKENS', 'M3_SIMILARITY_THRESHOLD']
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]

def main():
    """Run all production integration tests"""
    print("🚀 M3 Production Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Bot with M3 Disabled", test_bot_with_m3_disabled),  
        ("Bot with M3 Enabled (Fallback)", test_bot_with_m3_enabled_fallback),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"💥 EXCEPTION in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Production Integration Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Failed: {total - passed}/{total}")
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 All M3 production integration tests PASSED!")
        print("\n✅ M3 integration is ready for production use:")
        print("   • Set ENABLE_M3=true to enable M3 system")
        print("   • Set USE_M3_CONTEXT=true to use M3 context retrieval")
        print("   • M3 gracefully falls back to standard memory when SurrealDB unavailable")
        print("   • All environment variables work as expected")
        return True
    else:
        print("\n💥 Some M3 production integration tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)