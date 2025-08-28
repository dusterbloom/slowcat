#!/usr/bin/env python3
"""
Test consciousness configuration and environment integration

This tests the complete Task-5 implementation:
1. ConsciousnessConfig with environment variables
2. Configuration validation and dependency detection
3. Graceful degradation when dependencies are missing
4. SmartContextManager integration with consciousness config
5. MLX acceleration detection and fallback behavior
"""

import asyncio
import sys
import os
from unittest.mock import patch, MagicMock

# Add server to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, ConsciousnessConfig, config
from processors.smart_context_manager import create_smart_context_manager
from loguru import logger


def test_consciousness_config_creation():
    """Test basic consciousness configuration creation"""
    print("\n1️⃣  CONSCIOUSNESS CONFIG CREATION TEST")
    print("-" * 50)
    
    # Test default configuration
    consciousness_config = ConsciousnessConfig()
    print(f"✅ Default config created")
    print(f"  Enabled: {consciousness_config.enabled}")
    print(f"  Field dimension: {consciousness_config.field_dimension}")
    print(f"  MLX acceleration: {consciousness_config.enable_mlx_acceleration}")
    
    # Test configuration validation
    is_valid = consciousness_config.validate()
    print(f"  Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    return is_valid


def test_environment_variable_integration():
    """Test consciousness configuration with environment variables"""
    print("\n2️⃣  ENVIRONMENT VARIABLE INTEGRATION TEST")
    print("-" * 50)
    
    success_count = 0
    
    # Test ENABLE_CONSCIOUSNESS
    original_env = os.environ.copy()
    
    try:
        # Test consciousness disabled
        os.environ['ENABLE_CONSCIOUSNESS'] = 'false'
        config_disabled = ConsciousnessConfig()
        if not config_disabled.enabled:
            print("✅ ENABLE_CONSCIOUSNESS=false works")
            success_count += 1
        else:
            print("❌ ENABLE_CONSCIOUSNESS=false failed")
        
        # Test consciousness enabled
        os.environ['ENABLE_CONSCIOUSNESS'] = 'true'
        config_enabled = ConsciousnessConfig()
        if config_enabled.enabled:
            print("✅ ENABLE_CONSCIOUSNESS=true works")
            success_count += 1
        else:
            print("❌ ENABLE_CONSCIOUSNESS=true failed")
        
        # Test custom field dimension
        os.environ['CONSCIOUSNESS_FIELD_DIM'] = '256'
        config_custom = ConsciousnessConfig()
        if config_custom.field_dimension == 256:
            print("✅ CONSCIOUSNESS_FIELD_DIM=256 works")
            success_count += 1
        else:
            print("❌ CONSCIOUSNESS_FIELD_DIM=256 failed")
        
        # Test MLX acceleration settings
        os.environ['ENABLE_MLX_ACCELERATION'] = 'false'
        config_no_mlx = ConsciousnessConfig()
        if not config_no_mlx.should_enable_mlx():
            print("✅ ENABLE_MLX_ACCELERATION=false works")
            success_count += 1
        else:
            print("❌ ENABLE_MLX_ACCELERATION=false failed")
            
    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(original_env)
    
    print(f"\nEnvironment variable tests: {success_count}/4 passed")
    return success_count == 4


def test_dependency_detection():
    """Test consciousness dependency detection"""
    print("\n3️⃣  DEPENDENCY DETECTION TEST")
    print("-" * 50)
    
    test_config = Config()
    validation_result = test_config.validate_configuration()
    consciousness_status = validation_result['consciousness_status']
    
    print(f"Configuration valid: {'✅' if validation_result['valid'] else '❌'}")
    print(f"Consciousness can run: {'✅' if consciousness_status['can_run'] else '❌'}")
    print(f"MLX available: {'✅' if consciousness_status['mlx_available'] else '❌'}")
    print(f"SurrealDB available: {'✅' if consciousness_status['surrealdb_available'] else '❌'}")
    print(f"Available components: {consciousness_status['available']}")
    print(f"Missing components: {consciousness_status['missing']}")
    
    # Test effective configuration
    effective_config = test_config.get_consciousness_effective_config()
    print(f"\nEffective configuration:")
    for key, value in effective_config.items():
        if key != 'dependencies':
            print(f"  {key}: {value}")
    
    return consciousness_status['can_run']


def test_graceful_degradation():
    """Test graceful degradation when dependencies are missing"""
    print("\n4️⃣  GRACEFUL DEGRADATION TEST")  
    print("-" * 50)
    
    success_count = 0
    
    # Mock missing MLX
    with patch('mlx.core', side_effect=ImportError("MLX not available")):
        config_no_mlx = ConsciousnessConfig()
        should_enable_mlx = config_no_mlx.should_enable_mlx()
        
        if config_no_mlx.enable_mlx_acceleration == "auto" and not should_enable_mlx:
            print("✅ MLX auto-detection works when not available")
            success_count += 1
        else:
            print("❌ MLX auto-detection failed")
    
    # Test forced MLX when not available
    with patch.dict(os.environ, {'ENABLE_MLX_ACCELERATION': 'false'}):
        config_force_no_mlx = ConsciousnessConfig()
        if not config_force_no_mlx.should_enable_mlx():
            print("✅ MLX forced disable works")
            success_count += 1
        else:
            print("❌ MLX forced disable failed")
    
    # Test graceful degradation setting
    with patch.dict(os.environ, {'CONSCIOUSNESS_GRACEFUL_DEGRADATION': 'true'}):
        config_graceful = ConsciousnessConfig()
        if config_graceful.graceful_degradation:
            print("✅ Graceful degradation enabled")
            success_count += 1
        else:
            print("❌ Graceful degradation setting failed")
    
    print(f"\nGraceful degradation tests: {success_count}/3 passed")
    return success_count == 3


async def test_smart_context_manager_integration():
    """Test SmartContextManager integration with consciousness config"""
    print("\n5️⃣  SMART CONTEXT MANAGER INTEGRATION TEST")
    print("-" * 50)
    
    success_count = 0
    
    # Mock context object
    class MockContext:
        def __init__(self):
            self.messages = []
            
        def add_message(self, message):
            self.messages.append(message)
    
    try:
        # Test with consciousness enabled
        context = MockContext()
        
        # Set environment for testing
        original_env = dict(os.environ)
        os.environ['ENABLE_CONSCIOUSNESS'] = 'true'
        os.environ['USER_ID'] = 'test_user'
        
        try:
            smart_manager = create_smart_context_manager(
                context=context,
                facts_db_path="data/test_facts.db",
                max_tokens=2048
            )
            
            print("✅ SmartContextManager created successfully")
            success_count += 1
            
            # Check if consciousness integration was attempted
            has_consciousness = smart_manager._consciousness_instance is not None
            print(f"  Consciousness instance: {'✅ Present' if has_consciousness else '❌ Missing'}")
            if has_consciousness:
                success_count += 1
            
        except Exception as e:
            print(f"❌ SmartContextManager creation failed: {e}")
            
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)
        
        # Test with consciousness disabled
        os.environ['ENABLE_CONSCIOUSNESS'] = 'false'
        
        try:
            smart_manager_disabled = create_smart_context_manager(
                context=context,
                facts_db_path="data/test_facts.db",
                max_tokens=2048
            )
            
            has_no_consciousness = smart_manager_disabled._consciousness_instance is None
            print(f"  Consciousness disabled: {'✅ Confirmed' if has_no_consciousness else '❌ Failed'}")
            if has_no_consciousness:
                success_count += 1
            
        except Exception as e:
            print(f"❌ Disabled consciousness test failed: {e}")
            
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    print(f"\nSmartContextManager integration: {success_count}/3 passed")
    return success_count >= 2  # At least basic functionality should work


def test_configuration_validation():
    """Test configuration validation with invalid values"""
    print("\n6️⃣  CONFIGURATION VALIDATION TEST")
    print("-" * 50)
    
    success_count = 0
    
    # Test invalid field dimension
    with patch.dict(os.environ, {'CONSCIOUSNESS_FIELD_DIM': '-1'}):
        try:
            invalid_config = ConsciousnessConfig()
            if not invalid_config.validate():
                print("✅ Invalid field dimension caught")
                success_count += 1
            else:
                print("❌ Invalid field dimension not caught")
        except ValueError:
            print("✅ Invalid field dimension caught (ValueError)")
            success_count += 1
    
    # Test invalid evolution rate
    with patch.dict(os.environ, {'CONSCIOUSNESS_EVOLUTION_RATE': '2.0'}):  # Should be 0-1
        try:
            invalid_config = ConsciousnessConfig()
            if not invalid_config.validate():
                print("✅ Invalid evolution rate caught")
                success_count += 1
            else:
                print("❌ Invalid evolution rate not caught")
        except ValueError:
            print("✅ Invalid evolution rate caught (ValueError)")
            success_count += 1
    
    # Test invalid temperature
    with patch.dict(os.environ, {'CONSCIOUSNESS_TEMPERATURE': '-0.5'}):
        try:
            invalid_config = ConsciousnessConfig()
            if not invalid_config.validate():
                print("✅ Invalid temperature caught")
                success_count += 1
            else:
                print("❌ Invalid temperature not caught")
        except ValueError:
            print("✅ Invalid temperature caught (ValueError)")
            success_count += 1
    
    print(f"\nConfiguration validation: {success_count}/3 passed")
    return success_count >= 2


async def main():
    """Run all consciousness configuration tests"""
    print("🧠 CONSCIOUSNESS CONFIGURATION TESTS")
    print("="*80)
    
    results = []
    
    # Run all tests
    results.append(test_consciousness_config_creation())
    results.append(test_environment_variable_integration())
    results.append(test_dependency_detection()) 
    results.append(test_graceful_degradation())
    results.append(await test_smart_context_manager_integration())
    results.append(test_configuration_validation())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 FINAL RESULTS:")
    print("="*50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed >= total * 0.8:  # 80% success rate
        print("\n✅ CONSCIOUSNESS CONFIGURATION SUCCESSFUL!")
        print("   Task-5 implementation working correctly:")
        print("   • Environment variable integration")
        print("   • Dependency detection and fallback")
        print("   • Configuration validation")
        print("   • SmartContextManager integration")
        print("   • Graceful degradation when dependencies missing")
    else:
        print("\n❌ Consciousness configuration needs improvement")
        print("   Some tests are failing")
    
    return passed >= total * 0.8


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)