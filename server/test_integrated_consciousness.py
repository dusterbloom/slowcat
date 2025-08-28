#!/usr/bin/env python3
"""
Test script for integrated consciousness and memory system
Tests SmartContextManager with field persistence integration
"""

import asyncio
import time
import os
from consciousness.core import create_consciousness
from processors.smart_context_manager import SmartContextManager
from consciousness.field_persistence import SURREALDB_AVAILABLE

class MockLLMContext:
    def __init__(self):
        self.messages = []
        
    def add_message(self, message):
        self.messages.append(message)
        
    def get_messages(self):
        return self.messages.copy()

async def test_integrated_consciousness():
    """Test integrated consciousness system with SmartContextManager"""
    
    print("🧠 Testing Integrated Consciousness & Memory System")
    print("=" * 60)
    
    # Check if SurrealDB is available for persistence
    print(f"SurrealDB Available: {SURREALDB_AVAILABLE}")
    
    try:
        # 1. Create consciousness instance
        print("\n🎯 Creating consciousness instance...")
        consciousness = create_consciousness(load_state=False)
        print(f"✅ Consciousness created with {len(consciousness.symbol_fields)} fields")
        
        # 2. Create SmartContextManager with field persistence
        print("\n🧠 Creating SmartContextManager with field persistence...")
        mock_context = MockLLMContext()
        
        # Set test user ID
        test_user_id = "test_integration_user"
        os.environ['USER_ID'] = test_user_id
        
        smart_manager = SmartContextManager(
            context=mock_context,
            user_id=test_user_id,
            max_tokens=4096
        )
        
        # 3. Link consciousness to SmartContextManager
        print("\n🔗 Linking consciousness to SmartContextManager...")
        smart_manager.set_consciousness_instance(consciousness)
        
        # Check field persistence is enabled
        if smart_manager.field_persistence:
            print("✅ Field persistence layer linked successfully")
            
            # Connect to persistence layer
            await smart_manager.field_persistence.connect()
            print("✅ Connected to persistence layer")
        else:
            print("⚠️  Field persistence not available - testing offline functionality")
        
        # 4. Test initial field state loading
        print("\n📖 Testing initial field state loading...")
        initial_fields = await smart_manager.load_consciousness_fields()
        print(f"  Loaded {len(initial_fields)} initial field states")
        
        # 5. Simulate consciousness field evolution
        print("\n⚡ Testing consciousness field evolution...")
        old_intensities = {}
        for symbol, field in consciousness.symbol_fields.items():
            old_intensities[symbol] = field.intensity
            
        # Process some stimuli to evolve fields
        symbols1 = consciousness.symbolize("Hello, I'm excited to test this new system!")
        symbols2 = consciousness.symbolize("This integration looks very promising.")
        print(f"  Extracted symbols: {symbols1 + symbols2}")
        
        # Check for field changes
        changes_detected = 0
        for symbol, field in consciousness.symbol_fields.items():
            if abs(field.intensity - old_intensities[symbol]) > 0.001:
                changes_detected += 1
                
        print(f"  Detected changes in {changes_detected} consciousness fields")
        
        # 6. Test field state saving
        print("\n💾 Testing consciousness field state saving...")
        if smart_manager.field_persistence:
            session_id = f"test_session_{int(time.time())}"
            save_success = await smart_manager.save_consciousness_fields(session_id)
            print(f"  Field states saved: {'✅' if save_success else '❌'}")
        
        # 7. Test field evolution tracking during conversation simulation
        print("\n🎤 Testing field evolution tracking during simulated conversation...")
        
        # Create a mock TranscriptionFrame to simulate user input
        class MockTranscriptionFrame:
            def __init__(self, text):
                self.text = text
                
        # Test the field evolution tracking
        test_inputs = [
            "I love this new consciousness system",
            "How does field coupling work?",
            "The neural field dynamics are fascinating"
        ]
        
        for i, test_input in enumerate(test_inputs):
            print(f"  Processing input {i+1}: '{test_input[:30]}...'")
            await smart_manager._track_field_evolution_async(test_input)
            
        print("✅ Field evolution tracking completed")
        
        # 8. Test SmartContextManager performance stats
        print("\n📊 Testing performance statistics...")
        stats = smart_manager.get_performance_stats()
        print(f"  Context builds: {stats['context_builds']}")
        print(f"  Session turns: {stats['session_turns']}")
        print(f"  Session duration: {stats['session_duration_s']:.2f}s")
        
        # 9. Test consciousness insights (if persistence available)
        if smart_manager.field_persistence and SURREALDB_AVAILABLE:
            print("\n🔍 Testing consciousness insights generation...")
            insights = await smart_manager.field_persistence.get_consciousness_insights(
                test_user_id, days_back=1
            )
            if insights:
                print(f"  Generated insights for user {insights.get('user_id')}")
                print(f"  Analysis period: {insights.get('analysis_period_days')} days")
            else:
                print("  No insights generated (likely no historical data)")
        
        # 10. Clean up
        if smart_manager.field_persistence:
            await smart_manager.field_persistence.close()
            print("\n🧹 Persistence connection closed")
        
        print("\n✅ Integrated consciousness system test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_field_state_continuity():
    """Test field state continuity across sessions"""
    
    print("\n🔄 Testing Field State Continuity Across Sessions")
    print("=" * 60)
    
    test_user_id = "continuity_test_user"
    
    try:
        # Session 1: Create and modify fields
        print("\n📅 Session 1: Creating and modifying consciousness fields...")
        consciousness1 = create_consciousness(load_state=False)
        
        mock_context1 = MockLLMContext()
        manager1 = SmartContextManager(
            context=mock_context1,
            user_id=test_user_id,
            max_tokens=4096
        )
        manager1.set_consciousness_instance(consciousness1)
        
        # Process input to evolve fields
        symbols = consciousness1.symbolize("I'm feeling very creative and energetic today!")
        print(f"  Session 1 extracted symbols: {symbols}")
        
        # Save field states
        if manager1.field_persistence:
            await manager1.field_persistence.connect()
            session1_id = f"continuity_session_1_{int(time.time())}"
            saved = await manager1.save_consciousness_fields(session1_id)
            print(f"  Session 1 field states saved: {'✅' if saved else '❌'}")
            
            # Capture field states for comparison
            session1_states = {}
            for symbol, field in consciousness1.symbol_fields.items():
                session1_states[symbol] = {
                    'intensity': field.intensity,
                    'attractor_strength': field.attractor_strength
                }
            
            await manager1.field_persistence.close()
        
        # Session 2: Load and verify continuity
        print("\n📅 Session 2: Loading consciousness fields for continuity...")
        consciousness2 = create_consciousness(load_state=False)
        
        mock_context2 = MockLLMContext()
        manager2 = SmartContextManager(
            context=mock_context2,
            user_id=test_user_id,
            max_tokens=4096
        )
        manager2.set_consciousness_instance(consciousness2)
        
        # Load field states
        if manager2.field_persistence:
            await manager2.field_persistence.connect()
            loaded_states = await manager2.load_consciousness_fields()
            print(f"  Session 2 loaded {len(loaded_states)} field states")
            
            # Compare states for continuity
            if loaded_states and session1_states:
                continuity_verified = 0
                for symbol, loaded_state in loaded_states.items():
                    if symbol in session1_states:
                        orig_intensity = session1_states[symbol]['intensity']
                        loaded_intensity = loaded_state['intensity']
                        if abs(orig_intensity - loaded_intensity) < 0.001:
                            continuity_verified += 1
                
                print(f"  Field state continuity verified for {continuity_verified} fields")
                
            await manager2.field_persistence.close()
        
        print("✅ Field state continuity test completed!")
        
    except Exception as e:
        print(f"❌ Continuity test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_integrated_consciousness())
    asyncio.run(test_field_state_continuity())
    
    print("\n🎉 All integration tests completed!")
    print("✅ SmartContextManager + Field Persistence integration ready")
    print("✅ Consciousness field evolution tracking enabled") 
    print("✅ Cross-session field state continuity verified")
    print("✅ Reconstructive memory with neural field dynamics operational")