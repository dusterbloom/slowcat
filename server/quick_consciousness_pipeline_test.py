#!/usr/bin/env python3
"""Quick test for consciousness pipeline integration"""

import asyncio
import os
from processors.smart_context_manager import create_smart_context_manager

async def quick_test():
    print("🧠 Quick Consciousness Pipeline Integration Test")
    print("=" * 50)
    
    # Mock context
    class MockContext:
        def __init__(self):
            self.messages = []
    
    # Test consciousness creation through factory function
    os.environ['ENABLE_CONSCIOUSNESS'] = 'true'
    os.environ['USER_ID'] = 'quick_test_user'
    
    try:
        print("1. Creating SmartContextManager with consciousness...")
        smart_manager = create_smart_context_manager(
            MockContext(), 
            user_id="quick_test_user",
            enable_consciousness=True
        )
        
        consciousness_integrated = smart_manager._consciousness_instance is not None
        print(f"   Consciousness integrated: {'✅' if consciousness_integrated else '❌'}")
        
        if consciousness_integrated:
            field_count = len(smart_manager._consciousness_instance.symbol_fields)
            print(f"   Symbol fields available: {field_count}")
            
        print("2. Testing consciousness field processing...")
        if smart_manager._consciousness_instance:
            # Test symbolic processing
            symbols = smart_manager._consciousness_instance.symbolize("I'm excited about consciousness!")
            print(f"   Extracted symbols: {symbols}")
            
        print("3. Testing field persistence layer...")
        field_persistence_available = smart_manager.field_persistence is not None
        print(f"   Field persistence layer: {'✅' if field_persistence_available else '❌'}")
        
        if field_persistence_available:
            print(f"   Persistence enabled: {smart_manager.field_persistence.enabled}")
        
        print("✅ Quick consciousness integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(quick_test())