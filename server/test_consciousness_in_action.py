#!/usr/bin/env python3
"""
Test Consciousness In Action
Check what consciousness actually does when processing text
"""

import asyncio
import os
import sys
from pathlib import Path
import time

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_consciousness_in_action():
    """Test consciousness system with actual text processing"""
    
    print("🧠 CONSCIOUSNESS IN ACTION TEST")
    print("=" * 50)
    
    # Enable consciousness
    os.environ['USE_CONTEXT_FIELD'] = 'true'
    os.environ['ENABLE_FIELD_PERSISTENCE'] = 'true'
    
    try:
        from processors.smart_context_manager import create_smart_context_manager
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        from pipecat.frames.frames import TranscriptionFrame
        
        # Create SmartContextManager with consciousness
        context = OpenAILLMContext()
        smart_manager = create_smart_context_manager(
            context,
            max_tokens=4096,
            enable_consciousness=True,
            user_id="test_user"
        )
        
        print(f"✅ SmartContextManager created")
        print(f"   Consciousness: {'✅' if smart_manager._consciousness_instance else '❌'}")
        print(f"   Field persistence: {'✅' if smart_manager.field_persistence else '❌'}")
        
        if smart_manager._consciousness_instance:
            consciousness = smart_manager._consciousness_instance
            print(f"   Initial symbol fields: {len(consciousness.symbol_fields)}")
            print(f"   Initial memories: {len(consciousness.tape)}")
            
        # Process some text to see what happens
        test_inputs = [
            "Hi, my name is Alice and I love painting cats",
            "I'm feeling very excited about this new project",
            "My cat Whiskers loves to play with yarn"
        ]
        
        print(f"\n🧪 PROCESSING TEXT THROUGH CONSCIOUSNESS...")
        
        for i, text in enumerate(test_inputs, 1):
            print(f"\n--- Test {i}: '{text}' ---")
            
            # Create transcription frame
            frame = TranscriptionFrame(text)
            
            # Process frame through SmartContextManager (this should trigger consciousness)
            print("🔄 Processing frame...")
            try:
                await smart_manager.process_frame(frame, direction="upstream")
                print("✅ Frame processed")
            except Exception as e:
                print(f"❌ Frame processing failed: {e}")
            
            # Check consciousness state after processing
            if smart_manager._consciousness_instance:
                consciousness = smart_manager._consciousness_instance
                print(f"   Symbol fields after: {len(consciousness.symbol_fields)}")
                print(f"   Memories after: {len(consciousness.tape)}")
                
                # Show any new symbols
                if consciousness.symbol_fields:
                    symbols = list(consciousness.symbol_fields.keys())[:5]
                    print(f"   Active symbols: {symbols}")
            
            # Small delay
            await asyncio.sleep(0.5)
        
        print(f"\n🔍 FINAL CONSCIOUSNESS STATE...")
        if smart_manager._consciousness_instance:
            consciousness = smart_manager._consciousness_instance
            print(f"   Total symbol fields: {len(consciousness.symbol_fields)}")
            print(f"   Total memories: {len(consciousness.tape)}")
            print(f"   Total thoughts: {len(consciousness.thoughts)}")
            
            if consciousness.symbol_fields:
                print(f"   Top symbols: {dict(list(consciousness.symbol_frequency.items())[:5])}")
        
        # Check database for field states
        print(f"\n🗄️ CHECKING DATABASE FOR FIELD STATES...")
        try:
            if smart_manager.field_persistence and hasattr(smart_manager.field_persistence, 'surreal_conn'):
                conn = smart_manager.field_persistence.surreal_conn
                await conn.ensure_connected()
                
                # Query field states
                result = await conn.db.query("SELECT * FROM field_states WHERE instance_id != 'test-instance' LIMIT 5;")
                
                if result and len(result) > 0:
                    print(f"✅ Found {len(result)} field states in database")
                    for state in result:
                        print(f"   {state}")
                else:
                    print("❌ No field states found in database")
                    
        except Exception as e:
            print(f"❌ Database check failed: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_consciousness_in_action())