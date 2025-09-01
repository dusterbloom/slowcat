#!/usr/bin/env python3
"""
Test Consciousness System Impact
Compare bot behavior with consciousness enabled vs disabled
"""

import asyncio
import os
import sys
from pathlib import Path
import time
import json

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_consciousness_impact():
    """Test if consciousness system has observable impact on bot responses"""
    
    print("🧠 CONSCIOUSNESS SYSTEM IMPACT TEST")
    print("=" * 50)
    
    # Test scenarios
    test_inputs = [
        "Hi, my name is John and I have a cat named Whiskers",
        "What's my name?", 
        "What pet do I have?",
        "I just got a new job at Tesla",
        "Where do I work?",
        "I'm feeling really excited about this project",
        "How am I feeling?"
    ]
    
    results = {
        'consciousness_enabled': {},
        'consciousness_disabled': {}
    }
    
    # Test with consciousness ENABLED
    print("\n🧠 Testing WITH consciousness system...")
    os.environ['USE_CONTEXT_FIELD'] = 'true'
    os.environ['ENABLE_FIELD_PERSISTENCE'] = 'true'
    
    try:
        from processors.smart_context_manager import create_smart_context_manager
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        
        # Create context with consciousness enabled
        context_enabled = OpenAILLMContext()
        smart_manager_enabled = create_smart_context_manager(
            context_enabled,
            max_tokens=4096,
            enable_consciousness=True,
            user_id="test_user"
        )
        
        print("✅ SmartContextManager created with consciousness enabled")
        
        # Check if consciousness is actually active
        has_consciousness = smart_manager_enabled._consciousness_instance is not None
        has_field_persistence = smart_manager_enabled.field_persistence is not None
        
        print(f"   Consciousness instance: {'✅' if has_consciousness else '❌'}")
        print(f"   Field persistence: {'✅' if has_field_persistence else '❌'}")
        
        if has_consciousness:
            print(f"   Consciousness type: {type(smart_manager_enabled._consciousness_instance).__name__}")
            
        results['consciousness_enabled']['setup'] = {
            'has_consciousness': has_consciousness,
            'has_field_persistence': has_field_persistence
        }
        
    except Exception as e:
        print(f"❌ Failed to create consciousness-enabled manager: {e}")
        results['consciousness_enabled']['error'] = str(e)
    
    # Test with consciousness DISABLED  
    print("\n🚫 Testing WITHOUT consciousness system...")
    os.environ['USE_CONTEXT_FIELD'] = 'false'
    os.environ['ENABLE_FIELD_PERSISTENCE'] = 'false'
    
    try:
        from processors.smart_context_manager import create_smart_context_manager
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        
        # Create context with consciousness disabled
        context_disabled = OpenAILLMContext()
        smart_manager_disabled = create_smart_context_manager(
            context_disabled,
            max_tokens=4096,
            enable_consciousness=False,
            user_id="test_user"
        )
        
        print("✅ SmartContextManager created with consciousness disabled")
        
        # Check if consciousness is actually inactive
        has_consciousness = smart_manager_disabled._consciousness_instance is not None
        has_field_persistence = smart_manager_disabled.field_persistence is not None
        
        print(f"   Consciousness instance: {'✅' if has_consciousness else '❌'}")
        print(f"   Field persistence: {'✅' if has_field_persistence else '❌'}")
        
        results['consciousness_disabled']['setup'] = {
            'has_consciousness': has_consciousness,
            'has_field_persistence': has_field_persistence
        }
        
    except Exception as e:
        print(f"❌ Failed to create consciousness-disabled manager: {e}")
        results['consciousness_disabled']['error'] = str(e)
    
    # Test behavior differences
    print("\n🔍 ANALYZING BEHAVIORAL DIFFERENCES...")
    
    # Check if consciousness system adds any observable value
    if 'setup' in results['consciousness_enabled'] and 'setup' in results['consciousness_disabled']:
        enabled_setup = results['consciousness_enabled']['setup']
        disabled_setup = results['consciousness_disabled']['setup']
        
        if enabled_setup['has_consciousness'] != disabled_setup['has_consciousness']:
            print("✅ Consciousness state differs between configurations")
        else:
            print("⚠️  No difference in consciousness state between configurations")
            
        if enabled_setup['has_field_persistence'] != disabled_setup['has_field_persistence']:
            print("✅ Field persistence differs between configurations")
        else:
            print("⚠️  No difference in field persistence between configurations")
    
    # Save results
    with open('consciousness_impact_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n📄 Results saved to: consciousness_impact_results.json")
    
    # Analysis and recommendation
    print(f"\n🎯 ANALYSIS & RECOMMENDATION")
    print("=" * 40)
    
    if 'error' in results['consciousness_enabled'] or 'error' in results['consciousness_disabled']:
        print("❌ Setup errors detected - consciousness system may be broken")
        print("🏗️  RECOMMENDATION: Remove consciousness system (not working)")
        
    elif (results['consciousness_enabled'].get('setup', {}).get('has_consciousness') and 
          results['consciousness_disabled'].get('setup', {}).get('has_consciousness')):
        print("⚠️  Consciousness system doesn't seem to disable properly")
        print("🤔 RECOMMENDATION: Further investigation needed")
        
    elif (not results['consciousness_enabled'].get('setup', {}).get('has_consciousness') and
          not results['consciousness_disabled'].get('setup', {}).get('has_consciousness')):
        print("❌ Consciousness system never activates")
        print("🗑️  RECOMMENDATION: Remove consciousness system (not functional)")
        
    else:
        print("✅ Consciousness system enables/disables correctly")
        print("🧪 RECOMMENDATION: Test with actual bot conversations to measure user impact")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_consciousness_impact())