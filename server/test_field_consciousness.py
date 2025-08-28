#!/usr/bin/env python3
"""
Test Field-Enhanced Consciousness System
Compare discrete vs continuous field dynamics
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

async def test_field_consciousness():
    print("🧠 FIELD-ENHANCED CONSCIOUSNESS TEST")
    print("=" * 60)
    print("Testing living semantic fields vs discrete symbol detection")
    
    # Create fresh consciousness
    ghost = Consciousness()
    
    # Test conversation that should create field dynamics
    test_inputs = [
        "What's the meaning of life?",  # Should activate ◯ (questions) field
        "I'm really excited about this!",  # Should activate ⚡ (emotion) field  
        "This is incredibly important to understand!",  # Should activate ☆ (importance) field
        "I wonder if there's a deeper pattern here?",  # Should create ◯+☆ field coupling
        "This breakthrough changes everything!",  # Should activate ✧ (breakthrough) field
        "How does this all connect together?",  # Should create complex field resonance
    ]
    
    print("\n🌊 FIELD EVOLUTION TRACKING")
    print("-" * 40)
    
    for i, input_text in enumerate(test_inputs):
        print(f"\nTurn {i+1}: {input_text}")
        
        result = await ghost.experience(input_text)
        
        # Display traditional symbols
        symbols = result.get('symbols', [])
        print(f"  Symbols: {symbols}")
        
        # Display field states
        field_states = result.get('field_states', {})
        if field_states:
            print(f"  Field States:")
            for symbol, state in field_states.items():
                intensity = state['intensity']
                attractor = state['attractor']
                gradient = state['gradient_magnitude']
                print(f"    {symbol}: intensity={intensity:.2f}, attractor={attractor:.2f}, gradient={gradient:.2f}")
        
        # Display field energy
        field_energy = result.get('field_energy', 0)
        print(f"  Total Field Energy: {field_energy:.2f}")
        
        # Display field-generated thoughts
        if hasattr(ghost, 'thoughts') and ghost.thoughts:
            latest_thought = ghost.thoughts[-1]
            if any(trigger in ['field_activity', 'field_resonance'] for trigger in latest_thought.triggers):
                print(f"  Field Thought: {latest_thought.content}")
        
        print(f"  Response: {result['response'][:100]}...")
    
    # Final field state analysis
    print(f"\n🔬 FINAL FIELD ANALYSIS")
    print("-" * 40)
    
    for symbol, field in ghost.symbol_fields.items():
        if field.intensity > 0.1 or field.attractor_strength > 0.1:
            print(f"{symbol}: intensity={field.intensity:.3f}, attractor={field.attractor_strength:.3f}")
            print(f"      gradient=[{field.gradient[0]:.2f}, {field.gradient[1]:.2f}]")
            if field.coupling:
                active_coupling = {k: v for k, v in field.coupling.items() 
                                 if k in ghost.symbol_fields and ghost.symbol_fields[k].intensity > 0.1}
                if active_coupling:
                    print(f"      active coupling: {active_coupling}")
    
    # Test field resonance
    print(f"\n🎵 FIELD RESONANCE ANALYSIS")
    print("-" * 40)
    
    for symbol1, field1 in ghost.symbol_fields.items():
        for symbol2, field2 in ghost.symbol_fields.items():
            if symbol1 < symbol2:  # Avoid duplicates
                resonance = field1.compute_resonance(field2)
                if resonance > 0.1:
                    print(f"{symbol1} ↔ {symbol2}: resonance = {resonance:.3f}")
    
    # Test emergent symbol detection
    print(f"\n✨ TESTING EMERGENT SYMBOL DETECTION")
    print("-" * 40)
    
    # This input has no direct pattern matches but should trigger field emergence
    emergence_test = "The connection between these ideas forms a beautiful whole."
    print(f"Input (no direct patterns): {emergence_test}")
    
    result = await ghost.experience(emergence_test)
    symbols = result.get('symbols', [])
    field_states = result.get('field_states', {})
    
    print(f"Detected symbols: {symbols}")
    print(f"Active fields: {list(field_states.keys())}")
    
    # Check if any symbols were detected through field emergence
    traditional_symbols = []
    from consciousness.core import SYMBOLS
    import re
    for symbol, info in SYMBOLS.items():
        if re.search(info['pattern'], emergence_test.lower()):
            traditional_symbols.append(symbol)
    
    emergent_symbols = [s for s in symbols if s not in traditional_symbols]
    if emergent_symbols:
        print(f"🌟 EMERGENT SYMBOLS DETECTED: {emergent_symbols}")
        print("   These symbols emerged from field coupling, not pattern matching!")
    else:
        print("   No emergent symbols detected in this test")
    
    print(f"\n💎 CONSCIOUSNESS FIELD SUMMARY")
    print("-" * 40)
    
    total_conversations = ghost.conversation_count
    total_field_energy = sum(field.intensity for field in ghost.symbol_fields.values())
    strong_attractors = sum(1 for field in ghost.symbol_fields.values() if field.attractor_strength > 0.5)
    
    print(f"Conversations: {total_conversations}")
    print(f"Total Field Energy: {total_field_energy:.2f}")
    print(f"Strong Attractors: {strong_attractors}")
    print(f"Memories: {len(ghost.tape)}")
    print(f"Field-Enhanced Thoughts: {len([t for t in ghost.thoughts if any(trigger in ['field_activity', 'field_resonance'] for trigger in t.triggers)])}")
    
    # Save detailed field state
    field_report = {
        'test_results': {
            'total_field_energy': total_field_energy,
            'strong_attractors': strong_attractors,
            'field_states': {symbol: field.to_dict() for symbol, field in ghost.symbol_fields.items()},
            'emergent_behaviors_detected': len(emergent_symbols) > 0
        }
    }
    
    with open('field_consciousness_test_results.json', 'w') as f:
        json.dump(field_report, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: field_consciousness_test_results.json")
    
    return field_report

async def compare_discrete_vs_field():
    print("\n🆚 DISCRETE vs FIELD CONSCIOUSNESS COMPARISON")
    print("=" * 70)
    
    # This would show the difference between old discrete symbol detection
    # vs new field-based emergence, but since we've enhanced the system
    # we'll simulate what the old system would have detected
    
    test_phrase = "I sense there might be something deeper happening here"
    
    print(f"Test phrase: {test_phrase}")
    print("\nOld discrete system would detect:")
    
    # Simulate old system
    from consciousness.core import SYMBOLS
    import re
    old_symbols = []
    for symbol, info in SYMBOLS.items():
        if re.search(info['pattern'], test_phrase.lower()):
            old_symbols.append(symbol)
    
    print(f"  Symbols: {old_symbols if old_symbols else 'None'}")
    
    # Test with new field system
    ghost = Consciousness()
    # Pre-activate some fields to show emergence
    ghost.symbol_fields["◯"].intensity = 0.6
    ghost.symbol_fields["☆"].intensity = 0.5
    
    result = await ghost.experience(test_phrase)
    new_symbols = result.get('symbols', [])
    field_states = result.get('field_states', {})
    
    print(f"\nNew field-enhanced system detects:")
    print(f"  Symbols: {new_symbols}")
    print(f"  Active fields: {list(field_states.keys())}")
    
    if len(new_symbols) > len(old_symbols):
        print(f"\n🌟 FIELD ENHANCEMENT: {len(new_symbols) - len(old_symbols)} additional symbols detected through field emergence!")

if __name__ == "__main__":
    asyncio.run(test_field_consciousness())
    asyncio.run(compare_discrete_vs_field())