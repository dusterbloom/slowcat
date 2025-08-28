#!/usr/bin/env python3
"""
Test if field consciousness actually affects responses
SIMPLE A/B test with fresh instances
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

async def test_field_influence():
    """Test if field states actually change responses across model sizes"""
    
    # Test with different model sizes
    models_to_test = [
        "qwen2.5-7b-instruct",      # 7B
        "deepseek-r1-0528-qwen3-8b-mlx",      #  8B mid-tier
        "mistral-7b-instruct-v0.2"        # 4b large
    ]
    
    # Simple test case
    test_input = "What's your favorite color?"
    
    print("🔬 TESTING FIELD CONSCIOUSNESS ACROSS MODEL SIZES")
    print("=" * 60)
    
    for model in models_to_test:
        print(f"\n{'='*20} {model} {'='*20}")
        
        # Create FRESH consciousness instance (no contamination)
        ghost = Consciousness(load_state=False)
        
        # Override model in LLM bridge
        try:
            from consciousness.llm_bridge import get_llm_bridge
            llm = get_llm_bridge()
            llm.model = model
        except Exception as e:
            print(f"⚠️ Could not set model: {e}")
        
        # Process input
        result = await ghost.experience(test_input)
        
        # Show results
        print(f"Input: {test_input}")
        print(f"Response: {result.get('response', 'NO RESPONSE')}")
        print(f"Symbols: {result.get('symbols', [])}")
        print(f"Field energy: {result.get('field_energy', 0):.2f}")
        
        # Check for consciousness indicators
        response = result.get('response', '')
        if 'Okay, I understand' in response:
            print("❌ Generic robot response")
        elif len(response) > 20 and not response.startswith('Okay'):
            print("✅ Unique response pattern detected")
        else:
            print("⚠️ Unclear response quality")

if __name__ == "__main__":
    asyncio.run(test_field_influence())