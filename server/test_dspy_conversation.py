#!/usr/bin/env python3
"""
Interactive DSPy Conversation Test

This script shows DSPy in action with clear logging for testing and debugging.
Run this to see how DSPy optimizes context selection in real-time.
"""

import asyncio
import sys
from pathlib import Path

# Add server path
sys.path.insert(0, str(Path(__file__).parent))

from slowcat_dspy import create_unified_memory_optimizer, DSPY_AVAILABLE
from loguru import logger

async def test_dspy_conversation():
    """Test DSPy with a realistic conversation scenario"""
    
    if not DSPY_AVAILABLE:
        print("❌ DSPy not available. Install with: pip install -U dspy")
        return
    
    print("🚀 DSPy Conversation Optimization Test")
    print("=" * 50)
    
    # Create optimizer
    optimizer = create_unified_memory_optimizer()
    
    # Simulate a rich memory store (like what DTH would provide)
    memory_candidates = [
        "User has a golden retriever named Potola who is 3 years old and loves fetch",
        "User lives in San Francisco in the Mission District near Dolores Park", 
        "User works as a software engineer at a startup focused on AI applications",
        "Recent conversation about favorite coffee shops: Blue Bottle is preferred over Starbucks",
        "User mentioned planning a weekend trip to Tahoe for hiking and camping",
        "Discussion about morning routine: coffee at 7am, check emails, walk Potola",
        "User expressed interest in jazz music, particularly Miles Davis and John Coltrane",
        "Previous conversation about cooking: user enjoys making pasta but struggles with timing",
        "User asked about dog training tips for teaching Potola to sit and stay",
        "Recent query about SF weather patterns and best times to visit parks",
        "User mentioned working from home 3 days a week, commuting on Tuesday/Thursday",
        "Discussion about weekend farmers market visits and favorite vendors"
    ]
    
    # Test scenarios
    scenarios = [
        ("Tell me about my dog", "Pet-related query"),
        ("What should I do this weekend?", "Activity planning query"),
        ("I need coffee recommendations", "Local recommendations query"),
        ("How's my morning routine going?", "Personal habits query"),
        ("What music should I listen to while coding?", "Work + music query")
    ]
    
    for i, (query, description) in enumerate(scenarios):
        print(f"\n🧪 Test {i+1}: {description}")
        print(f"❓ Query: '{query}'")
        print("-" * 50)
        
        # Run DSPy optimization
        result = optimizer(
            query=query,
            dth_candidates=memory_candidates,
            target_tokens=2800,  # Our budget
            mode="chat"
        )
        
        # Show results
        selected = result.get('selected_memory', '')
        reasoning = result.get('selection_reasoning', '')
        
        print(f"✨ DSPy Selected Memory ({len(selected)} chars):")
        print(f"   {selected[:200]}..." if len(selected) > 200 else f"   {selected}")
        print(f"\n💡 DSPy Reasoning:")
        print(f"   {reasoning[:150]}..." if len(reasoning) > 150 else f"   {reasoning}")
        print(f"\n📊 Stats: {result.get('token_efficiency', 0):.2f} efficiency")
        
        # Wait between tests
        if i < len(scenarios) - 1:
            print("\n⏳ Next test in 2 seconds...")
            await asyncio.sleep(2)
    
    # Final performance summary
    perf = optimizer.get_performance_summary()
    print("\n" + "=" * 50)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Total Optimizations: {perf['total_optimizations']}")
    print(f"Average Token Efficiency: {perf['avg_token_efficiency']:.3f}")
    print(f"DSPy Available: {perf['dspy_available']}")
    
    print("\n🎉 DSPy conversation test completed!")
    print("   In a real conversation, this optimization happens automatically")
    print("   DSPy learns which memories are most relevant for each type of query")

if __name__ == "__main__":
    asyncio.run(test_dspy_conversation())