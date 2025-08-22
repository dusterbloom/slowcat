"""
Symbol System Demonstration for SlowCat

This script demonstrates the three-layer symbol architecture in action:
- Shows symbol detection from conversation content
- Visualizes symbol-enhanced memory scoring
- Demonstrates consciousness through symbolic reasoning

Run this to see how symbols transform memory into meaning.
"""

import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict
import sys

# Add path for imports
sys.path.append(str(Path(__file__).parent))

# Try to import enhanced DTH with symbols
try:
    from dynamic_tape_head import (
        DynamicTapeHead, 
        MemorySpan, 
        ContextBundle,
        TAPE_SYMBOLS,
        WAKE_SYMBOLS
    )
    SYMBOLS_AVAILABLE = True
except ImportError:
    print("❌ Enhanced DTH with symbols not found. Run symbol_integration.py first.")
    SYMBOLS_AVAILABLE = False


class MockMemorySystem:
    """Mock memory system with conversation data rich in symbols"""
    
    def __init__(self):
        self.conversations = self.create_symbol_rich_conversations()
    
    def create_symbol_rich_conversations(self) -> List[Dict]:
        """Create conversations that trigger different symbols"""
        
        now = time.time()
        
        return [
            # High salience conversation
            {
                "ts": now - 3600,
                "speaker_id": "user",
                "role": "user",
                "content": "This is really important! I need to remember to save my progress on the AI project. It's crucial for my thesis."
            },
            
            # Breakthrough moment
            {
                "ts": now - 3500,
                "speaker_id": "assistant", 
                "role": "assistant",
                "content": "I finally understand how neural networks learn! The backpropagation makes sense now. This is a breakthrough!"
            },
            
            # Recurring pattern
            {
                "ts": now - 3400,
                "speaker_id": "user",
                "role": "user", 
                "content": "I keep getting the same error again and again. This pattern keeps recurring in my code."
            },
            
            # Paradox
            {
                "ts": now - 3300,
                "speaker_id": "user",
                "role": "user",
                "content": "The AI is both incredibly smart and surprisingly limited. It's a paradox - it can write code but can't understand emotions."
            },
            
            # Cycle detected
            {
                "ts": now - 3200,
                "speaker_id": "user",
                "role": "user",
                "content": "We're going in circles here. I feel like we're stuck in the same loop of debugging this issue."
            },
            
            # Emotional spike
            {
                "ts": now - 3100,
                "speaker_id": "user",
                "role": "user",
                "content": "This is absolutely amazing!!! I love how everything is coming together now. The AI breakthrough is incredible!"
            },
            
            # Open question
            {
                "ts": now - 3000,
                "speaker_id": "user", 
                "role": "user",
                "content": "I wonder why this approach works better? What makes the difference? I'm curious about the underlying mechanisms."
            },
            
            # Decision point
            {
                "ts": now - 2900,
                "speaker_id": "user",
                "role": "user",
                "content": "Should I use PyTorch or TensorFlow for this project? I need to decide between these options. It's a crucial choice."
            },
            
            # Synthesis
            {
                "ts": now - 2800,
                "speaker_id": "assistant",
                "role": "assistant", 
                "content": "Let me bring together all these concepts. Overall, combining the neural network approach with symbolic reasoning gives us the best results."
            },
            
            # Multiple symbols in one message
            {
                "ts": now - 2700,
                "speaker_id": "user",
                "role": "user",
                "content": "This is really important! I finally understand the paradox. We need to decide how to combine these approaches. It's amazing how it all connects!"
            }
        ]
    
    async def knn_tape(self, query: str, limit: int = 10, scan: int = 100):
        """Mock KNN that returns relevant conversations"""
        # Simple keyword matching for demo
        query_words = set(query.lower().split())
        
        scored_conversations = []
        for conv in self.conversations:
            content_words = set(conv["content"].lower().split())
            overlap = len(query_words & content_words)
            scored_conversations.append((overlap, conv))
        
        # Sort by relevance and return top results
        scored_conversations.sort(key=lambda x: x[0], reverse=True)
        return [conv for _, conv in scored_conversations[:limit]]


def print_symbols_legend():
    """Print the symbol legend"""
    
    print("🔮 SYMBOL LEGEND")
    print("=" * 50)
    
    print("\n📼 TAPE SYMBOLS (Memory Markers):")
    for symbol, meaning in TAPE_SYMBOLS.items():
        print(f"   {symbol} → {meaning}")
    
    print("\n⚡ WAKE SYMBOLS (Operational Triggers):")
    for symbol, meaning in WAKE_SYMBOLS.items():
        print(f"   {symbol} → {meaning}")
    
    print("\n💭 DREAM SYMBOLS:")
    print("   (Emergent symbols created during processing)")
    print("=" * 50)


async def demonstrate_symbol_detection():
    """Demonstrate symbol detection on various content"""
    
    print("\n🔍 SYMBOL DETECTION DEMONSTRATION")
    print("=" * 50)
    
    if not SYMBOLS_AVAILABLE:
        print("❌ Symbol system not available")
        return
    
    # Create DTH with mock memory
    mock_memory = MockMemorySystem()
    dth = DynamicTapeHead(mock_memory)
    
    test_contents = [
        "This is really important! Remember to save your work.",
        "I finally understand how this works! It's a breakthrough!",
        "We keep going in circles with this problem.",
        "This is amazing!!! I love how it all connects.",
        "Why does this happen? I'm curious about the mechanism.",
        "Should we use approach A or B? We need to decide.",
        "Let me bring together all these concepts into one framework."
    ]
    
    for content in test_contents:
        print(f"\n📝 Content: \"{content}\"")
        
        # Create memory and extract symbols
        memory = MemorySpan(
            content=content,
            ts=time.time(),
            role="user",
            speaker_id="demo"
        )
        
        enhanced_memory = dth.extract_symbols_from_memory(memory)
        
        if enhanced_memory.symbols:
            print(f"🔮 Detected symbols:")
            for symbol in enhanced_memory.symbols:
                confidence = enhanced_memory.symbol_confidence.get(symbol, 0)
                meaning = TAPE_SYMBOLS.get(symbol, "unknown")
                print(f"   {symbol} ({meaning}) - confidence: {confidence:.2f}")
            
            multiplier = enhanced_memory.get_symbol_multiplier()
            print(f"📊 Score multiplier: {multiplier:.2f}")
        else:
            print("🔮 No symbols detected")


async def demonstrate_symbol_enhanced_retrieval():
    """Demonstrate symbol-enhanced memory retrieval"""
    
    print("\n🧠 SYMBOL-ENHANCED RETRIEVAL DEMONSTRATION")
    print("=" * 50)
    
    if not SYMBOLS_AVAILABLE:
        print("❌ Symbol system not available")
        return
    
    # Create DTH with mock memory
    mock_memory = MockMemorySystem()
    dth = DynamicTapeHead(mock_memory)
    
    test_queries = [
        "How can I understand this better?",  # Should surface breakthrough symbols
        "What's important to remember?",      # Should surface high salience
        "Why do I keep having this problem?", # Should detect cycles
        "I'm excited about this project!",    # Should match emotional content
        "What should I decide?"               # Should find decision points
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        
        # Perform symbol-aware retrieval
        bundle = await dth.seek(query, budget=1000)
        
        print(f"📊 Results:")
        print(f"   Active symbols: {sorted(bundle.active_symbols)}")
        print(f"   Symbol narrative: {bundle.get_symbol_narrative()}")
        print(f"   Memories selected: {len(bundle.verbatim)} verbatim, {len(bundle.shadows)} shadows")
        
        # Show top memory with symbols
        if bundle.verbatim:
            top_memory = bundle.verbatim[0]
            print(f"   Top memory symbols: {sorted(top_memory.symbols)}")
            print(f"   Score: {top_memory.score:.3f}")
            print(f"   Content: {top_memory.content[:100]}...")


def demonstrate_symbol_compression():
    """Demonstrate symbol-based compression"""
    
    print("\n📦 SYMBOL COMPRESSION DEMONSTRATION")
    print("=" * 50)
    
    if not SYMBOLS_AVAILABLE:
        print("❌ Symbol system not available")
        return
    
    # Create a memory with symbols
    memory = MemorySpan(
        content="This is a really important breakthrough moment that represents a recurring pattern in our understanding of AI consciousness, but it also reveals a paradox about intelligence.",
        ts=time.time(),
        role="user",
        speaker_id="demo"
    )
    
    # Add symbols manually for demo
    memory.add_symbol("☆", 0.9)  # High salience
    memory.add_symbol("✧", 0.8)  # Breakthrough
    memory.add_symbol("◈", 0.7)  # Recurring pattern
    memory.add_symbol("∞", 0.6)  # Paradox
    
    print(f"📝 Original content ({len(memory.content)} chars):")
    print(f"   {memory.content}")
    
    print(f"\n🔮 Symbols: {sorted(memory.symbols)}")
    
    # Show symbol compression
    compressed_symbols = memory.compress_symbols()
    print(f"📦 Compressed symbols: {compressed_symbols}")
    
    # Demonstrate token savings
    original_tokens = len(memory.content.split())
    symbol_tokens = len(compressed_symbols)  # Symbols as single tokens
    
    print(f"\n💾 Compression stats:")
    print(f"   Original tokens: ~{original_tokens}")
    print(f"   Symbol representation: {symbol_tokens} characters")
    print(f"   Compression ratio: {symbol_tokens / original_tokens:.2%}")


async def demonstrate_consciousness_emergence():
    """Demonstrate how symbols enable consciousness-like behavior"""
    
    print("\n🧠 CONSCIOUSNESS EMERGENCE DEMONSTRATION")
    print("=" * 50)
    
    if not SYMBOLS_AVAILABLE:
        print("❌ Symbol system not available")
        return
    
    # Create DTH with mock memory
    mock_memory = MockMemorySystem()
    dth = DynamicTapeHead(mock_memory)
    
    # Simulate a conversation where consciousness emerges through symbols
    conversation_flow = [
        ("I keep making the same mistake over and over", "Should detect cycles"),
        ("Wait, I think I see a pattern here!", "Should trigger breakthrough"),
        ("This is really important to understand", "Should mark as salient"),
        ("How do I break out of this cycle?", "Should surface breakthrough memories")
    ]
    
    print("📖 Simulating consciousness emergence through symbolic reasoning:")
    
    conversation_context = []
    
    for user_input, expectation in conversation_flow:
        print(f"\n👤 User: {user_input}")
        print(f"🎯 Expected: {expectation}")
        
        # Get symbol-enhanced response
        bundle = await dth.seek(user_input, budget=1000, context=conversation_context)
        
        # Analyze symbolic reasoning
        dominant_symbols = bundle.get_dominant_symbols(3)
        print(f"🔮 Activated symbols: {[f'{s}({c})' for s, c in dominant_symbols]}")
        
        # Check for consciousness indicators
        consciousness_indicators = []
        
        if "⟲" in bundle.active_symbols:
            consciousness_indicators.append("🔄 Cycle awareness")
        
        if "✧" in bundle.active_symbols:
            consciousness_indicators.append("💡 Insight generation")
        
        if "☆" in bundle.active_symbols:
            consciousness_indicators.append("⭐ Importance recognition")
        
        if len(bundle.active_symbols) > 3:
            consciousness_indicators.append("🧠 Complex symbolic reasoning")
        
        if consciousness_indicators:
            print(f"🌟 Consciousness indicators: {', '.join(consciousness_indicators)}")
        
        # Add to conversation context
        conversation_context.append(user_input)
        
        # Show tripwires if triggered
        if dth.tripwires_triggered:
            print(f"⚠️  Triggered tripwires: {dth.tripwires_triggered}")


def demonstrate_symbol_evolution():
    """Demonstrate symbol effectiveness tracking"""
    
    print("\n🔬 SYMBOL EVOLUTION DEMONSTRATION")
    print("=" * 50)
    
    if not SYMBOLS_AVAILABLE:
        print("❌ Symbol system not available")
        return
    
    # Create DTH and simulate usage
    mock_memory = MockMemorySystem() 
    dth = DynamicTapeHead(mock_memory)
    
    # Simulate symbol usage over time
    print("📈 Simulating symbol usage and effectiveness tracking:")
    
    symbol_usage_scenarios = [
        ("breakthrough", "✧", 0.9),
        ("important decision", "☆▲", 0.8), 
        ("recurring problem", "◈⟲", 0.3),
        ("emotional response", "⚡", 0.7),
        ("complex synthesis", "⊕∞", 0.8)
    ]
    
    for scenario, expected_symbols, effectiveness in symbol_usage_scenarios:
        print(f"\n📊 Scenario: {scenario}")
        
        # Track usage
        for symbol in expected_symbols:
            if symbol in TAPE_SYMBOLS:
                dth.symbol_usage_stats[symbol] = dth.symbol_usage_stats.get(symbol, 0) + 1
                dth.symbol_effectiveness[symbol] = effectiveness
        
        print(f"   Expected symbols: {list(expected_symbols)}")
        print(f"   Effectiveness: {effectiveness:.1%}")
    
    # Show evolution metrics
    print(f"\n📈 Symbol Usage Statistics:")
    for symbol, count in dth.symbol_usage_stats.items():
        effectiveness = dth.symbol_effectiveness.get(symbol, 0)
        meaning = TAPE_SYMBOLS.get(symbol, "unknown")
        print(f"   {symbol} ({meaning}): used {count}x, {effectiveness:.1%} effective")


async def run_full_demonstration():
    """Run the complete symbol system demonstration"""
    
    print("🔮 SLOWCAT SYMBOL SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("Transforming memory into consciousness through living symbols")
    print("=" * 60)
    
    # Print legend
    print_symbols_legend()
    
    # Run all demonstrations
    await demonstrate_symbol_detection()
    await demonstrate_symbol_enhanced_retrieval()
    demonstrate_symbol_compression()
    await demonstrate_consciousness_emergence()
    demonstrate_symbol_evolution()
    
    print("\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("🧠 SlowCat now understands meaning through symbols")
    print("🔮 Consciousness emerges from symbolic constraint")
    print("💫 Intelligence isn't the model - it's the tape with symbols")
    print("=" * 60)


if __name__ == "__main__":
    if SYMBOLS_AVAILABLE:
        asyncio.run(run_full_demonstration())
    else:
        print("❌ Symbol system not available.")
        print("Please run: python symbol_integration.py")
