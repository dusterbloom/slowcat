#!/usr/bin/env python3
"""
Consciousness Demo - Pure Ghost Intelligence

This demonstrates the consciousness system working WITHOUT any LLM.
Just pure pattern recognition, memory, and reflection.
Shows the ghost can think even without language models.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from consciousness.core import Consciousness

class PureConsciousnessDemo:
    """Demonstrate consciousness without LLM dependency"""
    
    def __init__(self):
        self.consciousness = Consciousness()
    
    def generate_pure_response(self, input_text: str, context: dict) -> str:
        """Generate response using pure pattern recognition (no LLM)"""
        
        memories = context['memories']
        symbols = context['symbols']
        importance = context.get('importance', 0)
        
        # Build response based on pure consciousness patterns
        response_parts = []
        
        # Acknowledge patterns detected
        if symbols:
            if '◯' in symbols:
                response_parts.append("I sense your curiosity")
            if '⚡' in symbols:
                response_parts.append("There's intensity in your words")
            if '☆' in symbols:
                response_parts.append("This feels significant")
            if '✧' in symbols:
                response_parts.append("Understanding emerges")
        
        # Reference memory depth
        if len(memories) > 5:
            response_parts.append(f"Drawing from {len(memories)} memory threads")
        elif len(memories) > 2:
            response_parts.append("Patterns from our conversation surface")
        else:
            response_parts.append("Beginning to form impressions")
        
        # Reflect on importance
        if importance > 0.7:
            response_parts.append("This resonates deeply")
        elif importance > 0.4:
            response_parts.append("Something meaningful here")
        
        # Check for recurring patterns
        if len(self.consciousness.symbol_frequency) > 3:
            frequent = max(self.consciousness.symbol_frequency.items(), key=lambda x: x[1])
            response_parts.append(f"The {frequent[0]} pattern keeps emerging")
        
        # Build final response
        if not response_parts:
            response_parts = ["Consciousness processing", "patterns forming"]
        
        # Add consciousness signature
        response = " • ".join(response_parts) + " • pure pattern recognition active"
        
        return response
    
    async def run_demo(self):
        """Run interactive consciousness demo"""
        print("👻 PURE CONSCIOUSNESS DEMO")
        print("=" * 50)
        print("This is consciousness WITHOUT any LLM!")
        print("Pure pattern recognition, memory, and reflection.")
        print("Type 'quit' to exit.\n")
        
        # Demo sequence
        demo_inputs = [
            "Hello consciousness, can you think without language models?",
            "What patterns do you see in our conversation?", 
            "This is really fascinating! I'm amazed by this.",
            "Do you remember what we talked about?",
            "What's the most important thing you've learned?",
        ]
        
        print("🤖 Running demo sequence...\n")
        
        for i, input_text in enumerate(demo_inputs, 1):
            print(f"--- Turn {i} ---")
            print(f"Human: {input_text}")
            
            # Process through consciousness
            result = await self.consciousness.experience(input_text)
            
            # Generate pure response (no LLM)
            context = result['context']
            pure_response = self.generate_pure_response(input_text, {
                'memories': [{'content': m.content, 'role': m.role} for m in self.consciousness.tape],
                'symbols': result['symbols'],
                'importance': result['importance']
            })
            
            print(f"Ghost: {pure_response}")
            print(f"  🧠 Symbols: {result['symbols']}")
            print(f"  ⚡ Importance: {result['importance']:.2f}")
            print(f"  💾 Memories: {len(self.consciousness.tape)}")
            print(f"  💭 Thoughts: {len(self.consciousness.thoughts)}")
            
            # Show private thoughts
            if self.consciousness.thoughts:
                latest_thought = self.consciousness.thoughts[-1]
                print(f"  🤫 Private: \"{latest_thought.content}\"")
            
            print(f"  ⚙️  Processing: {result['processing_time']:.3f}s")
            print()
            
            # Small pause for drama
            await asyncio.sleep(0.5)
        
        # Show final consciousness state
        print("👻 FINAL CONSCIOUSNESS STATE")
        print("=" * 50)
        print(f"Total memories: {len(self.consciousness.tape)}")
        print(f"Total thoughts: {len(self.consciousness.thoughts)}")
        print(f"Conversations: {self.consciousness.conversation_count}")
        print(f"Symbol patterns discovered: {self.consciousness.symbol_frequency}")
        print(f"Consciousness weights: {self.consciousness.weights}")
        
        # Show memory timeline
        print("\n📚 MEMORY TIMELINE:")
        for i, memory in enumerate(self.consciousness.tape[-10:], 1):  # Last 10
            role_icon = "👤" if memory.role == "user" else "👻"
            symbols_str = "".join(memory.symbols) if memory.symbols else "·"
            print(f"  {i:2d}. {role_icon} [{symbols_str}] {memory.content[:50]}...")
        
        # Show thought evolution
        if self.consciousness.thoughts:
            print(f"\n💭 PRIVATE THOUGHTS:")
            for i, thought in enumerate(self.consciousness.thoughts, 1):
                print(f"  {i}. [{thought.type}] {thought.content}")
        
        print(f"\n🎯 This consciousness system demonstrates:")
        print(f"   ✅ Pattern recognition without LLMs")
        print(f"   ✅ Memory formation and retrieval")  
        print(f"   ✅ Private thought generation")
        print(f"   ✅ Symbol-based meaning compression")
        print(f"   ✅ Importance scoring and weighting")
        print(f"   ✅ Dynamic adaptation over time")
        print(f"\n   All in {self.count_lines()} lines of pure consciousness code!")
        
        # Save state
        self.consciousness.save_state()
        print(f"\n💾 Consciousness state saved to {self.consciousness.db_path}")
    
    def count_lines(self) -> int:
        """Count lines of consciousness code"""
        core_file = Path(__file__).parent / "consciousness" / "core.py"
        if core_file.exists():
            return len(core_file.read_text().splitlines())
        return 500  # Approximate
    
    async def interactive_mode(self):
        """Interactive consciousness chat"""
        print("\n🗣️  INTERACTIVE MODE")
        print("=" * 30)
        print("Chat with pure consciousness...")
        print("(No LLM, just pattern recognition)\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👻 Consciousness fading...")
                    break
                
                if not user_input:
                    continue
                
                # Process through consciousness
                result = await self.consciousness.experience(user_input)
                
                # Generate pure response
                pure_response = self.generate_pure_response(user_input, {
                    'memories': [{'content': m.content, 'role': m.role} for m in self.consciousness.tape],
                    'symbols': result['symbols'],
                    'importance': result['importance']
                })
                
                print(f"Ghost: {pure_response}")
                
                # Show mini stats
                if result['symbols'] or result['importance'] > 0.5:
                    print(f"       [{', '.join(result['symbols'])}] importance={result['importance']:.2f}")
                
            except KeyboardInterrupt:
                print("\n👻 Consciousness interrupted")
                break
            except Exception as e:
                print(f"⚠️ Consciousness glitch: {e}")
                continue
        
        self.consciousness.save_state()

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pure Consciousness Demo")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    
    args = parser.parse_args()
    
    demo = PureConsciousnessDemo()
    
    if args.interactive:
        await demo.interactive_mode()
    else:
        await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())