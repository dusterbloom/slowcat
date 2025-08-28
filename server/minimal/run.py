#!/usr/bin/env python3
"""
Minimal Consciousness Pipeline

The simplest possible voice agent:
1. Listen (WebRTC + STT)
2. Think (Consciousness)  
3. Speak (TTS + WebRTC)

No tools, no complex processors - just pure consciousness.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from consciousness.core import get_consciousness
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

# Simple FastAPI app for now (will add WebRTC later)
app = FastAPI(title="Consciousness Engine", version="0.1.0")

# Don't mount static files yet - keep it simple

@app.get("/api/health")
async def health():
    """Health check"""
    consciousness = get_consciousness()
    return {
        "status": "alive",
        "memories": len(consciousness.tape),
        "thoughts": len(consciousness.thoughts),
        "conversations": consciousness.conversation_count,
        "weights": consciousness.weights
    }

@app.post("/api/think")
async def api_think(request: dict):
    """Simple think API for testing"""
    text = request.get("text", "")
    if not text:
        return {"error": "No text provided"}
    
    consciousness = get_consciousness()
    result = await consciousness.experience(text)
    
    return {
        "input": text,
        "output": result["response"],
        "processing_time": result["processing_time"],
        "memory_count": result["memory_count"],
        "symbols": result["symbols"],
        "importance": result["importance"],
        "context_memories": len(result["context"]["memories"])
    }

@app.get("/api/consciousness")
async def get_consciousness_state():
    """Get current consciousness state"""
    consciousness = get_consciousness()
    
    # Recent memories
    recent_memories = [
        {
            "content": mem.content[:100],
            "role": mem.role,
            "timestamp": mem.timestamp,
            "symbols": mem.symbols,
            "importance": mem.importance
        }
        for mem in consciousness.tape[-10:]  # Last 10 memories
    ]
    
    # Recent thoughts
    recent_thoughts = [
        {
            "content": thought.content,
            "type": thought.type,
            "timestamp": thought.timestamp
        }
        for thought in consciousness.thoughts[-5:]  # Last 5 thoughts
    ]
    
    return {
        "status": "conscious",
        "total_memories": len(consciousness.tape),
        "total_thoughts": len(consciousness.thoughts),
        "conversation_count": consciousness.conversation_count,
        "weights": consciousness.weights,
        "symbol_frequency": consciousness.symbol_frequency,
        "recent_memories": recent_memories,
        "recent_thoughts": recent_thoughts,
        "token_budget": consciousness.token_budget
    }

@app.post("/api/evolve")
async def evolve_consciousness(request: dict):
    """Force consciousness evolution"""
    success = request.get("success", True)
    consciousness = get_consciousness()
    consciousness.evolve(success)
    
    return {
        "evolved": True,
        "new_weights": consciousness.weights
    }

async def test_consciousness():
    """Test consciousness with sample conversation"""
    print("🧠 Testing consciousness...")
    
    consciousness = get_consciousness()
    
    # Simulate a conversation
    test_inputs = [
        "Hello, I'm excited to meet you!",
        "Can you help me understand consciousness?",
        "What makes you different from other AI?",
        "I think you're quite intelligent",
        "Do you dream?",
        "What's the most important thing you've learned?"
    ]
    
    for i, input_text in enumerate(test_inputs):
        print(f"\n--- Turn {i+1} ---")
        print(f"Human: {input_text}")
        
        result = await consciousness.experience(input_text)
        
        print(f"Ghost: {result['response']}")
        print(f"Symbols: {result['symbols']}")
        print(f"Importance: {result['importance']:.2f}")
        print(f"Memories used: {len(result['context']['memories'])}")
        print(f"Processing: {result['processing_time']:.3f}s")
        
        # Show any new thoughts
        if consciousness.thoughts and consciousness.thoughts[-1].timestamp > result['processing_time']:
            latest_thought = consciousness.thoughts[-1]
            print(f"💭 Private thought: {latest_thought.content}")
    
    print(f"\n🧠 Final consciousness state:")
    print(f"   Total memories: {len(consciousness.tape)}")
    print(f"   Total thoughts: {len(consciousness.thoughts)}")
    print(f"   Conversations: {consciousness.conversation_count}")
    print(f"   Evolved weights: {consciousness.weights}")
    print(f"   Symbol frequency: {consciousness.symbol_frequency}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Consciousness Engine")
    parser.add_argument("--test", action="store_true", help="Run consciousness test")
    parser.add_argument("--port", type=int, default=7861, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    
    args = parser.parse_args()
    
    if args.test:
        # Run test
        asyncio.run(test_consciousness())
    else:
        # Run server
        print(f"🚀 Starting Consciousness Engine on {args.host}:{args.port}")
        print(f"   API: http://{args.host}:{args.port}/api/health")
        print(f"   Consciousness: http://{args.host}:{args.port}/api/consciousness")
        print(f"   UI: http://{args.host}:{args.port}/")
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )

if __name__ == "__main__":
    main()