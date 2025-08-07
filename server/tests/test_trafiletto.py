#!/usr/bin/env python3
"""Test the trafiletto tool locally"""

import asyncio
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.handlers import tool_handlers

async def test_trafiletto():
    print("🧹 Testing trafiletto tool with Hacker News...")
    
    result = await tool_handlers.trafiletto('https://news.ycombinator.com', max_chars=800)
    
    print('\n📊 Trafiletto result:')
    print(f'   Title: {result.get("title", "N/A")}')
    print(f'   URL: {result.get("url", "N/A")}')
    print(f'   Length: {result.get("length", 0)} chars')
    print(f'   Truncated: {result.get("truncated", False)}')
    print(f'   Max chars: {result.get("max_chars", 0)}')
    
    if "error" in result:
        print(f'❌ Error: {result["error"]}')
    else:
        print('\n📄 Text preview (first 300 chars):')
        text = result.get('text', '')
        preview = text[:300] + '...' if len(text) > 300 else text
        print(preview)
        
        print(f'\n✨ Success! Clean text extracted: {len(text)} characters')
        
        # Show the difference with old methods
        if len(text) < 3000:
            print("🎯 Voice-friendly length - perfect for speech responses!")
        else:
            print("⚠️ Still quite long, but much cleaner than raw HTML")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_trafiletto())