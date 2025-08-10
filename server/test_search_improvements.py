#!/usr/bin/env python3
"""
Quick test for search improvements
"""

import asyncio
import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from tools.web_search_free import search_web_free
from tools.formatters import format_tool_response_for_voice
from tools.text_formatter import sanitize_for_voice

async def test_search_improvements():
    """Test the improved search functionality"""
    print("🧪 Testing search improvements...")
    print("-" * 50)
    
    # Test 1: Search functionality with better filtering
    print("1️⃣ Testing search with movie query...")
    try:
        result = await search_web_free("latest news from London", num_results=5)
        
        print(f"Results found: {result.get('result_count', 0)}")
        print(f"Provider: {result.get('provider', 'Unknown')}")
        
        if result.get('results'):
            print("\n📋 Raw results:")
            for i, item in enumerate(result['results'][:2], 1):
                print(f"  {i}. {item.get('title', '')[:60]}...")
                print(f"     {item.get('snippet', '')[:80]}...")
        
        # Test 2: Voice summary formatting
        print(f"\n🎤 Voice summary:")
        print(f"'{result.get('voice_summary', 'No voice summary')[:150]}...'")
        
        # Test 3: Formatter function
        print(f"\n🔧 Formatter output:")
        formatted = format_tool_response_for_voice("search_web_free", result)
        print(f"'{formatted[:150]}...'")
        
        # Test 4: TTS sanitization
        print(f"\n🧹 TTS sanitized:")
        sanitized = sanitize_for_voice(formatted)
        print(f"'{sanitized[:150]}...'")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Search improvements test completed!")

if __name__ == "__main__":
    asyncio.run(test_search_improvements())