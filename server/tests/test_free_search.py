#!/usr/bin/env python3
"""
Test the free web search implementation
"""

import asyncio
import sys
from loguru import logger

# Add parent directory to path
sys.path.insert(0, '.')

from tools.web_search_free import search_web_free

async def test_search():
    """Test various search queries"""
    
    test_queries = [
        "Python programming tutorials",
        "Latest AI news 2025",
        "How to make sourdough bread",
        "Weather in San Francisco"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print('='*60)
        
        try:
            result = await search_web_free(query, num_results=3)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Provider: {result.get('provider', 'Unknown')}")
                print(f"📊 Results found: {result.get('result_count', 0)}")
                print("\n📱 UI Formatted:")
                print(result.get('ui_formatted', 'No UI format'))
                print("\n🎤 Voice Summary:")
                print(result.get('voice_summary', 'No voice summary'))
                
                if result.get('results'):
                    print("\n📋 Raw Results:")
                    for i, r in enumerate(result['results'][:3], 1):
                        print(f"  {i}. {r.get('title', 'No title')}")
                        print(f"     URL: {r.get('url', 'No URL')}")
                        print(f"     Snippet: {r.get('snippet', 'No snippet')[:100]}...")
        
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()
        
        # Small delay between searches to be respectful
        await asyncio.sleep(1)

if __name__ == "__main__":
    print("🔍 Testing Free Web Search Implementation")
    print("This will test multiple search providers without API keys\n")
    
    asyncio.run(test_search())
    
    print("\n✅ Test complete!")