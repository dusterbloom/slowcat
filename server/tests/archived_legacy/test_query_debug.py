#!/usr/bin/env python3
"""
Debug specific query routing issues with Potola and complex queries
"""

import asyncio
import os
import sys
from pathlib import Path

# Add server path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

async def test_query_routing():
    """Test specific problematic queries"""
    
    print("🔍 Query Routing Debug Test")
    print("=" * 50)
    
    try:
        from memory import create_smart_memory_system
        memory_system = create_smart_memory_system()
        
        # Connect via the surreal_memory inside the adapter
        if hasattr(memory_system, 'surreal_memory'):
            await memory_system.surreal_memory.connect()
        else:
            print("No surreal_memory found in adapter")
        
        # Test individual queries that were failing
        test_queries = [
            "dog",
            "Potola", 
            "my dog name",
            "I was wondering if you could recall any other details about my dog",
            "what is my dog's age",
            "tell me about Potola",
            "dog breed",
            "my pet"
        ]
        
        for query in test_queries:
            try:
                print(f"\n🔍 Testing: '{query}'")
                
                # Direct router test
                if hasattr(memory_system, 'query_router'):
                    # Check router type and use appropriate parameters
                    router = memory_system.query_router
                    if hasattr(router, '__class__') and 'SurrealQueryRouter' in str(router.__class__):
                        response = await router.route_query(
                            query=query,
                            context={"speaker_id": "peppi"}
                        )
                    else:
                        response = await router.route_query(
                            query=query,
                            context={"speaker_id": "peppi"},
                            max_results=10
                        )
                    
                    result_count = len(response.results) if response.results else 0
                    print(f"   Results: {result_count}")
                    print(f"   Strategy: {response.strategy}")
                    print(f"   Confidence: {response.confidence:.2f}")
                    
                    if response.results and len(response.results) > 0:
                        print(f"   Sample: {str(response.results[0])[:100]}...")
                    
                    # Show query classification details
                    if hasattr(memory_system.query_router, 'query_classifier'):
                        classifier = memory_system.query_router.query_classifier
                        classification = await classifier.classify_query_async(query)
                        print(f"   Intent: {classification.intent}")
                        print(f"   Entity Type: {classification.entity_type}")
                        print(f"   Is Question: {classification.is_question}")
                        
                else:
                    print("   ❌ No query router available")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Test direct facts and tape searches
        print(f"\n📚 Direct Store Tests")
        print("=" * 30)
        
        # Direct facts search
        try:
            # Access via surreal_memory
            facts_results = await memory_system.surreal_memory.search_facts("dog", limit=5)
            print(f"Direct facts search 'dog': {len(facts_results) if facts_results else 0} results")
            if facts_results:
                for i, fact in enumerate(facts_results[:2]):
                    print(f"   Facts[{i}]: {fact}")
        except Exception as e:
            print(f"Direct facts search error: {e}")
        
        # Direct tape search  
        try:
            tape_results = await memory_system.surreal_memory.search_tape("Potola", limit=5)
            print(f"Direct tape search 'Potola': {len(tape_results) if tape_results else 0} results")
            if tape_results:
                for i, tape in enumerate(tape_results[:2]):
                    print(f"   Tape[{i}]: {tape.content if hasattr(tape, 'content') else str(tape)[:100]}")
        except Exception as e:
            print(f"Direct tape search error: {e}")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_query_routing())