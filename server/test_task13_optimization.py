#!/usr/bin/env python3
"""
Test Task-13 Optimization: Advanced SurrealDB and spaCy Usage

This tests the complete optimization pipeline:
1. Graph relationships with TYPE RELATION
2. Temporal extraction for events and dates
3. Coreference resolution across sentences
4. Entity resolution for deduplication
5. Graph traversal queries for complex information retrieval

The goal is to validate that queries like "When is my meeting with Sarah?" now work correctly.
"""

import asyncio
import sys
import os

# Add server to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.spacy_fact_extractor import extract_facts_from_text
from memory.surreal_memory import create_surreal_facts_graph
from memory.query_router import QueryRouter
from memory.coreference_resolver import resolve_coreferences
from memory.entity_resolver import resolve_entities_in_text, get_entity_resolver
from memory.temporal_extractor import extract_temporal_expressions, extract_events_from_text
from loguru import logger


async def test_complete_pipeline():
    """Test the complete Task-13 optimization pipeline"""
    
    logger.info("🧪 Testing Task-13 Advanced SurrealDB and spaCy Optimization")
    
    # Initialize components
    facts_graph = await create_surreal_facts_graph()
    query_router = QueryRouter(facts_graph=facts_graph)
    
    print("\n" + "="*80)
    print("TASK-13 OPTIMIZATION TEST")
    print("="*80)
    
    # Test cases that should now work with our improvements
    test_conversations = [
        # Meeting scenario with temporal and entity resolution
        [
            "I have a meeting with sarah tomorrow at 3 PM",
            "Sarah is a great project manager",
            "She mentioned the deadline is next Friday"
        ],
        
        # Pet scenario with coreference and entity resolution 
        [
            "My dog's name is Luna", 
            "She is a golden retriever",
            "Luna loves playing fetch in the park"
        ],
        
        # Complex personal facts
        [
            "I live in San Francisco",
            "My birthday is on December 15th", 
            "dr. smith is my doctor at UCSF"
        ]
    ]
    
    test_queries = [
        # These should now work with graph traversal
        "When is my meeting with Sarah?",
        "What's my dog's name?", 
        "Where do I live?",
        "When is my birthday?",
        "Who is my doctor?"
    ]
    
    # Store test conversations
    print("\n📝 STORING TEST CONVERSATIONS:")
    print("-" * 40)
    
    for i, conversation in enumerate(test_conversations, 1):
        print(f"\nConversation {i}:")
        for j, message in enumerate(conversation):
            print(f"  {j+1}. {message}")
            
            # Extract and store facts using optimized pipeline
            facts = extract_facts_from_text(message)
            print(f"     → Extracted {len(facts)} facts")
            
            # Store in facts graph  
            for fact in facts:
                await facts_graph.store_fact(fact)
    
    # Test individual components
    print("\n🔧 TESTING INDIVIDUAL COMPONENTS:")
    print("-" * 40)
    
    # Test 1: Coreference Resolution
    test_text = "Sarah went to the meeting. She presented the new proposal."
    resolved = resolve_coreferences(test_text)
    print(f"\n1. Coreference Resolution:")
    print(f"   Original: {test_text}")
    print(f"   Resolved: {resolved}")
    
    # Test 2: Entity Resolution
    entity_resolver = get_entity_resolver()
    canonical = entity_resolver.resolve_entity("dr. sarah smith", "PERSON")
    print(f"\n2. Entity Resolution:")
    print(f"   'dr. sarah smith' → '{canonical}'")
    
    # Test 3: Temporal Extraction
    temporal_text = "Meeting with John tomorrow at 3 PM"
    temporal_exprs = extract_temporal_expressions(temporal_text)
    events = extract_events_from_text(temporal_text)
    print(f"\n3. Temporal Extraction:")
    print(f"   Text: {temporal_text}")
    print(f"   Temporal expressions: {len(temporal_exprs)}")
    print(f"   Events: {len(events)}")
    
    # Test complex queries
    print("\n🔍 TESTING COMPLEX QUERIES:")
    print("-" * 40)
    
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            # Route query using optimized router
            response = await query_router.route_query(query)
            
            print(f"   Strategy: {response.strategy_used.value}")
            print(f"   Stores queried: {response.stores_queried}")
            print(f"   Results: {response.total_results}")
            print(f"   Time: {response.retrieval_time_ms:.1f}ms")
            
            if response.results:
                success_count += 1
                for j, result in enumerate(response.results[:2]):  # Show top 2
                    print(f"   Result {j+1}: {result.content[:100]}...")
            else:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Summary
    print(f"\n📊 RESULTS SUMMARY:")
    print("-" * 40)
    print(f"Successful queries: {success_count}/{len(test_queries)}")
    print(f"Success rate: {success_count/len(test_queries)*100:.1f}%")
    
    if success_count >= len(test_queries) * 0.8:  # 80% success rate
        print("✅ Task-13 optimization SUCCESSFUL!")
        print("   Advanced queries now work with graph relationships,")
        print("   temporal extraction, and entity resolution.")
    else:
        print("❌ Task-13 optimization needs improvement")
        print("   Some advanced queries still failing")
    
    print("\n" + "="*80)
    
    return success_count >= len(test_queries) * 0.8


async def main():
    """Run the complete test suite"""
    try:
        success = await test_complete_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())