#!/usr/bin/env python3
"""
Test end-to-end semantic search functionality
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def test_semantic_search():
    """Test semantic search with various queries"""
    
    conn = SurrealConnectionManager()
    await conn.connect()
    
    print("🔍 SEMANTIC SEARCH VALIDATION")
    print("="*50)
    
    # Test queries that should show semantic understanding
    test_queries = [
        "dog pet animal companion",
        "job work career profession", 
        "time temporal when date",
        "user person human individual",
        "cat feline kitty"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Use the search_knowledge_relations method directly
            results = await conn.search_knowledge_relations(query, limit=3)
            
            print(f"Found {len(results)} results:")
            
            for j, result in enumerate(results, 1):
                # Extract information from the result
                record_id = result.get('id', 'N/A')
                predicate = result.get('predicate', 'N/A')
                subject = result.get('subject', 'N/A') 
                obj = result.get('object', 'N/A')
                vector_score = result.get('vector_score', 0)
                combined_score = result.get('combined_score', 0)
                
                print(f"  [{j}] {record_id}")
                print(f"      Relation: {subject} --{predicate}--> {obj}")
                print(f"      Vector similarity: {vector_score:.4f}")
                print(f"      Combined score: {combined_score:.4f}")
                
        except Exception as e:
            print(f"❌ Error with query '{query}': {e}")
    
    # Test specific semantic relationships 
    print(f"\n🧠 SEMANTIC RELATIONSHIP TESTS")
    print("="*50)
    
    semantic_tests = [
        ("pet", "Should find animal-related facts"),
        ("work", "Should find job/career facts"),
        ("friend", "Should find social relationships"),
        ("home", "Should find location/place facts")
    ]
    
    for query, description in semantic_tests:
        print(f"\n🔍 Testing: {query} ({description})")
        
        try:
            results = await conn.search_knowledge_relations(query, limit=2)
            
            if results:
                print(f"✅ Found {len(results)} semantically related results:")
                for result in results:
                    predicate = result.get('predicate', 'N/A')
                    vector_score = result.get('vector_score', 0)
                    print(f"  • {predicate} (similarity: {vector_score:.4f})")
            else:
                print("  📄 No results found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(test_semantic_search())