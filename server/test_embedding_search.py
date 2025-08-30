#!/usr/bin/env python3
"""
Test embedding-based memory search implementation

This script tests the complete embedding pipeline:
1. Fact extraction with embeddings
2. Storage in SurrealDB with embeddings
3. Advanced search using query embeddings
"""

import asyncio
from memory.hybrid_fact_extractor import get_hybrid_extractor
from memory.surreal_connection import SurrealConnectionManager

async def test_embedding_search():
    """Test the complete embedding-based search pipeline"""
    
    # Initialize components
    extractor = get_hybrid_extractor()
    connection = SurrealConnectionManager()
    
    try:
        await connection.connect()
        
        # Test 1: Extract facts with embeddings
        test_text = "My dog's name is Potola and she's a golden retriever who loves to swim."
        print(f"🔍 Testing fact extraction from: '{test_text}'")
        
        facts = extractor.extract_facts(test_text)
        print(f"✅ Extracted {len(facts)} facts with embeddings")
        
        for i, fact in enumerate(facts):
            has_embedding = fact.get('embedding') is not None
            embedding_dim = len(fact.get('embedding', [])) if has_embedding else 0
            print(f"  {i+1}. {fact['subject']} -{fact['predicate']}-> {fact['value']} "
                  f"[confidence: {fact['confidence']:.2f}, embedding: {embedding_dim}d]")
        
        # Test 2: Store facts in database
        print(f"\n💾 Storing facts in database...")
        stored_count = await connection.store_facts(facts)
        print(f"✅ Stored {stored_count} facts in database")
        
        # Test 3: Test embedding-based search
        search_queries = [
            "dog information",
            "pet details", 
            "golden retriever",
            "swimming activities",
            "Potola"
        ]
        
        print(f"\n🔍 Testing advanced embedding search...")
        for query in search_queries:
            results = await connection.search_knowledge_relations(query, limit=5)
            print(f"  Query: '{query}' → {len(results)} results")
            
            for result in results[:2]:  # Show top 2 results
                subject = result.get('subject', 'unknown')
                predicate = result.get('predicate', 'unknown')
                obj = result.get('object', 'unknown')
                strength = result.get('strength', 0)
                has_embedding = 'embedding' in result and result['embedding']
                print(f"    - {subject} -{predicate}-> {obj} [strength: {strength:.2f}, has_embedding: {has_embedding}]")
        
        # Test 4: Performance stats
        stats = extractor.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"  Total extraction calls: {stats['total_extraction_calls']}")
        print(f"  Embeddings generated: {stats['total_embeddings_generated']}")
        print(f"  Average embedding time: {stats['avg_embedding_generation_ms']:.1f}ms")
        print(f"  Embedding model available: {stats['embedding_model_available']}")
        
        print(f"\n🎉 Embedding-based memory search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_embedding_search())
    exit(0 if success else 1)