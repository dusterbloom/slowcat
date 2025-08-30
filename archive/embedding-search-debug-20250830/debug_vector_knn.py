#!/usr/bin/env python3
"""
Debug script for K-nearest neighbors vector search in SurrealDB.
Test different syntaxes to understand what works and what doesn't.
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_knn_search():
    """Test different K-NN search syntaxes"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    # Get a sample embedding to search with
    sample_query = "SELECT embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if not sample_result or len(sample_result) == 0 or not sample_result[0].get('embedding'):
        print("❌ No embeddings found in database")
        return
    
    query_embedding = sample_result[0]['embedding']
    print(f"🔍 Using query embedding with {len(query_embedding)} dimensions")
    
    # Test different K-NN syntaxes
    test_cases = [
        # Basic K-NN operator
        {
            'name': 'Basic <|3|> operator',
            'query': 'SELECT id, subject, predicate, value, embedding <|3|> $embedding AS similarity FROM knowledge WHERE embedding IS NOT NONE'
        },
        
        # K-NN with distance function
        {
            'name': '<|3,COSINE|> with distance',
            'query': 'SELECT id, subject, predicate, value, embedding <|3,COSINE|> $embedding AS similarity FROM knowledge WHERE embedding IS NOT NONE'
        },
        
        # K-NN with literal embedding
        {
            'name': '<|3|> with literal embedding array',
            'query': f'SELECT id, subject, predicate, value, embedding <|3|> {query_embedding} AS similarity FROM knowledge WHERE embedding IS NOT NONE'
        },
        
        # Basic vector distance calculation
        {
            'name': 'Vector distance calculation',
            'query': 'SELECT id, subject, predicate, value, vector::distance::cosine(embedding, $embedding) AS distance FROM knowledge WHERE embedding IS NOT NONE LIMIT 3'
        },
        
        # Alternative distance syntax
        {
            'name': 'Alternative distance syntax',
            'query': 'SELECT id, subject, predicate, value, vector::distance(embedding, $embedding, "COSINE") AS distance FROM knowledge WHERE embedding IS NOT NONE LIMIT 3'
        },
        
        # Simple embedding check
        {
            'name': 'Embedding existence check',
            'query': 'SELECT count() as total_with_embeddings FROM knowledge WHERE embedding IS NOT NONE'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"Query: {test['query'][:100]}...")
        
        try:
            if '$embedding' in test['query']:
                result = await conn.db.query(test['query'], {'embedding': query_embedding})
            else:
                result = await conn.db.query(test['query'])
            
            print(f"✅ SUCCESS - Returned {len(result) if result else 0} results")
            
            if result:
                for j, record in enumerate(result[:3]):  # Show first 3 results
                    print(f"  [{j+1}] ID: {record.get('id', 'N/A')}")
                    if 'subject' in record:
                        print(f"      Fact: {record.get('subject')} {record.get('predicate')} {record.get('value')}")
                    if 'similarity' in record:
                        print(f"      Similarity: {record.get('similarity')}")
                    if 'distance' in record:
                        print(f"      Distance: {record.get('distance')}")
                    if 'total_with_embeddings' in record:
                        print(f"      Total records with embeddings: {record.get('total_with_embeddings')}")
            else:
                print("  📄 No results returned")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_knn_search())