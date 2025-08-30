#!/usr/bin/env python3
"""
Debug vector similarity score calculation in SurrealDB
Test different approaches to get actual similarity scores
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_similarity_scores():
    """Test different ways to get similarity scores from K-NN"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    # Get a sample embedding
    sample_query = "SELECT embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if not sample_result or not sample_result[0].get('embedding'):
        print("❌ No embeddings found in database")
        return
    
    query_embedding = sample_result[0]['embedding']
    print(f"🔍 Using query embedding with {len(query_embedding)} dimensions")
    
    # Test different similarity score approaches
    test_cases = [
        {
            'name': 'K-NN operator as field selector',
            'query': 'SELECT id, predicate, (embedding <|3|> $embedding) AS similarity FROM knowledge WHERE embedding IS NOT NONE'
        },
        
        {
            'name': 'K-NN operator with parentheses', 
            'query': 'SELECT id, predicate, embedding, (embedding <|3|> $embedding) AS scores FROM knowledge WHERE embedding IS NOT NONE'
        },
        
        {
            'name': 'K-NN as WHERE condition',
            'query': 'SELECT id, predicate FROM knowledge WHERE embedding <|3|> $embedding'
        },
        
        {
            'name': 'Simple array dot product',
            'query': 'SELECT id, predicate, array::dot(embedding, $embedding) AS dot_product FROM knowledge WHERE embedding IS NOT NONE LIMIT 3'
        },
        
        {
            'name': 'Manual cosine similarity calculation',
            'query': '''
            SELECT id, predicate, 
                   array::dot(embedding, $embedding) / 
                   (array::length(embedding) * array::length($embedding)) AS cosine_sim 
            FROM knowledge WHERE embedding IS NOT NONE LIMIT 3
            '''
        },
        
        {
            'name': 'Vector magnitude calculation',
            'query': '''
            SELECT id, predicate, 
                   array::dot(embedding, embedding) AS self_dot,
                   array::dot($embedding, $embedding) AS query_dot,
                   array::dot(embedding, $embedding) AS cross_dot
            FROM knowledge WHERE embedding IS NOT NONE LIMIT 3
            '''
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"Query: {test['query'][:80]}...")
        
        try:
            result = await conn.db.query(test['query'], {'embedding': query_embedding})
            
            print(f"✅ SUCCESS - Returned {len(result) if result else 0} results")
            
            if result:
                for j, record in enumerate(result[:3], 1):
                    print(f"  [{j}] ID: {record.get('id', 'N/A')}")
                    print(f"      Predicate: {record.get('predicate', 'N/A')}")
                    
                    # Show all similarity/score fields
                    for field in ['similarity', 'scores', 'dot_product', 'cosine_sim', 'self_dot', 'query_dot', 'cross_dot']:
                        if field in record:
                            value = record[field]
                            if isinstance(value, (list, tuple)) and len(value) > 3:
                                print(f"      {field}: [{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}, ...] ({len(value)} total)")
                            else:
                                print(f"      {field}: {value}")
            else:
                print("  📄 No results returned")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_similarity_scores())