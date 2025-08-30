#!/usr/bin/env python3
"""
Debug K-NN with correct SurrealDB syntax based on documentation
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_corrected_knn():
    """Test K-NN with correct syntax"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    # Check SurrealDB version first
    print("🔍 CHECKING SURREALDB VERSION")
    print("="*50)
    try:
        version_query = "SELECT VERSION() as version"
        version_result = await conn.db.query(version_query)
        if version_result:
            print(f"SurrealDB Version: {version_result[0].get('version', 'unknown')}")
    except Exception as e:
        print(f"Could not get version: {e}")
    
    # Get a sample embedding
    sample_query = "SELECT embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if not sample_result or not sample_result[0].get('embedding'):
        print("❌ No embeddings found in database")
        return
    
    query_embedding = sample_result[0]['embedding']
    print(f"Using query embedding with {len(query_embedding)} dimensions")
    
    # Test corrected K-NN syntaxes
    test_cases = [
        {
            'name': 'HNSW index K-NN with EF parameter',
            'query': 'SELECT id, predicate, embedding <|3,40|> $embedding AS similarity FROM knowledge WHERE embedding IS NOT NONE'
        },
        
        {
            'name': 'Brute force K-NN with COSINE',  
            'query': 'SELECT id, predicate, embedding <|3,COSINE|> $embedding AS similarity FROM knowledge WHERE embedding IS NOT NONE'
        },
        
        {
            'name': 'vector::distance::knn() function (requires 2.0+)',
            'query': 'SELECT id, predicate, vector::distance::knn(embedding, $embedding, 3) AS distance FROM knowledge WHERE embedding IS NOT NONE LIMIT 3'
        },
        
        {
            'name': 'K-NN with ORDER BY for exact results',
            'query': '''
            SELECT id, predicate, embedding <|3,COSINE|> $embedding AS similarity 
            FROM knowledge 
            WHERE embedding IS NOT NONE 
            ORDER BY similarity DESC 
            LIMIT 3
            '''
        },
        
        {
            'name': 'HNSW K-NN with different EF values',
            'query': 'SELECT id, predicate, embedding <|5,100|> $embedding AS scores FROM knowledge WHERE embedding IS NOT NONE LIMIT 5'
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
                    
                    # Show similarity/distance scores
                    for field in ['similarity', 'scores', 'distance']:
                        if field in record:
                            value = record[field]
                            if isinstance(value, (int, float)):
                                print(f"      {field}: {value:.6f}")
                            else:
                                print(f"      {field}: {value}")
            else:
                print("  📄 No results returned")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_corrected_knn())