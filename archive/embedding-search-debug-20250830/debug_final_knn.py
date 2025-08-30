#!/usr/bin/env python3
"""
Debug K-NN with CORRECT syntax from official SurrealDB docs
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_final_knn():
    """Test K-NN with official documented syntax"""
    
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
    
    # Test OFFICIAL K-NN syntaxes from SurrealDB docs
    test_cases = [
        {
            'name': 'Official K-NN syntax with distance function',
            'query': '''
            SELECT id, predicate, vector::distance::knn() as distance
            FROM knowledge 
            WHERE embedding <|3|> $embedding
            '''
        },
        
        {
            'name': 'K-NN without distance function',
            'query': 'SELECT id, predicate FROM knowledge WHERE embedding <|3|> $embedding'
        },
        
        {
            'name': 'K-NN with more results', 
            'query': '''
            SELECT id, predicate, vector::distance::knn() as distance
            FROM knowledge 
            WHERE embedding <|5|> $embedding
            '''
        },
        
        {
            'name': 'K-NN with ORDER BY distance',
            'query': '''
            SELECT id, predicate, vector::distance::knn() as distance
            FROM knowledge 
            WHERE embedding <|3|> $embedding
            ORDER BY distance ASC
            '''
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['name']}")
        print(f"{'='*60}")
        print(f"Query: {test['query'].strip()[:80]}...")
        
        try:
            result = await conn.db.query(test['query'], {'embedding': query_embedding})
            
            print(f"✅ SUCCESS - Returned {len(result) if result else 0} results")
            
            if result:
                for j, record in enumerate(result[:5], 1):
                    print(f"  [{j}] ID: {record.get('id', 'N/A')}")
                    print(f"      Predicate: {record.get('predicate', 'N/A')}")
                    
                    if 'distance' in record:
                        distance = record['distance']
                        if isinstance(distance, (int, float)):
                            print(f"      Distance: {distance:.6f}")
                        else:
                            print(f"      Distance: {distance}")
            else:
                print("  📄 No results returned")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_final_knn())