#!/usr/bin/env python3
"""
Test vector functions with confirmed version compatibility
Server: 2.3.7, Client: 1.0.6 (should work together)
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def test_vector_functions_final():
    """Test vector functions with confirmed compatible versions"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    print("🔍 TESTING VECTOR FUNCTIONS (Server 2.3.7, Client 1.0.6)")
    print("="*60)
    
    # Get a sample embedding
    sample_query = "SELECT embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if not sample_result or not sample_result[0].get('embedding'):
        print("❌ No embeddings found")
        return
    
    query_embedding = sample_result[0]['embedding']
    
    # Test vector functions that should work with version 2.0+
    vector_tests = [
        {
            'name': 'vector::distance::knn() function',
            'query': 'SELECT id, predicate, vector::distance::knn() AS distance FROM knowledge WHERE embedding <|3|> $embedding'
        },
        
        {
            'name': 'vector::similarity::cosine() function',
            'query': 'SELECT id, predicate, vector::similarity::cosine(embedding, $embedding) AS similarity FROM knowledge WHERE embedding IS NOT NONE LIMIT 3'
        },
        
        {
            'name': 'vector::distance::cosine() function',
            'query': 'SELECT id, predicate, vector::distance::cosine(embedding, $embedding) AS distance FROM knowledge WHERE embedding IS NOT NONE LIMIT 3'
        },
        
        {
            'name': 'K-NN operator alone',
            'query': 'SELECT id, predicate FROM knowledge WHERE embedding <|3|> $embedding'
        },
        
        {
            'name': 'Simple embedding existence check',
            'query': 'SELECT count() FROM knowledge WHERE embedding IS NOT NONE'
        }
    ]
    
    for i, test in enumerate(vector_tests, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 40)
        
        try:
            if '$embedding' in test['query']:
                result = await conn.db.query(test['query'], {'embedding': query_embedding})
            else:
                result = await conn.db.query(test['query'])
            
            print(f"✅ SUCCESS - {len(result) if result else 0} results")
            
            if result:
                for j, record in enumerate(result[:3], 1):
                    print(f"  [{j}] {record}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    # If K-NN doesn't work, implement manual similarity
    print(f"\n🛠️  IMPLEMENTING MANUAL COSINE SIMILARITY")
    print("="*60)
    
    manual_similarity_query = """
    SELECT id, predicate,
           array::len(embedding) as dimensions,
           array::max(embedding) as max_val,
           array::min(embedding) as min_val
    FROM knowledge 
    WHERE embedding IS NOT NONE 
    AND array::max(embedding) != array::min(embedding)
    ORDER BY id
    LIMIT 5
    """
    
    try:
        records = await conn.db.query(manual_similarity_query)
        
        if records:
            print("Manual similarity calculation for records:")
            
            for record in records:
                rec_embedding = (await conn.db.query(
                    "SELECT embedding FROM knowledge WHERE id = $id", 
                    {'id': record['id']}
                ))[0]['embedding']
                
                # Manual cosine similarity: dot(a,b) / (||a|| * ||b||)
                # We'll approximate this using available functions
                
                dot_product = sum(a * b for a, b in zip(query_embedding[:50], rec_embedding[:50]))  # First 50 dims
                norm_query = sum(x * x for x in query_embedding[:50]) ** 0.5
                norm_record = sum(x * x for x in rec_embedding[:50]) ** 0.5
                
                cosine_sim = dot_product / (norm_query * norm_record) if norm_query * norm_record > 0 else 0
                
                print(f"  {record['id']}: {record['predicate']} -> similarity: {cosine_sim:.4f}")
                
    except Exception as e:
        print(f"Manual similarity error: {e}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(test_vector_functions_final())