#!/usr/bin/env python3
"""
Debug manual vector similarity calculation without relying on HNSW
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_manual_similarity():
    """Try manual similarity calculation to test basic vector operations"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    print("🔍 MANUAL VECTOR SIMILARITY TEST")
    print("="*50)
    
    # Get a sample embedding to query against
    sample_query = "SELECT id, predicate, embedding FROM knowledge WHERE embedding IS NOT NONE AND array::max(embedding) != array::min(embedding) LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if not sample_result or not sample_result[0].get('embedding'):
        print("❌ No valid embeddings found")
        return
    
    query_record = sample_result[0]
    query_embedding = query_record['embedding']
    query_id = query_record['id']
    
    print(f"Using query record: {query_id} ({query_record['predicate']})")
    print(f"Query embedding: {len(query_embedding)} dimensions, range {min(query_embedding):.3f} to {max(query_embedding):.3f}")
    
    # Test manual dot product calculation using available functions
    print(f"\n🔍 MANUAL DOT PRODUCT CALCULATION")
    print("="*50)
    
    # Since we don't have array::dot, let's try to build it manually
    # We can use array::add to multiply corresponding elements, then sum
    dot_product_query = """
    SELECT id, predicate,
           array::len(embedding) as dims,
           array::max(embedding) as max_val,
           array::min(embedding) as min_val
    FROM knowledge 
    WHERE embedding IS NOT NONE 
    AND array::max(embedding) != array::min(embedding)
    LIMIT 3
    """
    
    dot_result = await conn.db.query(dot_product_query)
    
    if dot_result:
        print("Found valid embeddings for comparison:")
        for i, record in enumerate(dot_result, 1):
            print(f"  [{i}] {record.get('id')}: {record.get('predicate')}")
            print(f"      Range: {record.get('min_val'):.3f} to {record.get('max_val'):.3f}")
    
    # Try different K-NN approaches
    print(f"\n🔍 ALTERNATIVE K-NN APPROACHES")
    print("="*50)
    
    test_approaches = [
        {
            'name': 'K-NN with different query embeddings',
            'query': 'SELECT id, predicate FROM knowledge WHERE embedding <|10|> $embedding',
            'use_different_embedding': True
        },
        
        {
            'name': 'K-NN without HNSW index (drop index first)',
            'query': 'SELECT id, predicate FROM knowledge WHERE embedding <|3|> $embedding',
            'drop_index': True
        },
        
        {
            'name': 'Simple nearest record without K-NN',
            'query': 'SELECT id, predicate, embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 5',
            'manual_compare': True
        }
    ]
    
    for i, test in enumerate(test_approaches, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 40)
        
        try:
            if test.get('drop_index'):
                await conn.db.query("REMOVE INDEX hnsw_embedding_idx ON knowledge")
                print("Dropped HNSW index for this test")
            
            if test.get('use_different_embedding'):
                # Create a simple test embedding 
                test_embedding = [0.1] * 384
                result = await conn.db.query(test['query'], {'embedding': test_embedding})
            elif test.get('manual_compare'):
                result = await conn.db.query(test['query'])
                # For manual comparison, just show the records
                if result:
                    print(f"Found {len(result)} records to compare against:")
                    for j, record in enumerate(result[:3], 1):
                        rec_embedding = record.get('embedding', [])
                        if rec_embedding:
                            similarity = sum(a * b for a, b in zip(query_embedding[:10], rec_embedding[:10]))  # First 10 dims
                            print(f"  [{j}] {record.get('id')}: {record.get('predicate')} (partial similarity: {similarity:.3f})")
                continue
            else:
                result = await conn.db.query(test['query'], {'embedding': query_embedding})
            
            print(f"Results: {len(result) if result else 0} records")
            
            if result:
                for j, record in enumerate(result[:3], 1):
                    print(f"  [{j}] {record.get('id')}: {record.get('predicate')}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_manual_similarity())