#!/usr/bin/env python3
"""
Debug available vector/array functions in SurrealDB
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_vector_functions():
    """Test what vector functions are available"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    print("🔍 TESTING AVAILABLE VECTOR/ARRAY FUNCTIONS")
    print("="*60)
    
    # Get a sample embedding
    sample_query = "SELECT embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if not sample_result or not sample_result[0].get('embedding'):
        print("❌ No embeddings found in database")
        return
    
    query_embedding = sample_result[0]['embedding']
    print(f"Using embedding with {len(query_embedding)} dimensions")
    
    # Test available array functions
    array_functions = [
        'array::at',
        'array::len', 
        'array::length',
        'array::sum',
        'array::max',
        'array::min',
        'array::add',
        'array::concat',
        'array::slice',
        'math::dot',
        'math::cosine',
        'math::euclidean',
        'vector::add',
        'vector::distance',
        'vector::similarity',
        'vector::cosine',
        'vector::dot'
    ]
    
    for func in array_functions:
        print(f"\nTesting {func}:")
        
        try:
            if func in ['array::at']:
                # Test array::at with index
                test_query = f"SELECT {func}(embedding, 0) AS result FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
            elif func in ['array::len', 'array::length', 'array::sum', 'array::max', 'array::min']:
                # Test unary array functions
                test_query = f"SELECT {func}(embedding) AS result FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
            else:
                # Test binary functions with query embedding
                test_query = f"SELECT {func}(embedding, $embedding) AS result FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
                
            if '$embedding' in test_query:
                result = await conn.db.query(test_query, {'embedding': query_embedding})
            else:
                result = await conn.db.query(test_query)
                
            if result and result[0].get('result') is not None:
                res_val = result[0]['result']
                if isinstance(res_val, float):
                    print(f"  ✅ {func}: {res_val:.6f}")
                else:
                    print(f"  ✅ {func}: {res_val}")
            else:
                print(f"  ❓ {func}: No result")
                
        except Exception as e:
            error_msg = str(e)
            if 'Invalid function' in error_msg:
                print(f"  ❌ {func}: Function not available")
            else:
                print(f"  ❌ {func}: {error_msg[:50]}...")
    
    # Test if HNSW index is actually being used
    print(f"\n🔍 TESTING HNSW INDEX USAGE")
    print("="*60)
    
    try:
        # Check if index exists
        index_check = "SHOW INDEX ON knowledge"
        index_result = await conn.db.query(index_check)
        print(f"Indexes on knowledge table: {index_result}")
    except Exception as e:
        print(f"Could not check indexes: {e}")
    
    # Test direct HNSW query if possible  
    try:
        hnsw_query = "SELECT * FROM knowledge WHERE embedding @hnsw@ $embedding LIMIT 3"
        hnsw_result = await conn.db.query(hnsw_query, {'embedding': query_embedding})
        print(f"Direct HNSW query returned {len(hnsw_result)} results")
    except Exception as e:
        print(f"Direct HNSW query failed: {e}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_vector_functions())