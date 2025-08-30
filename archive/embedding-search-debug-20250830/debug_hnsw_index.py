#!/usr/bin/env python3
"""
Debug HNSW index status and effectiveness
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_hnsw_index():
    """Check HNSW index status and try to fix K-NN search"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    print("🔍 HNSW INDEX DIAGNOSTICS")
    print("="*50)
    
    # Check total records with embeddings
    count_query = "SELECT count() FROM knowledge WHERE embedding IS NOT NONE"
    count_result = await conn.db.query(count_query)
    total_embeddings = count_result[0]['count'] if count_result else 0
    print(f"Total records with embeddings: {total_embeddings}")
    
    # Check index status
    try:
        info_query = "INFO FOR TABLE knowledge"
        info_result = await conn.db.query(info_query)
        if info_result:
            print(f"Knowledge table info: {info_result[0]}")
    except Exception as e:
        print(f"Could not get table info: {e}")
    
    # Check if embeddings are actually valid (not all zeros/same values)
    print(f"\n🔍 EMBEDDING VALIDATION")
    print("="*50)
    
    validation_query = """
    SELECT id, predicate,
           array::len(embedding) as dim,
           array::max(embedding) as max_val,
           array::min(embedding) as min_val
    FROM knowledge 
    WHERE embedding IS NOT NONE 
    LIMIT 5
    """
    
    validation_result = await conn.db.query(validation_query)
    if validation_result:
        for i, record in enumerate(validation_result, 1):
            print(f"  [{i}] ID: {record.get('id', 'N/A')}")
            print(f"      Predicate: {record.get('predicate', 'N/A')}")
            print(f"      Dimensions: {record.get('dim', 'N/A')}")
            print(f"      Max value: {record.get('max_val', 'N/A')}")
            print(f"      Min value: {record.get('min_val', 'N/A')}")
            
            # Check if embedding is all same values (invalid)
            max_val = record.get('max_val')
            min_val = record.get('min_val') 
            if max_val == min_val:
                print(f"      ⚠️  WARNING: All values are identical ({max_val})")
            else:
                print(f"      ✅ Valid embedding with range {min_val:.3f} to {max_val:.3f}")
    
    # Try recreating the HNSW index
    print(f"\n🔨 RECREATING HNSW INDEX")
    print("="*50)
    
    try:
        # Drop existing index
        drop_query = "REMOVE INDEX hnsw_embedding_idx ON knowledge"
        await conn.db.query(drop_query)
        print("✅ Dropped existing HNSW index")
    except Exception as e:
        print(f"Note: Could not drop index (may not exist): {e}")
    
    try:
        # Create new HNSW index
        create_query = "DEFINE INDEX hnsw_embedding_idx ON knowledge FIELDS embedding HNSW DIMENSION 384 DIST COSINE"
        await conn.db.query(create_query)
        print("✅ Created new HNSW index with 384 dimensions and COSINE distance")
    except Exception as e:
        print(f"❌ Could not create HNSW index: {e}")
    
    # Test K-NN again after index recreation
    print(f"\n🔍 TESTING K-NN AFTER INDEX RECREATION")
    print("="*50)
    
    # Get a sample embedding for testing
    sample_query = "SELECT embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if sample_result and sample_result[0].get('embedding'):
        query_embedding = sample_result[0]['embedding']
        
        # Test simple K-NN
        knn_query = "SELECT id, predicate FROM knowledge WHERE embedding <|3|> $embedding"
        
        try:
            knn_result = await conn.db.query(knn_query, {'embedding': query_embedding})
            print(f"K-NN search returned {len(knn_result) if knn_result else 0} results")
            
            if knn_result:
                for j, record in enumerate(knn_result[:3], 1):
                    print(f"  [{j}] {record.get('id')}: {record.get('predicate')}")
            else:
                print("Still no results - possible issue with embeddings or index")
                
        except Exception as e:
            print(f"K-NN search error: {e}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_hnsw_index())