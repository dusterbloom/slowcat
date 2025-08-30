#!/usr/bin/env python3
"""
Debug the structure of knowledge records in SurrealDB
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_knowledge_structure():
    """Check the actual structure of knowledge records"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    # Check knowledge table structure
    print("🔍 KNOWLEDGE TABLE STRUCTURE")
    print("="*50)
    
    # Get sample records with all fields
    sample_query = "SELECT * FROM knowledge LIMIT 3"
    sample_result = await conn.db.query(sample_query)
    
    print(f"Total sample records: {len(sample_result) if sample_result else 0}")
    
    if sample_result:
        for i, record in enumerate(sample_result[:3], 1):
            print(f"\n[Record {i}] ID: {record.get('id', 'N/A')}")
            print(f"Available fields: {list(record.keys())}")
            
            # Check each field
            for field in ['subject', 'predicate', 'object', 'value', 'embedding', 'created_at', 'confidence']:
                if field in record:
                    value = record[field]
                    if field == 'embedding' and value:
                        print(f"  {field}: array of {len(value)} floats (first 3: {value[:3]})")
                    else:
                        print(f"  {field}: {value}")
                else:
                    print(f"  {field}: NOT PRESENT")
    
    # Check count of records with embeddings
    print(f"\n🔍 EMBEDDING STATISTICS")
    print("="*50)
    
    embed_count_query = "SELECT count() FROM knowledge WHERE embedding IS NOT NONE"
    embed_result = await conn.db.query(embed_count_query)
    if embed_result:
        print(f"Records with embeddings: {embed_result[0].get('count', 'unknown')}")
    
    total_count_query = "SELECT count() FROM knowledge"
    total_result = await conn.db.query(total_count_query)
    if total_result:
        print(f"Total knowledge records: {total_result[0].get('count', 'unknown')}")
    
    # Check HNSW index
    print(f"\n🔍 HNSW INDEX STATUS")
    print("="*50)
    
    try:
        index_query = "INFO FOR TABLE knowledge"
        index_result = await conn.db.query(index_query)
        if index_result:
            print(f"Table info: {index_result[0]}")
    except Exception as e:
        print(f"Could not get table info: {e}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_knowledge_structure())