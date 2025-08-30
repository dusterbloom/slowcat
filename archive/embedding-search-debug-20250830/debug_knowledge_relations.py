#!/usr/bin/env python3
"""
Debug knowledge relations and understand the in/out structure
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def debug_knowledge_relations():
    """Check the relationship structure in knowledge table"""
    
    # Initialize connection
    conn = SurrealConnectionManager()
    await conn.connect()
    
    print("🔍 KNOWLEDGE RELATIONS STRUCTURE")
    print("="*50)
    
    # Get records with in/out relationships
    relation_query = "SELECT *, ->in as subject_entity, ->out as object_entity FROM knowledge LIMIT 5"
    relation_result = await conn.db.query(relation_query)
    
    if relation_result:
        for i, record in enumerate(relation_result[:5], 1):
            print(f"\n[Relation {i}] ID: {record.get('id', 'N/A')}")
            print(f"  Predicate: {record.get('predicate', 'N/A')}")
            print(f"  In field: {record.get('in', 'N/A')}")  
            print(f"  Out field: {record.get('out', 'N/A')}")
            print(f"  Subject entity: {record.get('subject_entity', 'N/A')}")
            print(f"  Object entity: {record.get('object_entity', 'N/A')}")
            print(f"  Has embedding: {len(record.get('embedding', [])) > 0}")
    
    # Let's also check if there are entity records
    print(f"\n🔍 CHECKING FOR ENTITY RECORDS")
    print("="*50)
    
    entities_query = "SELECT * FROM entities LIMIT 3"
    try:
        entities_result = await conn.db.query(entities_query)
        if entities_result:
            print(f"Found {len(entities_result)} entity records:")
            for entity in entities_result[:3]:
                print(f"  Entity ID: {entity.get('id', 'N/A')}")
                print(f"  Name: {entity.get('name', 'N/A')}")
                print(f"  Fields: {list(entity.keys())}")
        else:
            print("No entity records found")
    except Exception as e:
        print(f"No entities table or error: {e}")
    
    # Test a simpler K-NN query with correct field names
    print(f"\n🔍 TESTING CORRECTED K-NN QUERY")
    print("="*50)
    
    # Get a sample embedding
    sample_query = "SELECT embedding FROM knowledge WHERE embedding IS NOT NONE LIMIT 1"
    sample_result = await conn.db.query(sample_query)
    
    if sample_result and sample_result[0].get('embedding'):
        query_embedding = sample_result[0]['embedding']
        print(f"Using embedding with {len(query_embedding)} dimensions")
        
        # Test K-NN with LIMIT to see if that helps
        knn_query = """
        SELECT id, predicate, embedding <|3|> $embedding AS similarity 
        FROM knowledge 
        WHERE embedding IS NOT NONE 
        LIMIT 3
        """
        
        try:
            knn_result = await conn.db.query(knn_query, {'embedding': query_embedding})
            print(f"K-NN query returned {len(knn_result) if knn_result else 0} results:")
            
            if knn_result:
                for j, record in enumerate(knn_result[:3], 1):
                    print(f"  [{j}] ID: {record.get('id', 'N/A')}")
                    print(f"      Predicate: {record.get('predicate', 'N/A')}")
                    print(f"      Similarity: {record.get('similarity', 'N/A')}")
                    
        except Exception as e:
            print(f"K-NN query failed: {e}")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_knowledge_relations())