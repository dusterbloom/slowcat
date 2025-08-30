#!/usr/bin/env python3
"""
Update the search function to use working vector::similarity::cosine()
"""

import asyncio
import sys
sys.path.append('.')

from memory.surreal_connection import SurrealConnectionManager

async def update_search_function():
    """Update fn::search_knowledge_advanced to use vector similarity"""
    
    conn = SurrealConnectionManager()
    await conn.connect()
    
    print("🔄 UPDATING SEARCH FUNCTION WITH VECTOR SIMILARITY")
    print("="*60)
    
    # First, let's see the current function
    try:
        current_function = await conn.db.query("SHOW FUNCTION fn::search_knowledge_advanced")
        print("Current function definition:")
        print(current_function)
    except Exception as e:
        print(f"Current function check: {e}")
    
    # Create the new function with vector similarity
    new_function = """
    DEFINE FUNCTION fn::search_knowledge_advanced($query: string, $query_embedding: array<float>, $limit: int) {
        -- Vector similarity search using working cosine similarity function
        LET $vector_results = (
            SELECT *, 
                   vector::similarity::cosine(embedding, $query_embedding) AS vector_score,
                   0.3f AS text_score
            FROM knowledge 
            WHERE embedding IS NOT NONE
            AND vector::similarity::cosine(embedding, $query_embedding) > 0.1
            ORDER BY vector_score DESC
            LIMIT $limit
        );
        
        -- Text search as fallback/supplement
        LET $text_results = (
            SELECT *, 
                   0f AS vector_score, 
                   1f AS text_score 
            FROM knowledge 
            WHERE (predicate != NONE AND string::contains(string::lowercase(predicate), string::lowercase($query))) 
            OR (in.canonical_name != NONE AND string::contains(string::lowercase(in.canonical_name), string::lowercase($query))) 
            OR (out.canonical_name != NONE AND string::contains(string::lowercase(out.canonical_name), string::lowercase($query))) 
            ORDER BY strength DESC, confidence DESC
            LIMIT $limit
        );
        
        -- Combine results, prioritizing vector matches
        LET $combined = array::union($vector_results, $text_results);
        
        -- Sort by combined score: vector_score * 0.7 + text_score * 0.3
        RETURN (
            SELECT *, 
                   (vector_score * 0.7 + text_score * 0.3) AS combined_score
            FROM $combined
            ORDER BY combined_score DESC
            LIMIT $limit
        );
    };
    """
    
    try:
        # Remove the old function first (if it exists)
        await conn.db.query("REMOVE FUNCTION fn::search_knowledge_advanced")
        print("✅ Removed old search function")
    except Exception as e:
        print(f"Note: Could not remove old function (may not exist): {e}")
    
    try:
        # Create the new function
        result = await conn.db.query(new_function)
        print("✅ Created new search function with vector similarity")
        print(f"Result: {result}")
    except Exception as e:
        print(f"❌ Error creating new function: {e}")
        return
    
    # Test the new function
    print(f"\n🧪 TESTING NEW SEARCH FUNCTION")
    print("="*50)
    
    # Generate a test embedding
    from memory.surreal_connection import get_query_sentence_transformer
    transformer = get_query_sentence_transformer()
    
    if transformer:
        test_query = "dog pet animal"
        test_embedding = transformer.encode(test_query, convert_to_numpy=True).tolist()
        
        try:
            test_result = await conn.db.query("""
                SELECT *, 
                       in.canonical_name as subject,
                       out.canonical_name as object
                FROM fn::search_knowledge_advanced($query, $query_embedding, $limit);
            """, {
                'query': test_query,
                'query_embedding': test_embedding,
                'limit': 5
            })
            
            print(f"✅ Search function test returned {len(test_result)} results:")
            
            for i, result in enumerate(test_result[:3], 1):
                print(f"  [{i}] {result.get('id')}: {result.get('predicate')}")
                print(f"      Vector score: {result.get('vector_score', 'N/A')}")
                print(f"      Text score: {result.get('text_score', 'N/A')}")
                print(f"      Combined score: {result.get('combined_score', 'N/A')}")
                print(f"      Subject: {result.get('subject', 'N/A')}")
                print(f"      Object: {result.get('object', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error testing search function: {e}")
    else:
        print("❌ Could not load sentence transformer for testing")
    
    await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(update_search_function())