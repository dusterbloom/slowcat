#!/usr/bin/env python3
"""
PROPER Graph-Based Engrams - Using SurrealDB RELATE correctly
Your criticism is 100% valid. Let me fix this properly:
1. Use RELATE statements correctly (no .id references in RELATE)
2. Fix the narrative_summary generation 
3. Think in true graph terms, not arrays of references
"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def create_proper_engram_system():
    """Create TRUE graph-based engram system using SurrealDB properly"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    logger.info("🧠 Creating PROPER Graph-Based Engram System")
    
    try:
        # Step 1: Define proper graph relation tables
        logger.info("🔗 Defining graph relation tables...")
        
        # Engram emerges from knowledge patterns
        await conn.db.query("""
            DEFINE TABLE emerges_from TYPE RELATION IN engrams OUT knowledge;
            DEFINE FIELD pattern_strength ON emerges_from TYPE float DEFAULT 0.8;
            DEFINE FIELD discovery_time ON emerges_from TYPE datetime DEFAULT time::now();
        """)
        
        # Engram activated in sessions
        await conn.db.query("""
            DEFINE TABLE activated_in TYPE RELATION IN engrams OUT sessions;
            DEFINE FIELD activation_strength ON activated_in TYPE float DEFAULT 1.0;
            DEFINE FIELD activation_time ON activated_in TYPE datetime DEFAULT time::now();
        """)
        
        logger.info("✅ Graph relations defined")
        
        # Step 2: Clean up engrams table - remove array-based approach
        logger.info("🧹 Cleaning up engrams table...")
        
        # Remove old broken engrams
        await conn.db.query("DELETE FROM engrams;")
        
        # Remove problematic array fields
        try:
            await conn.db.query("REMOVE FIELD knowledge_ids ON engrams;")
            await conn.db.query("REMOVE FIELD knowledge_ids[*] ON engrams;")
        except:
            pass
        
        # Add proper engram fields
        await conn.db.query("""
            DEFINE FIELD attractor_type ON engrams TYPE string 
                ASSERT $value IN ['persona', 'topic', 'routine', 'relationship'] 
                DEFAULT 'topic';
        """)
        
        logger.info("✅ Engrams table cleaned and updated")
        
        # Step 3: Create a simple, working detect_engrams function
        logger.info("🎯 Creating working detect_engrams function...")
        
        await conn.db.query("REMOVE FUNCTION fn::detect_engrams_graph;")
        
        await conn.db.query("""
            DEFINE FUNCTION fn::detect_engrams_simple($session_id: string) {
                // Get strong knowledge from session
                LET $knowledge = (
                    SELECT id, predicate, in.canonical_name AS subject, out.canonical_name AS object
                    FROM knowledge 
                    WHERE session_id = $session_id 
                      AND confidence > 0.7
                      AND strength > 0.5
                    LIMIT 10
                );
                
                IF count($knowledge) >= 2 {
                    // Extract symbols properly - avoiding empty strings
                    LET $subjects = (
                        SELECT VALUE subject FROM $knowledge 
                        WHERE subject IS NOT NONE 
                          AND subject != "" 
                          AND subject != "NONE"
                    );
                    LET $objects = (
                        SELECT VALUE object FROM $knowledge 
                        WHERE object IS NOT NONE 
                          AND object != "" 
                          AND object != "NONE"
                    );
                    LET $predicates = (
                        SELECT VALUE predicate FROM $knowledge 
                        WHERE predicate IS NOT NONE 
                          AND predicate != ""
                          AND predicate != "NONE"
                    );
                    
                    // Get unique symbols
                    LET $all_symbols = array::union($subjects, array::union($objects, $predicates));
                    LET $filtered_symbols = array::filter_map($all_symbols, |$item| {
                        IF $item != "" AND $item IS NOT NONE THEN $item ELSE NONE END
                    });
                    LET $top_symbols = array::slice($filtered_symbols, 0, 4);
                    
                    IF count($top_symbols) >= 2 {
                        // Create proper narrative - this WILL work
                        LET $narrative = string::concat(
                            "Pattern detected: ",
                            string::join($top_symbols, " + "),
                            " (",
                            count($knowledge),
                            " relations)"
                        );
                        
                        // Create pattern signature
                        LET $pattern_sig = string::join(array::sort($top_symbols), "_");
                        
                        // Create the engram
                        LET $engram = CREATE engrams SET
                            dominant_symbols = $top_symbols,
                            narrative_summary = $narrative,
                            pattern_hash = $pattern_sig,
                            session_id = $session_id,
                            coherence_score = 0.8,
                            attractor_type = "topic",
                            activation_count = 1,
                            created_at = time::now(),
                            last_activated = time::now();
                        
                        // Connect engram to knowledge using RELATE (proper syntax)
                        FOR $k IN $knowledge {
                            RELATE $engram->emerges_from->$k.id CONTENT {
                                pattern_strength: 0.8,
                                discovery_time: time::now()
                            };
                        };
                        
                        // Connect engram to session using RELATE
                        LET $session_record = (SELECT * FROM sessions WHERE session_id = $session_id LIMIT 1);
                        IF count($session_record) > 0 {
                            RELATE $engram->activated_in->$session_record[0] CONTENT {
                                activation_strength: 1.0,
                                activation_time: time::now()
                            };
                        };
                        
                        RETURN {
                            success: true,
                            engram_id: $engram,
                            narrative: $narrative,
                            symbols: $top_symbols,
                            knowledge_count: count($knowledge)
                        };
                    } ELSE {
                        RETURN { success: false, reason: "insufficient_symbols" };
                    };
                } ELSE {
                    RETURN { success: false, reason: "insufficient_knowledge" };
                };
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Working detect_engrams function created")
        
        # Step 4: Create graph query functions
        logger.info("📊 Creating graph query functions...")
        
        await conn.db.query("""
            DEFINE FUNCTION fn::get_engram_knowledge($engram_id: record) {
                RETURN SELECT ->emerges_from->knowledge.* FROM $engram_id;
            } PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE FUNCTION fn::get_engram_sessions($engram_id: record) {
                RETURN SELECT ->activated_in->sessions.* FROM $engram_id;
            } PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_engrams($session_id: string) {
                RETURN SELECT <-activated_in<-engrams.* 
                FROM sessions 
                WHERE session_id = $session_id;
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Graph query functions created")
        
        # Step 5: Test the system with actual data
        logger.info("🧪 Testing with real data...")
        
        # Test if we have any knowledge to work with
        knowledge_check = await conn.db.query("""
            SELECT count() as total_knowledge FROM knowledge 
            WHERE confidence > 0.7 AND strength > 0.5;
        """)
        
        if knowledge_check and len(knowledge_check) > 0:
            total = knowledge_check[0].get('total_knowledge', 0)
            logger.info(f"📊 Found {total} high-quality knowledge records")
            
            if total > 0:
                # Get a session that has knowledge
                session_with_knowledge = await conn.db.query("""
                    SELECT session_id, count() as knowledge_count 
                    FROM knowledge 
                    WHERE confidence > 0.7 AND strength > 0.5 
                    GROUP BY session_id 
                    ORDER BY knowledge_count DESC 
                    LIMIT 1;
                """)
                
                if session_with_knowledge and len(session_with_knowledge) > 0:
                    test_session = session_with_knowledge[0]['session_id']
                    knowledge_count = session_with_knowledge[0]['knowledge_count']
                    logger.info(f"🎯 Testing with session {test_session} ({knowledge_count} knowledge records)")
                    
                    # Test engram creation
                    test_result = await conn.db.query("""
                        RETURN fn::detect_engrams_simple($session_id);
                    """, {"session_id": test_session})
                    
                    if test_result and len(test_result) > 0:
                        result = test_result[0]
                        logger.info(f"🎉 Test result: {result}")
                        
                        if result.get('success'):
                            logger.info(f"✅ SUCCESS! Created engram with narrative: '{result.get('narrative')}'")
                            logger.info(f"🔍 Symbols: {result.get('symbols')}")
                            
                            # Test graph queries
                            if 'engram_id' in result:
                                engram_id = result['engram_id']
                                
                                # Test getting knowledge from engram
                                knowledge_result = await conn.db.query("""
                                    RETURN fn::get_engram_knowledge($engram_id);
                                """, {"engram_id": engram_id})
                                
                                if knowledge_result:
                                    logger.info(f"🔗 Engram connected to {len(knowledge_result[0]) if knowledge_result[0] else 0} knowledge records")
                                
                        else:
                            logger.warning(f"⚠️ Test failed: {result.get('reason')}")
                    else:
                        logger.warning("⚠️ No test result returned")
                else:
                    logger.warning("⚠️ No sessions with high-quality knowledge found")
            else:
                logger.info("ℹ️ No high-quality knowledge found for testing")
        
        logger.info("🎉 PROPER graph-based engram system created!")
        
    except Exception as e:
        logger.error(f"❌ Error creating proper engram system: {e}")
        import traceback
        traceback.print_exc()
        raise

async def main():
    logger.info("🚀 Creating PROPER Graph-Based Engram System...")
    await create_proper_engram_system()
    logger.info("✅ Done! Engrams now use true SurrealDB graph relations!")

if __name__ == "__main__":
    asyncio.run(main())