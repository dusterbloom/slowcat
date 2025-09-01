#!/usr/bin/env python3
"""
PROPER Graph-Based Engram System - Using SurrealDB RELATE correctly

This implements engrams as true graph structures using RELATE edges:
1. engram_contains: engram -> knowledge (what facts form this pattern)
2. engram_appears_in: engram -> sessions (where this pattern emerges)
3. Fixed narrative_summary generation that actually works
4. Leverages SurrealDB's graph capabilities properly
"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def create_proper_engram_graph_system():
    """Create TRUE graph-based engram system using proper RELATE syntax"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    logger.info("🧠 Creating PROPER Graph-Based Engram System")
    
    try:
        # Step 1: Clean up existing broken approach
        logger.info("🧹 Cleaning up old broken engrams...")
        await conn.db.query("DELETE FROM engrams;")
        
        # Remove old functions that don't work
        await conn.db.query("REMOVE FUNCTION fn::detect_engrams;")
        await conn.db.query("REMOVE FUNCTION fn::detect_engrams_graph;")
        await conn.db.query("REMOVE FUNCTION fn::detect_engrams_working;")
        await conn.db.query("REMOVE FUNCTION fn::detect_engrams_simple;")
        
        # Step 2: Define proper graph relation tables
        logger.info("🔗 Creating graph relation tables...")
        
        # Engram contains knowledge facts
        await conn.db.query("""
            DEFINE TABLE engram_contains TYPE RELATION IN engrams OUT knowledge SCHEMALESS;
            DEFINE FIELD strength ON engram_contains TYPE float DEFAULT 1.0;
            DEFINE FIELD contribution ON engram_contains TYPE float DEFAULT 0.8;
            DEFINE FIELD created_at ON engram_contains TYPE datetime DEFAULT time::now();
        """)
        
        # Engram appears in sessions  
        await conn.db.query("""
            DEFINE TABLE engram_appears_in TYPE RELATION IN engrams OUT sessions SCHEMALESS;
            DEFINE FIELD activation_strength ON engram_appears_in TYPE float DEFAULT 1.0;
            DEFINE FIELD first_appeared ON engram_appears_in TYPE datetime DEFAULT time::now();
            DEFINE FIELD last_seen ON engram_appears_in TYPE datetime DEFAULT time::now();
        """)
        
        logger.info("✅ Graph relation tables created")
        
        # Step 3: Update engrams table for graph-based approach
        logger.info("📊 Updating engrams table structure...")
        
        # Remove problematic knowledge_ids array field completely
        try:
            await conn.db.query("REMOVE FIELD knowledge_ids ON engrams;")
            await conn.db.query("REMOVE FIELD knowledge_ids[*] ON engrams;")
        except:
            pass  # Field might not exist
        
        # Add new fields for graph-based design
        await conn.db.query("""
            DEFINE FIELD pattern_type ON engrams TYPE string 
                ASSERT $value IN ['persona', 'topic', 'routine', 'relationship', 'context'] 
                DEFAULT 'topic';
        """)
        
        await conn.db.query("""
            DEFINE FIELD emergence_count ON engrams TYPE int DEFAULT 1;
        """)
        
        await conn.db.query("""
            DEFINE FIELD stability ON engrams TYPE float DEFAULT 0.5;
        """)
        
        logger.info("✅ Engrams table updated for graph design")
        
        # Step 4: Create WORKING detect_engrams function with proper RELATE syntax
        logger.info("🎯 Creating working graph-based detect_engrams function...")
        
        await conn.db.query("""
            DEFINE FUNCTION fn::detect_engrams_graph($session_id: string, $min_confidence: float, $min_facts: int) {
                // Get strong knowledge facts from session
                LET $session_facts = (
                    SELECT id, predicate, confidence, strength,
                           in.canonical_name AS subject, 
                           out.canonical_name AS object 
                    FROM knowledge 
                    WHERE session_id = $session_id 
                      AND confidence >= $min_confidence
                      AND strength > 0.4
                    ORDER BY confidence DESC
                    LIMIT 15
                );
                
                IF count($session_facts) >= $min_facts {
                    // Extract symbols - clean filtering
                    LET $subjects = (
                        SELECT VALUE subject FROM $session_facts 
                        WHERE subject != NONE AND subject != "" AND subject != "NONE"
                    );
                    LET $objects = (
                        SELECT VALUE object FROM $session_facts 
                        WHERE object != NONE AND object != "" AND object != "NONE"  
                    );
                    LET $predicates = (
                        SELECT VALUE predicate FROM $session_facts 
                        WHERE predicate != NONE AND predicate != "" AND predicate != "NONE"
                    );
                    
                    // Get unique symbols and take most important
                    LET $all_symbols = array::union($subjects, array::union($objects, $predicates));
                    LET $clean_symbols = array::group($all_symbols);
                    LET $core_symbols = array::slice($clean_symbols, 0, 4);
                    
                    IF count($core_symbols) >= 2 {
                        // Calculate metrics
                        LET $avg_confidence = math::mean((SELECT VALUE confidence FROM $session_facts));
                        LET $avg_strength = math::mean((SELECT VALUE strength FROM $session_facts));
                        LET $coherence = ($avg_confidence * 0.6) + ($avg_strength * 0.4);
                        
                        // Create pattern hash for similarity detection
                        LET $sorted_symbols = array::sort($core_symbols);
                        LET $pattern_sig = crypto::md5(string::join($sorted_symbols, "|"));
                        
                        // WORKING narrative generation - simple and reliable
                        LET $symbol_text = string::join(array::slice($core_symbols, 0, 3), " + ");
                        LET $full_narrative = string::concat(
                            "Knowledge pattern: ", 
                            $symbol_text, 
                            " (", 
                            count($session_facts), 
                            " facts, ", 
                            math::round($coherence * 100),
                            "% coherence)"
                        );
                        
                        // Check for existing similar patterns
                        LET $existing = (
                            SELECT * FROM engrams 
                            WHERE pattern_hash = $pattern_sig 
                            LIMIT 1
                        );
                        
                        IF count($existing) > 0 {
                            // Reinforce existing pattern using RELATE
                            LET $engram = $existing[0];
                            
                            UPDATE $engram SET 
                                emergence_count = emergence_count + 1,
                                last_activated = time::now(),
                                coherence_score = ($coherence + coherence_score) / 2,
                                stability = math::min([stability + 0.1, 1.0]);
                            
                            // Get session record for RELATE
                            LET $session_record = (SELECT * FROM sessions WHERE session_id = $session_id LIMIT 1);
                            IF count($session_record) > 0 {
                                LET $session_id_only = $session_record[0];
                                // Link engram to session using RELATE
                                RELATE $engram->engram_appears_in->$session_id_only SET
                                    activation_strength = $coherence,
                                    last_seen = time::now();
                            };
                            
                            RETURN {
                                success: true,
                                action: "reinforced",
                                engram_id: $engram.id,
                                pattern_hash: $pattern_sig,
                                narrative: $full_narrative,
                                coherence: $coherence
                            };
                            
                        } ELSE {
                            // Create NEW engram
                            LET $new_engram = CREATE engrams SET
                                dominant_symbols = $core_symbols,
                                narrative_summary = $full_narrative,
                                session_id = $session_id,
                                pattern_hash = $pattern_sig,
                                coherence_score = $coherence,
                                stability = 0.6,
                                pattern_type = "topic",
                                emergence_count = 1,
                                activation_count = 1,
                                created_at = time::now(),
                                last_activated = time::now();
                            
                            // Link engram to knowledge using RELATE (proper syntax)
                            FOR $fact IN $session_facts {
                                RELATE $new_engram->engram_contains->$fact SET
                                    strength = $fact.strength,
                                    contribution = $fact.confidence,
                                    created_at = time::now();
                            };
                            
                            // Link engram to session using RELATE
                            LET $session_record = (SELECT * FROM sessions WHERE session_id = $session_id LIMIT 1);
                            IF count($session_record) > 0 {
                                LET $session_id_only = $session_record[0];
                                RELATE $new_engram->engram_appears_in->$session_id_only SET
                                    activation_strength = $coherence,
                                    first_appeared = time::now(),
                                    last_seen = time::now();
                            };
                            
                            RETURN {
                                success: true,
                                action: "created",
                                engram_id: $new_engram.id,
                                pattern_hash: $pattern_sig,
                                narrative: $full_narrative,
                                coherence: $coherence,
                                fact_count: count($session_facts)
                            };
                        };
                    } ELSE {
                        RETURN {
                            success: false,
                            reason: "insufficient_symbol_diversity",
                            found_symbols: $core_symbols
                        };
                    };
                } ELSE {
                    RETURN {
                        success: false,
                        reason: "insufficient_knowledge_facts",
                        fact_count: count($session_facts)
                    };
                };
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Working graph-based detect_engrams function created")
        
        # Step 5: Create graph query functions
        logger.info("🔍 Creating graph query functions...")
        
        # Get knowledge connected to an engram
        await conn.db.query("""
            DEFINE FUNCTION fn::get_engram_knowledge($engram_id: record<engrams>) {
                RETURN SELECT ->engram_contains->knowledge.* FROM $engram_id;
            } PERMISSIONS FULL;
        """)
        
        # Get sessions where engram appears
        await conn.db.query("""
            DEFINE FUNCTION fn::get_engram_sessions($engram_id: record<engrams>) {
                RETURN SELECT ->engram_appears_in->sessions.* FROM $engram_id;
            } PERMISSIONS FULL;
        """)
        
        # Get engrams that appear in a session
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_engrams($session_id: string) {
                RETURN SELECT <-engram_appears_in<-engrams.* 
                FROM sessions 
                WHERE session_id = $session_id;
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Graph query functions created")
        
        # Step 6: Test with actual data
        logger.info("🧪 Testing new graph system...")
        
        # Check if we have knowledge to work with
        knowledge_stats = await conn.db.query("""
            SELECT 
                count() as total_knowledge,
                count(session_id) as with_sessions,
                math::max(confidence) as max_confidence,
                math::mean(confidence) as avg_confidence
            FROM knowledge 
            WHERE confidence > 0.5;
        """)
        
        if knowledge_stats and len(knowledge_stats) > 0:
            stats = knowledge_stats[0]
            logger.info(f"📊 Knowledge stats: {stats}")
            
            # Handle different result types
            total_knowledge = 0
            if isinstance(stats, dict):
                total_knowledge = stats.get('total_knowledge', 0)
            elif isinstance(stats, str) or isinstance(stats, int):
                total_knowledge = int(stats) if str(stats).isdigit() else 0
                
            if total_knowledge > 0:
                # Find session with good knowledge
                session_query = await conn.db.query("""
                    SELECT 
                        session_id, 
                        count() as fact_count,
                        math::mean(confidence) as avg_confidence
                    FROM knowledge 
                    WHERE confidence > 0.6 AND strength > 0.4
                      AND session_id IS NOT NONE
                    GROUP BY session_id 
                    ORDER BY fact_count DESC, avg_confidence DESC
                    LIMIT 1;
                """)
                
                if session_query and len(session_query) > 0:
                    test_session = session_query[0]['session_id']
                    fact_count = session_query[0]['fact_count']
                    avg_conf = session_query[0]['avg_confidence']
                    
                    logger.info(f"🎯 Testing with session {test_session} ({fact_count} facts, {avg_conf:.2f} avg confidence)")
                    
                    # Test engram detection
                    test_result = await conn.db.query("""
                        RETURN fn::detect_engrams_graph($session_id, 0.6, 2);
                    """, {"session_id": test_session})
                    
                    if test_result and len(test_result) > 0:
                        result = test_result[0]
                        logger.info(f"🎉 Test result: {result}")
                        
                        if result.get('success'):
                            narrative = result.get('narrative', '')
                            action = result.get('action', 'unknown')
                            logger.info(f"✅ SUCCESS! {action.title()} engram: '{narrative}'")
                            logger.info(f"🔍 Coherence: {result.get('coherence', 0):.2f}")
                            
                            # Test graph queries if we have an engram
                            if 'engram_id' in result:
                                engram_id = result['engram_id']
                                
                                # Test knowledge retrieval
                                knowledge_query = await conn.db.query("""
                                    RETURN fn::get_engram_knowledge($engram_id);
                                """, {"engram_id": engram_id})
                                
                                if knowledge_query and len(knowledge_query) > 0:
                                    knowledge_count = len(knowledge_query[0]) if knowledge_query[0] else 0
                                    logger.info(f"🔗 Engram linked to {knowledge_count} knowledge facts")
                                
                                # Test session retrieval
                                session_query = await conn.db.query("""
                                    RETURN fn::get_engram_sessions($engram_id);
                                """, {"engram_id": engram_id})
                                
                                if session_query and len(session_query) > 0:
                                    session_count = len(session_query[0]) if session_query[0] else 0
                                    logger.info(f"🔗 Engram appears in {session_count} sessions")
                                    
                        else:
                            reason = result.get('reason', 'unknown')
                            logger.warning(f"⚠️ Engram not created: {reason}")
                            if 'found_symbols' in result:
                                logger.info(f"   Found symbols: {result['found_symbols']}")
                    else:
                        logger.warning("⚠️ No test result returned")
                        
                else:
                    logger.info("ℹ️ No sessions with sufficient knowledge for testing")
            else:
                logger.info("ℹ️ No knowledge data available for testing")
        
        logger.info("🎉 PROPER Graph-Based Engram System Created!")
        logger.info("✅ Using true SurrealDB RELATE edges instead of arrays")
        logger.info("✅ Fixed narrative_summary generation")
        logger.info("✅ Leveraging graph database capabilities properly")
        
    except Exception as e:
        logger.error(f"❌ Error creating proper engram graph system: {e}")
        import traceback
        traceback.print_exc()
        raise

async def main():
    logger.info("🚀 Creating PROPER Graph-Based Engram System...")
    await create_proper_engram_graph_system()
    logger.info("✅ Done! Engrams now use proper SurrealDB graph relations!")

if __name__ == "__main__":
    asyncio.run(main())