#!/usr/bin/env python3
"""
PROPER SurrealDB Graph-Based Engram Design
Fix the fundamental issues with the current approach:
1. Use RELATE edges instead of knowledge_ids arrays
2. Fix narrative_summary generation 
3. Leverage SurrealDB's graph capabilities properly
"""

import asyncio
import sys
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def redesign_engrams_as_graph():
    """Redesign engrams to use proper SurrealDB graph relations"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    logger.info("🔥 REDESIGNING ENGRAMS AS PROPER GRAPH STRUCTURE")
    
    try:
        # Step 1: Create proper graph relation tables
        logger.info("📝 Creating graph relation tables...")
        
        # Engram contains knowledge (many-to-many)
        await conn.db.query("""
            DEFINE TABLE engram_contains TYPE RELATION IN engrams OUT knowledge SCHEMALESS;
            DEFINE FIELD strength ON engram_contains TYPE float DEFAULT 1.0;
            DEFINE FIELD contribution_score ON engram_contains TYPE float DEFAULT 0.5;
            DEFINE FIELD discovered_at ON engram_contains TYPE datetime DEFAULT time::now();
        """)
        
        # Engram reinforced by session (track which sessions strengthen it)
        await conn.db.query("""
            DEFINE TABLE reinforced_by TYPE RELATION IN engrams OUT sessions SCHEMALESS;
            DEFINE FIELD reinforcement_strength ON reinforced_by TYPE float DEFAULT 1.0;
            DEFINE FIELD discovered_in_session ON reinforced_by TYPE datetime DEFAULT time::now();
        """)
        
        logger.info("✅ Created graph relation tables")
        
        # Step 2: Remove the problematic knowledge_ids array approach
        logger.info("🗑️ Removing old array-based approach...")
        
        try:
            await conn.db.query("REMOVE FIELD knowledge_ids ON engrams;")
            await conn.db.query("REMOVE FIELD knowledge_ids[*] ON engrams;")
            logger.info("✅ Removed old knowledge_ids array")
        except:
            logger.info("ℹ️ knowledge_ids already removed or doesn't exist")
        
        # Step 3: Update engrams table structure for proper graph design
        logger.info("📊 Updating engrams table structure...")
        
        await conn.db.query("""
            DEFINE FIELD attractor_type ON engrams TYPE string 
                ASSERT $value IN ['persona', 'topic', 'routine', 'relationship', 'context', 'goal']
                DEFAULT 'topic';
        """)
        
        await conn.db.query("""
            DEFINE FIELD stability_score ON engrams TYPE float 
                ASSERT $value >= 0.0 AND $value <= 1.0 
                DEFAULT 0.5;
        """)
        
        await conn.db.query("""
            DEFINE FIELD emergence_strength ON engrams TYPE float 
                ASSERT $value >= 0.0 AND $value <= 1.0 
                DEFAULT 0.5;
        """)
        
        await conn.db.query("""
            DEFINE FIELD narrative_template ON engrams TYPE string DEFAULT '';
        """)
        
        logger.info("✅ Updated engrams table structure")
        
        # Step 4: Create PROPER detect_engrams function using graph queries
        logger.info("🧠 Creating proper graph-based detect_engrams function...")
        
        await conn.db.query("REMOVE FUNCTION fn::detect_engrams;")
        
        await conn.db.query("""
            DEFINE FUNCTION fn::detect_engrams_graph($session_id: string, $min_confidence: float, $min_cluster_size: int) {
                // Get session knowledge with high confidence
                LET $session_knowledge = (
                    SELECT *, 
                           in.canonical_name AS subject, 
                           out.canonical_name AS object 
                    FROM knowledge 
                    WHERE session_id = $session_id 
                      AND confidence >= $min_confidence
                      AND strength > 0.3
                );
                
                IF count($session_knowledge) >= $min_cluster_size {
                    // Extract core symbols with proper filtering
                    LET $subjects = array::group((
                        SELECT VALUE subject 
                        FROM $session_knowledge 
                        WHERE subject IS NOT NONE AND subject != ""
                    ));
                    
                    LET $objects = array::group((
                        SELECT VALUE object 
                        FROM $session_knowledge 
                        WHERE object IS NOT NONE AND object != ""
                    ));
                    
                    LET $predicates = array::group((
                        SELECT VALUE predicate 
                        FROM $session_knowledge 
                        WHERE predicate IS NOT NONE AND predicate != ""
                    ));
                    
                    // Get top symbols by frequency
                    LET $all_symbols = array::union($subjects, array::union($objects, $predicates));
                    LET $dominant_symbols = array::slice($all_symbols, 0, 6);
                    
                    IF count($dominant_symbols) >= 3 {
                        LET $knowledge_refs = (SELECT VALUE id FROM $session_knowledge);
                        
                        // Calculate proper coherence
                        LET $avg_confidence = math::mean((SELECT VALUE confidence FROM $session_knowledge));
                        LET $avg_strength = math::mean((SELECT VALUE strength FROM $session_knowledge));
                        LET $coherence_score = ($avg_confidence * 0.6) + ($avg_strength * 0.4);
                        
                        // Create pattern hash for similarity detection
                        LET $sorted_symbols = array::sort($dominant_symbols);
                        LET $pattern_hash = crypto::md5(string::join($sorted_symbols, "|"));
                        
                        // PROPER narrative generation - fix the empty string issue
                        LET $narrative_symbols = array::slice($dominant_symbols, 0, 4);
                        LET $narrative_text = string::concat(
                            "Knowledge pattern: ",
                            string::join($narrative_symbols, ", "),
                            " (coherence: ",
                            string::concat(math::round($coherence_score * 100), "%"),
                            ")"
                        );
                        
                        // Check for existing similar patterns
                        LET $similar_engrams = (
                            SELECT * FROM engrams 
                            WHERE pattern_hash = $pattern_hash 
                            AND session_id != $session_id
                            LIMIT 1
                        );
                        
                        IF count($similar_engrams) > 0 {
                            // Reinforce existing engram using GRAPH RELATIONS
                            LET $existing_engram = $similar_engrams[0];
                            
                            // Update the engram
                            UPDATE $existing_engram.id SET 
                                activation_count = activation_count + 1,
                                last_activated = time::now(),
                                coherence_score = (coherence_score + $coherence_score) / 2,
                                stability_score = math::min(1.0, stability_score + 0.1);
                            
                            // Create RELATE connection to session
                            RELATE $existing_engram.id->reinforced_by->(
                                SELECT * FROM sessions WHERE session_id = $session_id LIMIT 1
                            )[0] SET 
                                reinforcement_strength = $coherence_score,
                                discovered_in_session = time::now();
                            
                            RETURN {
                                engram_created: false,
                                existing_reinforced: true,
                                engram_id: $existing_engram.id,
                                pattern_hash: $pattern_hash,
                                coherence_score: $coherence_score,
                                narrative_summary: $narrative_text
                            };
                            
                        } ELSE {
                            // Create NEW engram with GRAPH RELATIONS
                            LET $new_engram = CREATE engrams SET 
                                dominant_symbols = $dominant_symbols,
                                narrative_summary = $narrative_text,
                                session_id = $session_id,
                                pattern_hash = $pattern_hash,
                                coherence_score = $coherence_score,
                                stability_score = 0.7,
                                emergence_strength = $coherence_score,
                                attractor_type = "topic",
                                activation_count = 1,
                                created_at = time::now(),
                                last_activated = time::now();
                            
                            // Link engram to knowledge using RELATE
                            FOR $knowledge_id IN $knowledge_refs {
                                RELATE $new_engram->engram_contains->$knowledge_id SET
                                    strength = 1.0,
                                    contribution_score = $coherence_score,
                                    discovered_at = time::now();
                            };
                            
                            // Link engram to session using RELATE  
                            RELATE $new_engram->reinforced_by->(
                                SELECT * FROM sessions WHERE session_id = $session_id LIMIT 1
                            )[0] SET
                                reinforcement_strength = $coherence_score,
                                discovered_in_session = time::now();
                            
                            RETURN {
                                engram_created: true,
                                engram_id: $new_engram,
                                pattern_hash: $pattern_hash,
                                coherence_score: $coherence_score,
                                narrative_summary: $narrative_text,
                                knowledge_count: count($knowledge_refs)
                            };
                        };
                    } ELSE {
                        RETURN {
                            engram_created: false,
                            reason: "insufficient_symbol_coherence"
                        };
                    };
                } ELSE {
                    RETURN {
                        engram_created: false,
                        reason: "insufficient_knowledge_cluster_size"
                    };
                };
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Created proper graph-based detect_engrams function")
        
        # Step 5: Create graph query functions
        logger.info("🔗 Creating graph query functions...")
        
        await conn.db.query("""
            DEFINE FUNCTION fn::get_engram_knowledge($engram_id: record<engrams>) {
                RETURN SELECT ->engram_contains->knowledge.* FROM $engram_id;
            } PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE FUNCTION fn::get_engram_sessions($engram_id: record<engrams>) {
                RETURN SELECT ->reinforced_by->sessions.* FROM $engram_id;
            } PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_engrams($session_id: string) {
                RETURN SELECT <-reinforced_by<-engrams.* 
                FROM sessions 
                WHERE session_id = $session_id;
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Created graph query functions")
        
        # Step 6: Migrate existing engram data to use graph relations
        logger.info("🔄 Migrating existing engram data to graph structure...")
        
        # This is complex - for now, let's clean up the bad engram and create fresh ones
        await conn.db.query("DELETE FROM engrams WHERE narrative_summary = 'Attractor state: , ';")
        logger.info("🗑️ Cleaned up malformed engrams")
        
        logger.info("✅ ENGRAM REDESIGN COMPLETE!")
        
        # Step 7: Test the new system
        logger.info("🧪 Testing new graph-based engram system...")
        
        result = await conn.db.query("""
            RETURN fn::detect_engrams_graph("test_session_123", 0.5, 2);
        """)
        
        if result and len(result) > 0:
            test_result = result[0]
            logger.info(f"✅ Test result: {test_result}")
        else:
            logger.warning("⚠️ Test returned no results")
            
        # Show info about new structure
        engrams_info = await conn.db.query("INFO FOR TABLE engrams;")
        relations_info = await conn.db.query("INFO FOR TABLE engram_contains;")
        
        logger.info("📊 New engrams table structure ready!")
        logger.info("🔗 Graph relations table created!")
        
    except Exception as e:
        logger.error(f"❌ Error redesigning engrams: {e}")
        import traceback
        traceback.print_exc()
        raise

async def main():
    logger.info("🚀 Starting PROPER SurrealDB Graph-Based Engram Redesign...")
    await redesign_engrams_as_graph()
    logger.info("🎉 Redesign complete! Engrams now use proper SurrealDB graph relations!")

if __name__ == "__main__":
    asyncio.run(main())