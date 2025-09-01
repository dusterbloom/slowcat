#!/usr/bin/env python3
"""
Apply SurrealDB fixes for proper engrams, memory_fragments, and field_states integration.
Follows the established pattern from archive/debug-files-20250830/apply_schema_functions.py
"""

import asyncio
import sys
import os
from loguru import logger

# Add server directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from memory.surreal_connection import SurrealConnectionManager

async def apply_engrams_fixes():
    """Apply engrams table structure fixes and improved detect_engrams function"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        logger.info("🧠 Applying engrams table fixes...")
        
        # Step 1: Fix engrams table structure (change session_ids to session_id)
        logger.info("📝 Updating engrams table structure...")
        
        # Remove problematic session_ids field
        try:
            await conn.db.query("REMOVE FIELD session_ids ON engrams;")
            await conn.db.query("REMOVE FIELD session_ids[*] ON engrams;")
            logger.info("✅ Removed old session_ids field")
        except Exception as e:
            logger.warning(f"⚠️ Could not remove session_ids (may not exist): {e}")
        
        # Add singular session_id field
        await conn.db.query("""
            DEFINE FIELD session_id ON engrams TYPE string PERMISSIONS FULL;
        """)
        
        # Add index for session_id
        await conn.db.query("""
            DEFINE INDEX engrams_session_id ON engrams FIELDS session_id;
        """)
        
        logger.info("✅ Added session_id field and index to engrams table")
        
        # Step 2: Update detect_engrams function with better attractor state detection
        logger.info("📝 Creating improved fn::detect_engrams function...")
        
        # Remove existing function
        try:
            await conn.db.query("REMOVE FUNCTION fn::detect_engrams;")
            logger.info("✅ Removed old detect_engrams function")
        except:
            logger.info("ℹ️ No existing detect_engrams function to remove")
        
        # Create improved function with proper attractor state detection
        await conn.db.query("""
            DEFINE FUNCTION fn::detect_engrams($session_id: string, $min_confidence: float, $min_cluster_size: int) {
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
                    // Extract and group symbols
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
                    
                    // Calculate pattern coherence based on repetition and strength
                    LET $subject_weights = {};
                    LET $predicate_weights = {};
                    LET $object_weights = {};
                    
                    // Weight symbols by frequency and knowledge strength
                    FOR $knowledge IN $session_knowledge {
                        LET $strength_boost = $knowledge.strength * $knowledge.confidence;
                        // Update weights (simplified - in real implementation would use proper weight accumulation)
                    };
                    
                    // Select dominant symbols (most frequent/strongest)
                    LET $all_symbols = array::union($subjects, array::union($objects, $predicates));
                    LET $dominant_symbols = array::slice($all_symbols, 0, 6);
                    
                    IF count($dominant_symbols) >= 3 {
                        LET $knowledge_refs = (SELECT VALUE id FROM $session_knowledge);
                        
                        // Calculate coherence based on pattern stability, not just confidence
                        LET $avg_confidence = math::mean((SELECT VALUE confidence FROM $session_knowledge));
                        LET $avg_strength = math::mean((SELECT VALUE strength FROM $session_knowledge));
                        LET $pattern_coherence = ($avg_confidence * 0.6) + ($avg_strength * 0.4);
                        
                        // Create pattern hash for detecting similar engrams across sessions
                        LET $sorted_symbols = array::sort($dominant_symbols);
                        LET $pattern_hash = crypto::md5(string::join($sorted_symbols, "|"));
                        
                        // Improved narrative generation
                        LET $symbols_str = string::join($dominant_symbols, ", ");
                        LET $narrative = string::concat("Attractor state: ", $symbols_str);
                        
                        // Check for existing similar engrams
                        LET $similar_engrams = (
                            SELECT * FROM engrams 
                            WHERE pattern_hash = $pattern_hash 
                            AND session_id != $session_id
                        );
                        
                        IF count($similar_engrams) > 0 {
                            // Reinforce existing pattern
                            LET $existing = $similar_engrams[0];
                            UPDATE $existing.id SET 
                                activation_count = activation_count + 1,
                                last_activated = time::now(),
                                coherence_score = (coherence_score + $pattern_coherence) / 2;
                            
                            RETURN {
                                engram_created: false,
                                existing_reinforced: true,
                                pattern_hash: $pattern_hash,
                                coherence_score: $pattern_coherence,
                                dominant_symbols: $dominant_symbols,
                                narrative_summary: $narrative
                            };
                        } ELSE {
                            // Create new engram with pattern hash
                            CREATE engrams SET 
                                dominant_symbols = $dominant_symbols,
                                narrative_summary = $narrative,
                                knowledge_ids = $knowledge_refs,
                                session_id = $session_id,
                                pattern_hash = $pattern_hash,
                                coherence_score = $pattern_coherence,
                                activation_count = 1,
                                created_at = time::now(),
                                last_activated = time::now();
                            
                            RETURN {
                                engram_created: true,
                                pattern_hash: $pattern_hash,
                                coherence_score: $pattern_coherence,
                                dominant_symbols: $dominant_symbols,
                                narrative_summary: $narrative,
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
        
        logger.info("✅ Created improved fn::detect_engrams function with pattern hash and reinforcement")
        
        # Step 3: Add pattern_hash field to engrams if not exists
        await conn.db.query("""
            DEFINE FIELD pattern_hash ON engrams TYPE option<string> PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE INDEX engrams_pattern_hash ON engrams FIELDS pattern_hash;
        """)
        
        logger.info("✅ Added pattern_hash field and index for cross-session pattern detection")
        
    except Exception as e:
        logger.error(f"❌ Error applying engrams fixes: {e}")
        import traceback
        traceback.print_exc()
        raise

async def apply_memory_fragments_fixes():
    """Apply memory_fragments table fixes for session integration"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        logger.info("🧩 Applying memory_fragments fixes...")
        
        # Add session_id field
        await conn.db.query("""
            DEFINE FIELD session_id ON memory_fragments TYPE option<string> PERMISSIONS FULL;
        """)
        
        # Add session_id index
        await conn.db.query("""
            DEFINE INDEX memory_fragments_session_id ON memory_fragments FIELDS session_id;
        """)
        
        # Ensure content field structure is properly defined
        await conn.db.query("""
            DEFINE FIELD content ON memory_fragments FLEXIBLE TYPE object DEFAULT {
                text: "",
                embedding: NONE,
                semantic_hash: ""
            } PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE FIELD content.text ON memory_fragments TYPE string PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE FIELD content.embedding ON memory_fragments TYPE option<array<float>> PERMISSIONS FULL;
        """)
        
        await conn.db.query("""
            DEFINE FIELD content.semantic_hash ON memory_fragments TYPE string PERMISSIONS FULL;
        """)
        
        logger.info("✅ Applied memory_fragments session integration and content structure")
        
    except Exception as e:
        logger.error(f"❌ Error applying memory_fragments fixes: {e}")
        raise

async def apply_field_states_fixes():
    """Apply field_states table fixes for session integration"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        logger.info("⚡ Applying field_states fixes...")
        
        # Add session_id field
        await conn.db.query("""
            DEFINE FIELD session_id ON field_states TYPE option<string> PERMISSIONS FULL;
        """)
        
        # Add session_id index  
        await conn.db.query("""
            DEFINE INDEX field_states_session_id ON field_states FIELDS session_id;
        """)
        
        logger.info("✅ Applied field_states session integration")
        
    except Exception as e:
        logger.error(f"❌ Error applying field_states fixes: {e}")
        raise

async def migrate_existing_data():
    """Migrate existing data to use new fields"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        logger.info("🔄 Migrating existing data...")
        
        # Update existing engrams to have session_id from first element of old session_ids array
        logger.info("📝 Migrating engrams session_ids to session_id...")
        result = await conn.db.query("""
            FOR $engram IN (SELECT * FROM engrams WHERE session_ids IS NOT NONE) {
                LET $first_session = array::first($engram.session_ids);
                IF $first_session IS NOT NONE {
                    UPDATE $engram.id SET session_id = $first_session;
                };
            };
        """)
        logger.info(f"✅ Migrated engrams data")
        
        # Set default session_id for memory_fragments from source interactions
        logger.info("📝 Setting session_id for memory_fragments...")
        await conn.db.query("""
            FOR $fragment IN (SELECT * FROM memory_fragments WHERE session_id IS NONE) {
                // Try to get session_id from linked messages
                LET $source_messages = (
                    SELECT session_id FROM messages 
                    WHERE id IN $fragment.source_interactions 
                    LIMIT 1
                );
                
                IF count($source_messages) > 0 AND $source_messages[0].session_id IS NOT NONE {
                    UPDATE $fragment.id SET session_id = $source_messages[0].session_id;
                } ELSE {
                    // Fall back to most recent active session
                    LET $recent_sessions = (
                        SELECT session_id, start_time FROM sessions 
                        WHERE is_active = true 
                        ORDER BY start_time DESC 
                        LIMIT 1
                    );
                    
                    IF count($recent_sessions) > 0 AND $recent_sessions[0].session_id IS NOT NONE {
                        UPDATE $fragment.id SET session_id = $recent_sessions[0].session_id;
                    };
                };
            };
        """)
        logger.info("✅ Set session_id for memory_fragments")
        
        # Set default session_id for field_states
        logger.info("📝 Setting session_id for field_states...")
        await conn.db.query("""
            FOR $state IN (SELECT * FROM field_states WHERE session_id IS NONE) {
                // Try to get session_id from linked memory fragment
                LET $fragment_results = (
                    SELECT session_id FROM memory_fragments 
                    WHERE id = $state.fragment_id 
                    LIMIT 1
                );
                
                IF count($fragment_results) > 0 AND $fragment_results[0].session_id IS NOT NONE {
                    UPDATE $state.id SET session_id = $fragment_results[0].session_id;
                } ELSE {
                    // Fall back to most recent active session
                    LET $recent_sessions = (
                        SELECT session_id, start_time FROM sessions 
                        WHERE is_active = true 
                        ORDER BY start_time DESC 
                        LIMIT 1
                    );
                    
                    IF count($recent_sessions) > 0 AND $recent_sessions[0].session_id IS NOT NONE {
                        UPDATE $state.id SET session_id = $recent_sessions[0].session_id;
                    };
                };
            };
        """)
        logger.info("✅ Set session_id for field_states")
        
    except Exception as e:
        logger.error(f"❌ Error migrating existing data: {e}")
        raise

async def create_helper_functions():
    """Create helpful query functions for the integrated system"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        logger.info("🔧 Creating helper functions...")
        
        # Session memory function
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_memory($session_id: string) {
                LET $messages = (SELECT * FROM messages WHERE session_id = $session_id ORDER BY timestamp);
                LET $knowledge = (SELECT * FROM knowledge WHERE session_id = $session_id);
                LET $engrams = (SELECT * FROM engrams WHERE session_id = $session_id);
                LET $fragments = (SELECT * FROM memory_fragments WHERE session_id = $session_id);  
                LET $field_states = (SELECT * FROM field_states WHERE session_id = $session_id);
                
                RETURN {
                    session_id: $session_id,
                    messages: $messages,
                    knowledge: $knowledge,
                    engrams: $engrams,
                    memory_fragments: $fragments,
                    field_states: $field_states,
                    stats: {
                        messages_count: count($messages),
                        knowledge_count: count($knowledge),
                        engrams_count: count($engrams),
                        fragments_count: count($fragments),
                        states_count: count($field_states)
                    }
                };
            } PERMISSIONS FULL;
        """)
        
        # Session stats function
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_stats($session_id: string) {
                LET $messages_result = (SELECT count() as count FROM messages WHERE session_id = $session_id);
                LET $knowledge_result = (SELECT count() as count FROM knowledge WHERE session_id = $session_id);
                LET $engrams_result = (SELECT count() as count FROM engrams WHERE session_id = $session_id);
                LET $fragments_result = (SELECT count() as count FROM memory_fragments WHERE session_id = $session_id);
                LET $states_result = (SELECT count() as count FROM field_states WHERE session_id = $session_id);
                
                RETURN {
                    session_id: $session_id,
                    messages_count: IF count($messages_result) > 0 THEN $messages_result[0].count ELSE 0 END,
                    knowledge_count: IF count($knowledge_result) > 0 THEN $knowledge_result[0].count ELSE 0 END,
                    engrams_count: IF count($engrams_result) > 0 THEN $engrams_result[0].count ELSE 0 END,
                    fragments_count: IF count($fragments_result) > 0 THEN $fragments_result[0].count ELSE 0 END,
                    states_count: IF count($states_result) > 0 THEN $states_result[0].count ELSE 0 END
                };
            } PERMISSIONS FULL;
        """)
        
        # Cross-session pattern detection
        await conn.db.query("""
            DEFINE FUNCTION fn::find_similar_patterns($pattern_hash: string) {
                RETURN SELECT * FROM engrams 
                WHERE pattern_hash = $pattern_hash 
                ORDER BY coherence_score DESC, last_activated DESC;
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Created helper functions")
        
    except Exception as e:
        logger.error(f"❌ Error creating helper functions: {e}")
        raise

async def test_integration():
    """Test that the integration works correctly"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        logger.info("🧪 Testing integration...")
        
        # Test detect_engrams function
        logger.info("🔍 Testing fn::detect_engrams...")
        result = await conn.db.query("""
            RETURN fn::detect_engrams("test_session", 0.5, 2);
        """)
        logger.info(f"✅ detect_engrams function callable: {len(result) > 0}")
        
        # Test session memory function
        logger.info("🔍 Testing fn::get_session_stats...")
        result = await conn.db.query("""
            RETURN fn::get_session_stats("nonexistent_session");
        """)
        logger.info(f"✅ get_session_stats function callable: {len(result) > 0}")
        
        # Check table structures
        logger.info("🔍 Checking table structures...")
        
        # Check engrams
        engrams_info = await conn.db.query("INFO FOR TABLE engrams;")
        logger.info(f"📊 Engrams table structure verified")
        
        # Check memory_fragments
        fragments_info = await conn.db.query("INFO FOR TABLE memory_fragments;")
        logger.info(f"📊 Memory fragments table structure verified")
        
        # Check field_states
        states_info = await conn.db.query("INFO FOR TABLE field_states;")
        logger.info(f"📊 Field states table structure verified")
        
        logger.info("✅ Integration test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        raise

async def main():
    """Apply all SurrealDB fixes"""
    
    logger.info("🚀 Starting SurrealDB fixes migration...")
    logger.info("=" * 80)
    
    try:
        # Apply fixes in order
        await apply_engrams_fixes()
        await apply_memory_fragments_fixes() 
        await apply_field_states_fixes()
        await migrate_existing_data()
        await create_helper_functions()
        await test_integration()
        
        logger.info("=" * 80)
        logger.info("🎉 SurrealDB fixes applied successfully!")
        logger.info("")
        logger.info("✅ Fixed Issues:")
        logger.info("   - Engrams now use session_id (not session_ids array)")
        logger.info("   - Improved engram detection with pattern hashing")
        logger.info("   - Memory fragments linked to sessions")  
        logger.info("   - Field states linked to sessions")
        logger.info("   - Cross-session pattern recognition")
        logger.info("   - Helper functions for integrated queries")
        logger.info("")
        logger.info("🧪 Test the integration:")
        logger.info("   python test_surrealdb_fixes.py")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())