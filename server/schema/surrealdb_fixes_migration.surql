-- ================================================================================
-- SURREALDB FIXES MIGRATION - Complete Integration of engrams, memory_fragments, field_states
-- ================================================================================
-- Fixes session_id consistency issues and table structure problems
-- Date: September 1, 2025
-- ================================================================================

USE NS slowcat;
USE DB memory_graph;

-- ================================================================================
-- STEP 1: FIX ENGRAMS TABLE STRUCTURE
-- ================================================================================

-- Fix engrams table to use singular session_id instead of session_ids array
-- This provides consistency with all other tables in the system

-- Remove the problematic session_ids field and add proper session_id
REMOVE FIELD session_ids ON engrams;
REMOVE FIELD session_ids[*] ON engrams;

-- Add singular session_id field for consistency 
DEFINE FIELD session_id ON engrams TYPE string PERMISSIONS FULL;

-- Add index for efficient session-based queries
DEFINE INDEX engrams_session_id ON engrams FIELDS session_id;

-- ================================================================================
-- STEP 2: FIX DETECT_ENGRAMS FUNCTION 
-- ================================================================================

-- Remove the existing function that has string join issues
REMOVE FUNCTION fn::detect_engrams;

-- Create fixed version with proper string handling and session_id (singular)
DEFINE FUNCTION fn::detect_engrams($session_id: string, $min_confidence: float, $min_cluster_size: int) {
    LET $session_knowledge = (
        SELECT *, 
               in.canonical_name AS subject, 
               out.canonical_name AS object 
        FROM knowledge 
        WHERE session_id = $session_id 
          AND confidence >= $min_confidence
    );
    
    IF count($session_knowledge) >= $min_cluster_size {
        LET $subjects = array::group((
            SELECT VALUE subject 
            FROM $session_knowledge 
            WHERE subject != NONE AND subject != ""
        ));
        
        LET $objects = array::group((
            SELECT VALUE object 
            FROM $session_knowledge 
            WHERE object != NONE AND object != ""
        ));
        
        LET $predicates = array::group((
            SELECT VALUE predicate 
            FROM $session_knowledge 
            WHERE predicate != NONE AND predicate != ""
        ));
        
        LET $all_symbols = array::union($subjects, array::union($objects, $predicates));
        LET $dominant_symbols = array::slice($all_symbols, 0, 6);
        
        IF count($dominant_symbols) >= 3 {
            LET $knowledge_refs = array::group((
                SELECT VALUE id FROM $session_knowledge
            ));
            
            LET $avg_confidence = math::mean((
                SELECT VALUE confidence FROM $session_knowledge
            ));
            
            -- Fixed narrative summary generation with proper string conversion
            LET $symbols_str = string::join($dominant_symbols, ", ");
            LET $narrative = string::concat("Attractor state: ", $symbols_str);
            
            CREATE engrams SET 
                dominant_symbols = $dominant_symbols,
                narrative_summary = $narrative,
                knowledge_ids = $knowledge_refs,
                session_id = $session_id,  -- Use singular session_id
                coherence_score = $avg_confidence,
                activation_count = 1,
                created_at = time::now(),
                last_activated = time::now();
            
            RETURN {
                coherence_score: $avg_confidence,
                dominant_symbols: $dominant_symbols,
                narrative_summary: $narrative,
                engram_created: true,
                knowledge_count: count($knowledge_refs)
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

-- ================================================================================
-- STEP 3: ADD SESSION_ID TO MEMORY_FRAGMENTS
-- ================================================================================

-- Add session_id field to memory_fragments for session linking
DEFINE FIELD session_id ON memory_fragments TYPE option<string> PERMISSIONS FULL;

-- Add index for efficient session-based fragment queries
DEFINE INDEX memory_fragments_session_id ON memory_fragments FIELDS session_id;

-- Add missing content field structure that's referenced in fragments_semantic_hash index
DEFINE FIELD content ON memory_fragments FLEXIBLE TYPE object DEFAULT {
    text: "",
    embedding: NONE,
    semantic_hash: ""
} PERMISSIONS FULL;

DEFINE FIELD content.text ON memory_fragments TYPE string PERMISSIONS FULL;
DEFINE FIELD content.embedding ON memory_fragments TYPE option<array<float>> PERMISSIONS FULL;
DEFINE FIELD content.semantic_hash ON memory_fragments TYPE string PERMISSIONS FULL;

-- ================================================================================
-- STEP 4: ADD SESSION_ID TO FIELD_STATES
-- ================================================================================

-- Add session_id field to field_states for session linking
DEFINE FIELD session_id ON field_states TYPE option<string> PERMISSIONS FULL;

-- Add index for efficient session-based field state queries
DEFINE INDEX field_states_session_id ON field_states FIELDS session_id;

-- ================================================================================
-- STEP 5: MIGRATE EXISTING DATA
-- ================================================================================

-- Update existing engrams to have session_id from first element of old session_ids array
FOR $engram IN (SELECT * FROM engrams WHERE session_ids IS NOT NONE) {
    LET $first_session = array::first($engram.session_ids);
    IF $first_session IS NOT NONE {
        UPDATE $engram.id SET session_id = $first_session;
    };
};

-- Set default session_id for memory_fragments that don't have one
-- This connects them to the most recent session for the fragment
FOR $fragment IN (SELECT * FROM memory_fragments WHERE session_id IS NONE) {
    -- Try to get session_id from source_interactions
    LET $source_session = (
        SELECT session_id FROM messages 
        WHERE id IN $fragment.source_interactions 
        LIMIT 1
    )[0].session_id;
    
    IF $source_session IS NOT NONE {
        UPDATE $fragment.id SET session_id = $source_session;
    } ELSE {
        -- Fall back to most recent active session
        LET $recent_session = (
            SELECT session_id FROM sessions 
            WHERE is_active = true 
            ORDER BY start_time DESC 
            LIMIT 1
        )[0].session_id;
        
        IF $recent_session IS NOT NONE {
            UPDATE $fragment.id SET session_id = $recent_session;
        };
    };
};

-- Set default session_id for field_states that don't have one  
FOR $state IN (SELECT * FROM field_states WHERE session_id IS NONE) {
    -- Try to get session_id from linked memory_fragment
    LET $fragment_session = (
        SELECT session_id FROM memory_fragments 
        WHERE id = $state.fragment_id 
        LIMIT 1
    )[0].session_id;
    
    IF $fragment_session IS NOT NONE {
        UPDATE $state.id SET session_id = $fragment_session;
    } ELSE {
        -- Fall back to most recent active session
        LET $recent_session = (
            SELECT session_id FROM sessions 
            WHERE is_active = true 
            ORDER BY start_time DESC 
            LIMIT 1
        )[0].session_id;
        
        IF $recent_session IS NOT NONE {
            UPDATE $state.id SET session_id = $recent_session;
        };
    };
};

-- ================================================================================
-- STEP 6: ADD HELPFUL QUERY FUNCTIONS
-- ================================================================================

-- Function to get all session-related data (messages, knowledge, engrams, fragments, states)
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

-- Function to get session statistics
DEFINE FUNCTION fn::get_session_stats($session_id: string) {
    RETURN {
        session_id: $session_id,
        messages_count: (SELECT count() FROM messages WHERE session_id = $session_id)[0].count,
        knowledge_count: (SELECT count() FROM knowledge WHERE session_id = $session_id)[0].count,
        engrams_count: (SELECT count() FROM engrams WHERE session_id = $session_id)[0].count,
        fragments_count: (SELECT count() FROM memory_fragments WHERE session_id = $session_id)[0].count,
        states_count: (SELECT count() FROM field_states WHERE session_id = $session_id)[0].count
    };
} PERMISSIONS FULL;

-- ================================================================================
-- VERIFICATION QUERIES
-- ================================================================================

-- Test engram creation with the fixed function
-- SELECT fn::detect_engrams("test_session", 0.5, 2);

-- Check session linkage across all tables
-- SELECT fn::get_session_stats("your_session_id_here");

-- View all engrams with their new structure  
-- SELECT * FROM engrams ORDER BY created_at DESC LIMIT 5;

-- ================================================================================
-- END OF MIGRATION
-- ================================================================================ 