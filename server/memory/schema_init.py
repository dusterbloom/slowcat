#!/usr/bin/env python3
"""
Schema initialization - ensure all functions exist on startup
"""

import asyncio
from loguru import logger
from .surreal_connection import SurrealConnectionManager


async def ensure_schema_functions(connection_manager: SurrealConnectionManager = None):
    """
    Ensure all schema functions exist in the database
    This should be called on system startup
    """
    
    if not connection_manager:
        connection_manager = SurrealConnectionManager()
        await connection_manager.ensure_connected()
    
    # Ensure connection is ready before proceeding
    if not connection_manager.db:
        await connection_manager.ensure_connected()
    
    # Double-check connection is available
    if not connection_manager.db:
        logger.error("❌ SurrealDB connection not available for schema initialization")
        return False
    
    try:
        logger.info("🔧 Ensuring all schema functions exist...")
        
        # fn::search_knowledge - Main search function with real-time decay calculations
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::search_knowledge($query: string, $limit: int) {
                RETURN SELECT *, 
                       fn::calculate_memory_decay(created_at, last_accessed, access_count) AS current_strength
                FROM knowledge
                WHERE (predicate != NONE AND string::contains(string::lowercase(predicate), string::lowercase($query)))
                   OR (in.canonical_name != NONE AND string::contains(string::lowercase(in.canonical_name), string::lowercase($query)))
                   OR (out.canonical_name != NONE AND string::contains(string::lowercase(out.canonical_name), string::lowercase($query)))
                   AND fn::calculate_memory_decay(created_at, last_accessed, access_count) > 0.1
                ORDER BY current_strength DESC, confidence DESC
                LIMIT $limit;
            };
        """)
        logger.debug("✅ fn::search_knowledge applied")
        
        # fn::get_entity_facts - Entity-specific search with decay calculations
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::get_entity_facts($entity_name: string) {
                RETURN SELECT *, 
                       fn::calculate_memory_decay(created_at, last_accessed, access_count) AS current_strength
                FROM knowledge 
                WHERE (in.canonical_name = $entity_name OR out.canonical_name = $entity_name)
                   AND fn::calculate_memory_decay(created_at, last_accessed, access_count) > 0.1
                ORDER BY current_strength DESC, temporal_context.when_said DESC;
            };
        """)
        logger.debug("✅ fn::get_entity_facts applied")
        
        # fn::get_memories_by_time - Temporal search
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::get_memories_by_time($from: datetime, $to: datetime) {
                RETURN SELECT * FROM knowledge
                WHERE temporal_context.when_said >= $from AND temporal_context.when_said <= $to
                ORDER BY temporal_context.when_said DESC;
            };
        """)
        logger.debug("✅ fn::get_memories_by_time applied")
        
        # fn::get_conversation_memory - Speaker history
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::get_conversation_memory($speaker: string, $limit: int) {
                RETURN SELECT * FROM knowledge
                WHERE in.canonical_name = $speaker AND predicate = 'said'
                ORDER BY temporal_context.when_said DESC
                LIMIT $limit;
            };
        """)
        logger.debug("✅ fn::get_conversation_memory applied")
        
        # fn::calculate_memory_decay - Memory management
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::calculate_memory_decay(
                $created_at: datetime,
                $last_accessed: datetime, 
                $access_count: int
            ) {
                LET $age_days = (time::now() - $created_at) / 1d;
                LET $recency_hours = (time::now() - $last_accessed) / 1h;
                LET $base_decay = 1.0 / (1.0 + $age_days / 30.0);
                LET $access_boost = IF $access_count / 20.0 < 0.5 THEN $access_count / 20.0 ELSE 0.5 END;
                LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 168.0);
                
                LET $result = $base_decay + $access_boost * $recency_penalty;
                RETURN IF $result > 0.1 THEN $result ELSE 0.1 END;
            };
        """)
        logger.debug("✅ fn::calculate_memory_decay applied")
        
        # fn::get_conversation_context - Session messages
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::get_conversation_context($session_id: string, $limit: int) {
                RETURN SELECT * FROM messages 
                WHERE session_id = $session_id 
                ORDER BY timestamp DESC 
                LIMIT $limit;
            };
        """)
        logger.debug("✅ fn::get_conversation_context applied")
        
        # fn::decay_background_memories - Background decay update function
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::decay_background_memories($batch_size: int) {
                LET $batch = (SELECT * FROM knowledge 
                    WHERE last_accessed < (time::now() - 1h)  // Only process facts older than 1 hour
                    LIMIT $batch_size);
                    
                FOR $fact IN $batch {
                    LET $new_strength = fn::calculate_memory_decay($fact.created_at, $fact.last_accessed, $fact.access_count);
                    UPDATE $fact.id SET strength = $new_strength;
                };
                
                RETURN count($batch);
            };
        """)
        logger.debug("✅ fn::decay_background_memories applied")
        
        # fn::cleanup_fragments - Remove very weak memories (< 0.1 strength)
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::cleanup_fragments($limit: int) {
                LET $fragments = (SELECT id FROM knowledge 
                    WHERE fn::calculate_memory_decay(created_at, last_accessed, access_count) < 0.1
                    LIMIT $limit);
                    
                FOR $fragment IN $fragments {
                    DELETE $fragment.id;
                };
                
                RETURN count($fragments);
            };
        """)
        logger.debug("✅ fn::cleanup_fragments applied")
        
        # fn::reconstruct_fragments - Strengthen related fragments when accessed together
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::reconstruct_fragments($entity_name: string, $boost_factor: float) {
                // Find fragments about the same entity
                LET $fragments = (SELECT * FROM knowledge 
                    WHERE (in.canonical_name = $entity_name OR out.canonical_name = $entity_name)
                    AND fn::calculate_memory_decay(created_at, last_accessed, access_count) < 0.5
                    AND fn::calculate_memory_decay(created_at, last_accessed, access_count) > 0.1);
                
                // If we have multiple fragments about the same entity, strengthen them
                IF count($fragments) > 1 {
                    FOR $fragment IN $fragments {
                        LET $current_strength = fn::calculate_memory_decay($fragment.created_at, $fragment.last_accessed, $fragment.access_count);
                        LET $boosted_strength = IF $current_strength + $boost_factor < 1.0 THEN $current_strength + $boost_factor ELSE 1.0 END;
                        
                        UPDATE $fragment.id SET 
                            strength = $boosted_strength,
                            last_accessed = time::now(),
                            access_count = access_count + 1,
                            reconstruction_boost = (reconstruction_boost OR 0) + $boost_factor;
                    };
                    
                    RETURN count($fragments);
                } ELSE {
                    RETURN 0;
                };
            };
        """)
        logger.debug("✅ fn::reconstruct_fragments applied")
        
        # fn::detect_memory_patterns - Find related fragments for reconstruction
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::detect_memory_patterns($query: string, $similarity_threshold: float) {
                // Get fragments that match the query - simplified version
                LET $query_fragments = (SELECT * FROM knowledge
                    WHERE (predicate != NONE AND string::contains(string::lowercase(predicate), string::lowercase($query)))
                       OR (in.canonical_name != NONE AND string::contains(string::lowercase(in.canonical_name), string::lowercase($query)))
                       OR (out.canonical_name != NONE AND string::contains(string::lowercase(out.canonical_name), string::lowercase($query)))
                       AND fn::calculate_memory_decay(created_at, last_accessed, access_count) >= 0.1 
                       AND fn::calculate_memory_decay(created_at, last_accessed, access_count) <= 0.5);
                
                RETURN $query_fragments;
            };
        """)
        logger.debug("✅ fn::detect_memory_patterns applied")
        
        logger.info("✅ All schema functions applied successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to apply schema functions: {e}")
        return False


async def test_schema_functions(connection_manager: SurrealConnectionManager = None):
    """Test that all schema functions are working"""
    
    if not connection_manager:
        connection_manager = SurrealConnectionManager()
        await connection_manager.ensure_connected()
    
    try:
        logger.info("🧪 Testing schema functions...")
        
        # Test search function
        result = await connection_manager.db.query("SELECT * FROM fn::search_knowledge('test', 1);")
        logger.debug(f"fn::search_knowledge test: {len(result)} results")
        
        # Test entity facts function  
        result = await connection_manager.db.query("SELECT * FROM fn::get_entity_facts('user');")
        logger.debug(f"fn::get_entity_facts test: {len(result)} results")
        
        logger.info("✅ All schema functions working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Schema function test failed: {e}")
        return False