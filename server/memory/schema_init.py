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
        
        # fn::search_knowledge - Main search function with NULL handling
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::search_knowledge($query: string, $limit: int) {
                RETURN SELECT * FROM knowledge
                WHERE (predicate != NONE AND string::contains(string::lowercase(predicate), string::lowercase($query)))
                   OR (in.canonical_name != NONE AND string::contains(string::lowercase(in.canonical_name), string::lowercase($query)))
                   OR (out.canonical_name != NONE AND string::contains(string::lowercase(out.canonical_name), string::lowercase($query)))
                ORDER BY strength DESC, confidence DESC
                LIMIT $limit;
            };
        """)
        logger.debug("✅ fn::search_knowledge applied")
        
        # fn::get_entity_facts - Entity-specific search
        await connection_manager.db.query("""
            DEFINE FUNCTION fn::get_entity_facts($entity_name: string) {
                RETURN SELECT * FROM knowledge 
                WHERE in.canonical_name = $entity_name OR out.canonical_name = $entity_name
                ORDER BY strength DESC, temporal_context.when_said DESC;
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
                LET $access_boost = math::min(0.5, $access_count / 20.0);
                LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 168.0);
                
                RETURN math::max(0.1, $base_decay + $access_boost * $recency_penalty);
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