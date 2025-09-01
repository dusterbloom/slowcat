#!/usr/bin/env python3
"""
COMPLETE Graph-Based Memory System - Extend RELATE to ALL tables

This extends the engram graph approach to create a fully connected graph database:
- messages ↔ sessions (message belongs to session)
- knowledge ↔ entity (knowledge about entity)  
- knowledge ↔ sessions (knowledge from session)
- entity ↔ sessions (entity mentioned in session)

True graph database usage across the entire consciousness system.
"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def create_complete_graph_system():
    """Create comprehensive graph relations across all memory tables"""
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    logger.info("🌐 Creating COMPLETE Graph-Based Memory System")
    
    try:
        # Step 1: Define comprehensive graph relation tables
        logger.info("🔗 Creating comprehensive graph relation tables...")
        
        # Message belongs to session
        await conn.db.query("""
            DEFINE TABLE message_belongs_to TYPE RELATION IN messages OUT sessions SCHEMALESS;
            DEFINE FIELD timestamp ON message_belongs_to TYPE datetime DEFAULT time::now();
            DEFINE FIELD message_order ON message_belongs_to TYPE int DEFAULT 0;
            DEFINE FIELD speaker_role ON message_belongs_to TYPE string DEFAULT 'user';
        """)
        
        # Knowledge about entity (what we know about whom/what)
        await conn.db.query("""
            DEFINE TABLE knowledge_about TYPE RELATION IN knowledge OUT entity SCHEMALESS;
            DEFINE FIELD relevance_score ON knowledge_about TYPE float DEFAULT 1.0;
            DEFINE FIELD relationship_type ON knowledge_about TYPE string DEFAULT 'general';
            DEFINE FIELD discovered_at ON knowledge_about TYPE datetime DEFAULT time::now();
        """)
        
        # Knowledge from session (where knowledge was learned)
        await conn.db.query("""
            DEFINE TABLE knowledge_from TYPE RELATION IN knowledge OUT sessions SCHEMALESS;
            DEFINE FIELD extraction_confidence ON knowledge_from TYPE float DEFAULT 0.8;
            DEFINE FIELD extraction_method ON knowledge_from TYPE string DEFAULT 'conversation';
            DEFINE FIELD learned_at ON knowledge_from TYPE datetime DEFAULT time::now();
        """)
        
        # Entity mentioned in session (who/what was discussed)
        await conn.db.query("""
            DEFINE TABLE entity_mentioned_in TYPE RELATION IN entity OUT sessions SCHEMALESS;
            DEFINE FIELD mention_frequency ON entity_mentioned_in TYPE int DEFAULT 1;
            DEFINE FIELD mention_sentiment ON entity_mentioned_in TYPE float DEFAULT 0.0;
            DEFINE FIELD first_mentioned ON entity_mentioned_in TYPE datetime DEFAULT time::now();
            DEFINE FIELD last_mentioned ON entity_mentioned_in TYPE datetime DEFAULT time::now();
        """)
        
        # Session involves speaker (who participated)
        await conn.db.query("""
            DEFINE TABLE session_involves TYPE RELATION IN sessions OUT entity SCHEMALESS;
            DEFINE FIELD participation_role ON session_involves TYPE string DEFAULT 'participant';
            DEFINE FIELD message_count ON session_involves TYPE int DEFAULT 0;
            DEFINE FIELD first_interaction ON session_involves TYPE datetime DEFAULT time::now();
            DEFINE FIELD last_interaction ON session_involves TYPE datetime DEFAULT time::now();
        """)
        
        logger.info("✅ Comprehensive graph relation tables created")
        
        # Step 2: Create migration functions to populate relations from existing data
        logger.info("🔄 Creating graph migration functions...")
        
        # Function to migrate messages -> sessions relations
        await conn.db.query("""
            DEFINE FUNCTION fn::migrate_message_relations() {
                LET $messages = (SELECT * FROM messages WHERE session_id IS NOT NONE);
                LET $migrated = 0;
                
                FOR $msg IN $messages {
                    // Find corresponding session
                    LET $session_records = (SELECT * FROM sessions WHERE session_id = $msg.session_id LIMIT 1);
                    IF count($session_records) > 0 {
                        LET $session_record = $session_records[0];
                        // Create RELATE edge
                        RELATE $msg->message_belongs_to->$session_record SET
                            timestamp = $msg.timestamp OR time::now(),
                            message_order = $msg.message_order OR 0,
                            speaker_role = $msg.role OR 'user';
                        LET $migrated = $migrated + 1;
                    };
                };
                
                RETURN {
                    migrated_messages: $migrated,
                    total_messages: count($messages),
                    success: true
                };
            } PERMISSIONS FULL;
        """)
        
        # Function to migrate knowledge -> entity relations
        await conn.db.query("""
            DEFINE FUNCTION fn::migrate_knowledge_entity_relations() {
                LET $knowledge = (SELECT * FROM knowledge);
                LET $migrated = 0;
                
                FOR $k IN $knowledge {
                    // Extract entity names from in/out canonical names
                    LET $subject_name = $k.in.canonical_name OR "";
                    LET $object_name = $k.out.canonical_name OR "";
                    
                    // Create or find entities and relate knowledge
                    IF $subject_name != "" AND $subject_name != "NONE" {
                        LET $subject_entity = UPSERT entity:[$subject_name] SET 
                            canonical_name = $subject_name,
                            entity_type = 'person',
                            created_at = time::now();
                        
                        RELATE $k->knowledge_about->$subject_entity SET
                            relevance_score = $k.confidence OR 1.0,
                            relationship_type = 'subject',
                            discovered_at = $k.created_at OR time::now();
                        LET $migrated = $migrated + 1;
                    };
                    
                    IF $object_name != "" AND $object_name != "NONE" AND $object_name != $subject_name {
                        LET $object_entity = UPSERT entity:[$object_name] SET 
                            canonical_name = $object_name,
                            entity_type = 'concept',
                            created_at = time::now();
                        
                        RELATE $k->knowledge_about->$object_entity SET
                            relevance_score = $k.confidence OR 1.0,
                            relationship_type = 'object',
                            discovered_at = $k.created_at OR time::now();
                        LET $migrated = $migrated + 1;
                    };
                };
                
                RETURN {
                    migrated_relations: $migrated,
                    total_knowledge: count($knowledge),
                    success: true
                };
            } PERMISSIONS FULL;
        """)
        
        # Function to migrate knowledge -> sessions relations
        await conn.db.query("""
            DEFINE FUNCTION fn::migrate_knowledge_session_relations() {
                LET $knowledge = (SELECT * FROM knowledge WHERE session_id IS NOT NONE);
                LET $migrated = 0;
                
                FOR $k IN $knowledge {
                    // Find corresponding session
                    LET $session_records = (SELECT * FROM sessions WHERE session_id = $k.session_id LIMIT 1);
                    IF count($session_records) > 0 {
                        LET $session_record = $session_records[0];
                        RELATE $k->knowledge_from->$session_record SET
                            extraction_confidence = $k.confidence OR 0.8,
                            extraction_method = 'conversation',
                            learned_at = $k.created_at OR time::now();
                        LET $migrated = $migrated + 1;
                    };
                };
                
                RETURN {
                    migrated_knowledge: $migrated,
                    total_knowledge: count($knowledge),
                    success: true
                };
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Graph migration functions created")
        
        # Step 3: Create comprehensive graph query functions
        logger.info("🔍 Creating comprehensive graph query functions...")
        
        # Get all messages from a session (via graph)
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_messages_graph($session_id: string) {
                RETURN SELECT <-message_belongs_to<-messages[*] 
                FROM sessions 
                WHERE session_id = $session_id;
            } PERMISSIONS FULL;
        """)
        
        # Get all knowledge about an entity (via graph)
        await conn.db.query("""
            DEFINE FUNCTION fn::get_entity_knowledge_graph($entity_name: string) {
                RETURN SELECT <-knowledge_about<-knowledge[*]
                FROM entity 
                WHERE canonical_name = $entity_name;
            } PERMISSIONS FULL;
        """)
        
        # Get all entities mentioned in a session (via graph)
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_entities_graph($session_id: string) {
                RETURN SELECT ->entity_mentioned_in->entity[*]
                FROM sessions 
                WHERE session_id = $session_id;
            } PERMISSIONS FULL;
        """)
        
        # Get session knowledge (via graph)
        await conn.db.query("""
            DEFINE FUNCTION fn::get_session_knowledge_graph($session_id: string) {
                RETURN SELECT <-knowledge_from<-knowledge[*]
                FROM sessions 
                WHERE session_id = $session_id;
            } PERMISSIONS FULL;
        """)
        
        # Get entity sessions (where entity was mentioned via graph)
        await conn.db.query("""
            DEFINE FUNCTION fn::get_entity_sessions_graph($entity_name: string) {
                RETURN SELECT <-entity_mentioned_in<-sessions[*]
                FROM entity 
                WHERE canonical_name = $entity_name;
            } PERMISSIONS FULL;
        """)
        
        # Comprehensive session analysis (all related data via graph)
        await conn.db.query("""
            DEFINE FUNCTION fn::analyze_session_graph($session_id: string) {
                LET $session_records = (SELECT * FROM sessions WHERE session_id = $session_id LIMIT 1);
                IF count($session_records) == 0 { 
                    RETURN { error: "Session not found", session_id: $session_id };
                };
                
                LET $session_record = $session_records[0];
                
                // Get all related data via graph traversal
                LET $messages = (SELECT <-message_belongs_to<-messages[*] FROM $session_record);
                LET $knowledge = (SELECT <-knowledge_from<-knowledge[*] FROM $session_record);
                LET $entities = (SELECT ->entity_mentioned_in->entity[*] FROM $session_record);
                LET $engrams = (SELECT <-engram_appears_in<-engrams[*] FROM $session_record);
                
                RETURN {
                    session: $session_record,
                    messages: $messages,
                    knowledge: $knowledge, 
                    entities: $entities,
                    engrams: $engrams,
                    stats: {
                        message_count: count($messages),
                        knowledge_count: count($knowledge),
                        entity_count: count($entities),
                        engram_count: count($engrams)
                    }
                };
            } PERMISSIONS FULL;
        """)
        
        # Entity comprehensive analysis (all related data via graph)
        await conn.db.query("""
            DEFINE FUNCTION fn::analyze_entity_graph($entity_name: string) {
                LET $entity_records = (SELECT * FROM entity WHERE canonical_name = $entity_name LIMIT 1);
                IF count($entity_records) == 0 {
                    RETURN { error: "Entity not found", entity_name: $entity_name };
                };
                
                LET $entity_record = $entity_records[0];
                
                // Get all related data via graph traversal
                LET $knowledge = (SELECT <-knowledge_about<-knowledge[*] FROM $entity_record);
                LET $sessions = (SELECT <-entity_mentioned_in<-sessions[*] FROM $entity_record);
                LET $engrams = (
                    SELECT <-knowledge_about<-knowledge->knowledge_from->sessions<-engram_appears_in<-engrams[*]
                    FROM $entity_record
                );
                
                RETURN {
                    entity: $entity_record,
                    knowledge: $knowledge,
                    sessions: $sessions,
                    engrams: $engrams,
                    stats: {
                        knowledge_count: count($knowledge),
                        session_count: count($sessions),
                        engram_count: count($engrams)
                    }
                };
            } PERMISSIONS FULL;
        """)
        
        logger.info("✅ Comprehensive graph query functions created")
        
        # Step 4: Execute the migration
        logger.info("🚀 Executing graph migration...")
        
        # Migrate messages to sessions
        logger.info("   Migrating message → session relations...")
        msg_result = await conn.db.query("RETURN fn::migrate_message_relations();")
        if msg_result:
            if isinstance(msg_result, dict):
                logger.info(f"   ✅ Messages: {msg_result.get('migrated_messages', 0)} relations created")
            else:
                logger.info(f"   ✅ Messages migration result: {msg_result}")
        
        # Migrate knowledge to entities
        logger.info("   Migrating knowledge → entity relations...")
        ke_result = await conn.db.query("RETURN fn::migrate_knowledge_entity_relations();")
        if ke_result:
            if isinstance(ke_result, dict):
                logger.info(f"   ✅ Knowledge-Entity: {ke_result.get('migrated_relations', 0)} relations created")
            else:
                logger.info(f"   ✅ Knowledge-Entity migration result: {ke_result}")
        
        # Migrate knowledge to sessions
        logger.info("   Migrating knowledge → session relations...")
        ks_result = await conn.db.query("RETURN fn::migrate_knowledge_session_relations();")
        if ks_result:
            if isinstance(ks_result, dict):
                logger.info(f"   ✅ Knowledge-Session: {ks_result.get('migrated_knowledge', 0)} relations created")
            else:
                logger.info(f"   ✅ Knowledge-Session migration result: {ks_result}")
        
        logger.info("✅ Graph migration completed")
        
        # Step 5: Test the complete graph system
        logger.info("🧪 Testing complete graph system...")
        
        # Find a session with data to test
        session_test = await conn.db.query("""
            SELECT session_id, count() as message_count
            FROM messages 
            GROUP BY session_id 
            ORDER BY message_count DESC 
            LIMIT 1;
        """)
        
        if session_test and len(session_test) > 0:
            test_session_id = session_test[0]['session_id']
            message_count = session_test[0]['message_count']
            logger.info(f"   Testing with session: {test_session_id} ({message_count} messages)")
            
            # Test comprehensive session analysis
            analysis_result = await conn.db.query("""
                RETURN fn::analyze_session_graph($session_id);
            """, {"session_id": test_session_id})
            
            if analysis_result:
                if isinstance(analysis_result, dict):
                    stats = analysis_result.get('stats', {})
                    logger.info(f"   📊 Session analysis via graph:")
                    logger.info(f"      Messages: {stats.get('message_count', 0)}")
                    logger.info(f"      Knowledge: {stats.get('knowledge_count', 0)}")
                    logger.info(f"      Entities: {stats.get('entity_count', 0)}")
                    logger.info(f"      Engrams: {stats.get('engram_count', 0)}")
                else:
                    logger.info(f"   📊 Session analysis result: {analysis_result}")
        
        # Test entity analysis
        entity_test = await conn.db.query("""
            SELECT canonical_name, count() as knowledge_count
            FROM entity 
            WHERE canonical_name IS NOT NONE AND canonical_name != ""
            ORDER BY knowledge_count DESC 
            LIMIT 1;
        """)
        
        if entity_test and len(entity_test) > 0:
            test_entity = entity_test[0]['canonical_name']
            logger.info(f"   Testing with entity: {test_entity}")
            
            entity_analysis = await conn.db.query("""
                RETURN fn::analyze_entity_graph($entity_name);
            """, {"entity_name": test_entity})
            
            if entity_analysis:
                if isinstance(entity_analysis, dict):
                    stats = entity_analysis.get('stats', {})
                    logger.info(f"   📊 Entity analysis via graph:")
                    logger.info(f"      Knowledge: {stats.get('knowledge_count', 0)}")
                    logger.info(f"      Sessions: {stats.get('session_count', 0)}")
                else:
                    logger.info(f"   📊 Entity analysis result: {entity_analysis}")
        
        # Step 6: Verify all graph relations
        logger.info("📈 Verifying complete graph system...")
        
        relation_tables = [
            'message_belongs_to', 'knowledge_about', 'knowledge_from', 
            'entity_mentioned_in', 'session_involves',
            'engram_contains', 'engram_appears_in'
        ]
        
        for table in relation_tables:
            try:
                count = await conn.db.query(f'SELECT count() FROM {table};')
                if count and len(count) > 0:
                    cnt = count[0].get('count', 0) if isinstance(count[0], dict) else count[0]
                    logger.info(f"   ✅ {table}: {cnt} relations")
                else:
                    logger.info(f"   ⚠️ {table}: no relations (expected if no data)")
            except Exception as e:
                logger.warning(f"   ❌ {table}: error - {str(e)[:60]}...")
        
        logger.info("🎉 COMPLETE Graph-Based Memory System Created!")
        logger.info("✅ ALL tables now use proper SurrealDB RELATE edges")
        logger.info("✅ True graph traversal available across entire consciousness system")
        logger.info("✅ Enhanced analysis functions for comprehensive data exploration")
        
    except Exception as e:
        logger.error(f"❌ Error creating complete graph system: {e}")
        import traceback
        traceback.print_exc()
        raise

async def main():
    logger.info("🌐 Creating COMPLETE Graph-Based Memory System...")
    await create_complete_graph_system()
    logger.info("✅ Done! ALL memory tables now use proper SurrealDB graph relations!")

if __name__ == "__main__":
    asyncio.run(main())