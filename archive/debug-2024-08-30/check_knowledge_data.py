#!/usr/bin/env python3
"""Check what knowledge data exists from the running bot"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def check_knowledge_data():
    """Check current state of knowledge and other tables"""
    
    conn = SurrealConnectionManager()
    
    try:
        await conn.ensure_connected()
        logger.info("✅ Connected to SurrealDB")
        
        # Check knowledge table
        logger.info("🔍 Checking knowledge table...")
        knowledge_result = await conn.db.query("SELECT * FROM knowledge ORDER BY created_at DESC LIMIT 10;")
        if knowledge_result and knowledge_result[0].get('result'):
            knowledge = knowledge_result[0]['result']
            logger.info(f"   📊 Found {len(knowledge)} knowledge records")
            
            for i, fact in enumerate(knowledge):
                subject = fact.get('subject', 'N/A')
                predicate = fact.get('predicate', 'N/A') 
                obj = fact.get('object', fact.get('value', 'N/A'))
                strength = fact.get('strength', 'N/A')
                created = fact.get('created_at', 'N/A')
                accessed = fact.get('last_accessed', 'N/A')
                count = fact.get('access_count', 'N/A')
                
                logger.info(f"   {i+1}. {subject} {predicate} {obj}")
                logger.info(f"      Strength: {strength}, Created: {created}, Accessed: {accessed}, Count: {count}")
        else:
            logger.info("   ❌ No knowledge records found")
        
        # Check entity table
        logger.info("\n🔍 Checking entity table...")
        entity_result = await conn.db.query("SELECT * FROM entity ORDER BY last_referenced DESC LIMIT 5;")
        if entity_result and entity_result[0].get('result'):
            entities = entity_result[0]['result']
            logger.info(f"   📊 Found {len(entities)} entity records")
            
            for i, entity in enumerate(entities):
                name = entity.get('canonical_name', 'N/A')
                entity_type = entity.get('type', 'N/A')
                ref_count = entity.get('reference_count', 'N/A')
                logger.info(f"   {i+1}. {name} ({entity_type}) - refs: {ref_count}")
        else:
            logger.info("   ❌ No entity records found")
        
        # Check messages table
        logger.info("\n🔍 Checking messages table...")
        messages_result = await conn.db.query("SELECT COUNT() AS count FROM messages GROUP ALL;")
        if messages_result and messages_result[0].get('result'):
            count_data = messages_result[0]['result']
            if count_data:
                msg_count = count_data[0].get('count', 0)
                logger.info(f"   📊 Found {msg_count} message records")
            else:
                logger.info("   ❌ No messages found")
        else:
            logger.info("   ❌ No messages table data")
        
        # Check facts table (legacy)
        logger.info("\n🔍 Checking facts table...")
        facts_result = await conn.db.query("SELECT COUNT() AS count FROM facts GROUP ALL;")
        if facts_result and facts_result[0].get('result'):
            count_data = facts_result[0]['result']
            if count_data:
                fact_count = count_data[0].get('count', 0)
                logger.info(f"   📊 Found {fact_count} fact records")
            else:
                logger.info("   ❌ No facts found")
        else:
            logger.info("   ❌ No facts table data")
            
        # Test a memory function with existing data if available
        if knowledge_result and knowledge_result[0].get('result') and knowledge_result[0]['result']:
            logger.info("\n🧪 Testing memory functions with real data...")
            
            # Test search function
            search_result = await conn.db.query("RETURN fn::search_knowledge('user', 5);")
            if search_result and len(search_result) > 0:
                results = search_result[0]
                logger.info(f"   Search 'user': {len(results) if results else 0} results")
            
            # Test entity facts
            entity_facts = await conn.db.query("RETURN fn::get_entity_facts('user');") 
            if entity_facts and len(entity_facts) > 0:
                results = entity_facts[0]
                logger.info(f"   Entity 'user' facts: {len(results) if results else 0} results")
        
        logger.info("\n✅ Data check completed!")
        
    except Exception as e:
        logger.error(f"Check failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if conn.connected:
            await conn.disconnect()

if __name__ == "__main__":
    asyncio.run(check_knowledge_data())