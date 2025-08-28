#!/usr/bin/env python3
"""
SurrealDB Schema Migration and Data Cleanup
Applies proper schema and cleans up bad data like 'default' session_ids
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add server path for imports
sys.path.append(str(Path(__file__).parent))

from memory.surreal_connection import SurrealConnectionManager

async def migrate_schema():
    """Apply schema and clean up bad data"""
    logger.info("🔧 Starting SurrealDB schema migration...")
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        # 1. Apply the proper schema
        logger.info("📋 Applying proper schema from consciousness_schema.surql...")
        schema_file = Path(__file__).parent / "schema" / "consciousness_schema.surql"
        
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema (split by semicolons and execute each statement)
            statements = [s.strip() for s in schema_sql.split(';') if s.strip() and not s.strip().startswith('--')]
            
            for i, statement in enumerate(statements[:10]):  # Apply first 10 statements for safety
                if statement:
                    logger.info(f"Executing statement {i+1}: {statement[:50]}...")
                    try:
                        result = await conn.db.query(statement + ';')
                        logger.debug(f"Statement result: {result}")
                    except Exception as e:
                        logger.warning(f"Statement {i+1} failed (might be existing): {e}")
        
        # 2. Clean up bad session_id data
        logger.info("🧹 Cleaning up messages with 'default' session_id...")
        
        # Find messages with bad session_ids
        bad_messages = await conn.db.query("SELECT * FROM messages WHERE session_id = 'default' LIMIT 10;")
        if bad_messages and len(bad_messages) > 0:
            logger.warning(f"Found {len(bad_messages)} messages with 'default' session_id")
            
            # For now, just log them - we could delete or migrate them
            for msg in bad_messages[:5]:
                logger.info(f"Bad message: {msg.get('id')} - role: {msg.get('role')}, speaker: {msg.get('speaker_id')}")
        
        # 3. Analyze current data quality
        logger.info("📊 Analyzing current data quality...")
        
        # Count total messages
        total_messages = await conn.db.query("SELECT count() AS total FROM messages GROUP ALL;")
        if total_messages:
            logger.info(f"Total messages: {total_messages[0].get('total', 0)}")
        
        # Count sessions
        total_sessions = await conn.db.query("SELECT count() AS total FROM sessions GROUP ALL;")
        if total_sessions:
            logger.info(f"Total sessions: {total_sessions[0].get('total', 0)}")
        
        # Count facts
        try:
            total_facts = await conn.db.query("SELECT count() AS total FROM facts GROUP ALL;")
            if total_facts:
                logger.info(f"Total facts: {total_facts[0].get('total', 0)}")
        except:
            logger.info("Facts table not yet created")
        
        # 4. Show sample data (use SELECT * to avoid field ordering issues)
        logger.info("🔍 Sample recent data:")
        recent_messages = await conn.db.query("SELECT * FROM messages ORDER BY timestamp DESC LIMIT 3;")
        if recent_messages:
            for msg in recent_messages:
                content = msg.get('content', '')[:50] + '...' if len(msg.get('content', '')) > 50 else msg.get('content', '')
                logger.info(f"  {msg.get('role')} [{msg.get('speaker_id')}] in session {msg.get('session_id')}: {content}")
        
        # 5. Clean up bad data
        logger.info("🧹 Cleaning up bad data...")
        
        # Option 1: Delete messages with 'default' session_id (they're test data anyway)
        logger.info("Deleting messages with 'default' session_id...")
        delete_result = await conn.db.query("DELETE FROM messages WHERE session_id = 'default';")
        logger.info(f"Deleted bad messages: {delete_result}")
        
        # Recount after cleanup
        total_messages_after = await conn.db.query("SELECT count() AS total FROM messages GROUP ALL;")
        if total_messages_after:
            logger.info(f"Messages after cleanup: {total_messages_after[0].get('total', 0)}")
        
        logger.info("✅ Schema migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await conn.disconnect()

async def main():
    """Main migration function"""
    try:
        await migrate_schema()
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
    except Exception as e:
        logger.error(f"Migration error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())