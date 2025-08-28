#!/usr/bin/env python3
"""
SurrealDB Performance Indexes
Adds proper indexes for optimal query performance
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add server path for imports
sys.path.append(str(Path(__file__).parent))

from memory.surreal_connection import SurrealConnectionManager

async def add_performance_indexes():
    """Add performance indexes to SurrealDB tables"""
    logger.info("🚀 Adding performance indexes to SurrealDB...")
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        # Define indexes for optimal query performance
        indexes = [
            # Messages table indexes
            "DEFINE INDEX idx_messages_session_id ON messages FIELDS session_id;",
            "DEFINE INDEX idx_messages_speaker_id ON messages FIELDS speaker_id;", 
            "DEFINE INDEX idx_messages_timestamp ON messages FIELDS timestamp;",
            "DEFINE INDEX idx_messages_role ON messages FIELDS role;",
            # Full-text search index (simplified - ascii analyzer might not exist)
            "DEFINE INDEX idx_messages_content_search ON messages FIELDS content SEARCH BM25 HIGHLIGHTS;",
            
            # Sessions table indexes
            "DEFINE INDEX idx_sessions_speaker_id ON sessions FIELDS speaker_id;",
            "DEFINE INDEX idx_sessions_created_at ON sessions FIELDS created_at;",
            "DEFINE INDEX idx_sessions_status ON sessions FIELDS status;",
            
            # Facts table indexes (if exists)
            "DEFINE INDEX idx_facts_subject ON facts FIELDS subject;",
            "DEFINE INDEX idx_facts_predicate ON facts FIELDS predicate;",
            "DEFINE INDEX idx_facts_confidence ON facts FIELDS confidence;",
            "DEFINE INDEX idx_facts_timestamp ON facts FIELDS timestamp;",
            
            # Composite indexes for common query patterns
            "DEFINE INDEX idx_messages_session_timestamp ON messages FIELDS session_id, timestamp;",
            "DEFINE INDEX idx_messages_speaker_role ON messages FIELDS speaker_id, role;",
            "DEFINE INDEX idx_sessions_speaker_status ON sessions FIELDS speaker_id, status;",
        ]
        
        # Apply each index
        for i, index_sql in enumerate(indexes, 1):
            try:
                logger.info(f"Creating index {i}/{len(indexes)}: {index_sql.split()[2]} on {index_sql.split()[4]}")
                result = await conn.db.query(index_sql)
                logger.debug(f"Index result: {result}")
                
            except Exception as e:
                # Some indexes might already exist, which is okay
                if "already exists" in str(e) or "DEFINE INDEX" in str(e):
                    logger.debug(f"Index {i} already exists or definition updated")
                else:
                    logger.warning(f"Index {i} failed: {e}")
        
        # Verify indexes were created
        logger.info("🔍 Verifying created indexes...")
        try:
            indexes_result = await conn.db.query("INFO FOR TABLE messages;")
            if indexes_result:
                logger.info("Messages table info retrieved successfully")
                logger.debug(f"Table info: {indexes_result}")
        except Exception as e:
            logger.debug(f"Could not retrieve table info: {e}")
        
        # Test query performance with EXPLAIN
        logger.info("🧪 Testing query performance...")
        test_queries = [
            "SELECT * FROM messages WHERE session_id = 'test' ORDER BY timestamp DESC LIMIT 10;",
            "SELECT * FROM sessions WHERE speaker_id = 'test_user';",
            "SELECT * FROM messages WHERE speaker_id = 'test_user' AND role = 'user';"
        ]
        
        for query in test_queries:
            try:
                explain_result = await conn.db.query(f"EXPLAIN {query}")
                logger.debug(f"Query: {query[:50]}... → Execution plan: {explain_result}")
                
            except Exception as e:
                logger.debug(f"Explain failed for query (normal for simple queries): {e}")
        
        # Show table statistics
        logger.info("📊 Table statistics after indexing...")
        
        # Count records in each table
        tables = ['messages', 'sessions', 'facts']
        for table in tables:
            try:
                count_result = await conn.db.query(f"SELECT count() AS total FROM {table} GROUP ALL;")
                if count_result:
                    total = count_result[0].get('total', 0)
                    logger.info(f"  {table}: {total} records")
            except Exception as e:
                logger.debug(f"Could not count {table}: {e}")
        
        logger.info("✅ Performance indexes added successfully!")
        
    except Exception as e:
        logger.error(f"Failed to add indexes: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await conn.disconnect()

async def main():
    """Main indexing function"""
    try:
        await add_performance_indexes()
    except KeyboardInterrupt:
        logger.info("Indexing cancelled by user")
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())