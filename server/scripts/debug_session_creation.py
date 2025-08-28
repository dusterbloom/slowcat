#!/usr/bin/env python3
"""
Debug Session Creation

Debug why sessions aren't being created in the migration script.
"""

import asyncio
import uuid
from surrealdb import AsyncSurreal
from loguru import logger

# Import the extraction logic
from extract_backup_data import SurrealQLParser

async def debug_session_creation():
    """Debug session creation step by step"""
    logger.info("🔍 Debugging session creation...")
    
    # Step 1: Extract data to see what we have
    backup_file = "/Users/peppi/Dev/macos-local-voice-agents/server/memory/localslowcat-2025-08-27.surql"
    parser = SurrealQLParser(backup_file)
    extracted_data = parser.parse_backup()
    
    logger.info("Extracted data summary:")
    for table, records in extracted_data.items():
        logger.info(f"  {table}: {len(records)} records")
        if records and table in ['sessions', 'session_summary']:
            logger.info(f"    Sample {table}: {records[0]}")
    
    # Step 2: Connect to database
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Step 3: Try to create sessions using the same logic as migration script
    logger.info("Testing session creation logic...")
    
    # Clear existing sessions for clean test
    await db.query("DELETE session")
    
    # Test session creation from sessions table
    sessions_data = extracted_data.get('sessions', [])
    logger.info(f"Processing {len(sessions_data)} sessions from 'sessions' table")
    
    for i, session_record in enumerate(sessions_data[:2]):  # Test first 2 only
        logger.info(f"Processing session {i+1}: {session_record}")
        
        try:
            # Extract speaker_id (mimic migration logic)
            speaker_id = session_record.get('speaker_id', session_record.get('user', 'unknown'))
            logger.info(f"  Speaker ID: {speaker_id}")
            
            # Get user_id (assuming user exists)
            user_id = f"user:peppi"  # Hardcode for testing
            logger.info(f"  User ID: {user_id}")
            
            # Generate session ID
            session_id = f"session:{uuid.uuid4().hex[:12]}"
            logger.info(f"  Session ID: {session_id}")
            
            # Create session using working f-string method
            result = await db.query(f"""
                CREATE {session_id} SET
                    user_id = $user_id,
                    agent_id = 'slowcat',
                    started_at = time::now(),
                    turn_count = $turn_count,
                    summary = $summary,
                    keywords = $keywords,
                    status = 'ended'
            """, {
                "user_id": user_id,
                "turn_count": session_record.get('total_turns', 0),
                "summary": session_record.get('summary', ''),
                "keywords": session_record.get('keywords', [])
            })
            
            logger.info(f"✅ Session {i+1} created: {result}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create session {i+1}: {e}")
    
    # Test session creation from session_summary table  
    session_summary_data = extracted_data.get('session_summary', [])
    logger.info(f"Processing {len(session_summary_data)} sessions from 'session_summary' table")
    
    for i, summary in enumerate(session_summary_data[:2]):  # Test first 2 only
        logger.info(f"Processing session summary {i+1}: {summary}")
        
        try:
            # Extract session_id from summary
            old_session_id = summary.get('session_id', '')
            logger.info(f"  Old session ID: {old_session_id}")
            
            # Extract speaker from session_id (e.g., "peppi:1756224...")
            speaker = old_session_id.split(':')[0] if ':' in old_session_id else 'unknown'
            logger.info(f"  Speaker from session_id: {speaker}")
            
            user_id = f"user:peppi"  # Hardcode for testing
            new_session_id = f"session:{uuid.uuid4().hex[:12]}"
            
            # Create session
            result = await db.query(f"""
                CREATE {new_session_id} SET
                    user_id = $user_id,
                    agent_id = 'slowcat',
                    started_at = time::now(),
                    turn_count = 0,
                    summary = $summary,
                    keywords = $keywords,
                    status = 'active'
            """, {
                "user_id": user_id,
                "summary": summary.get('summary', ''),
                "keywords": summary.get('keywords', [])
            })
            
            logger.info(f"✅ Session summary {i+1} created: {result}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create session summary {i+1}: {e}")
    
    # Check what was created
    try:
        result = await db.query("SELECT count() FROM session")
        count = result[0].get('count', 0) if result else 0
        logger.info(f"📊 Total sessions created: {count}")
        
        if count > 0:
            sessions = await db.query("SELECT * FROM session LIMIT 5")
            for session in sessions[:5]:
                logger.info(f"  Created session: {session}")
                
    except Exception as e:
        logger.error(f"Failed to check created sessions: {e}")
    
    await db.close()
    logger.info("✅ Session creation debugging completed")

if __name__ == "__main__":
    asyncio.run(debug_session_creation())