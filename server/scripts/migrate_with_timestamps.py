#!/usr/bin/env python3
"""
Migration with Proper Timestamps

Migrate data preserving original timestamps from the backup.
"""

import asyncio
import uuid
from datetime import datetime
from surrealdb import AsyncSurreal
from loguru import logger
from extract_backup_data import SurrealQLParser

def convert_timestamp(timestamp_str):
    """Convert ISO timestamp string to SurrealDB datetime format"""
    if not timestamp_str or timestamp_str == 'time::now()':
        return 'time::now()'
    
    try:
        # Parse ISO format like "2025-08-22T11:21:14.071981Z"
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        # Format for SurrealDB: d'2025-08-22T11:21:14.071981Z'
        return f"d'{timestamp_str}'"
    except Exception:
        return 'time::now()'

async def migrate_with_proper_timestamps():
    """Migrate sessions with original timestamps"""
    logger.info("🚀 Starting migration with proper timestamps...")
    
    # Extract data
    backup_file = "/Users/peppi/Dev/macos-local-voice-agents/server/memory/localslowcat-2025-08-27.surql"
    parser = SurrealQLParser(backup_file)
    extracted_data = parser.parse_backup()
    
    # Connect to database
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Clear existing sessions
    await db.query("DELETE session")
    logger.info("Cleared existing sessions")
    
    # Migration 1: Sessions from 'sessions' table (3 records)
    sessions_data = extracted_data.get('sessions', [])
    logger.info(f"Migrating {len(sessions_data)} sessions from 'sessions' table...")
    
    session_count = 0
    session_id_map = {}
    
    for session_record in sessions_data:
        try:
            # Get speaker and generate user_id
            speaker_id = session_record.get('speaker_id', 'unknown')
            if speaker_id == 'unknown':
                continue
                
            user_id = f"user:{speaker_id}"
            new_session_id = f"session:{uuid.uuid4().hex[:12]}"
            
            # Get original timestamps
            first_seen = convert_timestamp(session_record.get('first_seen'))
            last_interaction = convert_timestamp(session_record.get('last_interaction'))
            
            # Calculate duration if possible
            duration_secs = 0
            if 'first_seen' in session_record and 'last_interaction' in session_record:
                try:
                    start = datetime.fromisoformat(session_record['first_seen'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(session_record['last_interaction'].replace('Z', '+00:00'))
                    duration_secs = int((end - start).total_seconds())
                except Exception:
                    pass
            
            # Create session with original timestamps
            result = await db.query(f"""
                CREATE {new_session_id} SET
                    user_id = {user_id},
                    agent_id = 'slowcat',
                    started_at = {first_seen},
                    ended_at = {last_interaction},
                    turn_count = $turn_count,
                    summary = $summary,
                    keywords = $keywords,
                    status = 'ended',
                    duration_secs = $duration_secs
            """, {
                "turn_count": session_record.get('total_turns', 0),
                "summary": f"Session with {session_record.get('session_count', 0)} conversations",
                "keywords": [speaker_id, 'conversation'],
                "duration_secs": duration_secs
            })
            
            # Map old session record for reference
            old_id = session_record.get('id', '')
            session_id_map[old_id] = new_session_id
            session_count += 1
            
            logger.info(f"✅ Created session: {new_session_id} (duration: {duration_secs}s, turns: {session_record.get('total_turns', 0)})")
            
        except Exception as e:
            logger.error(f"❌ Failed to create session from sessions table: {e}")
    
    # Migration 2: Sessions from 'session_summary' table (80 records) 
    session_summary_data = extracted_data.get('session_summary', [])
    logger.info(f"Migrating {len(session_summary_data)} sessions from 'session_summary' table...")
    
    for summary_record in session_summary_data:
        try:
            # Extract session info
            old_session_id = summary_record.get('session_id', '')
            
            # Extract speaker from session_id (e.g., "peppi:1756134556")
            if ':' in old_session_id:
                speaker_id = old_session_id.split(':')[0]
            else:
                continue
                
            user_id = f"user:{speaker_id}"
            new_session_id = f"session:{uuid.uuid4().hex[:12]}"
            
            # Get original timestamp
            timestamp_str = summary_record.get('ts', '')
            session_timestamp = convert_timestamp(timestamp_str)
            
            # Get duration
            duration_secs = summary_record.get('duration_s', 0)
            
            # Create session
            result = await db.query(f"""
                CREATE {new_session_id} SET
                    user_id = {user_id},
                    agent_id = 'slowcat',
                    started_at = {session_timestamp},
                    ended_at = {session_timestamp},
                    turn_count = $turn_count,
                    summary = $summary,
                    keywords = $keywords,
                    status = 'ended',
                    duration_secs = $duration_secs
            """, {
                "turn_count": summary_record.get('turns', 0),
                "summary": summary_record.get('summary', ''),
                "keywords": summary_record.get('keywords', []),
                "duration_secs": duration_secs
            })
            
            # Map for message creation later
            session_id_map[old_session_id] = new_session_id
            session_count += 1
            
            logger.info(f"✅ Created session from summary: {new_session_id} (duration: {duration_secs}s, turns: {summary_record.get('turns', 0)})")
            
        except Exception as e:
            logger.error(f"❌ Failed to create session from session_summary: {e}")
    
    # Check results
    try:
        result = await db.query("SELECT count() FROM session")
        total_count = result[0].get('count', 0) if result else 0
        logger.info(f"📊 Total sessions created: {total_count}")
        
        if total_count > 0:
            # Show samples
            samples = await db.query("SELECT id, user_id, started_at, ended_at, turn_count, duration_secs FROM session LIMIT 5")
            for session in samples:
                logger.info(f"  📅 {session['id']}: {session['started_at']} → {session['ended_at']} ({session['turn_count']} turns, {session['duration_secs']}s)")
                
    except Exception as e:
        logger.error(f"Failed to check created sessions: {e}")
    
    await db.close()
    logger.info(f"✅ Migration with timestamps completed! Created {session_count} sessions")

if __name__ == "__main__":
    asyncio.run(migrate_with_proper_timestamps())