#!/usr/bin/env python3
"""
Check session data in SurrealDB to see end_time status
"""

import asyncio
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def check_sessions():
    """Check session data to see if end_time is being set"""
    
    try:
        # Connect to SurrealDB using actual .env config
        surreal = SurrealConnectionManager(
            url="ws://127.0.0.1:8000/rpc",
            namespace="slowcat", 
            database="memory_graph"
        )
        
        await surreal.connect()
        logger.info("✅ Connected to SurrealDB")
        
        # Check all sessions
        all_sessions = await surreal.db.query("SELECT * FROM sessions ORDER BY start_time DESC LIMIT 20")
        
        print("🔍 Recent Sessions:")
        print("=" * 80)
        print(f"{'Session ID':<20} {'Speaker':<10} {'Start Time':<20} {'End Time':<20} {'Active'}")
        print("-" * 80)
        
        sessions_with_end = 0
        sessions_without_end = 0
        
        for session in all_sessions:
            session_id = str(session.get('session_id', 'Unknown'))[:18]
            speaker = session.get('speaker_id', 'Unknown')[:10]
            start_time = str(session.get('start_time', 'None'))[:19] if session.get('start_time') else 'None'
            end_time = str(session.get('end_time', 'None'))[:19] if session.get('end_time') else 'None'
            is_active = session.get('is_active', False)
            
            print(f"{session_id:<20} {speaker:<10} {start_time:<20} {end_time:<20} {is_active}")
            
            if end_time != 'None':
                sessions_with_end += 1
            else:
                sessions_without_end += 1
        
        print("-" * 80)
        print(f"📊 Summary:")
        print(f"   Sessions WITH end_time: {sessions_with_end}")
        print(f"   Sessions WITHOUT end_time: {sessions_without_end}")
        
        # Check current active sessions
        active_sessions = await surreal.db.query("SELECT * FROM sessions WHERE is_active = true")
        print(f"   Currently active sessions: {len(active_sessions)}")
        
        # Check for peppi specifically
        peppi_sessions = await surreal.db.query(
            "SELECT * FROM sessions WHERE speaker_id = 'peppi' ORDER BY start_time DESC LIMIT 5"
        )
        
        print(f"\n🎯 Peppi's Recent Sessions:")
        print("-" * 60)
        for session in peppi_sessions:
            session_id = str(session.get('session_id', 'Unknown'))[:18]
            start_time = str(session.get('start_time', 'None'))[:19] if session.get('start_time') else 'None'
            end_time = str(session.get('end_time', 'None'))[:19] if session.get('end_time') else 'None'
            is_active = session.get('is_active', False)
            
            status = "🔴 ACTIVE" if is_active else "⚫ ENDED" if end_time != 'None' else "⚠️ NO END TIME"
            print(f"   {session_id} | {start_time} -> {end_time} | {status}")
        
        # Prefer disconnect(); close() remains for backward-compat
        await surreal.disconnect()
        
    except Exception as e:
        logger.error(f"❌ Failed to check sessions: {e}")

if __name__ == "__main__":
    asyncio.run(check_sessions())
