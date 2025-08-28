#!/usr/bin/env python3
"""
Test script to check sessions in SurrealDB
"""

import asyncio
import os
from surrealdb import AsyncSurreal
from datetime import datetime

async def check_surrealdb_sessions():
    """Check sessions stored in SurrealDB"""
    
    print("=" * 60)
    print("SURREALDB SESSION CHECK")
    print("=" * 60)
    
    # Connection parameters from environment
    url = os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc')
    user = os.getenv('SURREALDB_USER', 'root')
    password = os.getenv('SURREALDB_PASS', 'slowcat_secure_2024')
    namespace = os.getenv('SURREALDB_NAMESPACE', 'slowcat')
    database = os.getenv('SURREALDB_DATABASE', 'memory')
    
    print(f"Connecting to: {url}")
    print(f"Namespace: {namespace}, Database: {database}")
    print()
    
    # Connect to SurrealDB
    db = AsyncSurreal(url)
    try:
        await db.use(namespace, database)
        await db.signin({'user': user, 'pass': password})
        
        # Query sessions
        result = await db.query("SELECT * FROM sessions ORDER BY session_count DESC")
        
        if result and len(result) > 0:
            rows = result[0].get('result', []) if isinstance(result[0], dict) else result[0]
            
            print(f"Found {len(rows)} speaker(s) in SurrealDB:")
            print("-" * 60)
            
            for row in rows:
                if isinstance(row, dict):
                    speaker_id = row.get('speaker_id', 'N/A')
                    session_count = row.get('session_count', 0)
                    last_interaction = row.get('last_interaction', 0)
                    first_seen = row.get('first_seen', 0)
                    total_turns = row.get('total_turns', 0)
                    
                    # Format timestamps
                    last_str = datetime.fromtimestamp(last_interaction).strftime('%Y-%m-%d %H:%M:%S') if last_interaction else 'Never'
                    first_str = datetime.fromtimestamp(first_seen).strftime('%Y-%m-%d %H:%M:%S') if first_seen else 'Never'
                    
                    print(f"\nSpeaker: '{speaker_id}'")
                    print(f"  Sessions: {session_count}")
                    print(f"  Total turns: {total_turns}")
                    print(f"  First seen: {first_str}")
                    print(f"  Last seen: {last_str}")
                    
                    # Check against USER_ID
                    user_id = os.getenv('USER_ID', '')
                    if user_id and speaker_id == user_id:
                        print(f"  ✅ This matches USER_ID env var")
        else:
            print("No sessions found in SurrealDB")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await db.close()
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH SQLite")
    print("=" * 60)
    print("The SQLite database at data/facts.db shows:")
    print("  - 'unknown': 104 sessions")
    print("  - 'peppi': 8 sessions")
    print("\nBut since USE_SURREALDB=true, the app uses SurrealDB instead!")
    print("The SQLite sessions are from when the app ran without SurrealDB.")

if __name__ == "__main__":
    # Load .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    asyncio.run(check_surrealdb_sessions())