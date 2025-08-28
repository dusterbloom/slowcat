#!/usr/bin/env python3
"""
Test script to see what's actually happening with session tracking in production
"""

import sqlite3
import os
from datetime import datetime

# Check environment
print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

# Check if USER_ID is set
user_id = os.getenv('USER_ID', '')
print(f"USER_ID env var: '{user_id}' (empty={not user_id})")

# Check if .env is being loaded
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if 'USER_ID' in line:
                print(f".env contains: {line.strip()}")
                break

# Check database
print("\n" + "=" * 60)
print("DATABASE CHECK")
print("=" * 60)

db_path = "data/facts.db"
if not os.path.exists(db_path):
    print(f"ERROR: Database does not exist at {db_path}")
else:
    print(f"Database exists: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get all sessions
    cur.execute("SELECT * FROM sessions ORDER BY session_count DESC")
    rows = cur.fetchall()
    
    print(f"\nFound {len(rows)} speaker(s) in database:")
    print("-" * 60)
    
    for row in rows:
        speaker_id = row['speaker_id']
        session_count = row['session_count']
        last_interaction = row['last_interaction']
        first_seen = row['first_seen']
        total_turns = row['total_turns']
        
        # Format timestamps
        last_str = datetime.fromtimestamp(last_interaction).strftime('%Y-%m-%d %H:%M:%S') if last_interaction else 'Never'
        first_str = datetime.fromtimestamp(first_seen).strftime('%Y-%m-%d %H:%M:%S') if first_seen else 'Never'
        
        print(f"\nSpeaker: '{speaker_id}'")
        print(f"  Sessions: {session_count}")
        print(f"  Total turns: {total_turns}")
        print(f"  First seen: {first_str}")
        print(f"  Last seen: {last_str}")
        
        # Check if this is the expected speaker
        if user_id and speaker_id == user_id:
            print(f"  ✅ This matches USER_ID env var")
        elif speaker_id == 'unknown':
            print(f"  ⚠️ This is the default 'unknown' speaker")
        elif speaker_id == 'default_user':
            print(f"  ⚠️ This is the fallback 'default_user'")
    
    conn.close()

# Analysis
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

if user_id:
    print(f"✅ USER_ID is set to: {user_id}")
else:
    print("❌ USER_ID is not set or empty")
    
print("\nPOSSIBLE ISSUES:")
print("1. If 'unknown' has more sessions than your USER_ID:")
print("   - The USER_ID env var might not be loaded at startup")
print("   - Voice recognition might be overriding the USER_ID")
print("   - The app might have been run without the .env file")
print("\n2. If your USER_ID has low session count:")
print("   - The USER_ID was recently added")
print("   - Sessions before that were tracked as 'unknown'")
print("\n3. To fix:")
print("   - Ensure .env is loaded before bot starts")
print("   - Check if run_bot.sh sources .env properly")
print("   - Consider migrating 'unknown' sessions to your USER_ID")