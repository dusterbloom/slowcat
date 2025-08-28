#!/usr/bin/env python3
"""
Inspect Session Data from Backup

Check what timestamp and session data we have in the backup to preserve original times.
"""

import json
from pathlib import Path
from loguru import logger
from extract_backup_data import SurrealQLParser

def inspect_session_data():
    """Inspect the actual session data from backup"""
    logger.info("🔍 Inspecting session data from backup...")
    
    # Extract data
    backup_file = "/Users/peppi/Dev/macos-local-voice-agents/server/memory/localslowcat-2025-08-27.surql"
    parser = SurrealQLParser(backup_file)
    extracted_data = parser.parse_backup()
    
    # Check sessions table
    sessions_data = extracted_data.get('sessions', [])
    logger.info(f"📊 Sessions table: {len(sessions_data)} records")
    
    for i, session in enumerate(sessions_data[:3]):  # Show first 3
        logger.info(f"\nSession {i+1}:")
        for key, value in session.items():
            logger.info(f"  {key}: {value} ({type(value).__name__})")
    
    # Check session_summary table  
    session_summary_data = extracted_data.get('session_summary', [])
    logger.info(f"\n📊 Session_summary table: {len(session_summary_data)} records")
    
    for i, summary in enumerate(session_summary_data[:3]):  # Show first 3
        logger.info(f"\nSession Summary {i+1}:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value} ({type(value).__name__})")
    
    # Check tape data for session timing
    tape_data = extracted_data.get('tape', [])
    logger.info(f"\n📊 Tape table: {len(tape_data)} records")
    
    # Find unique session_ids from tape
    session_ids = set()
    for record in tape_data[:100]:  # Sample first 100
        session_id = record.get('session_id')
        if session_id:
            session_ids.add(session_id)
    
    logger.info(f"📊 Found {len(session_ids)} unique session_ids in tape:")
    for session_id in list(session_ids)[:5]:  # Show first 5
        logger.info(f"  - {session_id}")
    
    # Check timestamp formats in tape
    logger.info(f"\n📊 Timestamp formats in tape (first 3 records):")
    for i, record in enumerate(tape_data[:3]):
        logger.info(f"\nTape Record {i+1}:")
        for key, value in record.items():
            if 'time' in key.lower() or 'ts' in key.lower() or 'date' in key.lower():
                logger.info(f"  {key}: {value} ({type(value).__name__})")
    
    # Export sample data for inspection
    output_dir = Path("extracted_data")
    output_dir.mkdir(exist_ok=True)
    
    # Export sessions sample
    if sessions_data:
        with open(output_dir / "sessions_sample.json", 'w') as f:
            json.dump(sessions_data[:10], f, indent=2, default=str)
        logger.info(f"📝 Exported sessions sample to {output_dir}/sessions_sample.json")
    
    # Export session_summary sample
    if session_summary_data:
        with open(output_dir / "session_summary_sample.json", 'w') as f:
            json.dump(session_summary_data[:10], f, indent=2, default=str)
        logger.info(f"📝 Exported session_summary sample to {output_dir}/session_summary_sample.json")
    
    # Export tape sample
    if tape_data:
        with open(output_dir / "tape_sample.json", 'w') as f:
            json.dump(tape_data[:20], f, indent=2, default=str)
        logger.info(f"📝 Exported tape sample to {output_dir}/tape_sample.json")
    
    logger.info("✅ Session data inspection completed")

if __name__ == "__main__":
    inspect_session_data()