#!/usr/bin/env python3
"""Debug voice recognition issues"""

import numpy as np
import json
from pathlib import Path
from loguru import logger

# Check speaker profiles
profile_dir = Path("data/speaker_profiles/auto_enrolled")
logger.info(f"Checking profiles in {profile_dir}")

for profile_path in profile_dir.glob("*.json"):
    with open(profile_path) as f:
        data = json.load(f)
    
    name = data.get("name", "Unknown")
    fingerprints = data.get("fingerprints", [])
    
    logger.info(f"\nProfile: {name}")
    logger.info(f"  Enrolled: {data.get('enrolled_at', 'Unknown')}")
    logger.info(f"  Auto-enrolled: {data.get('auto_enrolled', False)}")
    logger.info(f"  Number of fingerprints: {len(fingerprints)}")
    
    if fingerprints:
        fp = np.array(fingerprints[0])
        non_zero = np.count_nonzero(fp)
        logger.info(f"  Fingerprint quality:")
        logger.info(f"    - Length: {len(fp)}")
        logger.info(f"    - Non-zero values: {non_zero}/{len(fp)} ({non_zero/len(fp)*100:.1f}%)")
        logger.info(f"    - Mean: {np.mean(fp):.4f}")
        logger.info(f"    - Std: {np.std(fp):.4f}")
        logger.info(f"    - Max: {np.max(fp):.4f}")

# Check speaker names
names_file = Path("data/speaker_profiles/speaker_names.json")
if names_file.exists():
    with open(names_file) as f:
        names_data = json.load(f)
    logger.info(f"\nSpeaker name mappings: {names_data.get('mappings', {})}")

# Check recent memory entries
import sqlite3
conn = sqlite3.connect("data/memory/memory.sqlite")
cursor = conn.cursor()

logger.info("\nRecent conversations by user:")
cursor.execute("""
    SELECT user_id, COUNT(*) as count, MAX(timestamp) as last_seen 
    FROM conversations 
    WHERE timestamp > datetime('now', '-1 hour')
    GROUP BY user_id
""")
for row in cursor.fetchall():
    logger.info(f"  {row[0]}: {row[1]} messages, last: {row[2]}")

conn.close()