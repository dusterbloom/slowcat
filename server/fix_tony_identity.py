#!/usr/bin/env python3
"""Fix the Tony identity issue by deleting the specific fact and creating peppi fact."""

import asyncio
import os
import sys
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

from memory.surreal_connection import SurrealConnectionManager

async def fix_identity_issue():
    """Delete the Tony fact and create the correct peppi fact."""
    
    # Initialize connection
    connection_manager = SurrealConnectionManager()
    
    try:
        await connection_manager.connect()
        
        print("🔍 Checking current Tony fact...")
        # Query the specific record ID from the logs
        result = await connection_manager.db.query('SELECT * FROM knowledge:yq1mwbnj0320tn2094wh;')
        print(f"Found Tony fact: {result}")
        
        print("❌ Deleting Tony fact...")
        # Delete the specific record
        delete_result = await connection_manager.db.query('DELETE knowledge:yq1mwbnj0320tn2094wh;')
        print(f"Delete result: {delete_result}")
        
        print("✅ Creating peppi fact...")
        # Create the correct peppi fact
        create_result = await connection_manager.db.query("""
            INSERT INTO knowledge (
                subject_id, 
                predicate, 
                object_id, 
                confidence,
                created_at,
                updated_at
            ) VALUES (
                'user',
                'has_person', 
                'peppi',
                0.95,
                time::now(),
                time::now()
            );
        """)
        print(f"Create result: {create_result}")
        
        print("🔍 Verifying the change...")
        # Query to verify the peppi fact exists
        verify_result = await connection_manager.db.query("""
            SELECT * FROM knowledge 
            WHERE subject_id = 'user' AND predicate = 'has_person'
            ORDER BY confidence DESC;
        """)
        print(f"Current user identity facts: {verify_result}")
        
        print("✅ Identity fix complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        await connection_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(fix_identity_issue())