#!/usr/bin/env python3
"""Check if consciousness schema (entity fields, engrams table) exists in DB."""

import asyncio
import sys
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

from memory.surreal_connection import SurrealConnectionManager

async def check_consciousness_schema():
    """Check if consciousness infrastructure exists in the database."""
    
    connection_manager = SurrealConnectionManager()
    
    try:
        await connection_manager.connect()
        print("✅ Connected to SurrealDB")
        
        print("\n1. Checking entity table fields...")
        entity_info = await connection_manager.db.query("INFO FOR TABLE entity;")
        print(f"Entity table info: {entity_info}")
        
        print("\n2. Checking if engrams table exists...")
        try:
            engrams_info = await connection_manager.db.query("INFO FOR TABLE engrams;")
            print(f"Engrams table info: {engrams_info}")
        except Exception as e:
            print(f"❌ Engrams table not found: {e}")
        
        print("\n3. Checking for any existing engrams...")
        try:
            engrams_count = await connection_manager.db.query("SELECT count() FROM engrams GROUP ALL;")
            print(f"Engrams count: {engrams_count}")
        except Exception as e:
            print(f"❌ Cannot query engrams: {e}")
        
        print("\n4. Checking entity records with consciousness fields...")
        try:
            entities = await connection_manager.db.query("""
                SELECT canonical_name, global_salience, last_activation 
                FROM entity 
                LIMIT 5;
            """)
            print(f"Sample entities with consciousness fields: {entities}")
        except Exception as e:
            print(f"❌ Cannot query consciousness fields: {e}")
        
        print("\n5. Checking for engram detection function...")
        try:
            function_info = await connection_manager.db.query("INFO FOR DB;")
            print(f"Database functions: {function_info}")
        except Exception as e:
            print(f"❌ Cannot check functions: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await connection_manager.disconnect()

if __name__ == "__main__":
    asyncio.run(check_consciousness_schema())