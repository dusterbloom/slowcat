#!/usr/bin/env python3
"""
Clear Database and Re-run Migration

Clean slate migration after fixing parameterized query issues.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger
from migrate_to_graph import main as migrate_main

async def clear_database():
    """Clear all data from database tables"""
    logger.info("🧹 Clearing database tables...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Clear all tables
    tables = ['user', 'session', 'message', 'concept', 'thought', 'knows', 'contains', 'reflects', 'mentions']
    
    for table in tables:
        try:
            result = await db.query(f"DELETE {table}")
            logger.info(f"Cleared table: {table}")
        except Exception as e:
            logger.warning(f"Failed to clear {table}: {e}")
    
    await db.close()
    logger.info("✅ Database cleared")

async def main():
    """Clear database and run migration"""
    logger.info("🚀 Starting clean migration...")
    
    # Step 1: Clear existing data  
    await clear_database()
    
    # Step 2: Run migration
    await migrate_main()
    
    logger.info("✅ Clean migration completed")

if __name__ == "__main__":
    asyncio.run(main())