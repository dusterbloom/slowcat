#!/usr/bin/env python3
"""
Simple Validation Test

Test basic queries with proper result handling.
"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def simple_validation():
    """Simple validation with proper result format"""
    logger.info("🔍 Starting simple validation...")
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({
        "username": "root",
        "password": "slowcat_secure_2024"
    })
    await db.use("slowcat", "memory")
    
    # Test direct queries
    tests = [
        ("Users", "SELECT count() FROM user"),
        ("Sessions", "SELECT count() FROM session"), 
        ("Messages", "SELECT count() FROM message"),
        ("Concepts", "SELECT count() FROM concept"),
        ("Knowledge Relations", "SELECT count() FROM knows"),
        ("Contains Relations", "SELECT count() FROM contains"),
        ("Reflects Relations", "SELECT count() FROM reflects")
    ]
    
    results = {}
    
    for name, query in tests:
        try:
            result = await db.query(query)
            logger.info(f"{name} query result: {result}")
            
            if result and len(result) > 0:
                count = result[0].get('count', 0) if isinstance(result[0], dict) else result[0]
                results[name] = count
                logger.info(f"✅ {name}: {count}")
            else:
                results[name] = 0 
                logger.info(f"❌ {name}: 0")
                
        except Exception as e:
            logger.error(f"❌ {name}: Error - {e}")
            results[name] = 0
    
    # Summary
    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)
    
    for name, count in results.items():
        status = "✅" if count > 0 else "❌"
        logger.info(f"{status} {name}: {count}")
    
    await db.close()
    logger.info("✅ Simple validation completed")

if __name__ == "__main__":
    asyncio.run(simple_validation())