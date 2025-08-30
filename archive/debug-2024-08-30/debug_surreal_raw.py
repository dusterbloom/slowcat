#!/usr/bin/env python3
"""Debug SurrealDB connection directly with raw queries"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def debug_raw_connection():
    """Test raw connection to match Surrealist"""
    
    surreal = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    
    try:
        # Authenticate exactly like Surrealist would
        await surreal.signin({"user": "root", "pass": "slowcat_secure_2024"})
        
        # Try different database combinations
        databases_to_try = [
            ("slowcat", "memory_graph"),
            ("slowcat", "memory"),
            ("macos-local-voice-agents", "memory"),
            ("macos-local-voice-agents", "memory_graph"),
        ]
        
        for ns, db in databases_to_try:
            try:
                print(f"\n🔍 Trying namespace='{ns}', database='{db}'")
                await surreal.use(ns, db)
                
                # Test basic query
                result = await surreal.query("SELECT * FROM entity LIMIT 5;")
                if result and result[0].get('result'):
                    entities = result[0]['result']
                    print(f"   ✅ Found {len(entities)} entities")
                    for i, entity in enumerate(entities[:3]):
                        print(f"      {i+1}: {entity}")
                else:
                    print(f"   ❌ No entities found")
                
                # Test facts table
                facts_result = await surreal.query("SELECT * FROM facts LIMIT 5;")
                if facts_result and facts_result[0].get('result'):
                    facts = facts_result[0]['result']
                    print(f"   ✅ Found {len(facts)} facts")
                    for i, fact in enumerate(facts[:2]):
                        print(f"      {i+1}: {fact}")
                else:
                    print(f"   ❌ No facts found")
                    
            except Exception as e:
                print(f"   ❌ Error with {ns}/{db}: {e}")
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
    
    finally:
        try:
            await surreal.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(debug_raw_connection())