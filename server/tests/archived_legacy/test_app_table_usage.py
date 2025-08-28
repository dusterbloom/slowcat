#!/usr/bin/env python3
"""
Test which tables the app is actually using for read/write operations
"""

import asyncio
from dotenv import load_dotenv
load_dotenv('.env')

from memory import create_smart_memory_system

async def test_app_table_usage():
    """Test which tables the memory system actually uses"""
    print("🔍 Testing which tables the app uses...")
    
    # Create memory system (same as the app does)
    memory = create_smart_memory_system()
    
    # Test 1: Store some facts
    print("\n1. Testing fact storage...")
    try:
        facts_stored = await memory.store_facts("I love pizza and ice cream")
        print(f"   Stored {facts_stored} facts")
    except Exception as e:
        print(f"   Error storing facts: {e}")
    
    # Test 2: Query for existing knowledge
    print("\n2. Testing knowledge query...")
    try:
        response = await memory.process_query("Potola")
        print(f"   Query returned {len(response.results)} results")
        for i, result in enumerate(response.results[:3]):
            print(f"     - {result.content} (from {result.source_store}) - {result.subject} -> {result.value}")
    except Exception as e:
        print(f"   Error querying: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check which database and tables it's connecting to
    print("\n3. Testing direct database connection...")
    try:
        if hasattr(memory, 'surreal_memory'):
            surreal_mem = memory.surreal_memory
            print(f"   Using database: {surreal_mem.database}")
            print(f"   Using namespace: {surreal_mem.namespace}")
            
            # Check if connected to our graph database
            from surrealdb import AsyncSurreal
            db = AsyncSurreal(f"ws://127.0.0.1:8000/rpc")
            await db.connect()
            await db.signin({"username": "root", "password": "slowcat_secure_2024"})
            await db.use("slowcat", "memory_graph")
            
            # Count records in different tables after our test
            print("\n   Table record counts after test:")
            tables = ['message', 'tape', 'session', 'sessions', 'concept', 'fact']
            for table in tables:
                try:
                    result = await db.query(f"SELECT count() FROM {table}")
                    count = result[0]['count'] if result and len(result) > 0 and 'count' in result[0] else 0
                    print(f"     {table}: {count} records")
                except Exception as e:
                    print(f"     {table}: table doesn't exist or error ({e})")
            
            await db.close()
            
    except Exception as e:
        print(f"   Error checking database: {e}")
    
    # Clean up
    await memory.close()
    print("\n✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_app_table_usage())