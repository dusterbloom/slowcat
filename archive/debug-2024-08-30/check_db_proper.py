#!/usr/bin/env python3
"""Check database using proper SurrealDB patterns from cheatsheet"""

import asyncio
from surrealdb import AsyncSurreal

async def check_db_proper():
    """Check database using correct patterns"""
    
    # Use AsyncSurreal for async operations
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    
    try:
        await db.connect()
        await db.signin({"username": "root", "password": "slowcat_secure_2024"})
        
        # Try different namespace/database combinations
        combinations = [
            ("slowcat", "memory_graph"),
            ("slowcat", "memory"),
            ("app", "demo"),  # Default from cheatsheet
            ("macos-local-voice-agents", "memory_graph"),
        ]
        
        for ns, database in combinations:
            try:
                print(f"\n🔍 Trying namespace='{ns}', database='{database}'")
                await db.use(namespace=ns, database=database)
                
                # Use proper SurrealQL introspection
                db_info = await db.query("INFO FOR DB;")
                print(f"   ✅ Connected successfully!")
                
                if db_info and len(db_info) > 0:
                    info = db_info[0]
                    if 'result' in info and info['result']:
                        tables = info['result'].get('tables', {})
                        print(f"   📊 Found {len(tables)} tables: {list(tables.keys())}")
                        
                        # Check for knowledge table specifically
                        if 'knowledge' in tables:
                            # Count records in knowledge
                            count_result = await db.query("SELECT count() AS total FROM knowledge GROUP ALL;")
                            if count_result and len(count_result) > 0 and count_result[0].get('result'):
                                total = count_result[0]['result'][0].get('total', 0) if count_result[0]['result'] else 0
                                print(f"   📈 Knowledge table: {total} records")
                                
                                if total > 0:
                                    # Show sample records
                                    sample = await db.query("SELECT * FROM knowledge LIMIT 3;")
                                    if sample and len(sample) > 0 and sample[0].get('result'):
                                        records = sample[0]['result']
                                        print(f"   📋 Sample records:")
                                        for i, record in enumerate(records):
                                            print(f"      {i+1}: {record}")
                            else:
                                print(f"   ❌ Knowledge table is empty")
                        else:
                            print(f"   ❓ No knowledge table found")
                            
                        # Check entity table
                        if 'entity' in tables:
                            count_result = await db.query("SELECT count() AS total FROM entity GROUP ALL;")
                            if count_result and len(count_result) > 0 and count_result[0].get('result'):
                                total = count_result[0]['result'][0].get('total', 0) if count_result[0]['result'] else 0
                                print(f"   📈 Entity table: {total} records")
                        
                        # Check messages table  
                        if 'messages' in tables:
                            count_result = await db.query("SELECT count() AS total FROM messages GROUP ALL;")
                            if count_result and len(count_result) > 0 and count_result[0].get('result'):
                                total = count_result[0]['result'][0].get('total', 0) if count_result[0]['result'] else 0
                                print(f"   📈 Messages table: {total} records")
                                
                        print(f"   🎉 SUCCESS! Found data in {ns}.{database}")
                        return  # Exit once we find data
                
            except Exception as e:
                print(f"   ❌ Failed to connect to {ns}.{database}: {e}")
        
        print("\n⚠️ No data found in any namespace/database combination")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
    finally:
        try:
            await db.close()
        except:
            pass  # Handle close gracefully

if __name__ == "__main__":
    asyncio.run(check_db_proper())