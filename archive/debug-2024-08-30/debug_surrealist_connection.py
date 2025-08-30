#!/usr/bin/env python3
"""Test connection exactly like Surrealist is using"""

import asyncio
from surrealdb import AsyncSurreal
from loguru import logger

async def test_surrealist_connection():
    """Test connection matching what you see in Surrealist"""
    
    # Try different authentication methods
    auth_methods = [
        # Method 1: Root signin
        {"method": "root", "user": "root", "pass": "slowcat_secure_2024"},
        # Method 2: User signin  
        {"method": "user", "user": "slowcat", "pass": "slowcat"},
        # Method 3: Different root
        {"method": "root", "user": "root", "pass": "root"},
    ]
    
    for auth in auth_methods:
        print(f"\n🔐 Testing authentication: {auth['method']} - {auth['user']}")
        
        surreal = AsyncSurreal("http://127.0.0.1:8000")
        
        try:
            if auth["method"] == "root":
                await surreal.signin({
                    "username": auth["user"],  # Using username instead of user
                    "password": auth["pass"]
                })
            else:
                await surreal.signin({
                    "user": auth["user"],
                    "pass": auth["pass"] 
                })
            
            print(f"   ✅ Authentication successful!")
            
            # Try different namespace/database combinations
            db_combinations = [
                ("slowcat", "memory_graph"),
                ("slowcat", "memory"),  
                ("macos-local-voice-agents", "memory_graph"),
                ("macos-local-voice-agents", "memory"),
                ("default", "memory_graph"),
                ("default", "memory"),
            ]
            
            for ns, db in db_combinations:
                try:
                    await surreal.use(ns, db)
                    print(f"   ✅ Connected to {ns}.{db}")
                    
                    # Quick test for entity table
                    entity_result = await surreal.query("SELECT * FROM entity LIMIT 2;")
                    if entity_result and entity_result[0].get('result') and len(entity_result[0]['result']) > 0:
                        entities = entity_result[0]['result']
                        print(f"      🎯 FOUND DATA! Entity table: {len(entities)} records")
                        for i, entity in enumerate(entities[:2]):
                            print(f"         {i+1}: {entity}")
                        
                        # Also check facts if entities found
                        facts_result = await surreal.query("SELECT * FROM facts LIMIT 2;")
                        if facts_result and facts_result[0].get('result') and len(facts_result[0]['result']) > 0:
                            facts = facts_result[0]['result'] 
                            print(f"      🎯 Facts table: {len(facts)} records")
                            for i, fact in enumerate(facts[:1]):
                                print(f"         {i+1}: {fact}")
                        
                        print(f"   🎉 SUCCESS! Found data in {ns}.{db}")
                        return  # Found data, exit early
                        
                    else:
                        print(f"      ❓ No entity data in {ns}.{db}")
                        
                except Exception as db_error:
                    print(f"      ❌ Failed to connect to {ns}.{db}: {db_error}")
            
            print(f"   ⚠️ No data found in any database combination")
            break  # Exit after testing first successful auth
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        finally:
            try:
                await surreal.close()
            except:
                pass

if __name__ == "__main__":
    asyncio.run(test_surrealist_connection())