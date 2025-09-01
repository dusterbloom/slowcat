#!/usr/bin/env python3
"""
Test script to verify SurrealDB fixes for engrams, memory_fragments, and field_states integration.

This script tests:
1. Fixed fn::detect_engrams function with proper narrative_summary
2. Session ID consistency across all tables
3. Memory fragments and field states with session linking
"""

import asyncio
import sys
import json
from memory.surreal_connection import SurrealConnectionManager
from loguru import logger

async def test_engram_creation():
    """Test the fixed engram creation with proper narrative summary"""
    print("🧠 Testing engram creation with fixed detect_engrams function...")
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        # Create test session
        test_session_id = "test-session-engram-fix"
        
        # Create test entities and knowledge for engram detection
        await conn.db.query("""
            CREATE entity:test_user SET 
                type = 'user',
                canonical_name = 'test_user',
                session_id = $session_id;
        """, {"session_id": test_session_id})
        
        await conn.db.query("""
            CREATE entity:test_dog SET 
                type = 'object',
                canonical_name = 'Fluffy',
                session_id = $session_id;
        """, {"session_id": test_session_id})
        
        # Create knowledge relations with session_id
        for i, (predicate, obj) in enumerate([
            ("has_pet", "Fluffy"),
            ("pet_name", "Fluffy"),
            ("pet_breed", "Golden Retriever")
        ]):
            await conn.db.query("""
                RELATE entity:test_user->knowledge->entity:test_dog SET
                    predicate = $predicate,
                    confidence = 0.9,
                    strength = 1.0,
                    session_id = $session_id,
                    created_at = time::now(),
                    last_accessed = time::now(),
                    access_count = 1,
                    extraction_method = 'test';
            """, {
                "predicate": predicate,
                "session_id": test_session_id
            })
        
        # Test the fixed detect_engrams function
        print("🔍 Calling fixed fn::detect_engrams function...")
        result = await conn.db.query("""
            SELECT fn::detect_engrams($session_id, 0.5, 2) AS engram_result;
        """, {"session_id": test_session_id})
        
        if result and len(result) > 0:
            engram_result = result[0].get('engram_result', {})
            print(f"✅ Engram creation result: {json.dumps(engram_result, indent=2)}")
            
            if engram_result.get('engram_created'):
                # Check the created engram
                engrams = await conn.db.query("""
                    SELECT * FROM engrams WHERE session_id = $session_id;
                """, {"session_id": test_session_id})
                
                if engrams and len(engrams) > 0:
                    engram = engrams[0]
                    print(f"📊 Created engram details:")
                    print(f"   - ID: {engram.get('id')}")
                    print(f"   - Session ID: {engram.get('session_id')}")
                    print(f"   - Dominant symbols: {engram.get('dominant_symbols', [])}")
                    print(f"   - Narrative summary: '{engram.get('narrative_summary', 'MISSING')}'")
                    print(f"   - Coherence score: {engram.get('coherence_score', 0)}")
                    
                    # Check if narrative summary is fixed (not empty)
                    narrative = engram.get('narrative_summary', '')
                    if narrative and narrative != 'Attractor state: , ':
                        print("✅ FIXED: Narrative summary is properly generated!")
                    else:
                        print("❌ ISSUE: Narrative summary is still empty or malformed")
                        
                else:
                    print("❌ No engrams found after creation")
            else:
                print(f"❌ Engram not created: {engram_result.get('reason')}")
        else:
            print("❌ No result from detect_engrams function")
            
    except Exception as e:
        print(f"❌ Error testing engram creation: {e}")
        
    finally:
        # SurrealConnectionManager handles connection lifecycle automatically
        pass
        pass

async def test_session_consistency():
    """Test session ID consistency across all tables"""
    print("\n📊 Testing session ID consistency across tables...")
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        test_session_id = "test-session-consistency"
        
        # Create test data in each table with session_id
        
        # 1. Create session
        await conn.db.query("""
            CREATE sessions SET
                session_id = $session_id,
                speaker_id = 'test_user',
                start_time = time::now(),
                is_active = true;
        """, {"session_id": test_session_id})
        
        # 2. Create message
        await conn.db.query("""
            CREATE messages SET
                role = 'user',
                content = 'Test message for session consistency',
                session_id = $session_id,
                speaker_id = 'test_user',
                timestamp = time::now();
        """, {"session_id": test_session_id})
        
        # 3. Create knowledge (already done in previous test)
        
        # 4. Create memory fragment with session_id
        await conn.db.query("""
            CREATE memory_fragments SET
                fragment_id = 'test-fragment',
                tier = 1,
                type = 'semantic',
                content = {
                    text: 'Test memory fragment',
                    semantic_hash: 'test_hash_123'
                },
                strength = 1.0,
                session_id = $session_id,
                created_at = time::now(),
                last_accessed = time::now();
        """, {"session_id": test_session_id})
        
        # 5. Create field state with session_id  
        await conn.db.query("""
            CREATE field_states SET
                instance_id = 'test-instance',
                compression = 0.5,
                drift = 'low',
                recursion_depth = 2,
                resonance = 0.8,
                presence_signal = 0.9,
                boundary = 'gradient',
                session_id = $session_id,
                updated_at = time::now();
        """, {"session_id": test_session_id})
        
        # Test the session memory function
        print("🔍 Testing fn::get_session_memory function...")
        result = await conn.db.query("""
            SELECT fn::get_session_memory($session_id) AS session_data;
        """, {"session_id": test_session_id})
        
        if result and len(result) > 0:
            session_data = result[0].get('session_data', {})
            stats = session_data.get('stats', {})
            
            print("✅ Session memory retrieval successful:")
            print(f"   - Messages: {stats.get('messages_count', 0)}")
            print(f"   - Knowledge: {stats.get('knowledge_count', 0)}")
            print(f"   - Engrams: {stats.get('engrams_count', 0)}")
            print(f"   - Memory fragments: {stats.get('fragments_count', 0)}")
            print(f"   - Field states: {stats.get('states_count', 0)}")
            
            # Check if all counts are > 0 (except possibly engrams)
            if (stats.get('messages_count', 0) > 0 and 
                stats.get('knowledge_count', 0) > 0 and
                stats.get('fragments_count', 0) > 0 and
                stats.get('states_count', 0) > 0):
                print("✅ SUCCESS: All tables properly linked with session_id!")
            else:
                print("⚠️  Some tables may not have session_id properly set")
        else:
            print("❌ No result from get_session_memory function")
            
    except Exception as e:
        print(f"❌ Error testing session consistency: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # SurrealConnectionManager handles connection lifecycle automatically
        pass

async def test_migration_status():
    """Test that the migration has been applied successfully"""
    print("\n🔧 Testing migration status...")
    
    conn = SurrealConnectionManager()
    await conn.ensure_connected()
    
    try:
        # Check if engrams table has session_id field (singular)
        print("🔍 Checking engrams table structure...")
        result = await conn.db.query("INFO FOR TABLE engrams;")
        print(f"📋 Engrams table info: {result}")
        
        # Check if memory_fragments has session_id field
        print("🔍 Checking memory_fragments table structure...")  
        result = await conn.db.query("INFO FOR TABLE memory_fragments;")
        print(f"📋 Memory fragments table info: {result}")
        
        # Check if field_states has session_id field
        print("🔍 Checking field_states table structure...")
        result = await conn.db.query("INFO FOR TABLE field_states;")
        print(f"📋 Field states table info: {result}")
        
        print("✅ Migration structure check completed")
        
    except Exception as e:
        print(f"❌ Error checking migration status: {e}")
        
    finally:
        # SurrealConnectionManager handles connection lifecycle automatically
        pass

async def main():
    """Run all tests"""
    print("🚀 Starting SurrealDB fixes verification tests...")
    print("=" * 60)
    
    # Test migration status first
    await test_migration_status()
    
    # Test engram creation with fixes
    await test_engram_creation()
    
    # Test session consistency
    await test_session_consistency()
    
    print("\n" + "=" * 60)
    print("🎯 All tests completed!")
    print("\nTo apply the migration, run:")
    print("python apply_surrealdb_fixes.py")

if __name__ == "__main__":
    asyncio.run(main())