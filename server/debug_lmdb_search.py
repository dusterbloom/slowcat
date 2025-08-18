#!/usr/bin/env python3
"""Debug script to search LMDB database for Potola mentions"""

import lmdb
import json
import lz4.frame
import gzip
from pathlib import Path

def search_lmdb_for_content(db_path, search_terms):
    """Search LMDB database for specific content"""
    env_path = Path(db_path)
    if not env_path.exists():
        print(f"❌ Database path doesn't exist: {db_path}")
        return
    
    print(f"🔍 Searching LMDB database: {db_path}")
    
    try:
        env = lmdb.open(str(env_path), readonly=True, max_dbs=10)
        
        # Get list of databases
        with env.begin() as txn:
            cursor = txn.cursor()
            print("\n📂 Available databases:")
            db_names = []
            for key, value in cursor:
                db_name = key.decode('utf-8', errors='ignore')
                if not db_name.startswith('\x00'):  # Skip special keys
                    db_names.append(db_name)
                    print(f"   - {db_name}")
        
        # Search each database
        if not db_names:
            # Search main database
            search_database(env, None, search_terms, "main")
        else:
            # Search named databases
            for db_name in ['hot', 'warm', 'cold']:
                try:
                    db = env.open_db(db_name.encode())
                    search_database(env, db, search_terms, db_name)
                except:
                    print(f"⚠️ Database '{db_name}' not found, trying main database")
                    search_database(env, None, search_terms, "main")
                    break
        
        env.close()
        
    except Exception as e:
        print(f"❌ Error opening LMDB: {e}")

def search_database(env, db, search_terms, db_name):
    """Search a specific database within LMDB"""
    print(f"\n🔍 Searching database: {db_name}")
    
    try:
        with env.begin(db=db) as txn:
            cursor = txn.cursor()
            found_count = 0
            total_count = 0
            
            for key, value in cursor:
                total_count += 1
                key_str = key.decode('utf-8', errors='ignore')
                
                # Try to decompress and decode value
                try:
                    # Try LZ4 first
                    try:
                        decompressed = lz4.frame.decompress(value)
                    except:
                        try:
                            # Try gzip
                            decompressed = gzip.decompress(value)
                        except:
                            # Use raw data
                            decompressed = value
                    
                    # Decode to string
                    try:
                        content = decompressed.decode('utf-8')
                    except:
                        content = decompressed.decode('latin-1', errors='ignore')
                    
                    # Try to parse as JSON
                    try:
                        data = json.loads(content)
                        content_text = data.get('content', str(data))
                    except:
                        content_text = content
                    
                    # Search for terms
                    content_lower = content_text.lower()
                    for term in search_terms:
                        if term.lower() in content_lower:
                            found_count += 1
                            print(f"✅ FOUND '{term}' in key: {key_str}")
                            print(f"   Content: {content_text[:200]}...")
                            print()
                            break
                            
                except Exception as e:
                    # Skip entries that can't be decoded
                    pass
            
            print(f"📊 Database '{db_name}': Found {found_count} matches in {total_count} total entries")
            
    except Exception as e:
        print(f"❌ Error searching database '{db_name}': {e}")

if __name__ == "__main__":
    # Search terms
    search_terms = ["Potola", "dog", "name", "pet"]
    
    # Database paths to check
    db_paths = [
        "/Users/peppi/Dev/macos-local-voice-agents/server/data/stateless_memory",
        "/Users/peppi/Dev/macos-local-voice-agents/server/data/debug_memory",
        "/Users/peppi/Dev/macos-local-voice-agents/server/data/test_hybrid",
    ]
    
    for db_path in db_paths:
        search_lmdb_for_content(db_path, search_terms)
        print("\n" + "="*60 + "\n")