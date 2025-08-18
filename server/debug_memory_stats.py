#!/usr/bin/env python3
"""Debug script to check memory database size and search for location info"""

import lmdb
import json
import os
from pathlib import Path

def get_db_size_info(db_path):
    """Get database size statistics"""
    env_path = Path(db_path)
    if not env_path.exists():
        print(f"❌ Database path doesn't exist: {db_path}")
        return
    
    # Get directory size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(env_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    print(f"📊 Database: {db_path}")
    print(f"   Total size: {total_size / (1024*1024):.2f} MB")
    
    try:
        env = lmdb.open(str(env_path), readonly=True, max_dbs=10)
        
        # Check each database
        db_stats = {}
        for db_name in ['hot', 'warm', 'cold', 'metadata']:
            try:
                db = env.open_db(db_name.encode())
                with env.begin(db=db) as txn:
                    stat = txn.stat()
                    db_stats[db_name] = {
                        'entries': stat['entries'],
                        'page_size': stat['psize'],
                        'pages': stat['leaf_pages'] + stat['branch_pages'] + stat['overflow_pages']
                    }
            except:
                continue
        
        for db_name, stats in db_stats.items():
            print(f"   {db_name}: {stats['entries']} entries, {stats['pages']} pages")
        
        env.close()
        
    except Exception as e:
        print(f"   Error reading LMDB stats: {e}")

def search_for_location_info(db_path, search_terms):
    """Search for location-related information"""
    env_path = Path(db_path)
    if not env_path.exists():
        return
    
    print(f"\n🔍 Searching for location info in: {db_path}")
    
    try:
        env = lmdb.open(str(env_path), readonly=True, max_dbs=10)
        
        location_matches = []
        
        # Search each tier
        for tier_name in ['hot', 'warm', 'cold']:
            try:
                db = env.open_db(tier_name.encode())
                with env.begin(db=db) as txn:
                    cursor = txn.cursor()
                    tier_matches = 0
                    
                    for key, value in cursor:
                        try:
                            # Try to decompress and decode
                            try:
                                import lz4.frame
                                data = lz4.frame.decompress(value)
                            except:
                                try:
                                    import gzip
                                    data = gzip.decompress(value)
                                except:
                                    data = value
                            
                            try:
                                content = data.decode('utf-8')
                            except:
                                content = data.decode('latin-1', errors='ignore')
                            
                            # Parse JSON if possible
                            try:
                                memory_data = json.loads(content)
                                content_text = memory_data.get('content', str(memory_data))
                            except:
                                content_text = content
                            
                            # Search for location terms
                            content_lower = content_text.lower()
                            for term in search_terms:
                                if term.lower() in content_lower:
                                    location_matches.append({
                                        'tier': tier_name,
                                        'key': key.decode('utf-8', errors='ignore'),
                                        'content': content_text[:150] + '...' if len(content_text) > 150 else content_text,
                                        'term_found': term
                                    })
                                    tier_matches += 1
                                    break
                        except:
                            continue
                    
                    print(f"   {tier_name} tier: {tier_matches} location matches")
                    
            except Exception as e:
                print(f"   Error searching {tier_name}: {e}")
        
        env.close()
        
        # Display matches
        if location_matches:
            print(f"\n✅ Found {len(location_matches)} location-related memories:")
            for i, match in enumerate(location_matches[:10]):  # Show first 10
                print(f"   {i+1}. [{match['tier']}] '{match['term_found']}' in: {match['content']}")
        else:
            print("\n📭 No location information found")
            
    except Exception as e:
        print(f"❌ Error searching for location info: {e}")

def check_bm25_index(db_path):
    """Check BM25 index size and corpus"""
    bm25_path = Path(db_path) / "bm25_index"
    corpus_path = Path(db_path) / "bm25_corpus.json"
    
    print(f"\n📚 BM25 Index Status:")
    
    if bm25_path.exists():
        # Get BM25 index size
        index_size = 0
        for file in bm25_path.glob("*"):
            index_size += file.stat().st_size
        print(f"   Index directory: {index_size / 1024:.2f} KB")
        print(f"   Index files: {list(bm25_path.glob('*'))}")
    else:
        print("   ❌ No BM25 index found")
    
    if corpus_path.exists():
        corpus_size = corpus_path.stat().st_size
        print(f"   Corpus file: {corpus_size / 1024:.2f} KB")
        
        # Check corpus content
        try:
            with open(corpus_path, 'r') as f:
                corpus_data = json.load(f)
                print(f"   Corpus documents: {len(corpus_data.get('corpus', []))}")
                print(f"   Memory IDs: {len(corpus_data.get('memory_ids', []))}")
        except Exception as e:
            print(f"   Error reading corpus: {e}")
    else:
        print("   ❌ No corpus file found")

if __name__ == "__main__":
    # Location search terms
    location_terms = [
        "location", "address", "live", "home", "city", "state", "country",
        "where", "place", "area", "neighborhood", "zip", "postal", 
        "street", "avenue", "road", "boulevard", "coordinates", "GPS",
        "latitude", "longitude", "timezone", "region"
    ]
    
    # Database paths to check
    db_paths = [
        "/Users/peppi/Dev/macos-local-voice-agents/server/data/stateless_memory",
        "/Users/peppi/Dev/macos-local-voice-agents/server/data/debug_memory",
    ]
    
    print("🔍 MEMORY DATABASE ANALYSIS")
    print("=" * 50)
    
    for db_path in db_paths:
        get_db_size_info(db_path)
        search_for_location_info(db_path, location_terms)
        check_bm25_index(db_path)
        print("\n" + "="*50 + "\n")