#!/usr/bin/env python3
"""
Check what's stored in the memory database
"""

import os
import json
from pathlib import Path

# Check the absolute path from config
from config import config

def check_memory_storage():
    """Check what's in the memory storage"""
    
    print("🔍 Checking Memory Storage")
    print("=" * 40)
    
    # Check stateless memory path
    stateless_path = Path(config.stateless_memory.db_path)
    print(f"Stateless memory path: {stateless_path}")
    print(f"Exists: {stateless_path.exists()}")
    
    if stateless_path.exists():
        print(f"Contents: {list(stateless_path.iterdir())}")
        
        # Check LMDB files
        lmdb_files = list(stateless_path.glob("*"))
        print(f"LMDB files: {lmdb_files}")
        
        # Try to read LMDB
        try:
            import lmdb
            env = lmdb.open(str(stateless_path), readonly=True)
            
            with env.begin() as txn:
                # Check all databases
                for db_name in [None, b'warm', b'cold', b'meta']:
                    try:
                        if db_name:
                            db = env.open_db(db_name, txn=txn)
                            cursor = txn.cursor(db=db)
                            print(f"\n📁 Database '{db_name.decode()}' contents:")
                        else:
                            cursor = txn.cursor()
                            print(f"\n📁 Main database contents:")
                        
                        count = 0
                        for key, value in cursor:
                            print(f"  Key: {key.decode()[:50]}...")
                            print(f"  Value: {value[:100]}...")
                            count += 1
                            if count >= 5:  # Limit output
                                print(f"  ... and {cursor.count() - count} more items")
                                break
                        
                        if count == 0:
                            print("  (empty)")
                            
                    except Exception as e:
                        print(f"  Error reading {db_name}: {e}")
            
            env.close()
            
        except ImportError:
            print("LMDB not available, cannot read database")
        except Exception as e:
            print(f"Error reading LMDB: {e}")
    
    # Check traditional memory path
    traditional_path = Path(config.memory.data_dir)
    print(f"\nTraditional memory path: {traditional_path}")
    print(f"Exists: {traditional_path.exists()}")
    
    if traditional_path.exists():
        memory_files = list(traditional_path.glob("*"))
        print(f"Memory files: {memory_files}")
        
        # Check JSON files
        for file_path in memory_files:
            if file_path.suffix == '.json':
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"\n📄 {file_path.name}:")
                    if isinstance(data, list):
                        print(f"  {len(data)} items")
                        if data:
                            print(f"  Latest: {data[-1]}")
                    elif isinstance(data, dict):
                        print(f"  Keys: {list(data.keys())}")
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")
    
    # Look for any other data directories
    server_data = Path("data")
    if server_data.exists():
        print(f"\nAll data directories:")
        for item in server_data.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}")
                # Check if it contains memory-related files
                memory_files = list(item.glob("*memory*")) + list(item.glob("*.db")) + list(item.glob("*.lmdb"))
                if memory_files:
                    print(f"    Memory files: {[f.name for f in memory_files]}")

if __name__ == "__main__":
    check_memory_storage()