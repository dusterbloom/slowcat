#!/usr/bin/env python3
"""
Quick script to check what memories are stored in the Chroma database
"""

import sqlite3
import json
from pathlib import Path

def check_chroma_memories():
    """Check what memories are stored in Chroma database"""
    
    chroma_db_path = Path("./data/chroma_db/chroma.sqlite3")
    
    if not chroma_db_path.exists():
        print("❌ No Chroma database found at:", chroma_db_path)
        return
    
    print(f"🔍 Checking memories in: {chroma_db_path}")
    
    try:
        # Connect to the Chroma SQLite database
        conn = sqlite3.connect(chroma_db_path)
        cursor = conn.cursor()
        
        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Database tables: {[table[0] for table in tables]}")
        
        # Check collections table
        cursor.execute("SELECT * FROM collections;")
        collections = cursor.fetchall()
        print(f"📚 Collections: {len(collections)}")
        
        for collection in collections:
            print(f"   Collection: {collection}")
        
        # Check embedding metadata if available
        try:
            cursor.execute("SELECT * FROM embedding_metadata LIMIT 10;")
            metadata = cursor.fetchall()
            print(f"🧠 Stored memories: {len(metadata)}")
            
            for i, memory in enumerate(metadata[:5]):  # Show first 5
                print(f"   Memory {i+1}: {memory}")
                
        except sqlite3.OperationalError as e:
            print(f"   No embedding_metadata table: {e}")
        
        # Try to get embeddings table
        try:
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            embedding_count = cursor.fetchone()[0]
            print(f"🎯 Total embeddings: {embedding_count}")
            
            if embedding_count > 0:
                cursor.execute("SELECT id, metadata FROM embeddings LIMIT 10;")
                embeddings = cursor.fetchall()
                
                for i, (emb_id, metadata) in enumerate(embeddings[:3]):
                    print(f"   Embedding {i+1}: ID={emb_id}")
                    if metadata:
                        try:
                            meta_dict = json.loads(metadata)
                            print(f"      Metadata: {meta_dict}")
                        except:
                            print(f"      Raw metadata: {metadata}")
                            
        except sqlite3.OperationalError as e:
            print(f"   No embeddings table: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")

if __name__ == "__main__":
    check_chroma_memories()