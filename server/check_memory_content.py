#!/usr/bin/env python3
"""
Check the actual memory content for dog name or any personal info
"""

import sqlite3
import json
from pathlib import Path

def check_memory_content():
    """Check actual memory content in Chroma database"""
    
    chroma_db_path = Path("./data/chroma_db/chroma.sqlite3")
    
    if not chroma_db_path.exists():
        print("❌ No Chroma database found")
        return
    
    try:
        conn = sqlite3.connect(chroma_db_path)
        cursor = conn.cursor()
        
        # Get all embeddings with their documents
        print("🔍 Searching for stored memories...")
        
        # Try different ways to get the actual memory content
        try:
            cursor.execute("SELECT * FROM embeddings;")
            embeddings = cursor.fetchall()
            
            print(f"📊 Found {len(embeddings)} embeddings:")
            for i, embedding in enumerate(embeddings):
                print(f"\nEmbedding {i+1}:")
                print(f"  ID: {embedding[0]}")
                print(f"  Collection: {embedding[1]}")
                if len(embedding) > 2:
                    print(f"  Document: {embedding[2]}")
                if len(embedding) > 3 and embedding[3]:
                    try:
                        metadata = json.loads(embedding[3])
                        print(f"  Metadata: {metadata}")
                    except:
                        print(f"  Raw metadata: {embedding[3]}")
                        
        except Exception as e:
            print(f"Error reading embeddings: {e}")
        
        # Try to get documents directly
        try:
            cursor.execute("PRAGMA table_info(embeddings);")
            columns = cursor.fetchall()
            print(f"\n📋 Embeddings table columns: {[col[1] for col in columns]}")
            
            # Try to find document column
            cursor.execute("SELECT id, collection_id, document FROM embeddings;")
            docs = cursor.fetchall()
            
            print(f"\n📝 Documents stored:")
            for doc_id, collection_id, document in docs:
                print(f"  Doc ID: {doc_id}")
                print(f"  Collection: {collection_id}")
                print(f"  Content: {document}")
                
                # Check if document mentions dog names
                if document and any(keyword in document.lower() for keyword in ['dog', 'pet', 'name', 'called']):
                    print(f"  🐕 POTENTIAL DOG INFO: {document}")
                
        except Exception as e:
            print(f"Error reading documents: {e}")
        
        # Check collections for user info
        cursor.execute("SELECT * FROM collections WHERE name = 'slowcat_default_user';")
        user_collection = cursor.fetchone()
        if user_collection:
            print(f"\n👤 User collection: {user_collection}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_memory_content()