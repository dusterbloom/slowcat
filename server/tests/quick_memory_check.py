#!/usr/bin/env python3
"""
Quick check of what's in memory right now
"""

import requests

def check_memory_status():
    print("🔍 Checking Memory Status...")
    
    # Check Qdrant collections
    try:
        qdrant_response = requests.get("http://localhost:6333/collections")
        collections = qdrant_response.json()
        print(f"📊 Qdrant Collections: {collections}")
        
        if collections.get('result', {}).get('collections'):
            for collection in collections['result']['collections']:
                collection_name = collection['name']
                print(f"\n📁 Collection: {collection_name}")
                
                # Get collection info
                info_response = requests.get(f"http://localhost:6333/collections/{collection_name}")
                info = info_response.json()
                print(f"   Points count: {info.get('result', {}).get('points_count', 0)}")
                print(f"   Vector size: {info.get('result', {}).get('config', {}).get('params', {}).get('vectors', {}).get('size', 'unknown')}")
                
                # Try to get some points
                points_response = requests.post(
                    f"http://localhost:6333/collections/{collection_name}/points/scroll",
                    json={"limit": 5}
                )
                points = points_response.json()
                print(f"   Sample points: {len(points.get('result', {}).get('points', []))}")
        else:
            print("   No collections found - memory not storing anything")
            
    except Exception as e:
        print(f"❌ Error checking Qdrant: {e}")

if __name__ == "__main__":
    check_memory_status()