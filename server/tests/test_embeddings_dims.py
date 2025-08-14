#!/usr/bin/env python3
"""Test to check the actual embedding dimensions from LM Studio"""

import requests
import json

# Test the embedding endpoint to check dimensions
url = "http://localhost:1234/v1/embeddings"
headers = {"Content-Type": "application/json"}
data = {
    "model": "text-embedding-nomic-embed-text-v1.5",
    "input": "test text"
}

try:
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        embedding = result['data'][0]['embedding']
        print(f"✅ Embedding dimensions: {len(embedding)}")
        print(f"Model: {result.get('model', 'unknown')}")
        print(f"First few values: {embedding[:5]}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")
    print("Make sure LM Studio is running on port 1234 with the embedding model loaded")