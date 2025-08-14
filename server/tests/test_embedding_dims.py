#!/usr/bin/env python3
"""
Test actual embedding dimensions from sentence-transformers
"""

from sentence_transformers import SentenceTransformer

def test_embedding_dimensions():
    print("🧪 Testing HuggingFace embedding dimensions...")
    
    # Load the model we're using
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test with a sample text
    test_text = "My favorite color is blue"
    embeddings = model.encode([test_text])
    
    print(f"Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Dimensions: {embeddings.shape[1]}")
    
    return embeddings.shape[1]

if __name__ == "__main__":
    actual_dims = test_embedding_dimensions()