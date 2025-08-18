#!/usr/bin/env python3
"""
Debug script to understand why memory search isn't working
"""

import asyncio
import tempfile
import shutil
from processors.enhanced_stateless_memory import EnhancedStatelessMemoryProcessor, MemoryItem

async def debug_memory_search():
    """Debug the memory search algorithm"""
    
    print("🔍 Memory Search Debug")
    print("=" * 40)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create processor
        processor = EnhancedStatelessMemoryProcessor(
            db_path=temp_dir,
            hot_tier_size=5
        )
        
        # Add some test conversations
        conversations = [
            ("What is machine learning?", "Machine learning is a subset of AI."),
            ("Explain neural networks", "Neural networks are computing systems."),
            ("Tell me about Python", "Python is a programming language."),
        ]
        
        print(f"\n📝 Storing {len(conversations)} conversations...")
        for i, (user_msg, assistant_msg) in enumerate(conversations):
            processor.current_user_message = user_msg
            await processor._store_exchange_three_tier(user_msg, assistant_msg)
            print(f"   {i+1}. User: '{user_msg}' -> Assistant: '{assistant_msg}'")
        
        print(f"\n🔥 Hot tier contents ({len(processor.hot_tier)} items):")
        for i, item in enumerate(processor.hot_tier):
            if isinstance(item, MemoryItem):
                print(f"   {i+1}. {item.speaker_id}: '{item.content}' (tier: {item.tier})")
        
        # Test relevance checking
        test_queries = [
            "What is machine learning?",
            "machine learning", 
            "neural networks",
            "Python",
            "programming",
            "AI"
        ]
        
        print(f"\n🔍 Testing relevance for queries...")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            # Check each hot tier item for relevance
            relevant_items = []
            for item in processor.hot_tier:
                if isinstance(item, MemoryItem):
                    is_relevant = processor._is_relevant_enhanced(item.content, query)
                    print(f"   '{item.content[:30]}...' -> {is_relevant}")
                    if is_relevant:
                        relevant_items.append(item)
            
            print(f"   Found {len(relevant_items)} relevant items")
            
            # Test actual search
            memories = await processor._search_hot_tier(query, "default_user", 200)
            print(f"   Hot tier search returned {len(memories)} memories")
            
            # Test full three-tier search
            all_memories = await processor._get_relevant_memories_three_tier(query, "default_user", 200)
            print(f"   Full search returned {len(all_memories)} memories")
        
        # Test keyword extraction with new algorithm
        print(f"\n🔤 Testing keyword extraction with enhanced algorithm...")
        test_text = "What is machine learning and neural networks?"
        stored_content = "Machine learning is a subset of AI."
        
        # Test the enhanced relevance function directly
        is_relevant = processor._is_relevant_enhanced(stored_content, test_text)
        print(f"   Query: '{test_text}'")
        print(f"   Memory: '{stored_content}'")
        print(f"   Is relevant: {is_relevant}")
        
        # Test AI matching specifically
        ai_relevant = processor._is_relevant_enhanced(stored_content, "AI")
        print(f"   AI query vs AI content: {ai_relevant}")
        
        # Test with punctuation variations
        punctuation_tests = [
            ("AI", "AI."),
            ("AI", "AI!"),
            ("programming", "programming."),
            ("networks", "networks?"),
        ]
        
        print(f"   Punctuation handling tests:")
        for query, content in punctuation_tests:
            relevant = processor._is_relevant_enhanced(content, query)
            print(f"     '{query}' vs '{content}' -> {relevant}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(debug_memory_search())