#!/usr/bin/env python3
"""Direct test of memory search for debugging"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from processors.local_memory import LocalMemoryProcessor
from loguru import logger

async def test_search():
    """Test searching the actual memory database"""
    logger.info("🔍 Testing memory search on real data")
    
    # Use the actual memory database
    memory = LocalMemoryProcessor(
        data_dir="data/memory",
        user_id="default_user",  # Use the actual user ID from database
        max_history_items=200,
        include_in_context=10
    )
    
    # Test searches
    searches = [
        "name",
        "Becpe", 
        "Pepe",
        "Peppy",
        "quote",
        "Bible"
    ]
    
    for query in searches:
        logger.info(f"\n📌 Searching for: '{query}'")
        results = await memory.search_conversations(query, limit=5)
        
        if results:
            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. [{result['role']}] {result['content'][:80]}...")
        else:
            logger.info("  No results found")
    
    # Get conversation summary
    logger.info("\n📊 Conversation Summary:")
    summary = await memory.get_conversation_summary(days_back=1)
    logger.info(f"  Total messages: {summary['total_messages']}")
    logger.info(f"  Recent topics: {summary.get('recent_topics', [])[:3]}")

if __name__ == "__main__":
    asyncio.run(test_search())