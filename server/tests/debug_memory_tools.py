#!/usr/bin/env python3
"""Debug script to verify memory tools are properly available"""

import sys
import os

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

from tools.definitions import get_tool_names, ALL_FUNCTION_SCHEMAS
from loguru import logger

# List all available tools
logger.info("🔧 Available tools:")
for tool_name in get_tool_names():
    logger.info(f"  - {tool_name}")

# Check if memory search tools are present
memory_tools = ["search_conversations", "get_conversation_summary"]
logger.info("\n🔍 Memory search tools:")
for tool in memory_tools:
    if tool in get_tool_names():
        logger.info(f"  ✅ {tool} is available")
        # Find and show the tool definition
        for schema in ALL_FUNCTION_SCHEMAS:
            if schema.name == tool:
                logger.info(f"     Description: {schema.description}")
    else:
        logger.error(f"  ❌ {tool} is NOT available")

# Show the actual tool schemas being passed to LLM
logger.info("\n📋 Tool schemas for LLM:")
from tools import get_tools
tools_schema = get_tools()
logger.info(f"Total tools available: {len(tools_schema.standard_tools)}")

# Test if handlers are properly mapped
logger.info("\n🔗 Testing tool handler mapping:")
from tools.handlers import execute_tool_call
import asyncio

async def test_handlers():
    # Test if handlers exist
    test_tools = ["search_conversations", "get_conversation_summary"]
    for tool_name in test_tools:
        try:
            # This will fail if not mapped, but that's ok for this test
            result = await execute_tool_call(tool_name, {})
            logger.info(f"  ✅ {tool_name} handler is mapped")
        except Exception as e:
            if "Memory is not enabled" in str(e):
                logger.info(f"  ✅ {tool_name} handler is mapped (needs memory processor)")
            else:
                logger.error(f"  ❌ {tool_name} handler error: {e}")

asyncio.run(test_handlers())