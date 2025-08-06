#!/usr/bin/env python3
"""
Test script for JSON-RPC 2.0 stdio MCP server discovery
"""

import asyncio
from services.simple_mcp_tool_manager import SimpleMCPToolManager
from loguru import logger

async def main():
    """Test the new stdio-based MCP discovery"""
    logger.info("🚀 Testing JSON-RPC 2.0 stdio MCP discovery")
    
    # Create tool manager
    manager = SimpleMCPToolManager()
    
    # Force refresh to test discovery
    logger.info("🔄 Forcing manifest refresh...")
    await manager._refresh_manifest()
    
    # Get tools for LM Studio
    logger.info("📋 Getting tools for LM Studio...")
    tools = await manager.get_tools_for_llm()
    
    logger.info(f"✅ Discovery complete: {len(tools)} tools found")
    for tool in tools:
        function = tool.get("function", {})
        logger.info(f"   📦 {function.get('name')}: {function.get('description')}")
    
    # Test tool routing info
    logger.info("🔍 Testing tool routing...")
    for tool in tools:
        function = tool.get("function", {})
        tool_name = function.get("name")
        if tool_name:
            routing_info = manager.get_routing_info(tool_name)
            if routing_info:
                logger.info(f"   🚦 {tool_name} -> {routing_info.get('mcp_server', 'unknown')}")
    
    logger.info("🏁 Test complete")

if __name__ == "__main__":
    asyncio.run(main())