#!/usr/bin/env python3
"""
MCP Integration Debug Script
Tests MCP tool discovery and execution with detailed logging
"""

import asyncio
import os
import sys
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

# Add server to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables first
load_dotenv(override=True)
logger.info(f"🔧 Loaded .env file, BRAVE_API_KEY loaded: {'✅ YES' if os.getenv('BRAVE_API_KEY') else '❌ NO'}")

# Configure logging for detailed output
logger.remove()
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

async def test_mcp_integration():
    """Test MCP integration components"""
    logger.info("🧪 Testing MCP Integration Components")
    
    # Test 1: Environment Variables
    logger.info("=" * 60)
    logger.info("📋 TEST 1: Environment Variables")
    logger.info("=" * 60)
    
    # Check .env file exists
    env_file = Path(".env")
    if env_file.exists():
        logger.info("✅ .env file exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if "BRAVE_API_KEY" in content:
                logger.info("✅ BRAVE_API_KEY found in .env")
                # Check if it has a value (look for = followed by non-empty value)
                import re
                match = re.search(r'BRAVE_API_KEY\s*=\s*"?([^"\n\r]+)"?', content)
                if match and match.group(1).strip():
                    logger.info(f"✅ BRAVE_API_KEY has value (length: {len(match.group(1).strip())})")
                else:
                    logger.warning("⚠️ BRAVE_API_KEY is empty")
            else:
                logger.warning("⚠️ BRAVE_API_KEY not found in .env")
    else:
        logger.error("❌ .env file not found")
    
    # Check environment variables
    brave_key = os.getenv('BRAVE_API_KEY')
    if brave_key:
        logger.info(f"✅ BRAVE_API_KEY loaded in environment (length: {len(brave_key)})")
        logger.info(f"   First 10 chars: {brave_key[:10]}...")
    else:
        logger.error("❌ BRAVE_API_KEY not in environment")
    
    memory_path = os.getenv('MEMORY_FILE_PATH', './data/tool_memory/memory.json')
    logger.info(f"📁 MEMORY_FILE_PATH: {memory_path}")
    
    # Test 2: MCP Tool Discovery
    logger.info("=" * 60)
    logger.info("🔍 TEST 2: MCP Tool Discovery")
    logger.info("=" * 60)
    
    try:
        from services.simple_mcp_tool_manager import SimpleMCPToolManager
        
        mcp_manager = SimpleMCPToolManager(language="en")
        logger.info(f"📡 Base URL: {mcp_manager.base_url}")
        
        manifest = await mcp_manager.discover_tools()
        logger.info(f"🛠️ Discovered {len(manifest)} MCP tools")
        
        for tool_name, description in manifest.items():
            logger.info(f"   🔧 {tool_name}: {description[:60]}...")
            
    except Exception as e:
        logger.error(f"❌ MCP Discovery failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test 3: Tool Schema Building
    logger.info("=" * 60)
    logger.info("🏗️ TEST 3: Tool Schema Building")
    logger.info("=" * 60)
    
    try:
        from tools.definitions import ALL_FUNCTION_SCHEMAS
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        from pipecat.adapters.schemas.function_schema import FunctionSchema
        
        logger.info(f"🏠 Local tools: {len(ALL_FUNCTION_SCHEMAS)}")
        for schema in ALL_FUNCTION_SCHEMAS:
            logger.info(f"   🔧 {schema.name}")
        
        # Create mock MCP schemas
        mcp_schemas = []
        for tool_name, description in manifest.items():
            mcp_schema = FunctionSchema(
                name=tool_name,
                description=description,
                properties={"type": "object", "properties": {}},
                required=[]
            )
            mcp_schemas.append(mcp_schema)
            
        logger.info(f"🌐 MCP schemas: {len(mcp_schemas)}")
        
        # Create unified schema
        unified_tools = list(ALL_FUNCTION_SCHEMAS) + mcp_schemas
        tools_schema = ToolsSchema(standard_tools=unified_tools)
        
        logger.info(f"🎯 Total unified tools: {len(unified_tools)}")
        logger.info(f"   Local: {len(ALL_FUNCTION_SCHEMAS)}")
        logger.info(f"   MCP: {len(mcp_schemas)}")
        
    except Exception as e:
        logger.error(f"❌ Schema building failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test 4: Check mcp.json configuration
    logger.info("=" * 60)
    logger.info("📄 TEST 4: MCP Configuration")
    logger.info("=" * 60)
    
    try:
        import json
        mcp_config_file = Path("mcp.json")
        if mcp_config_file.exists():
            logger.info("✅ mcp.json exists")
            with open(mcp_config_file, 'r') as f:
                mcp_config = json.load(f)
            
            servers = mcp_config.get("mcpServers", {})
            logger.info(f"🖥️ Configured MCP servers: {len(servers)}")
            
            for server_name, server_config in servers.items():
                logger.info(f"   🖥️ {server_name}:")
                logger.info(f"      Command: {server_config.get('command')}")
                logger.info(f"      Args: {server_config.get('args')}")
                env_vars = server_config.get('env', {})
                if env_vars:
                    logger.info(f"      Environment variables: {list(env_vars.keys())}")
        else:
            logger.error("❌ mcp.json not found")
            
    except Exception as e:
        logger.error(f"❌ MCP config check failed: {e}")
    
    logger.info("=" * 60)
    logger.info("🎯 MCP Integration Test Complete")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_mcp_integration())