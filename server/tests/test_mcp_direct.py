#!/usr/bin/env python3
"""
Direct test of MCP server execution to verify they work
"""

import asyncio
import subprocess
import json
import os
from pathlib import Path

async def test_memory_server():
    """Test the memory MCP server directly"""
    print("🧠 Testing Memory MCP Server directly...")
    
    try:
        # Set up environment
        env = os.environ.copy()
        env["MEMORY_FILE_PATH"] = "./data/tool_memory/memory.json"
        
        # Start memory server
        process = await asyncio.create_subprocess_exec(
            "npx", "-y", "@modelcontextprotocol/server-memory",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Test 1: List tools
        print("📋 Testing tools/list...")
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        request_json = json.dumps(request) + "\n"
        stdout, stderr = await asyncio.wait_for(
            process.communicate(request_json.encode()),
            timeout=10.0
        )
        
        if stdout:
            response = json.loads(stdout.decode().strip())
            tools = response.get("result", {}).get("tools", [])
            print(f"✅ Found {len(tools)} tools: {[t['name'] for t in tools]}")
        
        if stderr:
            print(f"⚠️ Stderr: {stderr.decode().strip()}")
            
    except Exception as e:
        print(f"❌ Memory server test failed: {e}")

async def test_memory_search():
    """Test memory search specifically"""
    print("\n🔍 Testing Memory Search...")
    
    try:
        env = os.environ.copy()
        env["MEMORY_FILE_PATH"] = "./data/tool_memory/memory.json"
        
        process = await asyncio.create_subprocess_exec(
            "npx", "-y", "@modelcontextprotocol/server-memory",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Test search_nodes for "favorite number"
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_nodes",
                "arguments": {
                    "query": "favorite number"
                }
            }
        }
        
        request_json = json.dumps(request) + "\n"
        stdout, stderr = await asyncio.wait_for(
            process.communicate(request_json.encode()),
            timeout=10.0
        )
        
        if stdout:
            response = json.loads(stdout.decode().strip())
            print(f"📤 Search result: {response}")
        
        if stderr:
            print(f"⚠️ Stderr: {stderr.decode().strip()}")
            
    except Exception as e:
        print(f"❌ Memory search test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_memory_server())
    asyncio.run(test_memory_search())