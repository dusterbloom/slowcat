#!/usr/bin/env python3
"""
Test MCP memory server format and create test entities
"""

import asyncio
import subprocess
import json
import os
from pathlib import Path

async def test_memory_create_and_search():
    """Test creating entities and searching them"""
    print("🧠 Testing MCP Memory Create & Search...")
    
    # Use a test memory file
    test_memory_file = "./data/tool_memory/test_mcp_memory.json"
    
    try:
        env = os.environ.copy()
        env["MEMORY_FILE_PATH"] = test_memory_file
        
        # Start memory server
        process = await asyncio.create_subprocess_exec(
            "npx", "-y", "@modelcontextprotocol/server-memory",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Create test entities
        print("📝 Creating test entities...")
        create_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "create_entities",
                "arguments": {
                    "entities": [
                        {
                            "name": "Test User",
                            "entityType": "Person", 
                            "observations": ["This user likes testing MCP servers"]
                        },
                        {
                            "name": "Favorite Number",
                            "entityType": "Number",
                            "observations": ["The favorite number is 319"]
                        }
                    ]
                }
            }
        }
        
        request_json = json.dumps(create_request) + "\n"
        stdout, stderr = await process.communicate(request_json.encode())
        
        if stdout:
            response = json.loads(stdout.decode().strip())
            print(f"📤 Create result: {response}")
        
        if stderr:
            print(f"⚠️ Stderr: {stderr.decode().strip()}")
        
        # Check the format of the created file
        if Path(test_memory_file).exists():
            print(f"\n📁 Created memory file format:")
            with open(test_memory_file, 'r') as f:
                content = f.read()
                print(content)
        else:
            print("❌ Test memory file was not created")
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")

async def test_search_existing():
    """Test searching with the actual memory file"""
    print("\n🔍 Testing search with existing memory file...")
    
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
        
        # First, try to read the entire graph
        read_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "read_graph",
                "arguments": {}
            }
        }
        
        request_json = json.dumps(read_request) + "\n"
        stdout, stderr = await process.communicate(request_json.encode())
        
        if stdout:
            response = json.loads(stdout.decode().strip())
            print(f"📖 Read graph result: {response}")
        
        if stderr:
            print(f"⚠️ Stderr: {stderr.decode().strip()}")
            
    except Exception as e:
        print(f"❌ Read graph test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_memory_create_and_search())
    asyncio.run(test_search_existing())