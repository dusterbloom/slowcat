#!/usr/bin/env python3
"""
Final test of memory search with fixed absolute path
"""

import asyncio
import subprocess
import json
import os

async def test_memory_search_final():
    """Test memory search for favorite number"""
    print("🔍 Testing Memory Search for Favorite Number...")
    
    try:
        # Use the same absolute path as in mcp.json
        abs_path = "/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json"
        
        env = os.environ.copy()
        env["MEMORY_FILE_PATH"] = abs_path
        
        # Start memory server
        process = await asyncio.create_subprocess_exec(
            "npx", "-y", "@modelcontextprotocol/server-memory",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Search for favorite number
        search_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_nodes",
                "arguments": {
                    "query": "favorite number"
                }
            }
        }
        
        request_json = json.dumps(search_request) + "\n"
        stdout, stderr = await process.communicate(request_json.encode())
        
        if stdout:
            response = json.loads(stdout.decode().strip())
            print(f"🎯 Search response: {response}")
            
            # Parse the result
            result = response.get("result", {})
            content = result.get("content", [])
            if content:
                for item in content:
                    text = item.get("text", "")
                    if "319" in text:
                        print("✅ Found favorite number 319 in search results!")
                    print(f"📄 Result: {text}")
        
        if stderr:
            print(f"⚠️ Stderr: {stderr.decode().strip()}")
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_memory_search_final())