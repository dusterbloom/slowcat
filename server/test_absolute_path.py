#!/usr/bin/env python3
"""
Test MCP memory server with absolute path
"""

import asyncio
import subprocess
import json
import os
from pathlib import Path

async def test_with_absolute_path():
    """Test MCP memory server with absolute path"""
    print("🧠 Testing MCP Memory with absolute path...")
    
    # Use absolute path
    abs_memory_path = "/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json"
    print(f"📁 Using absolute path: {abs_memory_path}")
    
    # Check if file exists
    if not Path(abs_memory_path).exists():
        print(f"❌ Memory file does not exist at: {abs_memory_path}")
        return
    
    print(f"✅ Memory file exists, size: {Path(abs_memory_path).stat().st_size} bytes")
    
    try:
        env = os.environ.copy()
        env["MEMORY_FILE_PATH"] = abs_memory_path
        
        # Start memory server
        process = await asyncio.create_subprocess_exec(
            "npx", "-y", "@modelcontextprotocol/server-memory",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Try to read the graph
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
            print(f"📖 Graph content: {response}")
            
            # Try to parse the graph content
            result = response.get("result", {})
            content = result.get("content", [])
            if content and len(content) > 0:
                text_content = content[0].get("text", "{}")
                try:
                    parsed = json.loads(text_content)
                    entities = parsed.get("entities", [])
                    relations = parsed.get("relations", [])
                    print(f"📊 Found {len(entities)} entities, {len(relations)} relations")
                    if entities:
                        print("🔍 Entities:")
                        for entity in entities:
                            print(f"   - {entity}")
                except json.JSONDecodeError as e:
                    print(f"❌ Could not parse graph content: {e}")
        
        if stderr:
            stderr_text = stderr.decode().strip()
            print(f"⚠️ Stderr: {stderr_text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_with_absolute_path())