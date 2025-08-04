#!/usr/bin/env python3
"""Test if the loaded model actually supports function calling"""

import requests
import json

# Check model capabilities
response = requests.get("http://localhost:1234/v1/models")
models = response.json()

print("=== LM Studio Models ===")
for model in models.get("data", []):
    print(f"\nModel ID: {model.get('id')}")
    print(f"Object: {model.get('object')}")
    
    # Check capabilities
    if 'capabilities' in model:
        caps = model.get('capabilities', [])
        print(f"Capabilities: {caps}")
        if 'tool_use' in caps or 'function_calling' in caps:
            print("✅ This model supports function calling!")
        else:
            print("❌ This model does NOT support function calling")
    else:
        print("⚠️  No capabilities info available")

# Test with simple function call
print("\n\n=== Testing Function Call ===")

tools = [{
    "type": "function",
    "function": {
        "name": "test_function",
        "description": "A test function",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}]

data = {
    "model": "local-model",
    "messages": [
        {"role": "system", "content": "You must use the test_function when asked."},
        {"role": "user", "content": "Please call the test function."}
    ],
    "tools": tools,
    "tool_choice": "required"
}

response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    json=data,
    headers={"Content-Type": "application/json"}
)

print(f"Status: {response.status_code}")
result = response.json()

if 'choices' in result:
    choice = result['choices'][0]
    if 'message' in choice:
        msg = choice['message']
        if 'tool_calls' in msg and msg['tool_calls']:
            print("✅ Model made a function call!")
            print(f"Tool calls: {json.dumps(msg['tool_calls'], indent=2)}")
        else:
            print("❌ No function call made")
            print(f"Response: {msg.get('content', '')}")
else:
    print(f"Error: {result}")