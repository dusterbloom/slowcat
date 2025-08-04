#!/usr/bin/env python3
"""
Test tool calling with LM Studio
"""
import asyncio
import json
import os
from openai import AsyncOpenAI
from config import config
from tools import get_tools, set_memory_processor
from processors.local_memory import LocalMemoryProcessor
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_tools():
    """Test if LM Studio can handle tool calls"""
    
    # Initialize memory processor
    memory_processor = LocalMemoryProcessor(user_id="test_user")
    set_memory_processor(memory_processor)
    
    # Get tools
    tools_schema = get_tools()
    
    # Convert to OpenAI format
    tools_list = []
    for tool in tools_schema.standard_tools:
        # FunctionSchema is already in the correct format
        tool_dict = {
            "type": "function", 
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.properties,
                    "required": tool.required
                }
            }
        }
        tools_list.append(tool_dict)
    
    print(f"Available tools: {[t['function']['name'] for t in tools_list]}")
    
    # Initialize OpenAI client for LM Studio
    # Use the actual base URL from environment
    base_url = os.getenv("OPENAI_BASE_URL", config.network.llm_base_url)
    print(f"Using base URL: {base_url}")
    
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed"  # LM Studio doesn't need API key
    )
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to memory search tools."},
        {"role": "user", "content": "Search my memory for any conversations about Python"}
    ]
    
    try:
        # Make request with tools
        print(f"\nSending request to: {base_url}")
        print(f"Model: {config.models.default_llm_model}")
        
        response = await client.chat.completions.create(
            model=config.models.default_llm_model,
            messages=messages,
            tools=tools_list,
            tool_choice="auto",
            temperature=0.7
        )
        
        print(f"\nResponse: {response}")
        
        # Check if model made tool calls
        message = response.choices[0].message
        if message.tool_calls:
            print(f"\n✅ Model made {len(message.tool_calls)} tool calls!")
            for tool_call in message.tool_calls:
                print(f"  - {tool_call.function.name}: {tool_call.function.arguments}")
        else:
            print("\n❌ Model did not make any tool calls")
            print(f"Response content: {message.content}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPossible issues:")
        print("1. LM Studio model doesn't support function calling")
        print("2. Model needs specific prompt engineering for tools")
        print("3. Check if model supports OpenAI function calling format")

if __name__ == "__main__":
    asyncio.run(test_tools())