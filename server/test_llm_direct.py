#!/usr/bin/env python3
"""Direct test of LLM tool calling"""

import asyncio
import json
from openai import AsyncOpenAI
from loguru import logger

async def test_direct_llm():
    """Test LLM directly with OpenAI client"""
    
    client = AsyncOpenAI(
        api_key="dummy",
        base_url="http://localhost:1234/v1"
    )
    
    # Define the search_conversations tool
    tools = [{
        "type": "function",
        "function": {
            "name": "search_conversations",
            "description": "Search through past conversation history for specific topics or information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in past conversations"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Filter by specific user (optional)",
                        "default": None
                    }
                },
                "required": ["query"]
            }
        }
    }]
    
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant with memory capabilities.

IMPORTANT: You MUST use the search_conversations tool when users ask about:
- Things they mentioned in previous conversations
- What they told you before
- Their name
- Any information from past conversations

When the user asks "What's my name?", you MUST use search_conversations with query "name"."""
        },
        {
            "role": "user",
            "content": "Do you remember what I told you about the Bible quote?"
        }
    ]
    
    logger.info("🤔 Testing direct LLM call with tools...")
    
    try:
        # Make the call
        response = await client.chat.completions.create(
            model="qwen2.5-7b-instruct:2",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
            temperature=0.1
        )
        
        logger.info(f"✅ Got response from LLM")
        
        # Check the response
        message = response.choices[0].message
        
        if message.content:
            logger.info(f"📝 Content: {message.content}")
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            logger.success(f"🛠️ Tool calls detected: {len(message.tool_calls)}")
            for tool_call in message.tool_calls:
                logger.info(f"  - Function: {tool_call.function.name}")
                logger.info(f"    Arguments: {tool_call.function.arguments}")
        else:
            logger.error("❌ No tool calls in response!")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_llm())