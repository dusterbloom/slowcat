#!/usr/bin/env python3
"""Test different models with search_conversations tool"""

import asyncio
import json
from openai import AsyncOpenAI
from loguru import logger

async def test_model(model_name: str, query: str):
    """Test a specific model with search query"""
    
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
                    }
                },
                "required": ["query"]
            }
        }
    }]
    
    messages = [
        {
            "role": "system",
            "content": """You are an AI assistant with memory search capabilities.

CRITICAL: You MUST use the search_conversations tool when asked about past conversations.

When someone mentions "Bible quote", "search memory", or asks what they told you before, you MUST IMMEDIATELY use search_conversations.

DO NOT ask for clarification. DO NOT say you're searching without using the tool."""
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"Query: {query}")
    
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=150,
            temperature=0.1
        )
        
        message = response.choices[0].message
        
        if message.content:
            logger.info(f"Response: {message.content}")
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            logger.success(f"✅ Tool calls: {len(message.tool_calls)}")
            for tool_call in message.tool_calls:
                logger.info(f"  - Function: {tool_call.function.name}")
                logger.info(f"    Arguments: {tool_call.function.arguments}")
        else:
            logger.error("❌ No tool calls!")
            
    except Exception as e:
        logger.error(f"Error: {e}")

async def main():
    """Test multiple models"""
    
    models = [
        "llama-3.2-3b-instruct-uncensored",
        "qwen2.5-7b-instruct:2",
        "gemma-3-12b-it-qat"
    ]
    
    queries = [
        "Search for Bible quote in our conversations",
        "What did I tell you about the Bible?",
        "Find Bible in memory"
    ]
    
    for query in queries:
        for model in models:
            await test_model(model, query)
            await asyncio.sleep(1)  # Avoid rate limiting

if __name__ == "__main__":
    asyncio.run(main())