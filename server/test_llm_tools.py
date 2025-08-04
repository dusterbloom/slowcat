#!/usr/bin/env python3
"""Test if LLM is receiving and using tools correctly"""

import asyncio
import sys
import os
from loguru import logger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.llm_with_tools import LLMWithToolsService
from tools import get_tools, set_memory_processor
from processors.local_memory import LocalMemoryProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from openai import NOT_GIVEN

async def test_llm_tools():
    """Test if LLM correctly uses search_conversations tool"""
    
    # Initialize memory processor
    memory = LocalMemoryProcessor(
        data_dir="data/memory",
        user_id="Peppi",  # Use Peppi as user
        max_history_items=200,
        include_in_context=10
    )
    
    # Set memory processor for tools
    set_memory_processor(memory)
    
    # Initialize LLM with tools
    llm = LLMWithToolsService(
        api_key="dummy",
        base_url="http://localhost:1234/v1",
        model="llama-3.2-3b-instruct",
        max_tokens=150
    )
    
    # Get tools
    tools = get_tools()
    logger.info(f"📋 Available tools: {[t.name for t in tools.standard_tools]}")
    
    # Create context with system prompt
    system_prompt = """You are a helpful AI assistant with memory capabilities.

IMPORTANT: You MUST use the search_conversations tool when users ask about:
- Things they mentioned in previous conversations
- What they told you before
- Past topics you discussed together
- Their preferences or information they shared
- When they ask you to "recall", "remember", or "quote" something
- Any reference to past conversations or prior discussions
- Questions like "what did I say about..." or "do you remember when..."

EXAMPLES of when to use search_conversations:
- User: "What's my name?" → Use search_conversations with query "name"
- User: "Can you recall what I said?" → Use search_conversations
- User: "Quote what I told you about X" → Use search_conversations with query "X"
- User: "Do you remember my favorite color?" → Use search_conversations with query "favorite color"

Always use tools instead of guessing."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What's my name? Search your memory."}
    ]
    
    context = OpenAILLMContext(messages, tools=tools)
    
    # Create context aggregator
    context_aggregator = llm.create_context_aggregator(context)
    
    # Process the context
    logger.info("🤔 Asking LLM: What's my name? Search your memory.")
    
    # Get response
    try:
        # Initialize the LLM properly (simulate pipeline initialization)
        from pipecat.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        
        pipeline = Pipeline([llm])
        task = PipelineTask(pipeline, params=PipelineParams())
        
        # This will initialize the task manager
        await task.run()
        
        # Now process the context
        processed_context = await llm._process_context(context)
        logger.info(f"✅ Context processed successfully")
        logger.info(f"📝 Messages: {len(processed_context.messages)}")
        logger.info(f"🛠️ Tools: {len(processed_context.tools.standard_tools) if processed_context.tools != NOT_GIVEN else 'None'}")
        
        # Try to get a completion
        stream = await llm._stream_chat_completions(processed_context)
        
        full_response = ""
        tool_calls = []
        
        async for chunk in stream:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                
                # Check for content
                if hasattr(delta, 'content') and delta.content:
                    full_response += delta.content
                
                # Check for tool calls
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if hasattr(tool_call, 'function') and tool_call.function:
                            tool_calls.append({
                                'name': tool_call.function.name,
                                'arguments': tool_call.function.arguments
                            })
        
        logger.info(f"\n📤 LLM Response: {full_response}")
        logger.info(f"🔧 Tool calls: {tool_calls}")
        
        if tool_calls:
            logger.success("✅ LLM is using tools!")
        else:
            logger.error("❌ LLM did not use any tools despite clear instructions")
            
    except Exception as e:
        logger.error(f"❌ Error testing LLM: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    await memory.cleanup()

if __name__ == "__main__":
    asyncio.run(test_llm_tools())