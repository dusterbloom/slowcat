"""
Unified LLM service with proper Pipecat tool/function calling support
Uses FunctionSchema and ToolsSchema for OpenAI-compatible tool calling with LM Studio
"""

from typing import Optional
import json
from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from tools.handlers import execute_tool_call
from tools.formatters import format_tool_response_for_voice
from tools.definitions import ALL_FUNCTION_SCHEMAS, get_tools


class LLMWithToolsService(OpenAILLMService):
    """
    Unified LLM service with proper Pipecat function calling support.
    Extends OpenAILLMService to handle tool calls with LM Studio.
    """
    
    def __init__(self, **kwargs):
        """Initialize the tool-enabled LLM service"""
        super().__init__(**kwargs)
        
        logger.info("🚀 Initializing LLMWithToolsService")
        logger.info(f"📍 Base URL: {kwargs.get('base_url')}")
        logger.info(f"🤖 Model: {kwargs.get('model')}")
        
        # Register all function handlers
        self._register_function_handlers()
        
        # Store tools schema for reference
        self._tools_schema = get_tools()
        logger.info(f"🛠️ Loaded {len(self._tools_schema.standard_tools)} tools")
    
    def _register_function_handlers(self):
        """Register function handlers using Pipecat's register_function method"""
        
        for function_schema in ALL_FUNCTION_SCHEMAS:
            function_name = function_schema.name
            
            # Register handler for this function
            self.register_function(
                function_name,
                self._create_handler(function_name),
                cancel_on_interruption=True
            )
            
            logger.debug(f"✅ Registered handler for: {function_name}")
        
        logger.info(f"🎯 Successfully registered {len(ALL_FUNCTION_SCHEMAS)} function handlers")
    
    def _create_handler(self, function_name: str):
        """
        Create an async handler for a specific function
        
        Args:
            function_name: Name of the function to handle
            
        Returns:
            Async handler function
        """
        async def handler(params: FunctionCallParams):
            """Handler that executes tool and returns result"""
            logger.info(f"🎯 Executing function: {function_name}")
            logger.debug(f"📥 Arguments: {params.arguments}")
            
            try:
                # Execute the tool
                result = await execute_tool_call(function_name, params.arguments)
                
                # Format result for voice if needed
                formatted_result = format_tool_response_for_voice(function_name, result)
                
                # Ensure result is string for LM Studio
                if isinstance(result, (dict, list)):
                    result_str = json.dumps(result)
                else:
                    result_str = str(result)
                
                # Truncate if too large
                if len(result_str) > 4000:
                    result_str = result_str[:3900] + "... [truncated]"
                    logger.warning(f"⚠️ Truncated large result for {function_name}")
                
                # Send result back through callback
                await params.result_callback(result_str)
                
                logger.info(f"✅ Function '{function_name}' completed successfully")
                logger.debug(f"📤 Result (first 200 chars): {result_str[:200]}...")
                
                # Optional: Queue voice-formatted response
                # This allows the LLM to use the raw result while providing
                # a pre-formatted voice response
                if hasattr(params, 'llm') and formatted_result != result_str:
                    logger.debug(f"🎤 Voice format available: {formatted_result[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Error in function '{function_name}': {e}", exc_info=True)
                error_msg = f"Error executing {function_name}: {str(e)}"
                await params.result_callback(error_msg)
        
        return handler
    
    async def _process_context(self, context):
        """Override to ensure tools are properly set"""
        # Log context details
        if hasattr(context, 'tools') and context.tools:
            if isinstance(context.tools, ToolsSchema):
                tool_count = len(context.tools.standard_tools)
                logger.debug(f"📋 Processing context with {tool_count} tools (ToolsSchema)")
            elif isinstance(context.tools, list):
                # Tools have been converted to list format by adapter
                tool_count = len(context.tools)
                logger.debug(f"📋 Processing context with {tool_count} tools (list format)")
                if tool_count > 0 and isinstance(context.tools[0], dict):
                    logger.debug(f"📋 First tool: {context.tools[0].get('function', {}).get('name', 'unknown')}")
            else:
                logger.warning(f"⚠️ Context tools in unexpected format: {type(context.tools)}")
        else:
            logger.debug("📋 Processing context without tools")
        
        return await super()._process_context(context)
    
    async def _stream_chat_completions(self, context):
        """Override to add debugging for tool calls"""
        logger.debug("🌊 Starting streaming chat completion")
        
        # Check if tools are in the messages
        if hasattr(context, '_messages') and context._messages:
            # Log if we're sending tools to the API
            messages = context._messages
            logger.debug(f"📨 Sending {len(messages)} messages to LLM")
        
        # Get the stream from parent class
        stream = await super()._stream_chat_completions(context)
        
        # Create a wrapper that logs tool calls
        async def debug_stream():
            tool_call_detected = False
            async for chunk in stream:
                # Log if we see tool calls
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        if not tool_call_detected:
                            logger.info(f"🔧 Tool call detected in stream")
                            tool_call_detected = True
                        for tool_call in delta.tool_calls:
                            if hasattr(tool_call, 'function') and tool_call.function:
                                if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                    logger.info(f"🎯 Tool requested: {tool_call.function.name}")
                yield chunk
        
        return debug_stream()
    
    def get_tools_schema(self) -> ToolsSchema:
        """
        Get the ToolsSchema object for use in context
        
        Returns:
            ToolsSchema containing all available tools
        """
        return self._tools_schema