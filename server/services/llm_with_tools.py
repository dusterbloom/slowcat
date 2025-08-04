"""
Unified LLM service with proper Pipecat tool/function calling support
Uses FunctionSchema and ToolsSchema for OpenAI-compatible tool calling with LM Studio
"""

from typing import Optional
import json
import asyncio
from loguru import logger
from openai import NOT_GIVEN
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from tools.handlers import execute_tool_call
from tools.formatters import format_tool_response_for_voice
from tools.definitions import ALL_FUNCTION_SCHEMAS, get_tools

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
)
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
            """Handler that executes tool and returns result with enhanced error handling"""
            logger.info(f"🎯 Executing function: {function_name}")
            logger.debug(f"📥 Arguments: {params.arguments}")
            
            try:
                # Validate arguments
                if not isinstance(params.arguments, dict):
                    logger.warning(f"Invalid arguments type for {function_name}: {type(params.arguments)}")
                    params.arguments = {}
                
                # Execute the tool with timeout protection
                try:
                    result = await asyncio.wait_for(
                        execute_tool_call(function_name, params.arguments),
                        timeout=30.0  # 30 second timeout for tool execution
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Timeout executing {function_name}")
                    error_msg = f"Tool execution timed out after 30 seconds"
                    await params.result_callback(error_msg)
                    return
                
                # Format result for voice if needed
                formatted_result = format_tool_response_for_voice(function_name, result)
                
                # Ensure result is string for LM Studio
                if isinstance(result, (dict, list)):
                    result_str = json.dumps(result, ensure_ascii=False)
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
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON error in function '{function_name}': {e}")
                error_msg = f"Error: Invalid JSON response from {function_name}"
                await params.result_callback(error_msg)
            except KeyError as e:
                logger.error(f"❌ Missing required parameter in '{function_name}': {e}")
                error_msg = f"Error: Missing required parameter '{e}' for {function_name}"
                await params.result_callback(error_msg)
            except Exception as e:
                logger.error(f"❌ Unexpected error in function '{function_name}': {e}", exc_info=True)
                error_msg = f"Error executing {function_name}: {str(e)}"
                await params.result_callback(error_msg)
        
        return handler
    
    async def _process_context(self, context):
        """
        Override to ensure tools are properly set and to handle the initial
        greeting where no user message is present.
        """
        has_user_message = any(
            msg.get("role") == "user" for msg in context.messages
        )

        # If there's no user message (i.e., the bot is speaking first) and tools are present,
        # create a temporary context object without tools for this specific API call.
        if not has_user_message and context.tools is not NOT_GIVEN:
            logger.debug("No user message in context. Creating temporary tool-less context for this turn.")
            # Create a new, temporary context instance without tools.
            tool_less_context = OpenAILLMContext(
                messages=context.messages,
                tools=NOT_GIVEN
            )
            # Process the temporary context. The original context remains unchanged for future turns.
            return await super()._process_context(tool_less_context)
        else:
            # For all other cases (or if no tools are defined), process the original context.
            return await super()._process_context(context)

    async def _stream_chat_completions(self, context):
        """Override to add debugging and push immediate feedback on tool calls."""
        logger.debug("🌊 Starting streaming chat completion")

        stream = await super()._stream_chat_completions(context)

        async def feedback_stream():
            tool_call_detected = False
            async for chunk in stream:
                if not tool_call_detected and hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_call_detected = True
                        # We only need to do this once per set of tool calls
                        try:
                            # Get the first function name to create a generic response
                            function_name = delta.tool_calls[0].function.name
                            if function_name:
                                feedback_text = "Just a moment."
                                if "search" in function_name or "browse" in function_name:
                                    feedback_text = "Searching for that."
                                elif "weather" in function_name:
                                    feedback_text = "One moment while I check the weather."
                                elif "calculate" in function_name:
                                    feedback_text = "Calculating that for you."
                                
                                logger.info(f"🎤 Pushing immediate TTS feedback for tool call: '{feedback_text}'")
                                # This TextFrame goes directly to the TTS service, providing instant feedback.
                                await self.push_frame(TextFrame(feedback_text))
                        except (AttributeError, IndexError):
                            # Fallback if the chunk structure is unexpected
                            logger.warning("Could not extract function name for immediate feedback.")
                yield chunk

        return feedback_stream()
    

    def get_tools_schema(self) -> ToolsSchema:
        """
        Get the ToolsSchema object for use in context
        
        Returns:
            ToolsSchema containing all available tools
        """
        return self._tools_schema