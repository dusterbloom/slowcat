"""
Tool-enabled LLM service for LM Studio integration
Extends OpenAILLMService to handle function calls
"""

from typing import List, Dict, Any
import json
from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.frames.frames import TextFrame

from tool_handlers import execute_tool_call
from tools_config import format_tool_response_for_voice


class ToolEnabledLLMService(OpenAILLMService):
    """
    LLM service with tool/function calling support for LM Studio.
    Handles tool execution and formats responses for voice.
    """
    
    def __init__(self, **kwargs):
        """Initialize the tool-enabled LLM service"""
        super().__init__(**kwargs)
        logger.info("Initialized ToolEnabledLLMService with function calling support")
        
        # Register all tools as function handlers
        self._register_tool_handlers()
    
    def _register_tool_handlers(self):
        """Register tool handlers with Pipecat's function registry"""
        from tools_config import AVAILABLE_TOOLS
        
        # Register each tool as a function handler
        for tool_def in AVAILABLE_TOOLS:
            function_name = tool_def["function"]["name"]
            
            # Create a closure to capture the function name
            def make_handler(fn_name):
                async def handler(function_call_params):
                    """Handler that executes tool and returns result"""
                    try:
                        # Extract arguments from the new params object
                        arguments = function_call_params.arguments
                        result_callback = function_call_params.result_callback
                        
                        # Execute the tool
                        result = await execute_tool_call(fn_name, arguments)
                        
                        # Ensure result is JSON serializable
                        if isinstance(result, (dict, list)):
                            result_content = json.dumps(result)
                        else:
                            result_content = str(result)
                        
                        # Truncate if too large
                        if len(result_content) > 4000:
                            result_content = result_content[:4000] + "... [truncated]"
                        
                        # Return result through callback
                        await result_callback(result_content)
                        
                    except Exception as e:
                        logger.error(f"Error executing {fn_name}: {e}")
                        if hasattr(function_call_params, 'result_callback'):
                            await function_call_params.result_callback(f"Error: {str(e)}")
                
                return handler
            
            # Register the handler
            self.register_function(
                function_name,
                make_handler(function_name),
                cancel_on_interruption=True
            )
            
        logger.info(f"Registered {len(AVAILABLE_TOOLS)} tool handlers")