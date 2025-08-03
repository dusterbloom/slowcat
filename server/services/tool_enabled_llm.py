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
    
    async def run_function_calls(self, function_calls: List[FunctionCallFromLLM]):
        """
        Execute function calls requested by the LLM.
        
        Args:
            function_calls: List of function calls to execute
        """
        logger.info(f"Running {len(function_calls)} function calls")
        
        for call in function_calls:
            try:
                # Extract function details
                function_name = call.function_name
                arguments = call.arguments
                tool_call_id = call.tool_call_id
                
                logger.info(f"Executing function: {function_name}")
                
                # Execute the tool
                result = await execute_tool_call(function_name, arguments)
                
                # Format result for voice
                voice_response = format_tool_response_for_voice(function_name, result)
                
                # Log the execution (limit size to prevent memory issues)
                result_str = str(result)
                if len(result_str) > 200:
                    logger.info(f"Tool {function_name} result: {result_str[:200]}...")
                else:
                    logger.info(f"Tool {function_name} result: {result_str}")
                logger.info(f"Voice response: {voice_response[:100]}..." if len(voice_response) > 100 else f"Voice response: {voice_response}")
                
                # Add tool response to context
                # The response will be sent back to the model in the next request
                if hasattr(call, 'context') and call.context:
                    # Ensure result is JSON serializable and not too large
                    try:
                        if isinstance(result, (dict, list)):
                            result_content = json.dumps(result)
                        else:
                            result_content = str(result)
                        
                        # Truncate if too large to prevent memory issues
                        if len(result_content) > 4000:
                            result_content = result_content[:4000] + "... [truncated]"
                            
                    except Exception as json_error:
                        logger.warning(f"Could not JSON serialize result: {json_error}")
                        result_content = str(result)[:4000]
                    
                    call.context.add_message({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content
                    })
                
                # NOTE: Disabled immediate voice feedback to prevent TTS conflicts
                # The model will generate its own response after tool execution
                # if self._should_announce_tool_use(function_name):
                #     announcement = self._get_tool_announcement(function_name)
                #     await self.push_frame(TextFrame(text=announcement))
                
            except Exception as e:
                logger.error(f"Error executing function {function_name}: {e}", exc_info=True)
                # Add error to context
                if hasattr(call, 'context') and call.context:
                    try:
                        call.context.add_message({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": f"Error: {str(e)}"
                        })
                    except Exception as context_error:
                        logger.error(f"Error adding error to context: {context_error}")
    
    def _should_announce_tool_use(self, function_name: str) -> bool:
        """
        Determine if we should announce tool usage for voice UX.
        
        Args:
            function_name: Name of the function being called
            
        Returns:
            True if we should announce, False otherwise
        """
        # Announce tools that might take time
        announce_tools = ["search_web", "get_weather"]
        return function_name in announce_tools
    
    def _get_tool_announcement(self, function_name: str) -> str:
        """
        Get voice announcement for tool usage.
        
        Args:
            function_name: Name of the function being called
            
        Returns:
            Announcement text
        """
        announcements = {
            "get_weather": "Let me check the weather for you.",
            "search_web": "Let me search for that information.",
            "remember_information": "I'll remember that.",
            "recall_information": "Let me recall that information."
        }
        return announcements.get(function_name, "Processing your request.")