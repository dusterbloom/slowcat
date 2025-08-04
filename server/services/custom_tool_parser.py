"""
Custom tool call parser for non-standard LLM outputs
Handles formats like: [function_name param1="value1" param2="value2"]
"""
import re
import json
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class CustomToolParser:
    """Parse custom tool call formats from LLMs"""
    
    # Pattern to match [function_name param="value"] format
    TOOL_PATTERN = re.compile(r'\[(\w+)(?:\s+(.+?))?\]')
    PARAM_PATTERN = re.compile(r'(\w+)="([^"]*)"')
    
    @classmethod
    def parse_content_for_tools(cls, content: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        Parse content for tool calls in custom format.
        Returns: (tool_calls, remaining_content)
        """
        if not content or '[' not in content:
            return None, content
            
        logger.debug(f"Checking for custom tool format in: {content[:100]}...")
            
        tool_calls = []
        remaining_content = content
        
        for match in cls.TOOL_PATTERN.finditer(content):
            function_name = match.group(1)
            params_str = match.group(2) or ""
            
            # Parse parameters
            arguments = {}
            if params_str:
                for param_match in cls.PARAM_PATTERN.finditer(params_str):
                    param_name = param_match.group(1)
                    param_value = param_match.group(2)
                    arguments[param_name] = param_value
            
            tool_call = {
                "id": str(hash(f"{function_name}{arguments}")),
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments)
                }
            }
            tool_calls.append(tool_call)
            
            # Remove the tool call from content
            remaining_content = remaining_content.replace(match.group(0), "").strip()
        
        if tool_calls:
            logger.info(f"Parsed {len(tool_calls)} custom format tool calls")
            for tc in tool_calls:
                logger.debug(f"  - {tc['function']['name']}: {tc['function']['arguments']}")
            return tool_calls, remaining_content
        
        return None, content
    
    @classmethod
    def convert_to_openai_format(cls, tool_calls: List[Dict]) -> List:
        """Convert parsed tool calls to OpenAI format"""
        from openai.types.chat import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function
        
        openai_tools = []
        for tc in tool_calls:
            tool_call = ChatCompletionMessageToolCall(
                id=tc["id"],
                type=tc["type"],
                function=Function(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"]
                )
            )
            openai_tools.append(tool_call)
        
        return openai_tools