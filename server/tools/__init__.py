"""
Tools module for Slowcat voice agent
Provides tool definitions, handlers, and formatters for LLM function calling
"""

from .definitions import (
    get_tools,
    get_tool_names,
    get_tool_by_name,
    ALL_FUNCTION_SCHEMAS
)

from .handlers import (
    execute_tool_call,
    tool_handlers,
    set_memory_processor
)

from .formatters import (
    format_tool_response_for_voice
)

__all__ = [
    # Definitions
    'get_tools',
    'get_tool_names', 
    'get_tool_by_name',
    'ALL_FUNCTION_SCHEMAS',
    
    # Handlers
    'execute_tool_call',
    'tool_handlers',
    'set_memory_processor',
    
    # Formatters
    'format_tool_response_for_voice'
]