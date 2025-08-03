"""
Tool definitions for Slowcat using Pipecat's FunctionSchema format
This is the single source of truth for all tool definitions
"""

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from typing import List

# Time and Date Tools
GET_CURRENT_TIME = FunctionSchema(
    name="get_current_time",
    description="Get the current date and time in various formats",
    properties={
        "format": {
            "type": "string",
            "enum": ["ISO", "human", "unix", "date_only", "time_only"],
            "description": "Format for the time output",
            "default": "human"
        },
        "timezone": {
            "type": "string",
            "description": "Timezone name (e.g. 'UTC', 'America/New_York', 'Europe/London')",
            "default": "UTC"
        }
    },
    required=[]
)

# Weather Tools
GET_WEATHER = FunctionSchema(
    name="get_weather",
    description="Get the current weather for a location",
    properties={
        "location": {
            "type": "string",
            "description": "City name or coordinates (e.g. 'Paris' or '48.8566,2.3522')"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature units",
            "default": "celsius"
        }
    },
    required=["location"]
)

# Web Search Tools
SEARCH_WEB = FunctionSchema(
    name="search_web",
    description="Search the web for current information",
    properties={
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return",
            "default": 3
        }
    },
    required=["query"]
)

BROWSE_URL = FunctionSchema(
    name="browse_url",
    description="Fetch and extract text content from a web page URL",
    properties={
        "url": {
            "type": "string",
            "description": "The URL to fetch and read"
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum length of text to return (default: 2000 characters)",
            "default": 2000
        }
    },
    required=["url"]
)

# Memory Tools
REMEMBER_INFORMATION = FunctionSchema(
    name="remember_information",
    description="Store information in memory for future conversations",
    properties={
        "key": {
            "type": "string",
            "description": "Key to store the information under"
        },
        "value": {
            "type": "string",
            "description": "Information to remember"
        }
    },
    required=["key", "value"]
)

RECALL_INFORMATION = FunctionSchema(
    name="recall_information",
    description="Retrieve previously stored information from memory",
    properties={
        "key": {
            "type": "string",
            "description": "Key to retrieve information for"
        }
    },
    required=["key"]
)

# Calculation Tools
CALCULATE = FunctionSchema(
    name="calculate",
    description="Perform basic mathematical calculations",
    properties={
        "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate (e.g. '2 + 2', '10 * 5', 'sqrt(16)')"
        }
    },
    required=["expression"]
)

# File System Tools
READ_FILE = FunctionSchema(
    name="read_file",
    description="Read the contents of a local file",
    properties={
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum characters to read (default: 5000)",
            "default": 5000
        }
    },
    required=["file_path"]
)

SEARCH_FILES = FunctionSchema(
    name="search_files",
    description="Search for files containing specific text",
    properties={
        "query": {
            "type": "string",
            "description": "Text to search for"
        },
        "directory": {
            "type": "string",
            "description": "Directory to search in (default: current directory)",
            "default": "."
        },
        "file_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "File extensions to search (e.g., ['.txt', '.md'])"
        }
    },
    required=["query"]
)

LIST_FILES = FunctionSchema(
    name="list_files",
    description="List files and directories in a folder",
    properties={
        "directory": {
            "type": "string",
            "description": "Directory to list (default: current directory)",
            "default": "."
        },
        "pattern": {
            "type": "string",
            "description": "Glob pattern for filtering (e.g., '*.txt')",
            "default": "*"
        }
    },
    required=[]
)

WRITE_FILE = FunctionSchema(
    name="write_file",
    description="Write content to a text or markdown file",
    properties={
        "file_path": {
            "type": "string",
            "description": "Path or filename (e.g., 'notes.txt' or '/Users/YourDesktop/todo.md')"
        },
        "content": {
            "type": "string",
            "description": "Content to write to the file"
        },
        "overwrite": {
            "type": "boolean",
            "description": "Whether to overwrite if file exists (default: false)",
            "default": False
        }
    },
    required=["file_path", "content"]
)

# List of all function schemas for easy access
ALL_FUNCTION_SCHEMAS: List[FunctionSchema] = [
    GET_CURRENT_TIME,
    GET_WEATHER,
    SEARCH_WEB,
    BROWSE_URL,
    REMEMBER_INFORMATION,
    RECALL_INFORMATION,
    CALCULATE,
    READ_FILE,
    SEARCH_FILES,
    LIST_FILES,
    WRITE_FILE
]

def get_tools() -> ToolsSchema:
    """
    Get all tools wrapped in Pipecat's ToolsSchema format
    
    Returns:
        ToolsSchema object containing all available tools
    """
    return ToolsSchema(standard_tools=ALL_FUNCTION_SCHEMAS)

def get_tool_names() -> List[str]:
    """
    Get list of all available tool names
    
    Returns:
        List of tool function names
    """
    return [schema.name for schema in ALL_FUNCTION_SCHEMAS]

def get_tool_by_name(name: str) -> FunctionSchema:
    """
    Get a specific tool schema by name
    
    Args:
        name: The function name to look up
        
    Returns:
        FunctionSchema for the requested tool
        
    Raises:
        ValueError: If tool name not found
    """
    for schema in ALL_FUNCTION_SCHEMAS:
        if schema.name == name:
            return schema
    raise ValueError(f"Tool '{name}' not found")