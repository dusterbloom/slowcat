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

# Conversation History Tools
SEARCH_CONVERSATIONS = FunctionSchema(
    name="search_conversations",
    description="Search through past conversation history for specific topics or information",
    properties={
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
    required=["query"]
)

GET_CONVERSATION_SUMMARY = FunctionSchema(
    name="get_conversation_summary",
    description="Get a summary of conversations within a date range or overall statistics",
    properties={
        "days_back": {
            "type": "integer",
            "description": "Number of days back to look (default: 7, use 0 for all time)",
            "default": 7
        },
        "user_id": {
            "type": "string",
            "description": "Filter by specific user (optional)",
            "default": None
        }
    },
    required=[]
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

# Time-aware task tools
START_TIMED_TASK = FunctionSchema(
    name="start_timed_task",
    description="Start a timed task that runs for a specific duration. Use this when the user wants something done for a certain amount of time.",
    properties={
        "description": {
            "type": "string",
            "description": "What the task is doing, e.g., 'Search for hotels in Paris', 'Research AI news'"
        },
        "duration_minutes": {
            "type": "number",
            "description": "How long to run the task in minutes (can be decimal, e.g., 0.5 for 30 seconds)"
        },
        "output_file": {
            "type": "string",
            "description": "Optional file path to save results, e.g., 'searches/hotels.md', 'research/ai_news.json'"
        }
    },
    required=["description", "duration_minutes"]
)

CHECK_TASK_STATUS = FunctionSchema(
    name="check_task_status",
    description="Check the status of a running timed task",
    properties={
        "task_id": {
            "type": "string",
            "description": "The task ID to check status for"
        }
    },
    required=["task_id"]
)

STOP_TIMED_TASK = FunctionSchema(
    name="stop_timed_task",
    description="Stop a timed task before it completes",
    properties={
        "task_id": {
            "type": "string",
            "description": "The task ID to stop"
        }
    },
    required=["task_id"]
)

ADD_TO_TIMED_TASK = FunctionSchema(
    name="add_to_timed_task",
    description="Add a result or finding to an active timed task",
    properties={
        "task_id": {
            "type": "string",
            "description": "The task ID to add results to"
        },
        "content": {
            "type": "object",
            "description": "The content to add (can include title, snippet, source, or any relevant data)"
        }
    },
    required=["task_id", "content"]
)

GET_ACTIVE_TASKS = FunctionSchema(
    name="get_active_tasks",
    description="Get a list of all currently active timed tasks",
    properties={},
    required=[]
)

# Music control tools
PLAY_MUSIC = FunctionSchema(
    name="play_music",
    description="Play music - search and play a song or resume playback",
    properties={
        "query": {
            "type": "string",
            "description": "Optional search query (e.g., 'jazz', 'Beatles', 'upbeat music'). Leave empty to resume."
        }
    },
    required=[]
)

PAUSE_MUSIC = FunctionSchema(
    name="pause_music",
    description="Pause the currently playing music",
    properties={},
    required=[]
)

SKIP_SONG = FunctionSchema(
    name="skip_song",
    description="Skip to the next song in the queue",
    properties={},
    required=[]
)

QUEUE_MUSIC = FunctionSchema(
    name="queue_music",
    description="Add songs to the play queue",
    properties={
        "query": {
            "type": "string",
            "description": "Search query for songs to add to queue"
        }
    },
    required=["query"]
)

SEARCH_MUSIC = FunctionSchema(
    name="search_music",
    description="Search the music library",
    properties={
        "query": {
            "type": "string",
            "description": "Search query for songs, artists, or albums"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results (default: 10)",
            "default": 10
        }
    },
    required=["query"]
)

GET_NOW_PLAYING = FunctionSchema(
    name="get_now_playing",
    description="Get information about the currently playing song",
    properties={},
    required=[]
)

SET_VOLUME = FunctionSchema(
    name="set_volume",
    description="Set the music playback volume",
    properties={
        "level": {
            "type": "integer",
            "description": "Volume level (0-100)"
        }
    },
    required=["level"]
)

CREATE_PLAYLIST = FunctionSchema(
    name="create_playlist",
    description="Create a playlist based on mood or genre",
    properties={
        "mood": {
            "type": "string",
            "description": "Mood or genre (e.g., 'relaxing', 'energetic', 'jazz', 'party')"
        },
        "count": {
            "type": "integer",
            "description": "Number of songs to add (default: 10)",
            "default": 10
        }
    },
    required=["mood"]
)

GET_MUSIC_STATS = FunctionSchema(
    name="get_music_stats",
    description="Get statistics about the music library",
    properties={},
    required=[]
)

# List of all function schemas for easy access
ALL_FUNCTION_SCHEMAS: List[FunctionSchema] = [
    GET_CURRENT_TIME,
    GET_WEATHER,
    SEARCH_WEB,
    BROWSE_URL,
    REMEMBER_INFORMATION,
    RECALL_INFORMATION,
    SEARCH_CONVERSATIONS,
    GET_CONVERSATION_SUMMARY,
    CALCULATE,
    READ_FILE,
    SEARCH_FILES,
    LIST_FILES,
    WRITE_FILE,
    START_TIMED_TASK,
    CHECK_TASK_STATUS,
    STOP_TIMED_TASK,
    ADD_TO_TIMED_TASK,
    GET_ACTIVE_TASKS,
    PLAY_MUSIC,
    PAUSE_MUSIC,
    SKIP_SONG,
    QUEUE_MUSIC,
    SEARCH_MUSIC,
    GET_NOW_PLAYING,
    SET_VOLUME,
    CREATE_PLAYLIST,
    GET_MUSIC_STATS
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