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

# ❌ REMOVED: Weather, Web Search, and Browser tools - now handled by MCP servers
# - get_weather -> handled by MCP or external API
# - search_web -> handled by brave_web_search MCP server  
# - browse_url -> handled by browser-text MCP server

# ❌ REMOVED: Memory tools - now handled by MCP memory server
# - store_memory -> memory_create_entities (MCP)
# - retrieve_memory -> memory_search_nodes (MCP) 
# - search_memory -> memory_search_nodes (MCP)
# - delete_memory -> memory_delete_entities (MCP)
# - update_memory -> memory_update_entities (MCP)

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

# ❌ REMOVED: File system tools - now handled by MCP filesystem server
# - read_file -> filesystem_read_file (MCP)
# - write_file -> filesystem_write_file (MCP)  
# - list_files -> filesystem_list_files (MCP)
# - search_files -> can be done with filesystem + text search

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

STOP_MUSIC = FunctionSchema(
    name="stop_music",
    description="Stop music playback completely",
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


# List of LOCAL-ONLY function schemas (MCP tools removed)
# These tools stay local for performance/hardware integration reasons
ALL_FUNCTION_SCHEMAS: List[FunctionSchema] = [
    # Time/date (performance)
    GET_CURRENT_TIME,
    
    # Math/calculation (as requested)
    CALCULATE,
    
    # Timed task execution (app state)
    START_TIMED_TASK,
    CHECK_TASK_STATUS, 
    STOP_TIMED_TASK,
    ADD_TO_TIMED_TASK,
    GET_ACTIVE_TASKS,
    
    # Music control (hardware integration)
    PLAY_MUSIC,
    PAUSE_MUSIC,
    SKIP_SONG,
    STOP_MUSIC,
    QUEUE_MUSIC,
    SEARCH_MUSIC,
    GET_NOW_PLAYING,
    SET_VOLUME,
    CREATE_PLAYLIST,
    GET_MUSIC_STATS
]

# MCP tools are now handled natively by LM Studio:
# - Web search: brave_web_search 
# - Memory: memory_create_entities, memory_search_nodes, etc.
# - Filesystem: filesystem_read_file, filesystem_write_file, etc.
# - Browser: browser automation via @playwright/mcp
# - JavaScript: run_javascript via @modelcontextprotocol/server-javascript

# Dictionary of translated tool descriptions
TOOL_DESCRIPTIONS_IT = {
    "play_music": "Riproduci musica - cerca e riproduci un brano o riprendi la riproduzione",
    "pause_music": "Metti in pausa la musica attualmente in riproduzione",
    "skip_song": "Salta al brano successivo nella coda",
    "stop_music": "Interrompi completamente la riproduzione musicale",
    "queue_music": "Aggiungi brani alla coda di riproduzione",
    "search_music": "Cerca nella libreria musicale",
    "get_now_playing": "Ottieni informazioni sul brano attualmente in riproduzione",
    "set_volume": "Imposta il volume di riproduzione della musica",
    "create_playlist": "Crea una playlist basata sull'umore o sul genere",
}

TOOL_DESCRIPTIONS_LANG = {
    "it": TOOL_DESCRIPTIONS_IT,
}

def get_tools(language: str = "en") -> ToolsSchema:
    """
    Get all tools wrapped in Pipecat's ToolsSchema format, with descriptions
    translated to the specified language if available.
    
    Returns:
        ToolsSchema object containing all available tools
    """
    if language == "en" or language not in TOOL_DESCRIPTIONS_LANG:
        return ToolsSchema(standard_tools=ALL_FUNCTION_SCHEMAS)

    translated_schemas = []
    lang_descriptions = TOOL_DESCRIPTIONS_LANG[language]
    
    for schema in ALL_FUNCTION_SCHEMAS:
        description = schema.description
        if schema.name in lang_descriptions:
            description = lang_descriptions[schema.name]
            
        new_schema = FunctionSchema(
            name=schema.name,
            description=description,
            properties=schema.properties,
            required=schema.required
        )
        
        translated_schemas.append(new_schema)
        
    return ToolsSchema(standard_tools=translated_schemas)

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