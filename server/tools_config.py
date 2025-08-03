"""
Tool definitions for LM Studio function calling integration
These tools enable Slowcat to interact with external services
"""

from typing import Dict, Any, List
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionToolParam

# Define available tools in OpenAI format
AVAILABLE_TOOLS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "remember_information",
            "description": "Store information in memory for future conversations",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to store the information under"
                    },
                    "value": {
                        "type": "string",
                        "description": "Information to remember"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_information",
            "description": "Retrieve previously stored information from memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to retrieve information for"
                    }
                },
                "required": ["key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g. '2 + 2', '10 * 5', 'sqrt(16)')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_url",
            "description": "Fetch and extract text content from a web page URL",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files containing specific text",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a folder",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a text or markdown file",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["file_path", "content"]
            }
        }
    }
]

# Tool choice strategies
TOOL_CHOICE_AUTO = "auto"  # Let model decide
TOOL_CHOICE_NONE = "none"  # Disable tools
TOOL_CHOICE_REQUIRED = "required"  # Force tool use

def get_voice_optimized_tools() -> List[ChatCompletionToolParam]:
    """
    Get tools optimized for voice interaction.
    May filter or modify tools based on voice UX requirements.
    """
    # For now, return all tools
    # In future, could filter based on context or user preferences
    return AVAILABLE_TOOLS

def format_tool_response_for_voice(tool_name: str, result: Any) -> str:
    """
    Format tool responses for natural speech output.
    
    Args:
        tool_name: Name of the tool that was called
        result: Raw result from the tool
        
    Returns:
        Human-friendly string for TTS
    """
    if tool_name == "get_weather":
        if isinstance(result, dict):
            if "error" in result:
                return f"I couldn't get the weather: {result['error']}"
            
            temp = result.get("temperature", "unknown")
            conditions = result.get("conditions", "unknown")
            location = result.get("location", "your location")
            wind = result.get("wind_speed", 0)
            units = result.get("units", "celsius")
            
            # Build natural response
            response = f"The weather in {location} is {temp} degrees {units} with {conditions}"
            if wind > 0:
                response += f" and wind speed of {wind} kilometers per hour"
            return response
        return str(result)
    
    elif tool_name == "search_web":
        if isinstance(result, list) and len(result) > 0:
            # Check for errors
            if result[0].get("title") == "Search Error":
                return "I had trouble searching for that information"
            
            # Check if we have good results from Brave
            has_brave_results = any(r.get("source") == "Brave Search" for r in result)
            
            # If it's a Brave AI Summary
            if result[0].get("source") == "Brave AI Summary":
                snippet = result[0].get("snippet", "")
                # Clean up snippet for voice
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                return snippet
            
            # If it's a quick answer or definition
            first_result = result[0]
            if first_result.get("title") in ["Quick Answer", "Definition", "Summary"]:
                snippet = first_result.get("snippet", "")
                # Clean up snippet for voice
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                return snippet
            
            # For Brave Search results with URLs
            if has_brave_results:
                summary = "I found: "
                for i, item in enumerate(result[:2]):  # Top 2 for voice
                    if isinstance(item, dict) and item.get("source") == "Brave Search":
                        title = item.get("title", "Unknown")
                        snippet = item.get("snippet", "")
                        # Clean up and shorten
                        if snippet:
                            snippet = snippet[:80] + "..."
                        if i > 0:
                            summary += " Also, "
                        summary += f"{snippet}"
                return summary
            
            # Otherwise fallback to original format
            summary = "Here's what I found: "
            for i, item in enumerate(result[:2]):  # Top 2 for voice
                if isinstance(item, dict):
                    title = item.get("title", "Unknown")
                    snippet = item.get("snippet", "")[:100]
                    if i > 0:
                        summary += " Also, "
                    summary += f"{title}: {snippet}"
            return summary
        return "I couldn't find any results for that search"
    
    elif tool_name == "remember_information":
        if isinstance(result, dict) and result.get("status") == "saved":
            return "I've saved that information"
        return "I had trouble saving that information"
    
    elif tool_name == "recall_information":
        if result:
            return f"I remember: {result}"
        return "I don't have any information saved about that"
    
    elif tool_name == "calculate":
        if isinstance(result, dict):
            if "error" in result:
                return f"I couldn't calculate that: {result['error']}"
            
            expr = result.get("expression", "")
            res = result.get("result", "")
            
            # Format numbers nicely for speech
            if isinstance(res, float):
                if res == int(res):
                    res = int(res)
                elif abs(res) < 0.0001 or abs(res) > 10000:
                    res = f"{res:.2e}"  # Scientific notation
                else:
                    res = round(res, 4)
            
            return f"{expr} equals {res}"
        return "I couldn't perform that calculation"
    
    elif tool_name == "browse_url":
        if isinstance(result, dict):
            if "error" in result:
                return f"I couldn't read that webpage: {result['error']}"
            
            title = result.get("title", "the page")
            content = result.get("content", "")
            
            # For voice, summarize the content briefly
            if content:
                # Extract first paragraph or 200 chars
                first_part = content.split('\n\n')[0][:200]
                if result.get("truncated"):
                    return f"From {title}: {first_part}... The page contains more content."
                else:
                    return f"From {title}: {first_part}"
            else:
                return f"I found the page {title} but couldn't extract any text."
        return "I couldn't read that webpage"
    
    elif tool_name == "read_file":
        if isinstance(result, dict):
            if "error" in result:
                return f"I couldn't read the file: {result['error']}"
            
            name = result.get("name", "the file")
            content = result.get("content", "")
            
            if content:
                # For voice, give a brief summary
                lines = content.split('\n')
                preview = ' '.join(lines[:3])[:200]  # First 3 lines, max 200 chars
                
                if result.get("truncated"):
                    return f"From {name}: {preview}... The file contains more content."
                else:
                    return f"From {name}: {preview}"
            else:
                return f"The file {name} appears to be empty."
        return "I couldn't read that file"
    
    elif tool_name == "search_files":
        if isinstance(result, list):
            if len(result) == 0:
                return "I didn't find any files matching that search."
            
            if result[0].get("error"):
                return f"Search error: {result[0]['error']}"
            
            # Summarize results
            count = len(result)
            if count == 1:
                return f"I found 1 file: {result[0]['name']}"
            else:
                files = ', '.join(r['name'] for r in result[:3])
                if count > 3:
                    return f"I found {count} files. The first few are: {files}, and more."
                else:
                    return f"I found {count} files: {files}"
        return "I couldn't search for files"
    
    elif tool_name == "list_files":
        if isinstance(result, dict):
            if "error" in result:
                return f"I couldn't list the directory: {result['error']}"
            
            files = result.get("files", [])
            dirs = result.get("directories", [])
            directory = result.get("directory", "the directory")
            
            # Extract just the folder name for voice
            if "/" in directory:
                folder_name = directory.split("/")[-1] or "your home"
            else:
                folder_name = directory
            
            response = f"In {folder_name}, "
            
            if dirs:
                response += f"I see {len(dirs)} folders"
                if len(dirs) <= 3:
                    # Clean folder names for voice (remove special chars)
                    clean_dirs = [d.replace("_", " ").replace("-", " ") for d in dirs]
                    response += f": {', '.join(clean_dirs)}"
                else:
                    response += f" including {dirs[0]} and {dirs[1]}"
            
            if files:
                if dirs:
                    response += ", and "
                else:
                    response += "I see "
                response += f"{len(files)} files"
                if len(files) <= 3 and files:
                    # Clean file names for voice (remove extensions and special chars)
                    clean_files = []
                    for f in files[:3]:
                        name = f['name']
                        # Remove file extension for cleaner speech
                        if '.' in name:
                            name = name.rsplit('.', 1)[0]
                        # Replace special characters
                        name = name.replace("_", " ").replace("-", " ")
                        clean_files.append(name)
                    response += f" including {', '.join(clean_files)}"
                elif len(files) > 3:
                    # Just mention first file type
                    first_file = files[0]['name']
                    if '.' in first_file:
                        ext = first_file.rsplit('.', 1)[1]
                        response += f", mostly {ext} files"
            
            if not dirs and not files:
                response = f"{folder_name} appears to be empty"
            
            return response
        return "I couldn't list that directory"
    
    elif tool_name == "write_file":
        if isinstance(result, dict):
            if "error" in result:
                error = result['error']
                if "already exists" in error:
                    return "That file already exists. Should I overwrite it?"
                elif "Only .txt and .md" in error:
                    return "I can only create text or markdown files"
                else:
                    return f"I couldn't write the file: {error}"
            
            if result.get("success"):
                name = result.get("name", "the file")
                # Clean filename for speech
                if '.' in name:
                    name = name.rsplit('.', 1)[0]
                name = name.replace("_", " ").replace("-", " ")
                
                return f"I've created {name} on your Desktop"
            
        return "I couldn't write that file"
    
    # Default formatting
    return str(result)