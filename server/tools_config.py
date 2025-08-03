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
            
            # Summarize results based on type
            first_result = result[0]
            
            # If it's a quick answer or definition
            if first_result.get("title") in ["Quick Answer", "Definition", "Summary"]:
                snippet = first_result.get("snippet", "")
                # Clean up snippet for voice
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."
                return snippet
            
            # Otherwise summarize multiple results
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
    
    # Default formatting
    return str(result)