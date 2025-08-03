# Tool Implementation Guide

This guide explains how tools (function calling) are implemented in Slowcat using Pipecat and LM Studio.

## Architecture Overview

The tool system is built on Pipecat's function calling framework and leverages LM Studio's OpenAI-compatible API:

```
tools/
├── definitions.py    # FunctionSchema definitions (single source of truth)
├── handlers.py       # Tool execution logic
├── formatters.py     # Voice formatting utilities
└── __init__.py       # Module exports

services/
└── llm_with_tools.py # Unified LLM service with tool support
```

## Best Practices

### 1. Tool Definition Structure

Follow the OpenAI function calling schema exactly:

```python
{
    "type": "function",
    "function": {
        "name": "tool_name",  # Snake_case, descriptive
        "description": "Clear, concise description for the LLM",
        "parameters": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string|number|boolean|array|object",
                    "description": "What this parameter does",
                    "enum": ["option1", "option2"],  # Optional
                    "default": "default_value"  # Optional
                }
            },
            "required": ["param1", "param2"]  # List required params
        }
    }
}
```

### 2. Handler Implementation Pattern

Every tool handler should follow this pattern:

```python
async def tool_name(self, param1: str, param2: int = None) -> Dict[str, Any]:
    """
    Tool description
    
    Args:
        param1: Description
        param2: Description with default
        
    Returns:
        Dict with consistent structure
    """
    try:
        # Input validation
        if not param1:
            return {"error": "param1 is required"}
            
        # Core logic here
        result = await self._do_something(param1, param2)
        
        # Return structured data
        return {
            "success": True,
            "data": result,
            # Include metadata for voice formatting
            "display_value": "Human readable result",
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "tool_name"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in tool_name: {e}")
        return {"error": str(e)}
```

### 3. Voice-Optimized Response Formatting

Add voice formatting in `format_tool_response_for_voice()`:

```python
elif tool_name == "your_tool":
    if isinstance(result, dict):
        if "error" in result:
            return f"I had trouble with that: {result['error']}"
        
        # Extract key information
        value = result.get("display_value", "")
        
        # Format for natural speech
        # - Keep responses concise
        # - Use natural language
        # - Avoid technical jargon
        # - Include only essential information
        
        return f"Here's what I found: {value}"
    return "I couldn't complete that task"
```

### 4. Language Support Considerations

Tools should work across all supported languages:

1. **Keep tool names in English** - The function names should remain consistent
2. **Localize descriptions** - Consider adding language-specific descriptions in the system prompt
3. **Format responses appropriately** - Voice formatting should adapt to the current language

### 5. Error Handling Best Practices

```python
# Always wrap in try-except
try:
    # Tool logic
except SpecificException as e:
    # Handle known errors gracefully
    logger.warning(f"Expected error in {tool_name}: {e}")
    return {"error": "User-friendly error message"}
except Exception as e:
    # Log unexpected errors
    logger.error(f"Unexpected error in {tool_name}: {e}")
    return {"error": "Something went wrong"}
```

### 6. Async Best Practices

- Always use `async def` for tool handlers
- Use `aiohttp` for HTTP requests, not `requests`
- Handle timeouts appropriately:

```python
async with aiohttp.ClientSession() as session:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            # Handle response
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}
```

### 7. Testing Tools

Test each tool in isolation:

```python
# test_tools.py
import asyncio
from tool_handlers import ToolHandlers

async def test_time_tool():
    handlers = ToolHandlers()
    
    # Test different formats
    result = await handlers.get_current_time(format="human", timezone="America/New_York")
    print(f"Human format: {result}")
    
    result = await handlers.get_current_time(format="ISO", timezone="UTC")
    print(f"ISO format: {result}")

if __name__ == "__main__":
    asyncio.run(test_time_tool())
```

## Example: Complete Tool Implementation

Here's the complete implementation of the time tool following all best practices:

### 1. Tool Definition (tools_config.py)

```python
{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current date and time in various formats",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["ISO", "human", "unix", "date_only", "time_only"],
                    "description": "Format for the time output",
                    "default": "human"
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g. 'UTC', 'America/New_York')",
                    "default": "UTC"
                }
            },
            "required": []
        }
    }
}
```

### 2. Handler Implementation (tool_handlers.py)

```python
async def get_current_time(self, format: str = "human", timezone: str = "UTC") -> Dict[str, Any]:
    """Get the current date and time in various formats"""
    try:
        # Validate timezone
        try:
            tz = pytz.timezone(timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone {timezone}, using UTC")
            tz = pytz.UTC
            
        # Get current time
        now = datetime.now(tz)
        
        # Format based on request
        if format == "ISO":
            time_str = now.isoformat()
        elif format == "unix":
            time_str = str(int(now.timestamp()))
        elif format == "date_only":
            time_str = now.strftime("%Y-%m-%d")
        elif format == "time_only":
            time_str = now.strftime("%H:%M:%S")
        else:  # human format
            time_str = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
        
        return {
            "time": time_str,
            "timezone": timezone,
            "format": format,
            "timestamp": int(now.timestamp()),
            "day_of_week": now.strftime("%A"),
            "date": now.strftime("%Y-%m-%d"),
            "time_24h": now.strftime("%H:%M:%S"),
            "time_12h": now.strftime("%I:%M %p")
        }
        
    except Exception as e:
        logger.error(f"Error getting current time: {e}")
        return {"error": str(e)}
```

### 3. Voice Formatting (tools_config.py)

```python
elif tool_name == "get_current_time":
    if isinstance(result, dict):
        if "error" in result:
            return f"I couldn't get the time: {result['error']}"
        
        time_str = result.get("time", "")
        format_type = result.get("format", "human")
        timezone = result.get("timezone", "UTC")
        
        # Natural timezone names for speech
        tz_spoken = {
            "UTC": "UTC",
            "America/New_York": "Eastern Time",
            "America/Chicago": "Central Time",
            "America/Los_Angeles": "Pacific Time",
            "Europe/London": "London time",
            "Asia/Tokyo": "Tokyo time"
        }
        
        tz_name = tz_spoken.get(timezone, timezone.split('/')[-1].replace('_', ' '))
        
        if format_type == "human":
            return f"It's {time_str}"
        elif format_type == "date_only":
            return f"Today's date is {time_str}"
        elif format_type == "time_only":
            return f"The time is {time_str} in {tz_name}"
        else:
            return f"The current time in {tz_name} is {time_str}"
            
    return "I couldn't get the current time"
```

### 4. Registration (tool_handlers.py)

```python
# In execute_tool_call function
elif function_name == "get_current_time":
    return await tool_handlers.get_current_time(**arguments)
```

## Troubleshooting Common Issues

### 1. Tools Not Being Called

- Verify LM Studio model supports function calling (check `/v1/models` endpoint)
- Ensure tools are properly formatted in OpenAI schema
- Check that `tools` and `tool_choice` are passed to the context

### 2. Language-Specific Issues

- System prompts should guide the model to use tools in any language
- Tool descriptions can include multilingual hints
- Voice formatting should adapt to the conversation language

### 3. Performance Considerations

- Keep tool execution under 2 seconds for voice interactions
- Use caching where appropriate
- Handle timeouts gracefully

### 4. Debugging Tips

Enable detailed logging:

```python
logger.info(f"Executing tool: {function_name} with args: {arguments}")
logger.debug(f"Tool result: {result}")
```

## Adding New Tools Checklist

- [ ] Define tool schema in `tools_config.py` AVAILABLE_TOOLS
- [ ] Implement handler in `tool_handlers.py`
- [ ] Add handler mapping in `execute_tool_call()`
- [ ] Add voice formatting in `format_tool_response_for_voice()`
- [ ] Update requirements.txt if new dependencies needed
- [ ] Test the tool in isolation
- [ ] Test with voice interaction
- [ ] Test in multiple languages
- [ ] Document any special considerations

## LM Studio Integration Notes

1. **Model Selection**: Use models with function calling support (e.g., Qwen, Llama 3.1)
2. **API Compatibility**: Slowcat uses OpenAI-compatible endpoints
3. **Streaming**: Tools work with both streaming and non-streaming responses
4. **Context Window**: Be mindful of context limits with many tools

## Future Enhancements

1. **Dynamic Tool Loading**: Load tools based on user preferences
2. **Tool Permissions**: User-configurable tool access
3. **Tool Analytics**: Track tool usage and performance
4. **Custom Tools**: Allow users to define their own tools
5. **Tool Chaining**: Support for complex multi-tool workflows