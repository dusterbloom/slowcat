"""
Voice formatting utilities for tool responses
Formats tool outputs to be natural for text-to-speech
"""

from typing import Dict, Any, Union
from loguru import logger

def format_tool_response_for_voice(function_name: str, result: Union[Dict, str, Any]) -> str:
    """
    Format tool responses for natural voice output
    
    Args:
        function_name: Name of the tool that was called
        result: Raw result from the tool
        
    Returns:
        Formatted string suitable for TTS
    """
    try:
        # Handle error responses
        if isinstance(result, dict) and "error" in result:
            return f"I encountered an error: {result['error']}"
        
        # Format based on function type
        if function_name == "get_weather":
            if isinstance(result, dict):
                location = result.get("location", "Unknown location")
                temp = result.get("temperature", "unknown")
                conditions = result.get("conditions", "unknown conditions")
                units = result.get("units", "celsius")
                unit_symbol = "°F" if units == "fahrenheit" else "°C"
                
                return (f"The weather in {location} is currently {conditions} "
                       f"with a temperature of {temp}{unit_symbol}.")
            
        elif function_name == "get_current_time":
            if isinstance(result, dict):
                time_str = result.get("time", "unknown time")
                timezone = result.get("timezone", "UTC")
                
                # Make the response more natural based on format
                format_type = result.get("format", "human")
                if format_type == "human":
                    return f"The current time is {time_str}."
                elif format_type == "date_only":
                    return f"Today's date is {time_str}."
                elif format_type == "time_only":
                    return f"The time is {time_str} in {timezone}."
                else:
                    return f"The time is: {time_str}"
                    
        elif function_name == "search_web":
            if isinstance(result, list) and result:
                # Format search results for voice
                response = "Here's what I found: "
                for i, item in enumerate(result[:3]):  # Limit to 3 results
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        snippet = item.get("snippet", "")
                        if title and snippet:
                            response += f"{title}: {snippet}. "
                return response.strip()
            else:
                return "I couldn't find any relevant search results."
                
        elif function_name == "search_web_free":
            # Handle free web search results with voice_summary
            if isinstance(result, dict):
                # Check if there's a voice_summary field (our preferred voice output)
                if "voice_summary" in result:
                    voice_summary = result["voice_summary"]
                    if voice_summary and voice_summary != "I couldn't search the web right now. Please try again.":
                        return voice_summary
                
                # Fallback: format from results array
                if "results" in result and result["results"]:
                    response = "Here's what I found: "
                    for i, item in enumerate(result["results"][:3]):  # Limit to 3 results
                        if isinstance(item, dict):
                            title = item.get("title", "").replace("…", "").strip()
                            snippet = item.get("snippet", "").replace("…", "").strip()
                            
                            # Skip results with non-English titles or obvious irrelevant content
                            if title and snippet:
                                # Clean up title and snippet for voice
                                title = title[:50]  # Limit title length
                                snippet = snippet[:100]  # Limit snippet length
                                response += f"{title}: {snippet}. "
                    return response.strip()
                    
                # Handle error cases
                elif "error" in result:
                    return f"I had trouble searching: {result['error']}"
                
            return "I couldn't find any relevant search results."
                
        elif function_name == "remember_information":
            if isinstance(result, dict) and result.get("status") == "saved":
                key = result.get("key", "that")
                return f"I've remembered {key} for you."
                
        elif function_name == "recall_information":
            if result is None:
                return "I don't have any information stored for that."
            else:
                return f"Here's what I remember: {result}"
                
        elif function_name == "calculate":
            if isinstance(result, dict):
                formatted = result.get("formatted", "")
                if formatted:
                    return f"The answer is: {formatted}"
                else:
                    return f"The result is {result.get('result', 'unknown')}"
                    
        elif function_name == "browse_url":
            if isinstance(result, dict):
                title = result.get("title", "the page")
                content = result.get("content", "")
                if content:
                    # Truncate for voice output
                    preview = content[:500].replace('\n', ' ').strip()
                    if result.get("truncated", False):
                        preview += "..."
                    return f"From {title}: {preview}"
                else:
                    return "I couldn't extract any content from that URL."
                    
        elif function_name in ["read_file", "write_file", "list_files", "search_files"]:
            # File operations - keep responses concise
            if isinstance(result, dict):
                if "error" in result:
                    return f"File operation failed: {result['error']}"
                elif function_name == "write_file" and result.get("status") == "success":
                    return f"I've successfully written to {result.get('file_path', 'the file')}."
                elif function_name == "list_files" and "files" in result:
                    files = result["files"]
                    count = len(files)
                    if count == 0:
                        return "No files found in that directory."
                    elif count <= 5:
                        file_list = ", ".join(files)
                        return f"I found {count} files: {file_list}"
                    else:
                        return f"I found {count} files in that directory."
                elif function_name == "search_files" and "results" in result:
                    matches = result["results"]
                    if not matches:
                        return "No files contain that search text."
                    else:
                        return f"I found {len(matches)} files containing that text."
        
        elif function_name == "extract_url_text":
            if isinstance(result, dict):
                # Extract the text content from URL extraction
                text_content = result.get("text", "")
                title = result.get("title", "")
                url = result.get("url", "")
                
                if text_content:
                    # Format for voice output - provide a brief intro then the content
                    intro = ""
                    if title:
                        intro = f"From {title}: "
                    elif url:
                        intro = f"From {url}: "
                    
                    # For voice output, provide substantial content (much longer limit)
                    max_length = 5000  # 5x increase - allow much more content
                    if len(text_content) > max_length:
                        truncated_text = text_content[:max_length].strip()
                        # Try to end at a sentence boundary
                        last_period = truncated_text.rfind('.')
                        if last_period > max_length * 0.7:  # If we find a period in the last 30%
                            truncated_text = truncated_text[:last_period + 1]
                        else:
                            truncated_text += " [content continues]"
                        return f"{intro}{truncated_text}"
                    else:
                        return f"{intro}{text_content}"
                else:
                    return "I extracted the content but couldn't find any readable text."
            elif isinstance(result, str):
                return result
        
        # Default formatting for any unhandled cases
        if isinstance(result, str):
            return result
        else:
            # For complex objects, try to extract key information
            return f"Operation completed successfully."
            
    except Exception as e:
        logger.error(f"Error formatting {function_name} response: {e}")
        return "I completed the operation but had trouble formatting the response."