"""
Text formatting utilities for dual-context output (voice + UI)
Optimizes text for TTS while preserving rich formatting for UI display
"""
import html
import re
from typing import Dict, Any, Optional


def sanitize_for_voice(text: str) -> str:
    """Clean text specifically for TTS output - removes voice-unfriendly formatting"""
    if not text:
        return ""
    
    # HTML decode and remove tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]*?>', '', text)
    
    # Remove voice-unfriendly formatting
    text = re.sub(r'\*+', '', text)  # Remove asterisks
    text = re.sub(r'[|]{2,}', ' ', text)  # Remove pipe separators  
    text = re.sub(r'[\[\](){}]', '', text)  # Remove brackets
    text = re.sub(r'[#]+\s*', '', text)  # Remove markdown headers
    text = re.sub(r'[-_]{3,}', '', text)  # Remove separator lines
    text = re.sub(r'&[a-zA-Z0-9]+;', '', text)  # Remove HTML entities
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Clean up common web artifacts
    text = re.sub(r'Read more\.\.\.', '', text)
    text = re.sub(r'Continue reading', '', text)
    text = re.sub(r'Click here', '', text)
    
    return text


def create_dual_context_result(
    title: str, 
    snippet: str, 
    url: Optional[str] = None,
    max_voice_length: int = 150
) -> Dict[str, Any]:
    """Create optimized result for both voice and UI contexts"""
    
    # Clean title and snippet for voice
    voice_title = sanitize_for_voice(title)
    voice_snippet = sanitize_for_voice(snippet)
    
    # Truncate voice content for brevity
    if len(voice_snippet) > max_voice_length:
        # Try to break at sentence boundary
        truncate_pos = voice_snippet.rfind('.', 0, max_voice_length)
        if truncate_pos > max_voice_length // 2:
            voice_snippet = voice_snippet[:truncate_pos + 1]
        else:
            voice_snippet = voice_snippet[:max_voice_length] + "..."
    
    return {
        # For TTS - clean, concise text
        "voice_title": voice_title,
        "voice_snippet": voice_snippet,
        "voice_text": f"{voice_title}. {voice_snippet}",
        
        # For UI - preserve original formatting
        "display_title": title,
        "display_snippet": snippet, 
        "url": url,
        
        # Backward compatibility
        "title": voice_title,  # Default to voice-optimized
        "snippet": voice_snippet
    }


def format_search_results_for_voice(results: list) -> str:
    """Format multiple search results as natural speech"""
    if not results:
        return "No search results found."
    
    # Create natural flowing speech
    if len(results) == 1:
        result = results[0]
        return f"I found this: {result.get('voice_text', result.get('title', ''))}."
    
    # Multiple results - create flowing summary
    voice_parts = []
    for i, result in enumerate(results[:3], 1):  # Limit to 3 for voice
        voice_text = result.get('voice_text', result.get('title', ''))
        if voice_text:
            voice_parts.append(f"Result {i}: {voice_text}")
    
    if len(results) > 3:
        voice_parts.append(f"And {len(results) - 3} more results.")
    
    return ". ".join(voice_parts) + "."


def create_search_response(
    query: str, 
    results: list,
    include_urls: bool = True
) -> Dict[str, Any]:
    """Create comprehensive search response for both voice and UI"""
    
    # Process each result for dual context
    processed_results = []
    for result in results:
        processed = create_dual_context_result(
            title=result.get('title', ''),
            snippet=result.get('snippet', ''),
            url=result.get('url', result.get('href'))
        )
        processed_results.append(processed)
    
    # Create voice-optimized summary
    voice_response = format_search_results_for_voice(processed_results)
    
    # Create UI-optimized formatted text with clickable links
    ui_response = format_search_results_for_ui(processed_results)
    
    return {
        "query": query,
        "results": processed_results,
        "voice_summary": voice_response,
        "ui_formatted": ui_response,  # NEW: Clean UI formatting
        "result_count": len(results)
    }


def format_search_results_for_ui(results: list) -> str:
    """Format search results with clean UI presentation and clickable links"""
    if not results:
        return "No search results found."
    
    formatted_items = []
    for i, result in enumerate(results[:5], 1):  # Limit to 5 for UI
        title = result.get('display_title', result.get('title', 'Untitled'))
        snippet = result.get('display_snippet', result.get('snippet', ''))
        url = result.get('url', '')
        
        # Create clean, compact UI format
        item_parts = [f"**{i}. {title}**"]
        
        if snippet:
            # Truncate snippet for UI readability
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            item_parts.append(f"*{snippet}*")
        
        if url:
            # Create clickable HTML link
            item_parts.append(f'<a href="{url}" target="_blank" rel="noopener">Visit website</a>')
        
        formatted_items.append(" - ".join(item_parts))
    
    return "\n\n".join(formatted_items)