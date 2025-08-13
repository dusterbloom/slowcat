"""
Text formatting utilities for dual-context output (voice + UI)
Optimizes text for TTS while preserving rich formatting for UI display
"""
import html
import re
from typing import Dict, Any, Optional


def sanitize_for_voice(text: str) -> str:
    """Clean text specifically for TTS output - removes voice-unfriendly formatting, emojis, and special characters"""
    if not text:
        return ""
    
    # HTML decode and remove tags
    text = html.unescape(text)
    text = re.sub(r'<[^>]*?>', '', text)
    
    # Remove emojis and emoticons (comprehensive Unicode ranges)
    # Use a more comprehensive emoji removal pattern
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001F100-\U0001F1FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF\U0000FE00-\U0000FE0F\u200D\u200C]', '', text)
    
    # Additional comprehensive emoji cleanup using simpler approach
    # Remove any remaining emoji-like characters
    text = re.sub(r'[🚗💨⚡🔥💯🎵🎤🎙️🔊🎯🎭🎪🎨🎬🎮🎲🃏🎰🎸🎺🎻🥁🎷📱💻⌨️🖥️🖨️📹📷📺📻📞☎️📟📠🔋🔌💡🔦🕯️🪔🧯🛢️💸💰💴💵💶💷💳💎⚖️🧰🔧🔨⚒️🛠️⛏️🔩⚙️🧱⛓️🧲🔫💣🧨🪓🔪⚔️🛡️🚬⚰️⚱️🏺🔮📿🧿💈⚗️🔭🔬🕳️🩹🩺💊💉🧪🧫🧬🦠💧🫧💦☔⭐🌟💫⚡☄️☀️🌤️⛅🌦️🌧️⚈🌩️🌨️❄️☃️⛄🌬️💨🌪️🌫️🌊💧🔥]', '', text)
    
    # Remove traditional emoticons
    text = re.sub(r'[:;=8][-~]?[)\]}>DPp(/\\|*+]', '', text)  # :) ;-) =D etc.
    text = re.sub(r'[)\]}>DPp(/\\|*+][-~]?[:;=8]', '', text)  # Reverse emoticons
    text = re.sub(r'<3', '', text)  # Heart
    text = re.sub(r'[¯°º][\\/_][¯°º]', '', text)  # Shrug variations
    
    # Remove special symbols that might confuse TTS
    text = re.sub(r'[★☆♪♫♯♭♮⚡⭐🔥💯]', '', text)  # Common special symbols
    text = re.sub(r'[→←↑↓↔↕]', '', text)  # Arrows
    text = re.sub(r'[™®©°±÷×§¶†‡•]', '', text)  # Trademark, copyright, etc.
    text = re.sub(r'[…‰‱]', '...', text)  # Convert ellipsis and permille to dots
    
    # Remove voice-unfriendly formatting
    text = re.sub(r'\*+', '', text)  # Remove asterisks
    text = re.sub(r'[|]{2,}', ' ', text)  # Remove pipe separators  
    text = re.sub(r'[\[\](){}]', '', text)  # Remove brackets
    text = re.sub(r'[#]+\s*', '', text)  # Remove markdown headers
    text = re.sub(r'[-_]{3,}', '', text)  # Remove separator lines
    text = re.sub(r'&[a-zA-Z0-9]+;', '', text)  # Remove HTML entities
    
    # Convert problematic characters to TTS-friendly alternatives
    text = re.sub(r'&', 'and', text)  # Ampersand to "and"
    text = re.sub(r'@', 'at', text)    # At symbol to "at"
    text = re.sub(r'#', 'number', text)  # Hash to "number"
    text = re.sub(r'\$', 'dollar', text)  # Dollar sign
    text = re.sub(r'%', 'percent', text)  # Percent sign
    text = re.sub(r'\+', 'plus', text)  # Plus sign
    
    # Remove other problematic characters
    text = re.sub(r'[`~^¨´]', '', text)  # Accent marks and tildes
    text = re.sub(r'[¡¿]', '', text)  # Inverted punctuation
    text = re.sub(r'[‚„""''‛]', '"', text)  # Convert fancy quotes to regular quotes
    text = re.sub(r'[–—]', '-', text)  # Convert em/en dashes to hyphens
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Clean up common web artifacts and search result formatting
    text = re.sub(r'Read more\.\.\.', '', text)
    text = re.sub(r'Continue reading', '', text)
    text = re.sub(r'Click here', '', text)
    text = re.sub(r'Learn more', '', text)
    
    # Remove URLs and markdown links that sound bad when spoken
    # Remove markdown links like [text](url) and keep just the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove standalone URLs (http/https)
    text = re.sub(r'https?://[^\s\]]+', '', text)
    
    # Remove common search result artifacts
    text = re.sub(r'via\s+[A-Za-z]+\s*(Library|Search)', '', text)  # "via DuckDuckGo Library"
    text = re.sub(r'Result\s+\d+:', '', text)  # "Result 1:"
    text = re.sub(r'Search results for:?', '', text)  # "Search results for:"
    
    # Clean up multiple periods from URL removal
    text = re.sub(r'\.{3,}', '...', text)
    
    # Remove domain names that might leak through
    text = re.sub(r'\b\w+\.(com|org|net|io|co|gov|edu)\b', '', text)
    
    # Final whitespace cleanup and spacing normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    # SAFE FIX: Conservative space normalization to avoid removing legitimate word boundaries
    # Focus on fixing excessive spaces while preserving normal word separation
    
    # Step 1: Normalize excessive whitespace (4+ spaces to single space)
    text = re.sub(r'    +', ' ', text)  # Replace 4+ spaces with single space
    
    # Step 2: Fix broken contractions like "can    '  t" -> "can't"
    text = re.sub(r'(\w+)\s+\'\s*(\w+)', r"\1'\2", text)  # Fix "can ' t" -> "can't"
    text = re.sub(r'(\w+)\s*\'\s+(\w+)', r"\1'\2", text)  # Fix "can' t" -> "can't"
    
    # Step 3: Clean up space around punctuation (but be conservative)
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,.!?;:])\s{2,}', r'\1 ', text)  # Normalize excessive space after punctuation
    
    # Step 4: Final conservative cleanup - only remove truly excessive spaces
    text = re.sub(r'\s{3,}', ' ', text)  # Replace 3+ spaces with single space
    
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