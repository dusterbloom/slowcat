# Free Web Search Implementation Guide

## Overview
This guide explains how web search works in Slowcat without requiring any API keys. The implementation uses multiple fallback providers to ensure reliability.

## The Problem
- The original implementation removed the `search_web` tool in favor of MCP's `brave_web_search`
- Brave Search requires an API key which costs money
- Without web search, the agent incorrectly searches music when asked for web information

## The Solution
We've implemented a **free web search tool** (`search_web_free`) that works without any API keys using multiple providers:

### 1. Primary Provider: DuckDuckGo Library
- Uses the `duckduckgo-search` Python package
- Most reliable and fastest option
- Returns clean, structured results
- No rate limiting for reasonable usage

### 2. Fallback Providers
- **DuckDuckGo HTML Scraping**: Direct HTML parsing
- **DuckDuckGo Lite**: Mobile interface scraping  
- **SearXNG Public Instances**: Privacy-focused metasearch
- **Google Cache**: Basic HTML version (last resort)

## Installation

### Step 1: Install the Library
```bash
cd server
source .venv/bin/activate  # or venv/bin/activate
pip install duckduckgo-search
```

### Step 2: Files Added/Modified
- `server/tools/web_search_free.py` - Free search implementation
- `server/tools/definitions.py` - Added `SEARCH_WEB_FREE` tool definition
- `server/tools/handlers.py` - Added handler for `search_web_free`
- `server/services/llm_with_tools.py` - Added to local tools list

## Usage

### In Voice Commands
Just ask naturally:
- "Search the web for Python tutorials"
- "What's the latest AI news?"
- "Find information about sourdough bread"

### In Code
```python
from tools.web_search_free import search_web_free

# Search with default 5 results
results = await search_web_free("Python programming")

# Search with custom number of results
results = await search_web_free("AI news", num_results=10)
```

### Response Format
```python
{
    "results": [...],  # Array of search results
    "ui_formatted": "...",  # Markdown formatted for UI display
    "voice_summary": "...",  # Clean text for TTS
    "result_count": 5,  # Number of results
    "provider": "DuckDuckGo Library"  # Which provider succeeded
}
```

## Alternative: MCP Servers (No API Keys)

If you prefer using MCP servers instead of the local implementation, here are free options:

### 1. DuckDuckGo MCP Server
```bash
npx @nickclyde/duckduckgo-mcp-server
```

### 2. Open WebSearch MCP
```bash
npx open-websearch@latest
```

### 3. One Search MCP (Multiple Engines)
```bash
npm install -g one-search-mcp
```

Add to your `mcp.json`:
```json
{
  "mcpServers": {
    "websearch": {
      "command": "npx",
      "args": ["@nickclyde/duckduckgo-mcp-server"]
    }
  }
}
```

## Troubleshooting

### Issue: "All search providers failed"
- Check your internet connection
- The providers might be rate limiting - wait a few seconds
- Try updating the duckduckgo-search library: `pip install -U duckduckgo-search`

### Issue: Agent searches music instead of web
- Ensure `search_web_free` is in `ALL_FUNCTION_SCHEMAS` in definitions.py
- Check that the tool is registered in handlers.py
- Verify it's listed in local_tools in llm_with_tools.py

### Issue: Results are not relevant
- Be more specific with your search query
- The free providers may have less sophisticated ranking than Google/Bing
- Consider using the `extract_url_text` tool to get full content from specific pages

## Performance Considerations
- **Speed**: DuckDuckGo library is fastest (~1-2 seconds)
- **Reliability**: Library method most reliable, HTML scraping can be blocked
- **Quality**: Results are good but may differ from Google/Bing
- **Rate Limits**: Be respectful - don't make hundreds of searches per minute

## Privacy Benefits
Unlike API-based solutions, this implementation:
- Doesn't require registration or API keys
- Doesn't track your searches (DuckDuckGo privacy)
- Can use SearXNG for additional privacy
- Keeps all searches local to your machine

## Future Enhancements
- Add more search providers (Qwant, Startpage, etc.)
- Implement result caching for repeated queries
- Add image and video search capabilities
- Support for advanced search operators
- Integration with local knowledge base