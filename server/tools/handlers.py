"""
Tool handlers for LM Studio function calling
Implements the actual functionality for each tool
"""

import json
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from loguru import logger
import os
from pathlib import Path
import math
import re
from bs4 import BeautifulSoup
import html2text
from datetime import datetime
import pytz
from file_tools import file_tools

class ToolHandlers:
    """Handles tool function execution for Slowcat"""
    
    def __init__(self, memory_dir: str = "data/tool_memory", memory_processor=None):
        """Initialize tool handlers with memory storage"""
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "tool_memory.json"
        self._load_memory()
        self.memory_processor = memory_processor
    
    def _load_memory(self):
        """Load persisted memory from file"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    self.memory = json.load(f)
            except Exception as e:
                logger.error(f"Error loading tool memory: {e}")
                self.memory = {}
        else:
            self.memory = {}
    
    def _save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tool memory: {e}")
    
    async def store_memory(self, key: str, value: str) -> Dict[str, Any]:
        """
        Store information in MCP-compatible memory
        Compatible with LM Studio's MCP memory server JSONL format
        REPLACES existing value (doesn't append)
        
        Args:
            key: The key to store the memory under
            value: The value to store (replaces any existing value)
            
        Returns:
            Success status
        """
        try:
            # Use MCP memory.json path for compatibility
            mcp_memory_file = Path("/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json")
            
            # Ensure directory exists
            mcp_memory_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing memories and deduplicate
            memories = []
            seen_keys = set()
            if mcp_memory_file.exists():
                with open(mcp_memory_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                memory = json.loads(line)
                                # Deduplicate by normalized key
                                norm_key = memory.get("name", "").lower().strip()
                                if norm_key and norm_key not in seen_keys:
                                    memories.append(memory)
                                    seen_keys.add(norm_key)
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON line: {line}")
            
            # Normalize the key for comparison
            key_normalized = key.lower().strip()
            
            # Check if this key already exists (case-insensitive)
            found = False
            for memory in memories:
                if memory.get("name", "").lower().strip() == key_normalized:
                    # REPLACE the observations with new value (not append)
                    memory["observations"] = [value]
                    found = True
                    break
            
            # If not found, create new entry
            if not found:
                memory_entry = {
                    "type": "entity",
                    "name": key,  # Keep original case for display
                    "entityType": "TEXT",
                    "observations": [value]
                }
                memories.append(memory_entry)
            
            # Write all memories back to file
            with open(mcp_memory_file, 'w') as f:
                for memory in memories:
                    f.write(json.dumps(memory) + '\n')
            
            logger.info(f"Stored memory: {key} = {value}")
            return {"success": True, "message": f"Stored '{key}' in memory"}
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return {"error": str(e)}
    
    async def retrieve_memory(self, key: str) -> Dict[str, Any]:
        """
        Retrieve information from MCP-compatible memory (JSONL format)
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value or error message
        """
        try:
            mcp_memory_file = Path("/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json")
            
            if not mcp_memory_file.exists():
                return {"error": "No memories stored yet"}
            
            # Read JSONL file
            memories = []
            with open(mcp_memory_file, 'r') as f:
                for line in f:
                    if line.strip():
                        memories.append(json.loads(line))
            
            # Find the memory with matching name/key
            for memory in memories:
                if memory.get("name") == key:
                    observations = memory.get("observations", [])
                    value = observations[-1] if observations else None  # Get latest observation
                    if value:
                        logger.info(f"Retrieved memory: {key} = {value}")
                        return {"success": True, "key": key, "value": value}
            
            return {"error": f"No memory found for key '{key}'"}
                
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return {"error": str(e)}
    
    async def search_memory(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search for information in memory (JSONL format)
        
        Args:
            query: Search query to find relevant memories
            max_results: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        try:
            mcp_memory_file = Path("/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json")
            
            if not mcp_memory_file.exists():
                return {"results": [], "message": "No memories stored yet"}
            
            # Read JSONL file
            memories = []
            with open(mcp_memory_file, 'r') as f:
                for line in f:
                    if line.strip():
                        memories.append(json.loads(line))
            
            # Search through memories
            query_lower = query.lower()
            results = []
            
            for memory in memories:
                name = memory.get("name", "")
                observations = memory.get("observations", [])
                
                # Check if query matches name or any observation
                if query_lower in name.lower():
                    for obs in observations:
                        if obs:
                            results.append({"key": name, "value": obs})
                            if len(results) >= max_results:
                                break
                else:
                    for obs in observations:
                        if obs and query_lower in str(obs).lower():
                            results.append({"key": name, "value": obs})
                            if len(results) >= max_results:
                                break
                
                if len(results) >= max_results:
                    break
            
            logger.info(f"Search found {len(results)} results for query: {query}")
            return {"success": True, "results": results, "count": len(results)}
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {"error": str(e)}
    
    async def update_memory(self, key: str, value: str) -> Dict[str, Any]:
        """
        Update/replace existing memory value
        Same as store_memory but explicitly for updates
        
        Args:
            key: The key to update
            value: The new value to replace the old one
            
        Returns:
            Success status
        """
        # Just call store_memory since it now replaces values
        return await self.store_memory(key, value)
    
    async def delete_memory(self, key: str) -> Dict[str, Any]:
        """
        Delete information from memory (JSONL format)
        
        Args:
            key: The key to delete
            
        Returns:
            Success status
        """
        try:
            mcp_memory_file = Path("/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json")
            
            if not mcp_memory_file.exists():
                return {"error": "No memories stored yet"}
            
            # Read all memories
            memories = []
            with open(mcp_memory_file, 'r') as f:
                for line in f:
                    if line.strip():
                        memories.append(json.loads(line))
            
            # Filter out the memory to delete
            updated_memories = [m for m in memories if m.get("name") != key]
            
            if len(updated_memories) == len(memories):
                return {"error": f"No memory found for key '{key}'"}
            
            # Write back the filtered memories
            with open(mcp_memory_file, 'w') as f:
                for memory in updated_memories:
                    f.write(json.dumps(memory) + '\n')
            
            logger.info(f"Deleted memory: {key}")
            return {"success": True, "message": f"Deleted memory for key '{key}'"}
                
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"error": str(e)}
    
    async def get_weather(self, location: str, units: str = "celsius") -> Dict[str, Any]:
        """
        Get weather for a location using Open-Meteo API (free, no key required)
        
        Args:
            location: City name or coordinates
            units: Temperature units (celsius/fahrenheit)
            
        Returns:
            Weather data dict
        """
        try:
            logger.info(f"Getting weather for {location} in {units}")
            
            # First, geocode the location using Open-Meteo's geocoding API
            # Use timeout for all requests
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Geocoding API
                geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
                async with session.get(geocode_url) as resp:
                    if resp.status != 200:
                        return {"error": f"Failed to geocode location: {location}"}
                    
                    geo_data = await resp.json()
                    if not geo_data.get("results"):
                        return {"error": f"Location not found: {location}"}
                    
                    # Get first result
                    result = geo_data["results"][0]
                    lat = result["latitude"]
                    lon = result["longitude"]
                    city_name = result.get("name", location)
                    country = result.get("country", "")
                    
                # Weather API
                temp_unit = "fahrenheit" if units == "fahrenheit" else "celsius"
                weather_url = (
                    f"https://api.open-meteo.com/v1/forecast?"
                    f"latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,weather_code,wind_speed_10m"
                    f"&temperature_unit={temp_unit}"
                )
                
                async with session.get(weather_url) as resp:
                    if resp.status != 200:
                        return {"error": "Failed to get weather data"}
                    
                    weather_data = await resp.json()
                    current = weather_data.get("current", {})
                    
                    # Map weather codes to conditions
                    weather_code = current.get("weather_code", 0)
                    conditions = self._get_weather_condition(weather_code)
                    
                    return {
                        "location": f"{city_name}, {country}" if country else city_name,
                        "temperature": round(current.get("temperature_2m", 0)),
                        "conditions": conditions,
                        "wind_speed": round(current.get("wind_speed_10m", 0)),
                        "units": units
                    }
            
        except Exception as e:
            logger.error(f"Error getting weather: {e}")
            return {"error": str(e)}
    
    def _get_weather_condition(self, code: int) -> str:
        """Convert Open-Meteo weather code to human-readable condition"""
        # Based on WMO Weather interpretation codes
        weather_codes = {
            0: "clear sky",
            1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 48: "foggy",
            51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
            61: "light rain", 63: "moderate rain", 65: "heavy rain",
            71: "light snow", 73: "moderate snow", 75: "heavy snow",
            80: "light showers", 81: "moderate showers", 82: "heavy showers",
            95: "thunderstorm", 96: "thunderstorm with hail"
        }
        return weather_codes.get(code, "unknown conditions")
    
    async def get_current_time(self, format: str = "human", timezone: str = "UTC") -> Dict[str, Any]:
        """
        Get the current date and time in various formats
        
        Args:
            format: Time format - "ISO", "human", "unix", "date_only", "time_only"
            timezone: Timezone name (e.g., 'UTC', 'America/New_York', 'Europe/London')
            
        Returns:
            Dict with time information
        """
        try:
            # Get timezone
            try:
                tz = pytz.timezone(timezone)
            except pytz.exceptions.UnknownTimeZoneError:
                logger.warning(f"Unknown timezone {timezone}, using UTC")
                tz = pytz.UTC
                
            # Get current time in specified timezone
            now = datetime.now(tz)
            
            # Format based on requested format
            if format == "ISO":
                time_str = now.isoformat()
            elif format == "unix":
                time_str = str(int(now.timestamp()))
            elif format == "date_only":
                time_str = now.strftime("%Y-%m-%d")
            elif format == "time_only":
                time_str = now.strftime("%H:%M:%S")
            else:  # human format (default)
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
    
    async def search_web(self, query: str, num_results: int = 3) -> list:
        """
        Search the web using Brave Search API or fallback to DuckDuckGo
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            logger.info(f"Searching web for: {query}")
            
            # Check if Brave Search API key is available
            from config import config
            brave_key = config.mcp.brave_search_api_key
            logger.info(f"Brave API key present: {bool(brave_key)}")
            if brave_key:
                logger.info(f"Using Brave Search (key length: {len(brave_key)}, first chars: {brave_key[:8]}...)")
                return await self._search_brave(query, num_results)
            else:
                logger.info("No Brave API key, falling back to DuckDuckGo")
                return await self._search_duckduckgo(query, num_results)
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return [{"title": "Search Error", "snippet": str(e), "source": "Error"}]
    
    async def _search_brave(self, query: str, num_results: int) -> list:
        """Search using Brave Search API"""
        try:
            from config import config
            api_key = config.mcp.brave_search_api_key
            if not api_key:
                logger.error("Brave API key is empty!")
                return await self._search_duckduckgo(query, num_results)
                
            logger.debug(f"Brave Search: query='{query}', num_results={num_results}")
            logger.debug(f"API Key length: {len(api_key)}, starts with: {api_key[:10]}...")
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = "https://api.search.brave.com/res/v1/web/search"
                headers = {
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json"
                }
                params = {
                    "q": query,
                    "count": num_results
                }
                
                logger.debug(f"Calling Brave API: {url}")
                async with session.get(url, headers=headers, params=params) as resp:
                    logger.debug(f"Brave API response status: {resp.status}")
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Brave Search API error {resp.status}: {error_text}")
                        # Fallback to DuckDuckGo
                        return await self._search_duckduckgo(query, num_results)
                    
                    data = await resp.json()
                    logger.debug(f"Brave API returned {len(data.get('web', {}).get('results', []))} results")
                    results = []
                    
                    # Extract web results
                    for result in data.get("web", {}).get("results", [])[:num_results]:
                        title = result.get("title", "")
                        snippet = result.get("description", "")
                        
                        # Clean up HTML from snippets
                        if snippet:
                            # Remove HTML tags
                            snippet = re.sub(r'<[^>]+>', '', snippet)
                            # Limit length for voice
                            if len(snippet) > 150:
                                snippet = snippet[:150] + "..."
                        
                        # Add all results regardless of language
                        if title and snippet:
                            results.append({
                                "title": title,
                                "snippet": snippet
                            })
                    
                    # Add answer box if available
                    if data.get("summarizer") and not results:
                        results.append({
                            "title": "Quick Answer",
                            "snippet": data["summarizer"].get("summary", ""),
                            "source": "Brave AI Summary"
                        })
                    
                    return results if results else [{"title": "No results found", "snippet": f"No results for: {query}", "source": "Brave Search"}]
                    
        except Exception as e:
            logger.error(f"Brave Search error: {e}")
            # Fallback to DuckDuckGo
            return await self._search_duckduckgo(query, num_results)
    
    async def _search_duckduckgo(self, query: str, num_results: int) -> list:
        """Original DuckDuckGo search implementation"""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # DuckDuckGo API
                url = "https://api.duckduckgo.com/"
                params = {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1"
                }
                
                async with session.get(url, params=params) as resp:
                    # DuckDuckGo API can return 202 or 200
                    if resp.status not in [200, 202]:
                        logger.error(f"Search API returned status {resp.status}")
                        return []
                    
                    # Get response text first
                    text = await resp.text()
                    if not text:
                        return []
                    
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse search response")
                        return []
                    
                    results = []
                    
                    # Abstract (summary)
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", "Summary"),
                            "snippet": data["Abstract"][:200] + "...",
                            "source": data.get("AbstractSource", "DuckDuckGo")
                        })
                    
                    # Definition
                    if data.get("Definition") and len(results) < num_results:
                        results.append({
                            "title": "Definition",
                            "snippet": data["Definition"][:200] + "...",
                            "source": data.get("DefinitionSource", "Dictionary")
                        })
                    
                    # Answer (for factual queries)
                    if data.get("Answer") and len(results) < num_results:
                        results.append({
                            "title": "Quick Answer",
                            "snippet": data["Answer"],
                            "source": data.get("AnswerType", "Instant Answer")
                        })
                    
                    # Related topics
                    for topic in data.get("RelatedTopics", [])[:num_results - len(results)]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("Text", "").split(" - ")[0][:50],
                                "snippet": topic.get("Text", "")[:200] + "...",
                                "source": "Related Topic"
                            })
                    
                    # If no structured results, return a generic response
                    if not results:
                        results.append({
                            "title": f"Search results for: {query}",
                            "snippet": "For more detailed results, try searching on a web browser.",
                            "source": "Limited API"
                        })
                    
                    return results[:num_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return [{"title": "Search Error", "snippet": str(e), "source": "Error"}]
    
    async def remember_information(self, key: str, value: str) -> Dict[str, str]:
        """
        Store information in persistent memory
        
        Args:
            key: Key to store under
            value: Information to store
            
        Returns:
            Confirmation dict
        """
        try:
            logger.info(f"Remembering: {key} = {value}")
            self.memory[key] = value
            self._save_memory()
            return {"status": "saved", "key": key}
            
        except Exception as e:
            logger.error(f"Error remembering information: {e}")
            return {"status": "error", "message": str(e)}
    
    async def recall_information(self, key: str) -> Optional[str]:
        """
        Retrieve information from memory
        
        Args:
            key: Key to retrieve
            
        Returns:
            Stored value or None
        """
        try:
            logger.info(f"Recalling: {key}")
            return self.memory.get(key)
            
        except Exception as e:
            logger.error(f"Error recalling information: {e}")
            return None
    
    async def calculate(self, expression: str) -> Dict[str, Any]:
        """
        Perform safe mathematical calculations
        
        Args:
            expression: Math expression to evaluate
            
        Returns:
            Calculation result
        """
        try:
            logger.info(f"Calculating: {expression}")
            
            # Clean the expression
            expression = expression.strip()
            
            # Define safe functions
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sqrt': math.sqrt, 'pow': pow, 'sum': sum,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'pi': math.pi, 'e': math.e
            }
            
            # Basic safety check - only allow numbers, operators, and safe functions
            if re.search(r'[^0-9+\-*/().\s\w]', expression):
                return {"error": "Invalid characters in expression"}
            
            # Evaluate the expression safely
            try:
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                return {
                    "expression": expression,
                    "result": result,
                    "formatted": f"{expression} = {result}"
                }
            except Exception as e:
                return {"error": f"Calculation error: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Error in calculate: {e}")
            return {"error": str(e)}
    
    async def browse_url(self, url: str, max_length: int = 2000) -> Dict[str, Any]:
        """
        Fetch and extract text content from a URL
        
        Args:
            url: URL to fetch
            max_length: Maximum characters to return
            
        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Browsing URL: {url}")
            
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Use timeout for all requests
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Set headers to appear as a browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status != 200:
                        return {"error": f"Failed to fetch URL: HTTP {resp.status}"}
                    
                    # Get content type
                    content_type = resp.headers.get('Content-Type', '')
                    
                    # Only process HTML/text content
                    if 'text' not in content_type and 'html' not in content_type:
                        return {"error": f"Unsupported content type: {content_type}"}
                    
                    # Read the content
                    html_content = await resp.text()
                    
                    # Parse HTML and extract text
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Remove script and style elements
                    for element in soup(['script', 'style', 'meta', 'link']):
                        element.decompose()
                    
                    # Get page title
                    title = soup.find('title')
                    title_text = title.text.strip() if title else "No title"
                    
                    # Convert to markdown for better structure
                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    h.ignore_images = True
                    h.skip_internal_links = True
                    h.inline_links = True
                    h.wrap_links = False
                    h.body_width = 0  # Don't wrap lines
                    
                    # Get text content
                    text_content = h.handle(str(soup))
                    
                    # Clean up excessive whitespace
                    text_content = re.sub(r'\n{3,}', '\n\n', text_content)
                    text_content = text_content.strip()
                    
                    # Truncate if needed
                    if len(text_content) > max_length:
                        text_content = text_content[:max_length] + "..."
                    
                    return {
                        "url": url,
                        "title": title_text,
                        "content": text_content,
                        "length": len(text_content),
                        "truncated": len(text_content) > max_length
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching URL: {url}")
            return {"error": "Request timed out"}
        except Exception as e:
            logger.error(f"Error browsing URL: {e}")
            return {"error": str(e)}
    
    async def search_conversations(self, query: str, limit: int = 10, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Search through past conversation history
        
        Args:
            query: Text to search for
            limit: Maximum results to return
            user_id: Optional user filter
            
        Returns:
            Search results or error
        """
        try:
            logger.info(f"Searching conversations for: {query}")
            
            if not self.memory_processor:
                return {
                    "error": "Memory is not enabled",
                    "results": []
                }
            
            # Call the memory processor's search method
            results = await self.memory_processor.search_conversations(query, limit, user_id)
            
            # Format results for voice output
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "role": result["role"],
                    "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "timestamp": result["timestamp"]
                })
            
            return {
                "query": query,
                "results_count": len(results),
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return {"error": str(e), "results": []}
    
    async def get_conversation_summary(self, days_back: int = 7, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of conversations
        
        Args:
            days_back: Number of days to look back (0 for all time)
            user_id: Optional user filter
            
        Returns:
            Conversation summary or error
        """
        try:
            logger.info(f"Getting conversation summary for {days_back} days")
            
            if not self.memory_processor:
                return {
                    "error": "Memory is not enabled",
                    "total_messages": 0
                }
            
            # Call the memory processor's summary method
            summary = await self.memory_processor.get_conversation_summary(days_back, user_id)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {"error": str(e), "total_messages": 0}

# Global instance
tool_handlers = ToolHandlers()

def set_memory_processor(memory_processor):
    """Set the memory processor for tool handlers"""
    global tool_handlers
    tool_handlers.memory_processor = memory_processor
    logger.info("Memory processor set for tool handlers")

async def execute_tool_call(function_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Execute a tool call by name with arguments
    
    Args:
        function_name: Name of the function to call
        arguments: Arguments for the function
        
    Returns:
        Tool execution result
    """
    logger.info(f"Executing tool: {function_name} with args: {arguments}")
    
    # Map function names to handlers
    if function_name == "get_weather":
        return await tool_handlers.get_weather(**arguments)
    elif function_name == "get_current_time":
        return await tool_handlers.get_current_time(**arguments)
    elif function_name == "search_web":
        return await tool_handlers.search_web(**arguments)
    elif function_name == "store_memory":
        return await tool_handlers.store_memory(**arguments)
    elif function_name == "retrieve_memory":
        return await tool_handlers.retrieve_memory(**arguments)
    elif function_name == "search_memory":
        return await tool_handlers.search_memory(**arguments)
    elif function_name == "update_memory":
        return await tool_handlers.update_memory(**arguments)
    elif function_name == "delete_memory":
        return await tool_handlers.delete_memory(**arguments)
    elif function_name == "remember_information":
        return await tool_handlers.remember_information(**arguments)
    elif function_name == "recall_information":
        return await tool_handlers.recall_information(**arguments)
    elif function_name == "calculate":
        return await tool_handlers.calculate(**arguments)
    elif function_name == "browse_url":
        return await tool_handlers.browse_url(**arguments)
    elif function_name == "read_file":
        return await file_tools.read_file(**arguments)
    elif function_name == "search_files":
        return await file_tools.search_files(**arguments)
    elif function_name == "list_files":
        return await file_tools.list_files(**arguments)
    elif function_name == "write_file":
        return await file_tools.write_file(**arguments)
    elif function_name == "search_conversations":
        return await tool_handlers.search_conversations(**arguments)
    elif function_name == "get_conversation_summary":
        return await tool_handlers.get_conversation_summary(**arguments)
    elif function_name == "start_timed_task":
        from .time_tools import start_timed_task
        return await start_timed_task(**arguments)
    elif function_name == "check_task_status":
        from .time_tools import check_task_status
        return await check_task_status(**arguments)
    elif function_name == "stop_timed_task":
        from .time_tools import stop_timed_task
        return await stop_timed_task(**arguments)
    elif function_name == "add_to_timed_task":
        from .time_tools import add_to_timed_task
        return await add_to_timed_task(**arguments)
    elif function_name == "get_active_tasks":
        from .time_tools import get_active_tasks
        return await get_active_tasks(**arguments)
    elif function_name == "play_music":
        from .music_tools import play_music
        return await play_music(**arguments)
    elif function_name == "pause_music":
        from .music_tools import pause_music
        return await pause_music(**arguments)
    elif function_name == "skip_song":
        from .music_tools import skip_song
        return await skip_song(**arguments)
    elif function_name == "stop_music":
        from .music_tools import stop_music
        return await stop_music(**arguments)
    elif function_name == "queue_music":
        from .music_tools import queue_music
        return await queue_music(**arguments)
    elif function_name == "search_music":
        from .music_tools import search_music
        return await search_music(**arguments)
    elif function_name == "get_now_playing":
        from .music_tools import get_now_playing
        return await get_now_playing(**arguments)
    elif function_name == "set_volume":
        from .music_tools import set_volume
        return await set_volume(**arguments)
    elif function_name == "create_playlist":
        from .music_tools import create_playlist
        return await create_playlist(**arguments)
    elif function_name == "get_music_stats":
        from .music_tools import get_music_stats
        return await get_music_stats(**arguments)
    # Handle LM Studio MCP memory tools (these are stubs since LM Studio handles them)
    elif function_name in ["store_memory", "retrieve_memory", "search_memory", "delete_memory"]:
        logger.info(f"MCP memory tool called: {function_name}")
        # Return a message indicating these are handled by LM Studio
        return {
            "status": "info",
            "message": f"Memory tool '{function_name}' is handled by LM Studio MCP. The memory operation has been noted.",
            "note": "Memory persistence works through LM Studio's MCP server."
        }
    else:
        logger.error(f"Unknown tool function: {function_name}")
        return {"error": f"Unknown function: {function_name}"}