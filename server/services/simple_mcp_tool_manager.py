u"""
Dynamic MCP Tool Manager with auto-discovery and hot-plug support
Implements your friend's brilliant dynamic polling architecture!
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
import subprocess
import json
import time
import os
import threading
import aiohttp
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field

# 👑 Import the KING'S Pure Algorithmic Negotiator - ZERO PEASANT RULES!
from core.pure_algorithmic_negotiator import PureAlgorithmicNegotiator

from config import config

# Cache directory for tool manifests
CACHE_DIR = Path("./data/mcp_schemas")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MCPToolInfo:
    """Lightweight tool information"""
    server_name: str
    tool_name: str
    full_name: str
    description: str
    path: str
    
    
@dataclass  
class SimpleMCPToolManager:
    """
    Dynamic MCP Tool Manager with auto-discovery and hot-plug support
    Implements TTL-based polling, disk caching, and zero-maintenance tool registry
    """
    language: str = "en"
    ttl_seconds: int = 60  # 60s refresh interval for development
    
    # Internal state
    _manifest: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {tool_name: schema}
    _last_refresh: float = field(default_factory=lambda: 0)
    _mcp_servers: Dict[str, str] = field(default_factory=dict)  # {server_name: url}
    _translations: Optional[Dict[str, Dict[str, str]]] = None

    def __post_init__(self):
        """Initialize MCP servers from mcp.json configuration"""
        # Load MCP server configurations from mcp.json
        self._mcp_servers = self._load_mcp_config()
        
        # HTTP connection pool for ultra-low latency
        self._http_session = None
        
        # 👑 Initialize the KING'S Pure Algorithmic Negotiator - ZERO MANUAL RULES!
        self.negotiator = PureAlgorithmicNegotiator()
        
        # Load cached manifest on startup for instant first call
        self._load_cached_manifest()
        
        logger.info(f"🔄 Dynamic MCP Tool Manager initialized")
        logger.info(f"   TTL: {self.ttl_seconds}s")
        logger.info(f"   Cache dir: {CACHE_DIR}")
        logger.info(f"   Configured servers: {list(self._mcp_servers.keys())}")
        logger.info(f"   HTTP pooling: enabled")
    
    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create persistent HTTP session with connection pooling"""
        # CRITICAL FIX: Always create fresh session in current event loop
        # This prevents "Event loop is closed" errors when crossing thread boundaries
        try:
            # Check if existing session is usable in current event loop
            if (self._http_session is not None and 
                not self._http_session.closed and
                self._http_session._loop == asyncio.get_running_loop()):
                return self._http_session
        except RuntimeError:
            # No running event loop or session from different loop
            pass
        
        # Create new session in current event loop
        connector = aiohttp.TCPConnector(
            limit=20,              # Total connection pool size
            limit_per_host=10,     # Max connections per host (MCPO)
            ttl_dns_cache=300,     # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=60,  # Keep connections alive 60s
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30.0,      # Total timeout
            connect=5.0,     # Connection timeout  
            sock_read=10.0   # Socket read timeout
        )
        
        self._http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Slowcat-MCP-Client/1.0'}
        )
        logger.debug(f"🔗 Created new HTTP session in current event loop")
        
        return self._http_session
    
    async def _close_session(self):
        """Close HTTP session gracefully"""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            logger.debug(f"🔒 HTTP session closed")
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_http_session') and self._http_session and not self._http_session.closed:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.create_task(self._close_session())
            except Exception:
                pass  # Best effort cleanup
    
    def _load_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Load MCP server configurations from mcp.json"""
        mcp_config_path = Path("./mcp.json")
        
        if not mcp_config_path.exists():
            logger.warning("❌ mcp.json not found, using empty configuration")
            return {}
        
        try:
            with open(mcp_config_path, 'r') as f:
                config = json.load(f)
            
            servers = {}
            mcp_servers = config.get("mcpServers", {})
            
            for server_name, server_config in mcp_servers.items():
                servers[server_name] = {
                    "command": server_config.get("command"),
                    "args": server_config.get("args", []),
                    "env": server_config.get("env", {})
                }
                logger.info(f"   📋 Loaded {server_name}: {server_config.get('command')} {' '.join(server_config.get('args', []))}")
            
            return servers
            
        except Exception as e:
            logger.error(f"❌ Failed to load mcp.json: {e}")
            return {}
    
    def _validate_environment(self) -> List[str]:
        """
        Validate required environment variables for MCP servers
        Returns list of missing environment variables
        """
        import os
        missing = []
        
        # Check for required environment variables based on mcp.json
        required_env = {
            "BRAVE_API_KEY": "brave-search server",  # Use BRAVE_API_KEY to match mcp.json
            # Add other env vars as needed
        }
        
        for env_var, description in required_env.items():
            if not os.getenv(env_var):
                missing.append(f"{env_var} (for {description})")
        
        return missing
    
    def _load_cached_manifest(self):
        """Load cached manifest from disk for instant startup"""
        cache_file = CACHE_DIR / "manifest.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self._manifest = json.load(f)
                logger.info(f"📁 Loaded {len(self._manifest)} tools from cache")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load cached manifest: {e}")
                self._manifest = {}
        else:
            # Bootstrap with static tools for first run
            self._manifest = self._get_static_fallback_tools()
            logger.info(f"🔧 Bootstrapped with {len(self._manifest)} static tools")
    
    def _save_manifest_to_cache(self):
        """Persist manifest to disk cache"""
        try:
            cache_file = CACHE_DIR / "manifest.json" 
            with open(cache_file, 'w') as f:
                json.dump(self._manifest, f, indent=2)
            logger.debug(f"💾 Saved manifest to cache ({len(self._manifest)} tools)")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save manifest cache: {e}")
    
    async def refresh_if_stale(self):
        """Auto-refresh manifest if TTL expired"""
        if time.time() - self._last_refresh < self.ttl_seconds:
            return  # Still fresh
        
        logger.info("🔄 Refreshing MCP tool manifest (TTL expired)")
        await self._refresh_manifest()
    
    async def _refresh_manifest(self):
        """Refresh tool manifest using MCPO HTTP endpoints"""
        import aiohttp
        new_manifest = {}
        
        # Auto-generate MCPO endpoints from mcp.json (truly dynamic!)
        mcpo_endpoints = {}
        for server_name in self._mcp_servers.keys():
            # Skip servers that failed to start (like javascript)
            mcpo_endpoints[server_name] = f"http://localhost:3001/{server_name}"
        
        logger.info(f"🔍 Auto-discovered MCPO endpoints: {list(mcpo_endpoints.keys())}")
        
        # Probe each MCPO HTTP endpoint
        for server_name, endpoint in mcpo_endpoints.items():
            try:
                tools = await self._probe_mcpo_server(server_name, endpoint)
                for tool in tools:
                    tool["mcp_server"] = server_name
                    tool["mcpo_endpoint"] = endpoint
                    new_manifest[tool["name"]] = tool
                    
                logger.info(f"✅ {server_name}: discovered {len(tools)} tools via MCPO")
                
            except Exception as e:
                logger.warning(f"⚠️ {server_name} MCPO probe failed: {e}")
        
        # If no tools discovered, use static fallback
        if not new_manifest:
            logger.warning("🔄 No tools discovered via MCPO, using static fallback")
            new_manifest = self._get_static_fallback_tools()
        
        # Update manifest and cache
        self._manifest = new_manifest
        self._last_refresh = time.time()
        self._save_manifest_to_cache()
        
        logger.info(f"🎯 Manifest refreshed via MCPO: {len(self._manifest)} total tools available")
    
    async def _probe_mcpo_server(self, server_name: str, endpoint: str) -> List[Dict[str, Any]]:
        """Probe MCPO HTTP server to get tool manifest from OpenAPI spec"""
        import aiohttp
        
        try:
            logger.info(f"🔍 Probing {server_name} via MCPO: {endpoint}/openapi.json")
            
            headers = {
                "Authorization": "Bearer slowcat-secret",
                "Accept": "application/json"
            }
            
            session = await self._get_http_session()
            async with session.get(f"{endpoint}/openapi.json", headers=headers, timeout=2.0) as response:
                    if response.status == 200:
                        openapi_spec = await response.json()
                        tools = self._extract_tools_from_openapi(openapi_spec)
                        logger.debug(f"📥 {server_name}: Found {len(tools)} tools in OpenAPI spec")
                        return tools
                    else:
                        error_text = await response.text()
                        logger.warning(f"⚠️ {server_name}: HTTP {response.status}: {error_text}")
                        return []
                        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ {server_name}: MCPO probe timeout")
            return []
        except Exception as e:
            logger.warning(f"❌ {server_name}: MCPO probe failed: {e}")
            return []
    
    def _extract_tools_from_openapi(self, openapi_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool definitions from OpenAPI specification"""
        tools = []
        
        paths = openapi_spec.get("paths", {})
        components = openapi_spec.get("components", {})
        schemas = components.get("schemas", {})
        
        for path, methods in paths.items():
            if "post" in methods:
                post_spec = methods["post"]
                
                # Extract tool name from path (e.g., /search_nodes -> search_nodes)
                tool_name = path.replace("/", "")
                
                # Extract description
                description = post_spec.get("summary", post_spec.get("description", f"Execute {tool_name}"))
                
                # Extract parameters from request body schema
                parameters = {"type": "object", "properties": {}}
                request_body = post_spec.get("requestBody", {})
                if request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    schema = json_content.get("schema", {})
                    
                    # Handle schema references
                    if "$ref" in schema:
                        ref_path = schema["$ref"].replace("#/components/schemas/", "")
                        if ref_path in schemas:
                            schema = schemas[ref_path]
                    
                    parameters = schema
                
                tools.append({
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters
                })
                
        return tools
    
    
    def _get_static_fallback_tools(self) -> Dict[str, Dict[str, Any]]:
        """Minimal static fallback tools based on expected MCP servers"""
        return {
            # Based on mcp.json servers - these are minimal placeholders
            "brave_web_search": {
                "name": "brave_web_search",
                "description": "Search the web using Brave Search API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                },
                "mcp_server": "brave-search"
            },
            "memory_create_entities": {
                "name": "memory_create_entities",
                "description": "Create and store entities in memory", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {"type": "array", "description": "Array of entities to create"}
                    }
                },
                "mcp_server": "memory"
            },
            "run_javascript": {
                "name": "run_javascript",
                "description": "Execute JavaScript code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "JavaScript code to execute"}
                    }
                },
                "mcp_server": "javascript"
            },
            "read_file": {
                "name": "read_file", 
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    }
                },
                "mcp_server": "filesystem"
            }
        }
    
    async def get_tools_for_llm(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        🚀 MAIN METHOD: Get OpenAI-compatible tools array for LM Studio
        This is the key method that implements your friend's architecture!
        """
        # Auto-refresh if TTL expired (unless using cached tools)
        if force_refresh:
            await self._refresh_manifest()
        else:
            await self.refresh_if_stale()
        
        # Convert internal manifest to OpenAI tools format
        tools = []
        for tool_name, tool_info in self._manifest.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_info["name"],
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("parameters", {"type": "object", "properties": {}})
                }
            })
        
        logger.info(f"🎯 Providing {len(tools)} tools to LM Studio")
        logger.debug(f"   Tool names: {[t['function']['name'] for t in tools]}")
        
        return tools
    
    def get_cached_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        🚀 CACHED VERSION: Get tools without any HTTP calls or refreshing
        Used during pipeline building for instant tool loading
        """
        # Convert internal manifest to OpenAI tools format (no refresh)
        tools = []
        for tool_name, tool_info in self._manifest.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_info["name"],
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("parameters", {"type": "object", "properties": {}})
                }
            })
        
        logger.info(f"🎯 Providing {len(tools)} CACHED tools to LM Studio (no HTTP calls)")
        logger.debug(f"   Tool names: {[t['function']['name'] for t in tools]}")
        
        return tools
    
    def get_routing_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get routing information for a specific tool (server, command, etc.)"""
        return self._manifest.get(tool_name)
    
    async def discover_tools(self) -> Dict[str, str]:
        """
        Legacy method for compatibility - now just returns tool names and descriptions
        """
        await self.refresh_if_stale()
        
        # Convert to simple name -> description mapping
        return {
            tool_name: tool_info.get("description", "MCP tool") 
            for tool_name, tool_info in self._manifest.items()
        }
    
    
    
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        🚀 DYNAMIC ROUTING: Call tool using routing info from manifest
        This implements the execute_tool_call logic from your friend's architecture!
        """
        logger.info(f"🔧 Calling tool: {tool_name} with params: {params}")
        
        # Get routing information
        routing_info = self.get_routing_info(tool_name)
        if not routing_info:
            logger.error(f"❌ Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            mcp_server = routing_info.get("mcp_server")
            
            # Route based on server type
            if mcp_server in ["static", "builtin"]:
                # Built-in static tools (our working implementations)
                return await self._call_static_tool(tool_name, params)
            else:
                # Call dynamic MCP server via JSON-RPC stdio
                return await self._call_dynamic_mcp_tool(routing_info, params)
                
        except Exception as e:
            logger.error(f"❌ Error calling tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def _call_static_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call our working static tool implementations"""
        if tool_name == "brave_web_search":
            return await self._call_brave_search(params)
        elif tool_name == "memory_create_entities":
            return await self._call_memory_create_entities(params)
        elif tool_name.startswith("memory_"):
            return await self._call_memory_tool(tool_name, params)
        else:
            return {"error": f"Unknown static tool: {tool_name}"}
    
    async def _call_dynamic_mcp_tool(self, routing_info: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool via MCPO HTTP endpoint"""
        import aiohttp
        
        server_name = routing_info["mcp_server"]
        tool_name = routing_info["name"]
        endpoint = routing_info.get("mcpo_endpoint")
        
        if not endpoint:
            logger.error(f"❌ No MCPO endpoint found for server: {server_name}")
            return {"error": f"No MCPO endpoint configured for {server_name}"}
        
        # 👑 THE KING'S APPROACH: Let the API teach us through negotiation!
        async def api_caller(negotiated_params):
            """Internal API caller for the negotiation protocol"""
            headers = {
                "Authorization": "Bearer slowcat-secret",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Map server name to correct endpoint path
            server_path_map = {
                "memory": "memory",
                "brave-search": "brave-search", 
                "filesystem": "filesystem"
            }
            server_path = server_path_map.get(server_name, server_name)
            tool_url = f"{endpoint}/{tool_name}"
            
            logger.debug(f"🔧 {server_name}: Executing {tool_name} via MCPO: {tool_url}")
            logger.debug(f"🔧 Params: {params}")
            
            session = await self._get_http_session()
            async with session.post(tool_url, headers=headers, json=params, timeout=30.0) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ {server_name}: {tool_name} executed successfully via MCPO")
                        
                        # Ensure result is a dict before processing
                        if not isinstance(result, dict):
                            logger.warning(f"Unexpected response type from MCPO: {type(result)}")
                            result = {"result": result}
                        
                        # 🔥 SMART RESPONSE FILTERING - prevent huge responses from breaking voice
                        result = self._filter_large_response(tool_name, result)
                        
                        # 🎯 SEARCH RESULT FORMATTING - add clickable links for UI
                        if tool_name == 'brave_web_search' and isinstance(result, dict):
                            result = self._format_brave_search_response(result)
                        
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ {server_name}: HTTP {response.status}: {error_text}")
                        
                        # Return the raw error - sanitizer should have prevented this
                        return {"error": f"HTTP {response.status}: {error_text}"}
            
        
        # 👑 PURE ALGORITHMIC NEGOTIATION: ZERO MANUAL RULES!
        try:
            return await self.negotiator.negotiate_call(tool_name, params, api_caller)
        except asyncio.TimeoutError:
            logger.error(f"⏱️ {server_name}: {tool_name} timeout via MCPO")
            return {"error": f"Tool execution timeout: {tool_name}"}
        except Exception as e:
            logger.error(f"❌ MCPO tool call failed: {e}")
            return {"error": f"Tool execution failed: {e}"}
    
    def _filter_large_response(self, tool_name: str, result: dict) -> dict:
        """Filter huge responses to prevent voice/token issues"""
        
        # Extract the actual result content
        if "result" not in result:
            return result
            
        content = result["result"]
        
        # Handle browser screenshot - keep for vision analysis but flag it
        if tool_name == "browser_take_screenshot" and isinstance(content, str):
            if content.startswith("data:image/") or "base64" in content:
                logger.info("🖼️ Screenshot with base64 data - keeping for vision analysis")
                # Keep the image but add a note - LLM can analyze images
                return result
        
        # Handle browser navigation - extract clean readable content
        if tool_name == "browser_navigate" and isinstance(content, str):
            if len(content) > 2000:
                logger.info(f"✂️ Truncated browser response from {len(content)} to 2000 chars")
                # Try to extract just readable text, not code/markup
                clean_content = self._extract_readable_content(content)
                result["result"] = clean_content
                return result
        
        # Handle any other massive text response
        if isinstance(content, str) and len(content) > 3000:
            logger.info(f"✂️ Truncated large response from {len(content)} to 3000 chars")  
            result["result"] = content[:3000] + "\n\n[Response truncated due to length]"
        
        return result
    
    def _extract_readable_content(self, content: str) -> str:
        """Extract clean readable text from browser content, skipping code/markup"""
        
        # If it's already short enough, return as-is
        if len(content) < 2000:
            return content
        
        lines = content.split('\n')
        readable_lines = []
        char_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Skip lines that look like code/markup/technical stuff
            if any(skip_pattern in line.lower() for skip_pattern in [
                'ref=', 'cursor=pointer', '/url:', 'button', 'generic', 'link',
                'listitem', 'navigation', 'complementary', 'img', 'tablist',
                'heading [level=', '[ref=', 'class=', 'id=', '<', '>', '{', '}'
            ]):
                continue
                
            # Skip very short lines (likely UI elements)
            if len(line) < 20:
                continue
                
            # This looks like readable content - add it
            readable_lines.append(line)
            char_count += len(line) + 1
            
            # Stop if we have enough content
            if char_count > 1500:
                break
        
        if not readable_lines:
            # Fallback - just truncate the original
            return content[:1500] + "\n\n[Content truncated]"
        
        clean_text = '\n'.join(readable_lines)
        
        # Add page info
        if 'Page URL:' in content:
            url_line = next((line for line in lines if 'Page URL:' in line), '')
            if url_line:
                clean_text = url_line.strip() + '\n\n' + clean_text
        
        if 'Page Title:' in content:
            title_line = next((line for line in lines if 'Page Title:' in line), '')
            if title_line:
                clean_text = title_line.strip() + '\n' + clean_text
                
        return clean_text[:2000] + "\n\n[Content filtered for readability]"
    
    async def _call_brave_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Brave Search directly with API"""
        try:
            import aiohttp
            api_key = os.getenv('BRAVE_API_KEY')
            if not api_key:
                return {"error": "Brave API key not found"}
            
            query = params.get('query', '')
            if not query:
                return {"error": "No search query provided"}
            
            logger.info(f"🔍 Brave Search: {query}")
            
            session = await self._get_http_session()
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "X-Subscription-Token": api_key,
                "Accept": "application/json"
            }
            search_params = {"q": query, "count": 3}
            
            async with session.get(url, headers=headers, params=search_params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        
                        for result in data.get("web", {}).get("results", [])[:3]:
                            results.append({
                                "title": result.get("title", ""),
                                "snippet": result.get("description", ""),
                                "url": result.get("url", "")
                            })
                        
                        # Format with dual-context response immediately
                        from tools.text_formatter import create_search_response
                        formatted_response = create_search_response(query, results)
                        
                        return {
                            "ui_formatted": formatted_response["ui_formatted"],  # HTML links for UI
                            "voice_summary": formatted_response["voice_summary"],  # Clean text for TTS
                            "result_count": len(results),
                            "query": query,
                            "raw_results": results  # Original data for compatibility
                        }
                    else:
                        error_text = await resp.text()
                        return {"error": f"Brave Search API error: {error_text}"}
                        
        except Exception as e:
            logger.error(f"Brave Search error: {e}")
            return {"error": str(e)}
    
    async def _call_memory_create_entities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create memory entities in JSONL format"""
        try:
            entities = params.get('entities', [])
            if not entities:
                return {"error": "No entities provided"}
            
            # Use the existing memory file path
            memory_file = Path("/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json")
            
            # Read existing memories
            memories = []
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            memories.append(json.loads(line))
            
            created_count = 0
            for entity in entities:
                name = entity.get('name')
                entity_type = entity.get('entityType', 'TEXT')
                observations = entity.get('observations', [])
                
                if name and observations:
                    # Check if entity already exists
                    existing = next((m for m in memories if m.get('name', '').lower() == name.lower()), None)
                    if existing:
                        # Add new observations to existing entity
                        existing['observations'].extend(observations)
                    else:
                        # Create new entity
                        new_entity = {
                            "type": "entity",
                            "name": name,
                            "entityType": entity_type,
                            "observations": observations
                        }
                        memories.append(new_entity)
                    created_count += 1
            
            # Write back to file
            with open(memory_file, 'w') as f:
                for memory in memories:
                    f.write(json.dumps(memory) + '\n')
            
            logger.info(f"✅ Created/updated {created_count} memory entities")
            return {
                "success": True,
                "created_count": created_count,
                "message": f"Successfully created/updated {created_count} memory entities"
            }
            
        except Exception as e:
            logger.error(f"Error creating memory entities: {e}")
            return {"error": str(e)}
    
    async def _call_memory_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory operations"""
        # For now, return a success message - can be enhanced later
        logger.info(f"🧠 Memory tool {tool_name} called")
        return {"success": f"Memory operation {tool_name} completed", "params": params}
    
    async def _call_filesystem_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle filesystem operations"""
        logger.info(f"📁 Filesystem tool {tool_name} called")
        return {"success": f"Filesystem operation {tool_name} completed", "params": params}
    
    def _format_brave_search_response(self, response) -> dict:
        """Parse and format brave search response from MCPO server"""
        try:
            from tools.text_formatter import create_search_response
            
            # First check if response has 'result' field (MCPO format)
            if isinstance(response, dict) and 'result' in response:
                inner_result = response['result']
                # Handle different inner result formats
                if isinstance(inner_result, str):
                    # Parse the raw text response format from brave-search MCP server
                    search_results = self._parse_brave_text_response(inner_result)
                elif isinstance(inner_result, list):
                    # Already structured format
                    search_results = inner_result
                elif isinstance(inner_result, dict) and 'results' in inner_result:
                    # Wrapped in results field
                    search_results = inner_result['results']
                else:
                    search_results = inner_result
            # Handle direct response formats
            elif isinstance(response, str):
                # Parse the raw text response format from brave-search MCP server
                search_results = self._parse_brave_text_response(response)
            elif isinstance(response, list):
                # Already structured format
                search_results = response
            elif isinstance(response, dict) and 'results' in response:
                # Wrapped in results field
                search_results = response['results']
            else:
                logger.warning(f"Unknown brave search response format: {type(response)}")
                return response
            
            if not search_results:
                return response
            
            # Use our dual-context formatter
            formatted_response = create_search_response("search query", search_results)
            
            return {
                "ui_formatted": formatted_response["ui_formatted"],  # Clean HTML links
                "voice_summary": formatted_response["voice_summary"],  # Clean TTS text  
                "result_count": len(search_results),
                "raw_results": search_results  # Keep original for compatibility
            }
            
        except Exception as e:
            logger.warning(f"Failed to format brave search response: {e}")
            return response  # Return original on error
    
    def _parse_brave_text_response(self, text_response: str) -> list:
        """Parse brave search text response into structured format"""
        results = []
        
        # Split by double newlines to separate results
        sections = text_response.split('\n\n')
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            
            # Look for Title:, Description:, URL: pattern
            title = ""
            description = ""
            url = ""
            
            for line in lines:
                if line.startswith('Title: '):
                    title = line[7:].strip()
                elif line.startswith('Description: '):
                    description = line[13:].strip()
                elif line.startswith('URL: '):
                    url = line[5:].strip()
                elif '://' in line and not title and not description:
                    # Might be a URL-only line
                    url = line.strip()
                elif not title and line and not line.startswith(('http', 'URL:')):
                    # First non-URL line might be title
                    title = line
                elif not description and line != title and not line.startswith(('http', 'URL:')):
                    # Second line might be description
                    description = line
            
            # Add result if we have at least a URL or title
            if url or title:
                results.append({
                    'title': title or 'Search Result',
                    'snippet': description or 'No description available',
                    'url': url or ''
                })
        
        return results
    
    async def _call_browser_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle browser operations"""
        logger.info(f"🌐 Browser tool {tool_name} called")
        return {"success": f"Browser operation {tool_name} completed", "params": params}
    
    async def _call_javascript_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JavaScript execution"""
        logger.info(f"⚡ JavaScript tool called")
        return {"success": "JavaScript execution completed", "params": params}
    
    def _get_localized_description(self, tool_name: str, default_desc: str) -> str:
        """Get localized description for a tool"""
        if self.language == "en":
            return default_desc
        
        # Load translations if not already loaded
        if self._translations is None:
            self._load_translations()
        
        # Get translation or fallback to English
        if self._translations and self.language in self._translations:
            lang_translations = self._translations[self.language]
            return lang_translations.get(tool_name, default_desc)
        
        return default_desc
    
    def _load_translations(self):
        """Load tool translations"""
        try:
            from config.mcp_tool_translations import MCP_TOOL_TRANSLATIONS
            self._translations = MCP_TOOL_TRANSLATIONS
        except ImportError:
            logger.debug("No tool translations available")
            self._translations = {}
    
    async def _get_tool_schema(self, server_name: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get OpenAPI schema for a specific tool - Powers the Universal Parameter Sanitizer"""
        try:
            # Use MCPO proxy URL instead of raw server URL
            server_path_map = {
                "memory": "memory",
                "brave-search": "brave-search", 
                "filesystem": "filesystem",
                "javascript": "javascript"
            }
            
            if server_name not in server_path_map:
                return None
            
            # Get OpenAPI spec from MCPO proxy
            openapi_url = f"http://localhost:3001/{server_path_map[server_name]}/openapi.json"
            session = await self._get_http_session()
            
            async with session.get(openapi_url) as response:
                if response.status == 200:
                    spec = await response.json()
                    
                    # Extract schema for this specific tool
                    paths = spec.get("paths", {})
                    for path, methods in paths.items():
                        if path.endswith(f"/{tool_name}"):
                            post_method = methods.get("post", {})
                            request_body = post_method.get("requestBody", {})
                            content = request_body.get("content", {})
                            json_content = content.get("application/json", {})
                            schema_ref = json_content.get("schema", {})
                            
                            # Resolve schema reference
                            if "$ref" in schema_ref:
                                ref_path = schema_ref["$ref"]  # e.g., "#/components/schemas/add_observations_form_model"
                                schema_name = ref_path.split("/")[-1]
                                components = spec.get("components", {})
                                schemas = components.get("schemas", {})
                                return schemas.get(schema_name, {})
                            else:
                                return schema_ref
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get schema for {server_name}/{tool_name}: {e}")
            return None
    

# Global MCP tool manager cache
_global_mcp_managers: Dict[str, "SimpleMCPToolManager"] = {}
_mcp_managers_lock = threading.Lock()

def get_global_mcp_manager(language: str = "en") -> "SimpleMCPToolManager":
    """Get or create a global MCP tool manager for a language"""
    with _mcp_managers_lock:
        if language not in _global_mcp_managers:
            _global_mcp_managers[language] = SimpleMCPToolManager(language=language)
        return _global_mcp_managers[language]

async def discover_mcp_tools_background(language: str = "en"):
    """Background MCP tool discovery during startup"""
    logger.info(f"🔍 Starting background MCP tool discovery for language: {language}")
    manager = get_global_mcp_manager(language)
    await manager.refresh_if_stale()
    logger.info(f"✅ Background MCP tool discovery completed for language: {language}")
    
