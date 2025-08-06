"""
Simple MCP Tool Manager for lazy loading and minimal context usage
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
import aiohttp
from loguru import logger
from dataclasses import dataclass, field


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
    Manages MCP tools with lazy loading and minimal context footprint
    """
    base_url: str = "http://localhost:8001"
    language: str = "en"
    tool_manifest: Dict[str, MCPToolInfo] = field(default_factory=dict)
    loaded_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _translations: Optional[Dict[str, Dict[str, str]]] = None
    
    async def discover_tools(self) -> Dict[str, str]:
        """
        Discover available MCP tools and build lightweight manifest
        Returns dict of tool_name -> description
        """
        logger.info("🔍 Discovering MCP tools (lightweight mode)...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get main OpenAPI schema
                async with session.get(f"{self.base_url}/openapi.json") as resp:
                    if resp.status != 200:
                        logger.warning("MCPO not available")
                        return {}
                    
                    main_schema = await resp.json()
                
                # Try to get schemas from known MCP servers
                servers = ["memory", "brave-search", "filesystem", "browser-text", "javascript"]
                
                for server_name in servers:
                    try:
                        async with session.get(f"{self.base_url}/{server_name}/openapi.json") as resp:
                            if resp.status == 200:
                                server_schema = await resp.json()
                                self._extract_tool_info(server_name, server_schema)
                    except Exception as e:
                        logger.debug(f"Server {server_name} not available: {e}")
                
                # Build simple manifest
                manifest = {}
                for tool_info in self.tool_manifest.values():
                    # Get localized description if available
                    desc = self._get_localized_description(tool_info.full_name, tool_info.description)
                    manifest[tool_info.full_name] = desc
                
                logger.info(f"✅ Discovered {len(manifest)} MCP tools")
                return manifest
                
        except Exception as e:
            logger.error(f"Failed to discover MCP tools: {e}")
            return {}
    
    def _extract_tool_info(self, server_name: str, schema: Dict[str, Any]):
        """Extract lightweight tool information from OpenAPI schema"""
        paths = schema.get("paths", {})
        
        for path, methods in paths.items():
            if "post" in methods:
                post_spec = methods["post"]
                tool_name = path.strip("/").split("/")[-1]
                full_name = f"{server_name}_{tool_name}"
                
                # Get description (prefer summary for brevity)
                description = post_spec.get("summary") or post_spec.get("description", f"MCP tool from {server_name}")
                
                # Store tool info
                self.tool_manifest[full_name] = MCPToolInfo(
                    server_name=server_name,
                    tool_name=tool_name,
                    full_name=full_name,
                    description=description,
                    path=path
                )
                
                logger.debug(f"  📌 Found tool: {full_name}")
    
    async def get_tool_schema(self, tool_full_name: str) -> Optional[Dict[str, Any]]:
        """
        Lazy load full schema for a specific tool
        """
        # Check cache first
        if tool_full_name in self.loaded_schemas:
            return self.loaded_schemas[tool_full_name]
        
        # Get tool info
        tool_info = self.tool_manifest.get(tool_full_name)
        if not tool_info:
            logger.error(f"Unknown tool: {tool_full_name}")
            return None
        
        try:
            # Fetch full schema from MCPO
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/{tool_info.server_name}/openapi.json"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        schema = await resp.json()
                        
                        # Extract the specific tool's schema
                        paths = schema.get("paths", {})
                        tool_path = paths.get(tool_info.path, {})
                        post_spec = tool_path.get("post", {})
                        
                        # Resolve any $ref in the schema
                        if "requestBody" in post_spec:
                            content = post_spec["requestBody"].get("content", {})
                            if "application/json" in content:
                                json_schema = content["application/json"].get("schema", {})
                                
                                if "$ref" in json_schema:
                                    resolved = self._resolve_ref(json_schema["$ref"], schema)
                                    post_spec["resolved_schema"] = resolved
                        
                        # Cache it
                        self.loaded_schemas[tool_full_name] = post_spec
                        logger.info(f"📥 Loaded schema for {tool_full_name}")
                        return post_spec
                        
        except Exception as e:
            logger.error(f"Failed to load schema for {tool_full_name}: {e}")
            return None
    
    def _resolve_ref(self, ref: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a $ref in the schema"""
        path = ref.replace("#/", "").split("/")
        result = schema
        for part in path:
            result = result.get(part, {})
        return result
    
    async def call_tool(self, tool_full_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool through MCPO
        """
        tool_info = self.tool_manifest.get(tool_full_name)
        if not tool_info:
            return {"error": f"Unknown tool: {tool_full_name}"}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/{tool_info.server_name}/{tool_info.tool_name}"
                
                async with session.post(
                    url,
                    json=params,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        error_text = await resp.text()
                        return {"error": error_text}
                        
        except Exception as e:
            logger.error(f"Error calling tool {tool_full_name}: {e}")
            return {"error": str(e)}
    
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
    
    def get_manifest_for_llm(self) -> List[str]:
        """
        Get tool names for LLM enum
        """
        return list(self.tool_manifest.keys())
    
    def get_manifest_with_descriptions(self) -> Dict[str, str]:
        """
        Get tool manifest with localized descriptions
        """
        manifest = {}
        for tool_info in self.tool_manifest.values():
            desc = self._get_localized_description(tool_info.full_name, tool_info.description)
            manifest[tool_info.full_name] = desc
        return manifest