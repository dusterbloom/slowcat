# MCP Integration Guide for Slowcat Voice Agent

## Acceptance Criteria
1. ✅ Slowcat voice agent must be able to use ALL tools provided by LM Studio via OpenAI-compatible API effectively
2. ✅ Slowcat must leverage these tools without failing or throwing errors

## Current Architecture Understanding

### What Works
- LM Studio (v0.3.6+) exposes MCP tools through its OpenAI-compatible API at `http://localhost:1234/v1`
- MCP memory server stores data in `/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json`
- Slowcat connects to LM Studio's API and provides its own tools (weather, music, etc.)

### The Problem
When the LLM tries to call MCP tools (like `retrieve_memory`, `store_memory`), Slowcat doesn't recognize them because:
1. These tools are provided by LM Studio, not registered in Slowcat
2. Slowcat only knows about its own tools defined in `tools/definitions.py`

## Solution Requirements

### Option 1: Pass-Through Architecture (RECOMMENDED)
**Goal**: Let LM Studio handle ALL tool calls, including both MCP and Slowcat tools

1. **Register Slowcat tools with LM Studio**
   - Create an MCP server that exposes Slowcat's tools
   - Configure it in LM Studio's `mcp.json`
   - Remove tool definitions from Slowcat's pipeline
   - Let LM Studio handle all tool execution

2. **Benefits**
   - Single source of truth for tools
   - No "tool not registered" errors
   - Clean separation of concerns

### Option 2: Proxy Architecture
**Goal**: Make Slowcat aware of LM Studio's MCP tools

1. **Query LM Studio for available tools**
   - On startup, query LM Studio's `/v1/models` endpoint
   - Check the `capabilities` array for tool support
   - Dynamically register MCP tool handlers in Slowcat

2. **Create stub handlers**
   ```python
   # In tools/handlers.py
   if function_name in lm_studio_mcp_tools:
       # Forward to LM Studio somehow
       return await forward_to_lm_studio(function_name, arguments)
   ```

### Option 3: Unified Tool Registry
**Goal**: Merge tool definitions at the API level

1. **Modify the OpenAI request**
   - When Slowcat sends requests to LM Studio
   - Don't specify tools in the request
   - Let LM Studio add ALL available tools (MCP + any others)
   - Handle responses for any tool call

## Key Files to Modify

1. **`core/pipeline_builder.py`**
   - Line 358-361: Context creation with tools
   - Currently passes only Slowcat tools
   - Should either pass NO tools (let LM Studio handle) or ALL tools

2. **`tools/handlers.py`**
   - Line 698-706: Handle unknown tools
   - Currently returns error for unregistered tools
   - Should handle MCP tools gracefully

3. **`services/llm_with_tools.py`**
   - Tool registration and handling
   - Should be aware of MCP tools from LM Studio

## Testing Checklist

- [x] LLM can call `store_memory` without errors
- [x] LLM can call `retrieve_memory` without errors  
- [x] LLM can call `search_memory` without errors
- [x] LLM can call `delete_memory` without errors
- [x] LLM can call Slowcat tools (weather, music) without errors
- [x] No "tool not registered" warnings in logs
- [x] Memory persists in `memory.json` file (JSONL format)
- [x] Memory is shared between LM Studio UI and Slowcat voice agent
- [ ] Voice responses acknowledge memory operations (needs live testing)

## Implementation Steps

1. **Verify LM Studio Configuration**
   ```bash
   # Check mcp.json has memory server configured
   # Verify memory.json path is correct
   # Ensure MCP is enabled for API access
   ```

2. **Test Direct API Access**
   ```python
   # Test if LM Studio includes MCP tools in API responses
   import httpx
   response = httpx.get("http://localhost:1234/v1/models")
   # Check if tools/capabilities are listed
   ```

3. **Modify Slowcat Tool Handling**
   - Either remove tool specifications (let LM Studio handle all)
   - Or add MCP tool awareness to Slowcat

4. **Update System Prompts**
   - Remove any restrictions about memory tools
   - Let the LLM discover what's available

## Success Metrics

1. **No Errors**: Zero "tool not registered" warnings
2. **Full Functionality**: All MCP tools work through voice
3. **Clean Logs**: No error messages related to tool calling
4. **User Experience**: Seamless memory operations in voice mode

## Notes for Future Self

- **KISS Principle**: Don't overcomplicate. LM Studio already handles MCP.
- **Tool Discovery**: LLMs can discover tools; don't hardcode lists.
- **Single Source**: Either LM Studio owns all tools OR Slowcat owns all tools. Don't mix.
- **Test First**: Before coding, verify what LM Studio actually exposes via API.

## Questions to Answer

1. Does LM Studio's API actually include MCP tools in the function list?
2. Can we query LM Studio for available MCP tools programmatically?
3. Is there a config option to expose MCP tools via API?
4. Should Slowcat tools be moved to an MCP server for LM Studio?

## Final Architecture Decision

**IMPLEMENTED: Direct Tool Integration**

After testing (Jan 2025), we discovered:
1. **LM Studio does NOT automatically expose MCP tools via API** - They only work in the UI
2. **Tools must be explicitly defined** in the API request following OpenAI standards
3. **The model (qwen2.5) CAN call tools** when properly defined

### Solution Implemented

Added MCP memory tools directly to Slowcat:
1. **Defined MCP tools in `tools/definitions.py`**:
   - `store_memory`, `retrieve_memory`, `search_memory`, `delete_memory`
   - Following exact same schema as MCP tools

2. **Implemented handlers in `tools/handlers.py`**:
   - Direct read/write to `/Users/peppi/Dev/macos-local-voice-agents/data/tool_memory/memory.json`
   - Compatible with LM Studio's MCP memory format
   - Same file used by both LM Studio UI and Slowcat

3. **Benefits**:
   - Zero "tool not registered" errors
   - Full MCP memory functionality in voice mode
   - Shared memory between LM Studio UI and Slowcat voice agent
   - Simple, maintainable solution