# MCP Tools Integration Plan for Slowcat

## Overview
This document outlines how to integrate LM Studio's MCP (Model Context Protocol) tools to enhance Slowcat's capabilities as a powerful voice assistant.

## Current Architecture
- **Voice Input**: Whisper STT (MLX)
- **LLM**: Local model via LM Studio (OpenAI-compatible API)
- **Voice Output**: Kokoro TTS
- **Memory**: Local JSON-based conversation memory
- **Speaker Recognition**: Automatic enrollment and identification
- **Video**: Webcam sampling capability

## Proposed MCP Tool Integration

### 1. Memory Enhancement
Replace or augment current JSON memory with MCP Memory Service:
- **Semantic Memory**: Use ChromaDB/SQLite-vec for vector-based memory storage
- **Context Retrieval**: More intelligent context retrieval based on semantic similarity
- **Consolidation**: Automatic memory organization and compression

### 2. Browser Capabilities
Add web interaction abilities:
- **Web Search**: Search and summarize web content in real-time
- **Website Interaction**: Navigate and extract information from websites
- **Real-time Information**: Access current news, weather, documentation

### 3. Filesystem Access
Enable file system operations:
- **File Management**: Read, write, create, delete files
- **Code Analysis**: Analyze code files in projects
- **Document Processing**: Read and summarize documents

### 4. Enhanced Tool Usage
Additional MCP servers for specific tasks:
- **GitHub Integration**: Access repositories, issues, PRs
- **Database Access**: Query local databases
- **API Interactions**: Make HTTP requests to various services

## Implementation Strategy

### Phase 1: System Prompt Updates
Update the system prompts to inform the model about available tools:

```python
# Add to system_instruction in config.py
"""
You have access to the following tools through MCP:

1. **Memory Tools**: You can store and retrieve information from previous conversations using semantic search. Use this to remember user preferences, ongoing projects, and important context.

2. **Browser Tools**: You can search the web, read websites, and get real-time information. Use this when users ask about current events, need documentation, or want information not in your training data.

3. **File System Tools**: You can read and write files, analyze code, and manage documents. Use this to help users with coding tasks, document creation, or file organization.

4. **Specialized Tools**: You have access to GitHub, databases, and other APIs as configured.

When using tools:
- Be proactive in using tools when they would enhance your response
- Explain what you're doing when using tools
- Summarize results concisely for voice output
- Ask for permission before writing or modifying files
"""
```

### Phase 2: LM Studio Configuration
Create an MCP configuration file for LM Studio:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["@modelcontextprotocol/memory-server"],
      "config": {
        "storage_path": "./data/mcp_memory"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/filesystem-server"],
      "config": {
        "allowed_directories": ["./data", "./documents"]
      }
    },
    "fetch": {
      "command": "npx",
      "args": ["@modelcontextprotocol/fetch-server"]
    },
    "github": {
      "command": "npx",
      "args": ["@modelcontextprotocol/github-server"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Phase 3: Context Management
Modify the context aggregator to handle tool responses:

1. **Tool Response Processing**: Parse and format tool outputs for voice
2. **Context Window Management**: Prioritize relevant tool outputs
3. **Error Handling**: Gracefully handle tool failures

### Phase 4: Voice-Optimized Tool Usage
Adapt tool usage for voice interaction:

1. **Confirmation Prompts**: Ask before executing actions
2. **Progress Updates**: Provide verbal feedback during long operations
3. **Result Summarization**: Condense tool outputs for speech

## Configuration Changes Needed

### 1. Environment Variables
Add to `.env`:
```bash
# MCP Configuration
ENABLE_MCP_TOOLS=true
MCP_CONFIG_PATH="./mcp.json"
GITHUB_TOKEN="your-github-token"
```

### 2. Config.py Updates
Add MCP configuration section:
```python
@dataclass
class MCPConfig:
    """MCP tools configuration"""
    enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_MCP_TOOLS", "false").lower() == "true")
    config_path: str = field(default_factory=lambda: os.getenv("MCP_CONFIG_PATH", "./mcp.json"))
    require_confirmation: bool = True  # Ask before executing tools
    allowed_tools: List[str] = field(default_factory=lambda: ["memory", "fetch", "filesystem"])
```

### 3. System Prompt Updates
Enhance prompts to leverage tools effectively while maintaining voice-first design.

## Benefits

1. **Enhanced Memory**: Semantic search and better context retention
2. **Real-time Information**: Access to current data beyond training cutoff
3. **Task Automation**: File operations, code analysis, API interactions
4. **Persistent Knowledge**: Better long-term memory across sessions
5. **Multimodal Integration**: Combine voice, vision, and tool usage

## Considerations

1. **Latency**: Tool calls may add delay - use streaming responses
2. **Security**: Implement proper permissions and confirmations
3. **Voice UX**: Adapt tool outputs for spoken responses
4. **Error Recovery**: Handle tool failures gracefully
5. **Context Size**: Manage token usage with tool responses

## Next Steps

1. Install required MCP servers via npm
2. Configure LM Studio with MCP settings
3. Update system prompts
4. Test individual tool integrations
5. Optimize for voice interaction patterns
6. Add voice-specific tool usage guidelines

This integration would transform Slowcat from a conversational assistant into a powerful voice-controlled agent capable of real-world actions and information retrieval.