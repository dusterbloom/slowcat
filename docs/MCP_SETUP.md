# MCP Setup Guide for Slowcat

## Overview
This guide explains how to enable MCP (Model Context Protocol) tools in Slowcat, giving your voice assistant access to memory, web browsing, weather, filesystem, and more.

## Prerequisites
1. LM Studio v0.3.17 or later (with MCP support)
2. Node.js and npm installed
3. Slowcat server running

## Setup Steps

### 1. Configure LM Studio

1. Open LM Studio
2. Go to Settings → MCP Servers
3. Click "Edit mcp.json"
4. Copy the contents from `server/mcp.json` into LM Studio's configuration
5. Save and restart LM Studio

### 2. Install MCP Servers

The MCP servers will be automatically installed when LM Studio starts, but you can pre-install them:

```bash
# Install MCP servers globally
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-puppeteer
npm install -g open-meteo-mcp
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-fetch
```

### 3. Configure Slowcat

The system prompts have already been updated to inform the model about available MCP tools. No additional configuration needed.

### 4. Start Using MCP Tools

With MCP enabled, you can now ask Slowcat to:

#### Memory Examples
- "Remember that my favorite color is blue"
- "What did we talk about yesterday?"
- "Save this information for later: [details]"

#### Browser Examples
- "Search for the latest news about AI"
- "What's on the OpenAI website?"
- "Find documentation about Python decorators"

#### Weather Examples
- "What's the weather like in San Francisco?"
- "Will it rain tomorrow in London?"
- "Give me a 5-day forecast for Tokyo"

#### Filesystem Examples
- "Read the README.md file"
- "Create a new file called notes.txt with my meeting notes"
- "What files are in the documents folder?"

#### Fetch Examples
- "Get the content from this API endpoint: [URL]"
- "Check if this website is online: [URL]"

## How It Works

1. **Voice Input**: You speak to Slowcat
2. **STT**: Your speech is converted to text
3. **LLM with MCP**: The model processes your request and can call MCP tools
4. **Tool Execution**: MCP servers execute the requested actions
5. **Response Generation**: The model incorporates tool results into its response
6. **TTS**: The response is converted back to speech

## Security Considerations

- **File System**: Limited to specific directories (./data, ./documents)
- **Browser**: Runs in a sandboxed Puppeteer instance
- **Confirmation**: LM Studio will ask for confirmation before executing tools
- **Permissions**: Always ask Slowcat to request permission before modifying files

## Troubleshooting

### MCP Tools Not Working
1. Ensure LM Studio is running with MCP enabled
2. Check that npm/npx is in your system PATH
3. Verify the mcp.json configuration in LM Studio
4. Check LM Studio logs for MCP server errors

### Tool Timeouts
- MCP tools may take time to execute
- The voice interface will provide updates like "Let me check that for you"
- Default timeout is 30 seconds per tool call

### Memory Issues
- Memory is stored locally by the MCP memory server
- Clear memory by restarting the memory MCP server
- Memory persists across Slowcat restarts

## Best Practices

1. **Voice-First Design**: Tools enhance but don't replace conversation
2. **Concise Responses**: Tool outputs are summarized for speech
3. **Progressive Disclosure**: Slowcat explains what it's doing
4. **Error Handling**: Graceful fallbacks when tools fail
5. **Privacy**: All tools run locally, no data leaves your machine

## Advanced Configuration

### Custom MCP Servers
You can add more MCP servers to the configuration:

```json
{
  "mcpServers": {
    "your-custom-server": {
      "command": "node",
      "args": ["path/to/your/server.js"]
    }
  }
}
```

### Environment Variables
Some MCP servers need API keys:

```bash
export GITHUB_TOKEN="your-token"
export BRAVE_API_KEY="your-key"
```

## Next Steps

1. Experiment with different tool combinations
2. Create custom MCP servers for specific needs
3. Fine-tune prompts for your use cases
4. Share feedback to improve the integration