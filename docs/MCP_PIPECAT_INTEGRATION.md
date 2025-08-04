# MCP Integration with Pipecat and LM Studio

## How It Works

MCP (Model Context Protocol) tools work **through** LM Studio, not directly in Pipecat. Here's the architecture:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Voice     │────►│  Pipecat    │────►│  LM Studio  │────►│ MCP Servers │
│   Input     │     │  (STT)      │     │  (LLM)      │     │  (Tools)    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                               │                      │
                                               └──────────────────────┘
                                                    Tool Results
```

## Key Points

1. **No Pipecat Code Changes Required**: Pipecat 0.0.77 already supports function calling through the OpenAI API
2. **MCP Runs in LM Studio**: All MCP servers are managed by LM Studio
3. **Transparent Integration**: The model decides when to use tools, LM Studio handles execution
4. **Voice-Optimized Prompts**: We only updated system prompts to inform the model about available tools

## What Pipecat Does

1. **Sends Messages**: Pipecat sends transcribed text to LM Studio via OpenAI API
2. **Supports Function Calling**: The `OpenAILLMService` already handles function call responses
3. **Receives Responses**: Gets the final response (including tool results) from LM Studio
4. **Converts to Speech**: Uses TTS to speak the response

## What LM Studio Does

1. **Receives Messages**: Gets the user message from Pipecat
2. **Model Reasoning**: The LLM decides if tools are needed
3. **Executes MCP Tools**: Runs the appropriate MCP servers
4. **Formats Response**: Combines tool results with natural language
5. **Returns to Pipecat**: Sends the complete response back

## Configuration

### 1. LM Studio Setup
- Copy `mcp.json` to LM Studio's configuration directory
- LM Studio automatically starts MCP servers
- Tool confirmations appear in LM Studio UI

### 2. Pipecat Configuration
- System prompts updated to mention available tools
- No code changes needed in pipeline
- Existing OpenAI integration handles everything

### 3. Environment
- `ENABLE_MCP=true` in `.env` (for future use)
- API keys for specific tools (GitHub, search, etc.)

## Example Flow

1. User says: "What's the weather in Paris?"
2. Pipecat STT → "What's the weather in Paris?"
3. Sent to LM Studio via OpenAI API
4. Model decides to use weather tool
5. LM Studio executes MCP weather server
6. Model receives: Paris, 18°C, partly cloudy
7. Model generates: "The weather in Paris is 18 degrees with partly cloudy skies"
8. Response sent back to Pipecat
9. Pipecat TTS → Voice output

## No Tool Processor Needed

Unlike direct tool integration, MCP through LM Studio means:
- No `ToolProcessor` class needed
- No `LLMToolCallFrame` handling
- No custom tool execution code
- Everything happens in LM Studio

## Testing

1. Start LM Studio with MCP enabled
2. Run Slowcat normally: `python bot.py`
3. Ask questions that require tools:
   - "What's the weather?"
   - "Search for Python tutorials"
   - "Remember my favorite color is blue"
4. Watch LM Studio console for tool execution
5. Hear the voice response with tool results

## Troubleshooting

### "Cannot import LLMToolCallFrame"
- This means someone tried to add direct tool support
- Remove any `tool_processor.py` files
- Remove `ToolProcessor` from imports
- MCP doesn't need this - it works through the API

### Tools Not Working
- Check LM Studio has MCP servers running
- Verify model supports function calling
- Ensure system prompt mentions available tools
- Check LM Studio logs for MCP errors

### Voice Delays
- Tool execution adds latency
- Use phrases like "Let me check that for you"
- Consider caching frequent requests
- Optimize MCP server startup time