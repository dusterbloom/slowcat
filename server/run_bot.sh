#!/bin/bash

# Function to handle script exit
on_exit() {
    echo ""
    echo "🛑 Shutting down..."
    exit
}

# Register cleanup on exit signals
trap on_exit EXIT INT TERM

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found (looked for venv/ and .venv/)"
fi

# Set environment variables to help with multiprocessing on macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export no_proxy=*

# Check if MCP integration is enabled
if [ "$ENABLE_MCP" = "true" ]; then
    echo "🔧 MCP integration enabled - tools handled natively by LM Studio"
    echo "📦 MCP servers configured in mcp.json"
else
    echo "🤖 MCP integration disabled - using local tools only"
fi

# Run the bot with all arguments passed through
echo "Starting bot with environment fixes..."
echo "Arguments: $@"
python bot_v2.py "$@"