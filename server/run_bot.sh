#!/bin/bash

# Store process PIDs for cleanup
MCPO_PID=""

# Function to handle script exit
on_exit() {
    echo ""
    echo "🛑 Shutting down..."
    
    # Kill MCPO server if running
    if [ ! -z "$MCPO_PID" ]; then
        echo "🔥 Stopping MCPO server (PID: $MCPO_PID)"
        kill "$MCPO_PID" 2>/dev/null
        wait "$MCPO_PID" 2>/dev/null
    fi
    
    # Kill any remaining mcpo processes started by this user
    pkill -u "$(whoami)" -f "mcpo --port ${MCPO_PORT:-3001}" 2>/dev/null
    
    echo "✅ Cleanup complete"
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

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env"
    set -a  # automatically export all variables
    source .env
    set +a  # stop automatically exporting
    
    # Force override system OpenAI variables with .env values for local LM Studio
    if [ ! -z "${OPENAI_API_KEY}" ]; then
        export OPENAI_API_KEY="${OPENAI_API_KEY}"
        echo "🔑 Overriding system OPENAI_API_KEY with .env value"
    fi
    if [ ! -z "${OPENAI_BASE_URL}" ]; then
        export OPENAI_BASE_URL="${OPENAI_BASE_URL}" 
        echo "🌐 Overriding system OPENAI_BASE_URL with .env value"
    fi
fi

# Set environment variables to help with multiprocessing on macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export no_proxy="${NO_PROXY:-localhost,127.0.0.1}"

# Check if Mem0 is enabled and setup Chroma database
ENABLE_MEM0=${ENABLE_MEM0:-false}

if [ "$ENABLE_MEM0" = "true" ]; then
    echo "🧠 Mem0 memory system enabled - using Chroma vector database"
    
    # Create data directory for Chroma storage
    mkdir -p ./data/chroma_db
    
    echo "✅ Chroma database directory ready"
    echo "💾 Data will be stored in: ./data/chroma_db/"
    echo "📊 No external services required - fully local setup"
else
    echo "🧠 Mem0 memory system disabled"
fi

# Check if MCP integration is enabled (default: true)
ENABLE_MCP=${ENABLE_MCP:-true}

if [ "$ENABLE_MCP" = "true" ]; then
    echo "🔧 MCP integration enabled - starting MCPO server"
    
    # Check if mcp.json exists
    if [ ! -f "mcp.json" ]; then
        echo "❌ mcp.json not found - MCP integration disabled"
        ENABLE_MCP=false
    else
        echo "🚀 Starting MCPO HTTP proxy server..."
        
        # Kill any existing mcpo processes first
        pkill -f "mcpo" 2>/dev/null
        sleep 1
        
        # Start MCPO server in background
        MEMORY_FILE_PATH="${MCPO_MEMORY_FILE_PATH:-$PWD/data/tool_memory/memory.json}" \
        mcpo --host 127.0.0.1 --port "${MCPO_PORT:-3001}" --api-key "${MCPO_API_KEY:-slowcat-secret}" --config mcp.json --name mcpo-proxy > mcpo.log 2>&1 &
        
        # Store the PID for cleanup
        MCPO_PID=$!
        
        # Wait a moment for startup and check if it's running
        sleep 3
        
        if kill -0 $MCPO_PID 2>/dev/null; then
            echo "✅ MCPO server started successfully (PID: $MCPO_PID)"
            echo "📡 HTTP endpoints available at http://localhost:3001/"
            echo "📄 Logs: mcpo.log"
        else
            echo "❌ MCPO server failed to start - check mcpo.log for details"
            MCPO_PID=""
        fi
    fi
else
    echo "🤖 MCP integration disabled - using local tools only"
fi

# Run the bot with all arguments passed through
echo "Starting bot with environment fixes..."
echo "Arguments: $@"
python bot_v2.py "$@"