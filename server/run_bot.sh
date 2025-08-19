#!/bin/bash

# Production-ready run_bot.sh with comprehensive error handling

# Store MCPO process PID for cleanup
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
    echo "❌ Error: No virtual environment found (looked for venv/ and .venv/)"
    echo "Please create a virtual environment first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Error: Virtual environment not activated"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Set environment variables to help with multiprocessing on macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export no_proxy="${NO_PROXY:-localhost,127.0.0.1}"
# Reduce noisy warnings/logs by default (overridable)
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
# Suppress deprecation about LLMMessagesFrame until we migrate frames
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:LLMMessagesFrame is deprecated:DeprecationWarning}"

# Memory system configuration
export USE_STATELESS_MEMORY=${USE_STATELESS_MEMORY:-true}  # Default to true now

# Check Python version (must be 3.12 or earlier for MLX)
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $PYTHON_VERSION"

if python -c "import sys; exit(0 if sys.version_info[:2] <= (3, 12) else 1)"; then
    echo "✅ Python version compatible with MLX"
else
    echo "❌ Error: Python version too new for MLX compatibility"
    echo "   MLX requires Python 3.12 or earlier"
    echo "   Current version: $PYTHON_VERSION"
    exit 1
fi

# Check critical dependencies
echo "🔍 Checking critical dependencies..."

# Check Pipecat version
PIPECAT_VERSION=$(python -c "import pipecat; print(pipecat.__version__)" 2>/dev/null || echo "unknown")
echo "   Pipecat version: $PIPECAT_VERSION"

if [ "$PIPECAT_VERSION" != "0.0.80" ]; then
    echo "   ⚠️  Warning: Expected Pipecat 0.0.80, found $PIPECAT_VERSION"
fi

# Check for required packages
MISSING_PACKAGES=""

for package in "fastapi" "uvicorn" "loguru" "mlx" "pydantic"; do
    python -c "import $package" 2>/dev/null || MISSING_PACKAGES="$MISSING_PACKAGES $package"
done

if [ ! -z "$MISSING_PACKAGES" ]; then
    echo "   ❌ Missing packages:$MISSING_PACKAGES"
    echo "   Please install missing packages or run: pip install -r requirements.txt"
    exit 1
fi

echo "   ✅ All critical dependencies satisfied"

echo "🧠 Memory system configuration:"
if [ "$USE_STATELESS_MEMORY" = "true" ]; then
    echo "   Using STATELESS memory system (constant performance)"
    export USE_STATELESS_MEMORY=true
    
    # Check for required dependencies
    echo "   Checking stateless memory dependencies..."
    python -c "import lmdb, lz4; print('   ✅ Dependencies satisfied')" 2>/dev/null || {
        echo "   ❌ Missing dependencies for stateless memory"
        echo "   Installing required packages..."
        pip install lmdb lz4 || {
            echo "   ❌ Failed to install dependencies"
            echo "   Please run: pip install lmdb lz4"
            exit 1
        }
        echo "   ✅ Dependencies installed successfully"
    }
    
    # Check for additional dependencies that might be missing
    python -c "import sentence_transformers; print('   ✅ Sentence transformers available')" 2>/dev/null || {
        echo "   ⚠️  Warning: sentence-transformers not available, will use fallback similarity"
    }
else
    echo "   Using TRADITIONAL memory system"
    export USE_STATELESS_MEMORY=false
fi

# Check if MCP integration is enabled (default: true)
ENABLE_MCP=${ENABLE_MCP:-true}

if [ "$ENABLE_MCP" = "true" ]; then
    echo "🔧 MCP integration enabled - starting MCPO server"
    
    # Check if mcp.json exists
    if [ ! -f "mcp.json" ]; then
        echo "❌ mcp.json not found - MCP integration disabled"
        ENABLE_MCP=false
        export ENABLE_MCP=false
    else
        echo "🚀 Starting MCPO HTTP proxy server..."
        
        # Kill any existing mcpo processes first
        pkill -f "mcpo" 2>/dev/null
        sleep 1
        
        # Check if mcpo is available
        if ! command -v mcpo &> /dev/null; then
            echo "❌ mcpo command not found - MCP integration disabled"
            echo "   Install with: pip install mcp"
            ENABLE_MCP=false
            export ENABLE_MCP=false
        else
            # Start MCPO server in background
            MEMORY_FILE_PATH="${MCPO_MEMORY_FILE_PATH:-$PWD/data/tool_memory/memory.json}" \
            mcpo --host 127.0.0.1 --port "${MCPO_PORT:-3001}" --api-key "${MCPO_API_KEY:-$(openssl rand -hex 16)}" --config mcp.json --name mcpo-proxy > mcpo.log 2>&1 &
            
            # Store the PID for cleanup
            MCPO_PID=$!
            
            # Wait a moment for startup and check if it's running
            sleep 3
            
            if kill -0 $MCPO_PID 2>/dev/null; then
                echo "✅ MCPO server started successfully (PID: $MCPO_PID)"
                echo "📡 HTTP endpoints available at http://localhost:3001/"
                echo "📄 Logs: mcpo.log"
                export ENABLE_MCP=true
            else
                echo "❌ MCPO server failed to start - check mcpo.log for details"
                echo "   MCP integration disabled"
                MCPO_PID=""
                export ENABLE_MCP=false
            fi
        fi
    fi
else
    echo "🤖 MCP integration disabled - using local tools only"
    export ENABLE_MCP=false
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/stateless_memory
mkdir -p data/debug_memory
mkdir -p data/tool_memory
mkdir -p data/dictation
mkdir -p data/speaker_profiles
echo "✅ Directories created"

# Set production-ready environment variables
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTHONASYNCIODEBUG=0  # Disable for production

# Add debugging support if requested
if [ "$DEBUG_PIPELINE" = "true" ]; then
    echo "🐛 Debug mode enabled - adding frame monitoring"
    export PYTHONPATH="${PWD}/debug:${PYTHONPATH}"
fi

# Performance optimizations
export TOKENIZERS_PARALLELISM=false  # Avoid threading issues
export OMP_NUM_THREADS=1  # Optimize for single-threaded performance

echo ""
echo "🚀 Starting Slowcat Bot with comprehensive error handling..."
echo "🔧 Arguments: $@"
echo "🌍 Environment:"
echo "   - Memory system: $([ "$USE_STATELESS_MEMORY" = "true" ] && echo "Stateless" || echo "Traditional")"
echo "   - MCP integration: $ENABLE_MCP"
echo "   - Python: $PYTHON_VERSION"
echo "   - Working directory: $PWD"
echo ""

# Run the bot with comprehensive error handling
set -e  # Exit on any error

# Function to handle Python errors
run_bot_with_error_handling() {
    python bot_v2.py "$@" 2>&1 | while IFS= read -r line; do
        echo "$line"
        
        # Check for critical errors that require immediate attention
        if echo "$line" | grep -q "AttributeError.*_FrameProcessor.*__input_queue"; then
            echo ""
            echo "🚨 CRITICAL ERROR: Frame processor initialization issue detected!"
            echo "   This is likely due to missing super().__init__() calls"
            echo "   Check all custom processors for proper initialization"
            exit 1
        fi
        
        if echo "$line" | grep -q "StartFrame.*not received yet"; then
            echo ""
            echo "🚨 CRITICAL ERROR: StartFrame race condition detected!"
            echo "   Some processor is receiving frames before StartFrame"
            echo "   Check RTVIProcessor ordering and frame forwarding"
            exit 1
        fi
        
        if echo "$line" | grep -q "pipeline.*blocking\|frame.*not.*forwarded"; then
            echo ""
            echo "🚨 CRITICAL ERROR: Pipeline blocking detected!"
            echo "   Some processor is not forwarding frames properly"
            echo "   Check all process_frame methods call push_frame"
            exit 1
        fi
    done
}

# Run with error monitoring
echo "▶️  Launching bot_v2.py..."
run_bot_with_error_handling "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ Bot completed successfully"
else
    echo "❌ Bot exited with error code: $exit_code"
    echo ""
    echo "🔍 Troubleshooting tips:"
    echo "   1. Check logs above for specific error messages"
    echo "   2. Verify all dependencies are installed: pip install -r requirements.txt"
    echo "   3. Check virtual environment is activated"
    echo "   4. For memory issues, try: USE_STATELESS_MEMORY=false ./run_bot.sh"
    echo "   5. For debugging, run: DEBUG_PIPELINE=true ./run_bot.sh"
    echo "   6. Check MCPO logs if MCP is enabled: cat mcpo.log"
fi

exit $exit_code
