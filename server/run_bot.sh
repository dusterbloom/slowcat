#!/bin/bash

# Store process PIDs for cleanup
MCPO_PID=""
MEMOBASE_PID=""

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
    
    # Kill MemoBase server/container if running
    if [ ! -z "$MEMOBASE_PID" ]; then
        echo "🔥 Stopping MemoBase server..."
        # Check if it's a Docker container
        if docker ps -q -f name=memobase-slowcat >/dev/null 2>&1; then
            echo "🐳 Stopping MemoBase Docker containers"
            docker stop memobase-slowcat memobase-redis memobase-postgres 2>/dev/null
            docker rm memobase-slowcat memobase-redis memobase-postgres 2>/dev/null
        else
            # Legacy process cleanup
            kill "$MEMOBASE_PID" 2>/dev/null
            wait "$MEMOBASE_PID" 2>/dev/null
        fi
    fi
    
    # Kill any remaining processes started by this user
    pkill -u "$(whoami)" -f "mcpo --port ${MCPO_PORT:-3001}" 2>/dev/null
    pkill -u "$(whoami)" -f "memobase.*server" 2>/dev/null
    
    # Clean up any orphaned MemoBase containers
    docker stop memobase-slowcat 2>/dev/null || true
    docker rm memobase-slowcat 2>/dev/null || true
    
    # Clean up database containers
    docker stop memobase-redis memobase-postgres 2>/dev/null || true
    docker rm memobase-redis memobase-postgres 2>/dev/null || true
    
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

# LM Studio Model Management - Extract memo model info early
echo ""
echo "📊 LM Studio Model Management"

# Parse arguments to extract memo model (preserve original args)
MEMO_MODEL=""
MAIN_LLM_MODEL=""
MEMO_PORT="1235"  # Default port for memory model
MAIN_PORT="1234"  # Default port for main model

# Store original arguments
ORIGINAL_ARGS=("$@")

# Parse command line arguments to extract models
for ((i=1; i<=${#ORIGINAL_ARGS[@]}; i++)); do
    if [[ "${ORIGINAL_ARGS[i]}" == "--memo" && $((i+1)) -le ${#ORIGINAL_ARGS[@]} ]]; then
        MEMO_MODEL="${ORIGINAL_ARGS[$((i+1))]}"
    elif [[ "${ORIGINAL_ARGS[i]}" == "--llm" && $((i+1)) -le ${#ORIGINAL_ARGS[@]} ]]; then
        MAIN_LLM_MODEL="${ORIGINAL_ARGS[$((i+1))]}"
    fi
done

# Function to check if LM Studio is running and get loaded models
check_lmstudio_models() {
    local port=$1
    if curl -s --connect-timeout 3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q "object.*list"; then
        curl -s "http://localhost:$port/v1/models" | grep -o '"id":"[^"]*"' | sed 's/"id":"//g' | sed 's/"//g'
    else
        echo ""
    fi
}

# Function to load model in LM Studio headless
load_model_headless() {
    local model_name=$1
    local port=$2
    
    echo "🚀 Starting LM Studio headless for model: $model_name on port $port"
    
    # Check if LM Studio CLI is available
    if ! command -v lms >/dev/null 2>&1; then
        echo "❌ LM Studio CLI (lms) not found - cannot auto-load models"
        echo "💡 Install LM Studio CLI or manually load the model"
        return 1
    fi
    
    # Start LM Studio server headless with specific model
    echo "📦 Loading $model_name on port $port..."
    lms server start --port "$port" --model "$model_name" --headless &
    
    # Wait for server to start
    echo "⏳ Waiting for model to load..."
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -s --connect-timeout 2 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "✅ Model $model_name loaded successfully on port $port"
            return 0
        fi
        sleep 2
        retries=$((retries + 1))
        echo "   Retry $retries/30..."
    done
    
    echo "❌ Failed to load model $model_name on port $port"
    return 1
}

# Check main LM Studio instance
echo "🔍 Checking main LM Studio instance on port $MAIN_PORT..."
MAIN_MODELS=$(check_lmstudio_models $MAIN_PORT)
if [ -n "$MAIN_MODELS" ]; then
    echo "✅ Main LM Studio running with models:"
    echo "$MAIN_MODELS" | sed 's/^/   - /'
    
    # Check if specified main model is loaded
    if [ -n "$MAIN_LLM_MODEL" ]; then
        if echo "$MAIN_MODELS" | grep -q "$MAIN_LLM_MODEL"; then
            echo "✅ Main model '$MAIN_LLM_MODEL' is already loaded"
        else
            echo "⚠️ Main model '$MAIN_LLM_MODEL' not loaded, but LM Studio is running"
            echo "💡 You may need to manually switch to the desired model"
        fi
    fi
else
    echo "❌ Main LM Studio not running on port $MAIN_PORT"
    echo "💡 Please start LM Studio and load your main conversation model"
fi

# Check memo model if specified
if [ -n "$MEMO_MODEL" ]; then
    echo ""
    echo "🧠 Checking memo model: $MEMO_MODEL"
    
    # First check if it's already loaded on main port
    if echo "$MAIN_MODELS" | grep -q "$MEMO_MODEL"; then
        echo "✅ Memo model '$MEMO_MODEL' found on main LM Studio instance"
        echo "💡 Will use main instance for both conversation and memory"
    else
        # Check memo port
        echo "🔍 Checking dedicated memo port $MEMO_PORT..."
        MEMO_MODELS=$(check_lmstudio_models $MEMO_PORT)
        
        if [ -n "$MEMO_MODELS" ]; then
            echo "✅ LM Studio running on memo port $MEMO_PORT with models:"
            echo "$MEMO_MODELS" | sed 's/^/   - /'
            
            if echo "$MEMO_MODELS" | grep -q "$MEMO_MODEL"; then
                echo "✅ Memo model '$MEMO_MODEL' is already loaded on port $MEMO_PORT"
                # Set memo model URL for use in MemoBase config
                export MEMO_LLM_BASE_URL="http://localhost:$MEMO_PORT/v1"
                echo "🔗 Set MEMO_LLM_BASE_URL=$MEMO_LLM_BASE_URL"
            else
                echo "⚠️ Memo model '$MEMO_MODEL' not found on port $MEMO_PORT"
                echo "💡 Available models don't include the requested memo model"
            fi
        else
            echo "🚀 No LM Studio instance on memo port $MEMO_PORT"
            echo "🤖 Attempting to auto-load memo model..."
            
            # Try to load memo model headless
            if load_model_headless "$MEMO_MODEL" "$MEMO_PORT"; then
                echo "✅ Memo model auto-loaded successfully"
                # Update environment for MemoBase to use memo port
                export MEMO_LLM_BASE_URL="http://localhost:$MEMO_PORT/v1"
                echo "🔗 Set MEMO_LLM_BASE_URL=$MEMO_LLM_BASE_URL"
            else
                echo "❌ Failed to auto-load memo model"
                echo "💡 Will fall back to using main LM Studio instance"
            fi
        fi
    fi
else
    echo "🧠 No separate memo model specified - using main LLM for memory operations"
fi

echo ""
echo "🚀 Model configuration complete"

# Check if MemoBase is enabled and setup external memory service
ENABLE_MEMOBASE=${ENABLE_MEMOBASE:-false}

if [ "$ENABLE_MEMOBASE" = "true" ]; then
    echo "🧠 MemoBase external memory system enabled"
    
    # Check if memobase client is installed
    if ! python -c "import memobase" 2>/dev/null; then
        echo "📦 Installing MemoBase client package..."
        pip install memobase>=0.1.0
        if [ $? -eq 0 ]; then
            echo "✅ MemoBase client package installed successfully"
        else
            echo "❌ Failed to install MemoBase package"
            echo "🔧 You can install it manually with: pip install memobase"
        fi
    else
        echo "✅ MemoBase client package already installed"
    fi
    
    # Check Docker availability for server auto-start
    if [ "$MEMOBASE_AUTO_START" = "true" ]; then
        if ! command -v docker >/dev/null 2>&1; then
            echo "❌ Docker not found - required for MemoBase server auto-start"
            echo "📦 Install Docker from: https://docker.com/get-started"
            echo "🔄 Will fallback to local memory mode"
            MEMOBASE_AUTO_START=false
        else
            echo "✅ Docker available for MemoBase server management"
        fi
    fi
    
    # Set default MemoBase configuration if not set
    MEMOBASE_PROJECT_URL=${MEMOBASE_PROJECT_URL:-"http://localhost:8019"}
    MEMOBASE_API_KEY=${MEMOBASE_API_KEY:-"secret"}
    MEMOBASE_FALLBACK_TO_LOCAL=${MEMOBASE_FALLBACK_TO_LOCAL:-true}
    
    echo "🌐 MemoBase Configuration:"
    echo "   Project URL: $MEMOBASE_PROJECT_URL"
    echo "   API Key: $MEMOBASE_API_KEY"
    echo "   Fallback to local: $MEMOBASE_FALLBACK_TO_LOCAL"
    
    # Test connection to MemoBase server
    echo "🔍 Testing MemoBase server connection..."
    
    # Use curl or Python to test the connection
    if command -v curl >/dev/null 2>&1; then
        # Try curl first (more reliable and faster)
        if curl -s --connect-timeout 3 "$MEMOBASE_PROJECT_URL" >/dev/null 2>&1; then
            CONNECTION_TEST="connected"
        else
            CONNECTION_TEST="unavailable"
        fi
    else
        # Fallback to Python urllib (no external dependencies)
        CONNECTION_TEST=$(python -c "
import urllib.request
import socket
try:
    urllib.request.urlopen('$MEMOBASE_PROJECT_URL', timeout=3)
    print('connected')
except:
    print('unavailable')
" 2>/dev/null)
    fi

    if [ "$CONNECTION_TEST" = "connected" ]; then
        echo "✅ MemoBase server is running and accessible"
        echo "🧠 Using external semantic memory via MemoBase"
    else
        echo "⚠️  MemoBase server not accessible at $MEMOBASE_PROJECT_URL"
        
        # Check if we should auto-start MemoBase server
        MEMOBASE_AUTO_START=${MEMOBASE_AUTO_START:-false}
        if [ "$MEMOBASE_AUTO_START" = "true" ]; then
            echo "🚀 Auto-starting MemoBase server..."
            
            # Start MemoBase server using Docker
            echo "🐳 Starting MemoBase server via Docker..."
            
            # Create MemoBase config directory
            mkdir -p ./data/memobase
            
            # Extract port from URL (default: 8019)
            MEMOBASE_PORT=$(echo "$MEMOBASE_PROJECT_URL" | sed -n 's/.*:\([0-9]*\).*/\1/p')
            MEMOBASE_PORT=${MEMOBASE_PORT:-8019}
            
            # Stop any existing MemoBase containers FIRST
            echo "🧹 Cleaning up existing MemoBase containers..."
            docker stop memobase-slowcat memobase-redis memobase-postgres 2>/dev/null || true
            docker rm memobase-slowcat memobase-redis memobase-postgres 2>/dev/null || true
            
            # Create data directories for persistence
            mkdir -p ./data/memobase/redis-data
            mkdir -p ./data/memobase/postgres-data
            
            # Start Redis container for MemoBase
            echo "📦 Starting Redis container..."
            docker run -d \
                --name memobase-redis \
                --restart=no \
                -p 6379:6379 \
                -v "$(pwd)/data/memobase/redis-data:/data" \
                redis:7-alpine
            
            # Start PostgreSQL container with pgvector extension for MemoBase
            echo "📦 Starting PostgreSQL container with pgvector..."
            docker run -d \
                --name memobase-postgres \
                --restart=no \
                -e POSTGRES_DB=memobase \
                -e POSTGRES_USER=memobase \
                -e POSTGRES_PASSWORD=memobase123 \
                -p 5432:5432 \
                -v "$(pwd)/data/memobase/postgres-data:/var/lib/postgresql/data" \
                pgvector/pgvector:pg15
            
            # Wait for databases to be ready
            echo "⏳ Waiting for databases to start..."
            sleep 20
            
            # Verify database connections before proceeding
            echo "🔍 Verifying database connections..."
            RETRIES=0
            while [ $RETRIES -lt 10 ]; do
                if docker exec memobase-postgres pg_isready -U memobase -d memobase >/dev/null 2>&1 && \
                   docker exec memobase-redis redis-cli ping >/dev/null 2>&1 && \
                   docker exec memobase-postgres psql -U memobase -d memobase -c "SELECT 1;" >/dev/null 2>&1; then
                    echo "✅ Databases are ready and can execute queries"
                    break
                else
                    echo "⏳ Databases still initializing... (attempt $((RETRIES + 1))/10)"
                    sleep 3
                    RETRIES=$((RETRIES + 1))
                fi
            done
            
            if [ $RETRIES -eq 10 ]; then
                echo "❌ Databases failed to start properly"
                echo "🔍 Database logs:"
                docker logs memobase-postgres 2>&1 | tail -10
                docker logs memobase-redis 2>&1 | tail -5
                if [ "$MEMOBASE_FALLBACK_TO_LOCAL" = "true" ]; then
                    echo "🔄 Will fallback to local memory mode"
                fi
            else
                # Give databases a bit more time to fully stabilize
                echo "⏳ Allowing databases to fully stabilize..."
                sleep 5
            fi
            
            # Create environment file for Docker container
            # Convert localhost to host.docker.internal for container networking
            DOCKER_LLM_URL=$(echo "${OPENAI_BASE_URL:-http://localhost:1234/v1}" | sed 's/localhost/host.docker.internal/g')
            
            # Use memo model URL if available, otherwise use main LLM URL
            MEMO_LLM_URL=${MEMO_LLM_BASE_URL:-$DOCKER_LLM_URL}
            MEMO_LLM_URL=$(echo "$MEMO_LLM_URL" | sed 's/localhost/host.docker.internal/g')
            
            # Use memo model name if specified, otherwise use default
            MEMO_MODEL_NAME=${MEMO_MODEL:-${DEFAULT_LLM_MODEL:-qwen2.5-7b-instruct}}
            
            cat > ./data/memobase/env.list << EOF
# MemoBase environment variables with database configuration
MEMOBASE_API_KEY=$MEMOBASE_API_KEY
llm_api_key=lm-studio
llm_base_url=$MEMO_LLM_URL
best_llm_model=$MEMO_MODEL_NAME
language=en

# Database configuration
DATABASE_URL=postgresql://memobase:memobase123@host.docker.internal:5432/memobase
REDIS_URL=redis://host.docker.internal:6379
EOF
            
            # Test if LM Studio is accessible from host
            echo "🔍 Testing LM Studio connection..."
            if curl -s --connect-timeout 3 "${OPENAI_BASE_URL:-http://localhost:1234/v1}/models" >/dev/null 2>&1; then
                echo "✅ LM Studio is accessible"
            else
                echo "⚠️ LM Studio not accessible - MemoBase may not work properly"
                echo "💡 Make sure LM Studio is running on ${OPENAI_BASE_URL:-http://localhost:1234/v1}"
            fi
            
            # Create simple config.yaml for MemoBase (following Ollama pattern)
            cat > ./data/memobase/config.yaml << EOF
# MemoBase Configuration - LM Studio Local Setup (based on Ollama example)
max_chat_blob_buffer_token_size: 50
buffer_flush_interval: 10

llm_api_key: lm-studio
llm_base_url: $MEMO_LLM_URL
best_llm_model: $MEMO_MODEL_NAME

# Embedding configuration for LM Studio (768 dimensions)
embedding_provider: openai
embedding_api_key: lm-studio
embedding_base_url: $MEMO_LLM_URL
embedding_dim: 768
embedding_model: text-embedding-nomic-embed-text-v1.5

language: en
EOF
            
            # Pull latest MemoBase image
            echo "📦 Pulling MemoBase Docker image..."
            docker pull ghcr.io/memodb-io/memobase:latest
            
            # Start MemoBase container
            echo "🚀 Starting MemoBase container..."
            
            # First, try to run interactively to see immediate errors
            echo "🔍 Testing container startup..."
            
            CONTAINER_ID=$(docker run -d \
                --name memobase-slowcat \
                --env-file ./data/memobase/env.list \
                -v "$(pwd)/data/memobase/config.yaml:/app/config.yaml" \
                -p "$MEMOBASE_PORT:8000" \
                --restart=no \
                ghcr.io/memodb-io/memobase:latest)
            
            echo "$CONTAINER_ID" > memobase.log
            
            if [ $? -eq 0 ] && [ ! -z "$CONTAINER_ID" ]; then
                echo "📋 Container ID: $CONTAINER_ID"
                
                # Check if container is still running after a few seconds
                sleep 3
                
                if docker ps -q -f name=memobase-slowcat >/dev/null 2>&1; then
                    echo "✅ Container is running, waiting for service to start..."
                    sleep 7
                    
                    # Test connection
                    if curl -s --connect-timeout 5 "$MEMOBASE_PROJECT_URL" >/dev/null 2>&1; then
                        echo "✅ MemoBase Docker container started successfully"
                        echo "🧠 Using external semantic memory via MemoBase"
                        echo "📄 Container logs: docker logs memobase-slowcat"
                        CONNECTION_TEST="connected"
                        
                        # Set PID to container ID for cleanup reference
                        MEMOBASE_PID=$(docker ps -q -f name=memobase-slowcat)
                    else
                        echo "❌ MemoBase service not responding"
                        echo "🔍 Container logs:"
                        docker logs memobase-slowcat 2>&1 | head -20
                        docker stop memobase-slowcat 2>/dev/null
                        docker rm memobase-slowcat 2>/dev/null
                        MEMOBASE_PID=""
                        if [ "$MEMOBASE_FALLBACK_TO_LOCAL" = "true" ]; then
                            echo "🔄 Will fallback to local memory mode"
                        fi
                    fi
                else
                    echo "❌ Container exited immediately - checking logs:"
                    docker logs "$CONTAINER_ID" 2>&1 | head -20
                    
                    echo ""
                    echo "🔍 Debugging info:"
                    echo "   Config file: $(cat ./data/memobase/config.yaml | head -5)"
                    echo "   Env file: $(cat ./data/memobase/env.list)"
                    
                    docker rm "$CONTAINER_ID" 2>/dev/null
                    MEMOBASE_PID=""
                    if [ "$MEMOBASE_FALLBACK_TO_LOCAL" = "true" ]; then
                        echo "🔄 Will fallback to local memory mode"
                    fi
                fi
            else
                echo "❌ Failed to start MemoBase Docker container"
                if [ "$MEMOBASE_FALLBACK_TO_LOCAL" = "true" ]; then
                    echo "🔄 Will fallback to local memory mode"
                fi
            fi
        else
            if [ "$MEMOBASE_FALLBACK_TO_LOCAL" = "true" ]; then
                echo "🔄 Will fallback to local memory mode (graceful degradation)"
                echo "💡 To auto-start MemoBase server, set MEMOBASE_AUTO_START=true in .env"
                echo "💡 Or start manually: https://docs.memobase.io/quickstart"
            else
                echo "❌ MemoBase required but unavailable - bot may not function properly"
            fi
        fi
    fi
    
    echo "📊 MemoBase features enabled:"
    echo "   • Semantic memory with OpenAI client patching"
    echo "   • Per-user memory isolation with voice recognition"
    echo "   • Automatic session management and context injection"
    
else
    echo "🧠 MemoBase external memory system disabled"
fi

# Memory System Summary
echo ""
echo "📊 Memory System Summary:"
if [ "$ENABLE_MEMOBASE" = "true" ]; then
    echo "   🧠 Primary: MemoBase (external semantic memory)"
    if [ "$CONNECTION_TEST" = "connected" ]; then
        if [ ! -z "$MEMOBASE_PID" ]; then
            echo "   ✅ Status: Auto-started via Docker and ready (Container: $MEMOBASE_PID)"
        else
            echo "   ✅ Status: Connected and ready"
        fi
    else
        if [ "$MEMOBASE_AUTO_START" = "true" ]; then
            echo "   ⚠️  Status: Docker auto-start failed, fallback to local mode"
        else
            echo "   ⚠️  Status: Will fallback to local mode"
        fi
    fi
elif [ "$ENABLE_MEM0" = "true" ]; then
    echo "   🧠 Primary: Mem0 (local vector database)"
    echo "   ✅ Status: Local Chroma database ready"
else
    echo "   🧠 Primary: Local memory only (basic conversation history)"
    echo "   ℹ️  Status: No semantic memory features"
fi
echo ""

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

# LM Studio Model Management - Auto-load memo model if specified
echo ""
echo "📊 LM Studio Model Management"

# Parse arguments to extract memo model (preserve original args)
MEMO_MODEL=""
MAIN_LLM_MODEL=""
MEMO_PORT="1235"  # Default port for memory model
MAIN_PORT="1234"  # Default port for main model

# Store original arguments
ORIGINAL_ARGS=("$@")

# Parse command line arguments to extract models
for ((i=1; i<=${#ORIGINAL_ARGS[@]}; i++)); do
    if [[ "${ORIGINAL_ARGS[i]}" == "--memo" && $((i+1)) -le ${#ORIGINAL_ARGS[@]} ]]; then
        MEMO_MODEL="${ORIGINAL_ARGS[$((i+1))]}"
    elif [[ "${ORIGINAL_ARGS[i]}" == "--llm" && $((i+1)) -le ${#ORIGINAL_ARGS[@]} ]]; then
        MAIN_LLM_MODEL="${ORIGINAL_ARGS[$((i+1))]}"
    fi
done

# Function to check if LM Studio is running and get loaded models
check_lmstudio_models() {
    local port=$1
    if curl -s --connect-timeout 3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q "object.*list"; then
        curl -s "http://localhost:$port/v1/models" | grep -o '"id":"[^"]*"' | sed 's/"id":"//g' | sed 's/"//g'
    else
        echo ""
    fi
}

# Function to load model in LM Studio headless
load_model_headless() {
    local model_name=$1
    local port=$2
    
    echo "🚀 Starting LM Studio headless for model: $model_name on port $port"
    
    # Check if LM Studio CLI is available
    if ! command -v lms >/dev/null 2>&1; then
        echo "❌ LM Studio CLI (lms) not found - cannot auto-load models"
        echo "💡 Install LM Studio CLI or manually load the model"
        return 1
    fi
    
    # Start LM Studio server headless with specific model
    echo "📦 Loading $model_name on port $port..."
    lms server start --port "$port" --model "$model_name" --headless &
    
    # Wait for server to start
    echo "⏳ Waiting for model to load..."
    local retries=0
    while [ $retries -lt 30 ]; do
        if curl -s --connect-timeout 2 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "✅ Model $model_name loaded successfully on port $port"
            return 0
        fi
        sleep 2
        retries=$((retries + 1))
        echo "   Retry $retries/30..."
    done
    
    echo "❌ Failed to load model $model_name on port $port"
    return 1
}

# Check main LM Studio instance
echo "🔍 Checking main LM Studio instance on port $MAIN_PORT..."
MAIN_MODELS=$(check_lmstudio_models $MAIN_PORT)
if [ -n "$MAIN_MODELS" ]; then
    echo "✅ Main LM Studio running with models:"
    echo "$MAIN_MODELS" | sed 's/^/   - /'
    
    # Check if specified main model is loaded
    if [ -n "$MAIN_LLM_MODEL" ]; then
        if echo "$MAIN_MODELS" | grep -q "$MAIN_LLM_MODEL"; then
            echo "✅ Main model '$MAIN_LLM_MODEL' is already loaded"
        else
            echo "⚠️ Main model '$MAIN_LLM_MODEL' not loaded, but LM Studio is running"
            echo "💡 You may need to manually switch to the desired model"
        fi
    fi
else
    echo "❌ Main LM Studio not running on port $MAIN_PORT"
    echo "💡 Please start LM Studio and load your main conversation model"
fi

# Check memo model if specified
if [ -n "$MEMO_MODEL" ]; then
    echo ""
    echo "🧠 Checking memo model: $MEMO_MODEL"
    
    # First check if it's already loaded on main port
    if echo "$MAIN_MODELS" | grep -q "$MEMO_MODEL"; then
        echo "✅ Memo model '$MEMO_MODEL' found on main LM Studio instance"
        echo "💡 Will use main instance for both conversation and memory"
    else
        # Check memo port
        echo "🔍 Checking dedicated memo port $MEMO_PORT..."
        MEMO_MODELS=$(check_lmstudio_models $MEMO_PORT)
        
        if [ -n "$MEMO_MODELS" ]; then
            echo "✅ LM Studio running on memo port $MEMO_PORT with models:"
            echo "$MEMO_MODELS" | sed 's/^/   - /'
            
            if echo "$MEMO_MODELS" | grep -q "$MEMO_MODEL"; then
                echo "✅ Memo model '$MEMO_MODEL' is already loaded on port $MEMO_PORT"
            else
                echo "⚠️ Memo model '$MEMO_MODEL' not found on port $MEMO_PORT"
                echo "💡 Available models don't include the requested memo model"
            fi
        else
            echo "🚀 No LM Studio instance on memo port $MEMO_PORT"
            echo "🤖 Attempting to auto-load memo model..."
            
            # Try to load memo model headless
            if load_model_headless "$MEMO_MODEL" "$MEMO_PORT"; then
                echo "✅ Memo model auto-loaded successfully"
                # Update environment for MemoBase to use memo port
                export MEMO_LLM_BASE_URL="http://localhost:$MEMO_PORT/v1"
                echo "🔗 Set MEMO_LLM_BASE_URL=$MEMO_LLM_BASE_URL"
            else
                echo "❌ Failed to auto-load memo model"
                echo "💡 Will fall back to using main LM Studio instance"
            fi
        fi
    fi
else
    echo "🧠 No separate memo model specified - using main LLM for memory operations"
fi

echo ""
echo "🚀 Model configuration complete - starting bot..."

# Run the bot with all original arguments passed through
echo "Starting bot with environment fixes..."
echo "Arguments: ${ORIGINAL_ARGS[*]}"
python bot_v2.py "${ORIGINAL_ARGS[@]}"