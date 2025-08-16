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
MEMO_PORT="1234"  # Default port for memory model
MAIN_PORT="1234"  # Default port for main model

# Store original arguments
ORIGINAL_ARGS=("$@")

# Parse command line arguments to extract models
for ((i=0; i<${#ORIGINAL_ARGS[@]}; i++)); do
    if [[ "${ORIGINAL_ARGS[i]}" == "--memo" && $((i+1)) -lt ${#ORIGINAL_ARGS[@]} ]]; then
        MEMO_MODEL="${ORIGINAL_ARGS[$((i+1))]}"
    elif [[ "${ORIGINAL_ARGS[i]}" == "--llm" && $((i+1)) -lt ${#ORIGINAL_ARGS[@]} ]]; then
        MAIN_LLM_MODEL="${ORIGINAL_ARGS[$((i+1))]}"
    fi
done

echo "🔍 Parsed arguments: MAIN_LLM_MODEL='$MAIN_LLM_MODEL', MEMO_MODEL='$MEMO_MODEL'"

# Function to check if LM Studio is running and get loaded models
check_lmstudio_models() {
    local port=$1
    if curl -s --connect-timeout 3 "http://localhost:$port/v1/models" 2>/dev/null | grep -q "object.*list"; then
        curl -s "http://localhost:$port/v1/models" | grep -o '"id": "[^"]*"' | sed 's/"id": "//g' | sed 's/"//g'
    else
        echo ""
    fi
}

# Function to load model in LM Studio headless
load_model_headless() {
    local model_name=$1
    local port=$2
    
    echo "🚀 Starting LM Studio for model: $model_name on port $port"
    
    # Check if LM Studio CLI is available
    if ! command -v lms >/dev/null 2>&1; then
        echo "❌ LM Studio CLI (lms) not found - cannot auto-load models"
        echo "💡 Install LM Studio CLI or manually load the model"
        return 1
    fi
    
    # First, start server on specific port
    echo "🌐 Starting LM Studio server on port $port..."
    lms server start --port "$port" --quiet &
    
    # Wait a moment for server to initialize
    sleep 5
    
    # Then load the model into the running server
    echo "📦 Loading model: $model_name..."
    if ! lms load "$model_name" --port "$port" --yes --quiet; then
        echo "❌ Failed to load model $model_name"
        echo "💡 Make sure the model is downloaded in LM Studio"
        return 1
    fi
    
    # Wait for server to start
    echo "⏳ Waiting for server to start..."
    local retries=0
    while [ $retries -lt 20 ]; do
        if curl -s --connect-timeout 2 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "✅ Model $model_name loaded successfully on port $port"
            return 0
        fi
        sleep 3
        retries=$((retries + 1))
        echo "   Retry $retries/20..."
    done
    
    echo "❌ Failed to start server with model $model_name on port $port"
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
    
    # Auto-load main model if specified
    if [ -n "$MAIN_LLM_MODEL" ]; then
        echo "🤖 Attempting to auto-load main model: $MAIN_LLM_MODEL"
        if load_model_headless "$MAIN_LLM_MODEL" "$MAIN_PORT"; then
            echo "✅ Main model auto-loaded successfully"
            # Refresh the models list
            MAIN_MODELS=$(check_lmstudio_models $MAIN_PORT)
        else
            echo "❌ Failed to auto-load main model"
            echo "💡 Please start LM Studio manually and load your conversation model"
        fi
    else
        echo "💡 No main model specified - please start LM Studio and load your conversation model"
    fi
fi

# Check memo model if specified
if [ -n "$MEMO_MODEL" ]; then
    echo ""
    echo "🧠 Checking memo model: $MEMO_MODEL"
    
    # Always use separate port for memo model when specified (for MemoBase isolation)
    echo "🔧 Using dedicated memo port for MemoBase isolation (even if same model)"
    
    # Check memo port directly (skip checking main port)
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
else
    echo "🧠 No separate memo model specified - using main LLM for memory operations"
fi

# Auto-load embedding model if needed
EMBEDDING_MODEL="text-embedding-nomic-embed-text-v1.5"
echo ""
echo "📊 Checking embedding model: $EMBEDDING_MODEL"

# Use the same base URL for everything (LM Studio limitation: only one server)
echo "🔗 Using unified LM Studio server from OPENAI_BASE_URL"

# Extract port from OPENAI_BASE_URL
EMBEDDING_PORT=$(echo "${OPENAI_BASE_URL:-http://localhost:1234/v1}" | sed -n 's/.*:\([0-9]*\).*/\1/p')
EMBEDDING_PORT=${EMBEDDING_PORT:-1234}
EMBEDDING_CONTEXT="base URL port"

echo "🔗 Using port $EMBEDDING_PORT for all LM Studio connections (LLM + embeddings)"

echo "🔍 Checking embedding model on $EMBEDDING_CONTEXT ($EMBEDDING_PORT)..."

# Check if embedding model is loaded on the target port
EMBEDDING_MODELS=$(check_lmstudio_models $EMBEDDING_PORT)

# Debug: Show what models were found
if [ -n "$EMBEDDING_MODELS" ]; then
    echo "🔍 Found models on port $EMBEDDING_PORT:"
    echo "$EMBEDDING_MODELS" | sed 's/^/   - /'
else
    echo "🔍 No models found on port $EMBEDDING_PORT"
fi

# Debug: Show what we're searching for
echo "🔍 Looking for embedding model: '$EMBEDDING_MODEL'"

if echo "$EMBEDDING_MODELS" | grep -q "$EMBEDDING_MODEL"; then
    echo "✅ Embedding model '$EMBEDDING_MODEL' already loaded on port $EMBEDDING_PORT"
else
    echo "🚀 Embedding model not found - attempting to auto-load..."
    
    # Check if there's a server running on the target port
    if [ -n "$EMBEDDING_MODELS" ]; then
        # Server exists, try to load the embedding model
        echo "📦 Loading embedding model on existing server (port $EMBEDDING_PORT)..."
        
        # First try the standard LMS load command
        if lms load "$EMBEDDING_MODEL" --port "$EMBEDDING_PORT" --yes --quiet 2>/dev/null; then
            echo "✅ Embedding model loaded successfully"
        else
            echo "⚠️ Standard load failed - embedding models may not load via LMS CLI"
            echo "💡 This is a known issue with embedding models in LM Studio"
            echo "📋 Please manually load '$EMBEDDING_MODEL' in LM Studio GUI on port $EMBEDDING_PORT"
            echo "🔍 The model should appear as 'text-embedding-nomic-embed-text-v1.5' in the models list"
        fi
    else
        # No server on target port - for embedding models, recommend manual setup
        echo "⚠️ No LM Studio server on port $EMBEDDING_PORT"
        echo "💡 Embedding models often require manual loading in LM Studio GUI"
        echo "📋 Please:"
        echo "   1. Open LM Studio"
        echo "   2. Load '$EMBEDDING_MODEL' manually"
        echo "   3. Start server on port $EMBEDDING_PORT"
        echo "🔍 Known issue: Embedding models may not auto-load via headless mode"
    fi
fi

# Final embedding model validation
echo ""
echo "🔍 Final embedding model validation..."

# Test embedding endpoint availability using OPENAI_BASE_URL
echo "🧪 Testing embedding endpoint using OPENAI_BASE_URL..."

# Extract the base URL for embeddings (replace /v1 with /v1/embeddings)
EMBEDDING_ENDPOINT=$(echo "${OPENAI_BASE_URL:-http://localhost:1234/v1}" | sed 's|/v1$|/v1/embeddings|')

# Try to make a test embedding request
if curl -s --connect-timeout 3 "$EMBEDDING_ENDPOINT" \
    -H "Content-Type: application/json" \
    -d '{"input": "test", "model": "text-embedding-nomic-embed-text-v1.5"}' \
    2>/dev/null | grep -q "embedding"; then
    echo "✅ Embedding endpoint is working at $EMBEDDING_ENDPOINT"
else
    echo "⚠️ Embedding endpoint test failed at $EMBEDDING_ENDPOINT"
    echo "💡 This may affect MemoBase semantic search functionality"
    echo "🔧 To fix:"
    echo "   1. Ensure '$EMBEDDING_MODEL' is loaded in LM Studio"
    echo "   2. Verify the model appears as 'text-embedding-nomic-embed-text-v1.5'"
    echo "   3. Check that LM Studio server is running at ${OPENAI_BASE_URL:-http://localhost:1234/v1}"
    echo "🔄 Alternative: Set ENABLE_MEMOBASE=false to use basic local memory instead"
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
            # Use OPENAI_BASE_URL for everything (unified approach)
            DOCKER_LLM_URL=$(echo "${OPENAI_BASE_URL:-http://localhost:1234/v1}" | sed 's/localhost/host.docker.internal/g')
            
            echo "🔗 Using OPENAI_BASE_URL for MemoBase: $DOCKER_LLM_URL"
            
            # Use memo model name if specified, otherwise use default
            MEMO_MODEL_NAME=${MEMO_MODEL:-${DEFAULT_LLM_MODEL:-qwen2.5-7b-instruct}}
            
            # Use the same URL for both LLM and embeddings (unified LM Studio server)
            MEMO_LLM_URL="$DOCKER_LLM_URL"
            EMBEDDING_URL="$DOCKER_LLM_URL"
            
            echo "🔗 MemoBase unified configuration: all connections use same server"
            
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
            
            # Test if LM Studio is accessible from host - BLOCKING REQUIREMENT
            echo "🔍 Testing LM Studio connection for MemoBase..."
            
            # Test the actual URL that MemoBase will use (without docker transformation)
            MEMOBASE_TEST_URL=$(echo "$MEMO_LLM_URL" | sed 's/host.docker.internal/localhost/g')
            
            # Wait for LM Studio to be ready with retries
            LLM_READY=false
            for i in {1..10}; do
                if curl -s --connect-timeout 3 "$MEMOBASE_TEST_URL/models" >/dev/null 2>&1; then
                    echo "✅ LM Studio accessible at $MEMOBASE_TEST_URL"
                    LLM_READY=true
                    break
                else
                    echo "⏳ Attempt $i/10: Waiting for LM Studio at $MEMOBASE_TEST_URL..."
                    sleep 2
                fi
            done
            
            if [ "$LLM_READY" != "true" ]; then
                echo "❌ BLOCKING: LM Studio not accessible at $MEMOBASE_TEST_URL after 10 attempts"
                echo "💡 MemoBase requires working LLM connection - skipping MemoBase startup"
                
                # Test main port as fallback info
                if curl -s --connect-timeout 3 "${OPENAI_BASE_URL:-http://localhost:1234/v1}/models" >/dev/null 2>&1; then
                    echo "ℹ️ Main LM Studio (port 1234) is accessible - consider using main port"
                fi
                
                if [ "$MEMOBASE_FALLBACK_TO_LOCAL" = "true" ]; then
                    echo "🔄 Will fallback to local memory mode"
                else
                    echo "❌ MemoBase startup aborted - fix LLM connection first"
                fi
                
                # Skip MemoBase startup entirely
                CONNECTION_TEST="unavailable"
            else
                # LLM is ready, proceed with MemoBase startup
                echo "🚀 LLM connection verified - proceeding with MemoBase startup"
                
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
embedding_base_url: $EMBEDDING_URL
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
            fi  # End of LLM_READY condition
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

# Run the bot with all original arguments passed through
echo "Starting bot with environment fixes..."
echo "Arguments: ${ORIGINAL_ARGS[*]}"
python bot_v2.py "${ORIGINAL_ARGS[@]}"