#!/bin/bash
# SurrealDB Startup Script for Slowcat
# Starts SurrealDB server with proper authentication and data persistence

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SURREALDB_HOST="${SURREALDB_HOST:-127.0.0.1}"
SURREALDB_PORT="${SURREALDB_PORT:-8000}"
SURREALDB_USER="${SURREALDB_USER:-root}"
SURREALDB_PASS="${SURREALDB_PASS:-slowcat_secure_2024}"
SURREALDB_DATA_DIR="${SURREALDB_DATA_DIR:-$(pwd)/data/surrealdb}"
SURREALDB_PID_FILE="${SURREALDB_PID_FILE:-$(pwd)/data/surrealdb.pid}"
SURREALDB_LOG_FILE="${SURREALDB_LOG_FILE:-$(pwd)/data/surrealdb.log}"

echo -e "${BLUE}🚀 Starting SurrealDB for Slowcat...${NC}"

# Check if SurrealDB is installed
if ! command -v surreal &> /dev/null; then
    echo -e "${RED}❌ SurrealDB not found. Install with: brew install surrealdb/tap/surreal${NC}"
    exit 1
fi

# Create data directory
mkdir -p "$(dirname "$SURREALDB_DATA_DIR")"
mkdir -p "$(dirname "$SURREALDB_PID_FILE")"
mkdir -p "$(dirname "$SURREALDB_LOG_FILE")"

# Check if SurrealDB is already running
if [ -f "$SURREALDB_PID_FILE" ]; then
    if ps -p "$(cat "$SURREALDB_PID_FILE")" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  SurrealDB already running (PID: $(cat "$SURREALDB_PID_FILE"))${NC}"
        echo -e "${GREEN}✅ SurrealDB is ready at ws://$SURREALDB_HOST:$SURREALDB_PORT/rpc${NC}"
        exit 0
    else
        echo -e "${YELLOW}🧹 Cleaning up stale PID file${NC}"
        rm -f "$SURREALDB_PID_FILE"
    fi
fi

# Start SurrealDB server
echo -e "${BLUE}📊 Starting SurrealDB server...${NC}"
echo -e "${BLUE}   Host: $SURREALDB_HOST:$SURREALDB_PORT${NC}"
echo -e "${BLUE}   Data: $SURREALDB_DATA_DIR${NC}"
echo -e "${BLUE}   User: $SURREALDB_USER${NC}"
echo -e "${BLUE}   Logs: $SURREALDB_LOG_FILE${NC}"

# Use RocksDB for persistent storage
SURREALDB_PATH="rocksdb://$SURREALDB_DATA_DIR"

# Start SurrealDB in background with proper authentication
nohup surreal start "$SURREALDB_PATH" \
    --bind "$SURREALDB_HOST:$SURREALDB_PORT" \
    --user "$SURREALDB_USER" \
    --pass "$SURREALDB_PASS" \
    --log info \
    --allow-all \
    > "$SURREALDB_LOG_FILE" 2>&1 &

# Save PID
SURREALDB_PID=$!
echo $SURREALDB_PID > "$SURREALDB_PID_FILE"

echo -e "${BLUE}⏳ Waiting for SurrealDB to start...${NC}"

# Wait for SurrealDB to be ready
WAIT_COUNT=0
MAX_WAIT=30

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://$SURREALDB_HOST:$SURREALDB_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ SurrealDB is ready!${NC}"
        echo -e "${GREEN}   URL: ws://$SURREALDB_HOST:$SURREALDB_PORT/rpc${NC}"
        echo -e "${GREEN}   PID: $SURREALDB_PID${NC}"
        echo -e "${GREEN}   Data: $SURREALDB_DATA_DIR${NC}"
        exit 0
    fi
    
    if ! ps -p $SURREALDB_PID > /dev/null 2>&1; then
        echo -e "${RED}❌ SurrealDB process died during startup${NC}"
        echo -e "${RED}📋 Last few lines of log:${NC}"
        tail -n 10 "$SURREALDB_LOG_FILE" || echo "No log file found"
        rm -f "$SURREALDB_PID_FILE"
        exit 1
    fi
    
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    echo -n "."
done

echo -e "\n${RED}❌ SurrealDB failed to start within ${MAX_WAIT} seconds${NC}"
echo -e "${RED}📋 Check logs: $SURREALDB_LOG_FILE${NC}"

# Clean up
if ps -p $SURREALDB_PID > /dev/null 2>&1; then
    kill $SURREALDB_PID
fi
rm -f "$SURREALDB_PID_FILE"
exit 1