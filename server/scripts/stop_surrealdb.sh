#!/bin/bash
# SurrealDB Stop Script for Slowcat
# Gracefully stops SurrealDB server and cleans up

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SURREALDB_PID_FILE="${SURREALDB_PID_FILE:-$(pwd)/data/surrealdb.pid}"
SURREALDB_LOG_FILE="${SURREALDB_LOG_FILE:-$(pwd)/data/surrealdb.log}"

echo -e "${BLUE}🛑 Stopping SurrealDB...${NC}"

# Check if PID file exists
if [ ! -f "$SURREALDB_PID_FILE" ]; then
    echo -e "${YELLOW}⚠️  No SurrealDB PID file found. Checking for running processes...${NC}"
    
    # Find SurrealDB processes
    SURREAL_PIDS=$(pgrep -f "surreal start" || true)
    if [ -z "$SURREAL_PIDS" ]; then
        echo -e "${GREEN}✅ SurrealDB is not running${NC}"
        exit 0
    else
        echo -e "${YELLOW}🔍 Found SurrealDB processes: $SURREAL_PIDS${NC}"
        for pid in $SURREAL_PIDS; do
            echo -e "${BLUE}🛑 Stopping SurrealDB process $pid...${NC}"
            kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        done
        echo -e "${GREEN}✅ SurrealDB processes stopped${NC}"
        exit 0
    fi
fi

# Read PID from file
SURREALDB_PID=$(cat "$SURREALDB_PID_FILE")

# Check if process is running
if ! ps -p "$SURREALDB_PID" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  SurrealDB process $SURREALDB_PID not found${NC}"
    rm -f "$SURREALDB_PID_FILE"
    echo -e "${GREEN}✅ Cleaned up stale PID file${NC}"
    exit 0
fi

echo -e "${BLUE}🛑 Stopping SurrealDB process $SURREALDB_PID...${NC}"

# Try graceful shutdown first
kill -TERM "$SURREALDB_PID" 2>/dev/null || true

# Wait for graceful shutdown
WAIT_COUNT=0
MAX_WAIT=10

while [ $WAIT_COUNT -lt $MAX_WAIT ] && ps -p "$SURREALDB_PID" > /dev/null 2>&1; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    echo -n "."
done

# Force kill if still running
if ps -p "$SURREALDB_PID" > /dev/null 2>&1; then
    echo -e "\n${YELLOW}⚡ Force stopping SurrealDB...${NC}"
    kill -KILL "$SURREALDB_PID" 2>/dev/null || true
    sleep 2
fi

# Verify stopped
if ps -p "$SURREALDB_PID" > /dev/null 2>&1; then
    echo -e "${RED}❌ Failed to stop SurrealDB process $SURREALDB_PID${NC}"
    exit 1
fi

# Clean up PID file
rm -f "$SURREALDB_PID_FILE"

echo -e "${GREEN}✅ SurrealDB stopped successfully${NC}"

# Optionally show log tail
if [ -f "$SURREALDB_LOG_FILE" ] && [ "$1" = "--show-logs" ]; then
    echo -e "${BLUE}📋 Last few lines of SurrealDB log:${NC}"
    tail -n 5 "$SURREALDB_LOG_FILE" || true
fi