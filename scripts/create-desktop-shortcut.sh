#!/bin/bash

# Create desktop shortcut for Slowcat
# This creates a shortcut that opens localhost:3000 in default browser

DESKTOP_PATH="$HOME/Desktop"
SHORTCUT_NAME="Slowcat.command"
SHORTCUT_PATH="$DESKTOP_PATH/$SHORTCUT_NAME"

cat > "$SHORTCUT_PATH" << 'EOF'
#!/bin/bash
# Slowcat Desktop Launcher
echo "🐱 Starting Slowcat..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Function to check if server is running
check_server() {
    if curl -s http://localhost:7860 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to check if client is running
check_client() {
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start server if not running
if ! check_server; then
    echo "🔧 Starting Slowcat server..."
    cd "$PROJECT_DIR/server" && ./run_bot.sh &
    SERVER_PID=$!
    echo "⏳ Waiting for server to start..."
    sleep 5
fi

# Start client if not running
if ! check_client; then
    echo "🌐 Starting Slowcat client..."
    cd "$PROJECT_DIR/client" && npm run dev &
    CLIENT_PID=$!
    echo "⏳ Waiting for client to start..."
    sleep 3
fi

# Open browser
echo "🚀 Opening Slowcat..."
open "http://localhost:3000"

echo "✅ Slowcat is ready!"
echo "🔴 Press Ctrl+C to stop"

# Keep script running
wait
EOF

# Make the shortcut executable
chmod +x "$SHORTCUT_PATH"

echo "✅ Desktop shortcut created at: $SHORTCUT_PATH"
echo "🎯 Double-click 'Slowcat.command' on your desktop to launch!"