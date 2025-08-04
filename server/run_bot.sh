#!/bin/bash

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

# Run the bot with all arguments passed through
echo "Starting bot with environment fixes..."
echo "Arguments: $@"
python bot.py "$@"