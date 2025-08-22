#!/bin/bash

echo "Testing .env loading mechanism..."
echo "================================="

# Simulate the same loading logic from run_bot.sh
if [ -f .env ]; then
    echo "📄 Loading environment variables from .env"
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
            # Only export valid KEY=VALUE pairs
            if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
                echo "  Exporting: $line"
                export "$line"
            else
                echo "  Skipping invalid line: $line"
            fi
        fi
    done < .env
fi

echo ""
echo "Checking USER_ID after loading:"
echo "USER_ID = '$USER_ID'"

echo ""
echo "Python check:"
source .venv/bin/activate
python -c "import os; print(f\"Python sees USER_ID = '{os.getenv('USER_ID', 'NOT SET')}'\") "