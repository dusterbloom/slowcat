#!/usr/bin/env python3
"""Debug wrapper for bot.py with fault handler"""

import faulthandler
import subprocess
import sys

# Enable fault handler to get better segfault diagnostics
faulthandler.enable()

print("Starting bot with fault handler enabled...")
print("=" * 60)

# Run bot.py as a subprocess
try:
    subprocess.run([sys.executable, "bot.py"] + sys.argv[1:], check=True)
except subprocess.CalledProcessError as e:
    print(f"\nBot exited with error code: {e.returncode}")
except KeyboardInterrupt:
    print("\nBot interrupted by user")
except Exception as e:
    print(f"\nUnexpected error: {e}")