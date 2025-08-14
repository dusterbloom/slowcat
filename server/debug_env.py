#!/usr/bin/env python3
"""
Debug environment variable loading
"""

import os
from dotenv import load_dotenv

print("🔍 Debugging Environment Variables...")

# Load .env file
load_dotenv()

enable_mem0 = os.getenv("ENABLE_MEM0", "false")
print(f"ENABLE_MEM0 = '{enable_mem0}'")
print(f"ENABLE_MEM0 lowercased = '{enable_mem0.lower()}'")
print(f"ENABLE_MEM0 == 'true'? {enable_mem0.lower() == 'true'}")

# Check other Mem0 related vars
print(f"MEM0_CHAT_MODEL = '{os.getenv('MEM0_CHAT_MODEL', 'NOT SET')}'")
print(f"MEM0_EMBEDDING_MODEL = '{os.getenv('MEM0_EMBEDDING_MODEL', 'NOT SET')}'")
print(f"MEM0_MAX_CONTEXT = '{os.getenv('MEM0_MAX_CONTEXT', 'NOT SET')}'")