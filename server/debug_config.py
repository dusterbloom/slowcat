#!/usr/bin/env python3

"""Debug what the bot actually sees for configuration"""

import os
from dotenv import load_dotenv

print("=== Before loading .env ===")
print(f"ENABLE_MEMOBASE: {os.getenv('ENABLE_MEMOBASE', 'NOT SET')}")
print(f"ENABLE_MEM0: {os.getenv('ENABLE_MEM0', 'NOT SET')}")
print(f"ENABLE_MEMORY: {os.getenv('ENABLE_MEMORY', 'NOT SET')}")

print("\n=== Loading .env file ===")
load_dotenv()

print(f"ENABLE_MEMOBASE: {os.getenv('ENABLE_MEMOBASE', 'NOT SET')}")
print(f"ENABLE_MEM0: {os.getenv('ENABLE_MEM0', 'NOT SET')}")
print(f"ENABLE_MEMORY: {os.getenv('ENABLE_MEMORY', 'NOT SET')}")

print("\n=== Config object ===")
import config
import importlib
importlib.reload(config)

print(f"config.memobase.enabled: {config.config.memobase.enabled}")
print(f"config.memory.enabled: {config.config.memory.enabled}")

print("\n=== Service Factory Test ===")
from core.service_factory import ServiceFactory

class MockArgs:
    def __init__(self):
        self.host = "localhost"
        self.port = 7860
        self.language = "en"
        self.llm_model = None
        self.stt = None
        self.mode = "server"

args = MockArgs()
factory = ServiceFactory(args)

# Test the memory service creation logic
enable_memobase = config.config.memobase.enabled
enable_mem0 = os.getenv("ENABLE_MEM0", "false").lower() == "true"

print(f"\nMemory service selection:")
print(f"  enable_memobase: {enable_memobase}")
print(f"  enable_mem0: {enable_mem0}")

if enable_memobase:
    print("  -> Should create MemoBase processor")
elif enable_mem0:
    print("  -> Should create Mem0 service")
else:
    print("  -> No memory service will be created")