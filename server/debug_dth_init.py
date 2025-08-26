#!/usr/bin/env python3
"""
Debug DTH initialization in SmartContextManager
"""

import os
import sys
from pathlib import Path

# Add server path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

def debug_dth_init():
    """Debug DTH initialization step by step"""
    
    print("🔍 DTH Initialization Debug")
    print("=" * 40)
    
    # Check environment
    enable_dth = os.getenv('ENABLE_DTH', 'false')
    print(f"ENABLE_DTH env var: '{enable_dth}' -> {enable_dth.lower() == 'true'}")
    
    # Check DynamicTapeHead import
    try:
        from memory.dynamic_tape_head import DynamicTapeHead
        print(f"DynamicTapeHead import: ✅ Success")
    except Exception as e:
        print(f"DynamicTapeHead import: ❌ Failed - {e}")
        return
    
    # Check memory system
    try:
        from memory import create_smart_memory_system
        memory_system = create_smart_memory_system()
        print(f"Memory system: ✅ Created")
    except Exception as e:
        print(f"Memory system: ❌ Failed - {e}")
        return
    
    # Try DTH initialization
    try:
        tape_head = DynamicTapeHead(memory_system)
        print(f"DTH initialization: ✅ Success")
    except Exception as e:
        print(f"DTH initialization: ❌ Failed - {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎯 DTH should be working in SmartContextManager!")

if __name__ == "__main__":
    debug_dth_init()