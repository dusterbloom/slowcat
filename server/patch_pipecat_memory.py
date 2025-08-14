#!/usr/bin/env python3
"""
Patch Pipecat's Mem0MemoryService to fix the memory formatting bug
"""

import os

def patch_pipecat_memory_service():
    """
    Fix the bug in Pipecat's memory service where memories["results"] 
    might not contain the expected structure.
    """
    
    memory_service_path = ".venv/lib/python3.12/site-packages/pipecat/services/mem0/memory.py"
    
    if not os.path.exists(memory_service_path):
        print(f"❌ Memory service file not found at {memory_service_path}")
        return False
        
    # Read the file
    with open(memory_service_path, 'r') as f:
        content = f.read()
    
    # Find the problematic code section
    old_code = '''        # Format memories as a message
        memory_text = self.system_prompt
        for i, memory in enumerate(memories["results"], 1):
            memory_text += f"{i}. {memory.get('memory', '')}\\n\\n"'''
            
    new_code = '''        # Format memories as a message
        memory_text = self.system_prompt
        
        # Handle different memory result structures
        memory_results = []
        if isinstance(memories, dict) and "results" in memories:
            memory_results = memories["results"]
        elif isinstance(memories, list):
            memory_results = memories
        
        # Format each memory
        for i, memory in enumerate(memory_results, 1):
            memory_content = ""
            if isinstance(memory, dict):
                memory_content = memory.get('memory', memory.get('text', str(memory)))
            else:
                memory_content = str(memory)
            
            if memory_content.strip():
                memory_text += f"{i}. {memory_content}\n\n"'''
    
    if old_code not in content:
        print("❌ Could not find the target code section to patch")
        return False
    
    # Apply the patch
    patched_content = content.replace(old_code, new_code)
    
    # Create backup
    backup_path = memory_service_path + ".backup"
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Write patched version
    with open(memory_service_path, 'w') as f:
        f.write(patched_content)
    
    print("✅ Successfully patched Pipecat memory service")
    print(f"   Backup saved to {backup_path}")
    return True

if __name__ == "__main__":
    success = patch_pipecat_memory_service()
    if success:
        print("\n🎉 Memory service patched! The 'Previously you learned:' section should now work correctly.")
    else:
        print("\n🚨 Patch failed. Memory injection may still be broken.")