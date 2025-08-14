#!/usr/bin/env python3
"""
Patch Mem0 to add better debugging for LLM response format issues
"""
import os

# Path to the Mem0 main.py file
mem0_main_path = ".venv/lib/python3.12/site-packages/mem0/memory/main.py"

# Read the current content
with open(mem0_main_path, 'r') as f:
    content = f.read()

# Add debugging before the JSON parsing
old_code = '''        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []'''

new_code = '''        try:
            response = remove_code_blocks(response)
            logger.debug(f"DEBUG: LLM raw response: {repr(response)}")
            parsed_response = json.loads(response)
            logger.debug(f"DEBUG: Parsed JSON keys: {list(parsed_response.keys())}")
            new_retrieved_facts = parsed_response["facts"]
            logger.debug(f"DEBUG: Extracted facts: {new_retrieved_facts}")
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            logger.error(f"DEBUG: Raw response was: {repr(response) if 'response' in locals() else 'No response'}")
            new_retrieved_facts = []'''

# Apply the patch
if old_code in content:
    new_content = content.replace(old_code, new_code)
    
    # Write back the patched version
    with open(mem0_main_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Mem0 debug patch applied successfully!")
    print("This will show exactly what the LLM is returning in the logs.")
else:
    print("❌ Could not find the target code to patch")