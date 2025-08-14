#!/usr/bin/env python3
"""
Patch Pipecat Mem0MemoryService to add debugging for storage failures
"""

# Path to the Pipecat memory service
pipecat_memory_path = ".venv/lib/python3.12/site-packages/pipecat/services/mem0/memory.py"

# Read the current content
with open(pipecat_memory_path, 'r') as f:
    content = f.read()

# Find and replace the _store_messages method to add more debugging
old_store_code = '''        try:
            logger.debug(f"Storing {len(messages)} messages in Mem0")
            params = {
                "messages": messages,
                "metadata": {"platform": "pipecat"},
                "output_format": "v1.1",
            }
            for id in ["user_id", "agent_id", "run_id"]:
                if getattr(self, id):
                    params[id] = getattr(self, id)

            if isinstance(self.memory_client, Memory):
                del params["output_format"]
            # Note: You can run this in background to avoid blocking the conversation
            self.memory_client.add(**params)
        except Exception as e:
            logger.error(f"Error storing messages in Mem0: {e}")'''

new_store_code = '''        try:
            logger.debug(f"Storing {len(messages)} messages in Mem0")
            logger.debug(f"DEBUG PIPECAT: Messages to store: {messages}")
            params = {
                "messages": messages,
                "metadata": {"platform": "pipecat"},
                "output_format": "v1.1",
            }
            for id in ["user_id", "agent_id", "run_id"]:
                if getattr(self, id):
                    params[id] = getattr(self, id)

            if isinstance(self.memory_client, Memory):
                del params["output_format"]
            
            logger.debug(f"DEBUG PIPECAT: Calling memory_client.add with params: {params}")
            logger.debug(f"DEBUG PIPECAT: Memory client type: {type(self.memory_client)}")
            
            # Note: You can run this in background to avoid blocking the conversation
            result = self.memory_client.add(**params)
            logger.debug(f"DEBUG PIPECAT: Memory storage result: {result}")
        except Exception as e:
            logger.error(f"Error storing messages in Mem0: {e}")
            import traceback
            logger.error(f"DEBUG PIPECAT: Full traceback: {traceback.format_exc()}")'''

# Apply the patch
if old_store_code in content:
    new_content = content.replace(old_store_code, new_store_code)
    
    # Write back the patched version
    with open(pipecat_memory_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Pipecat memory service debug patch applied!")
    print("This will show detailed debugging for memory storage calls.")
else:
    print("❌ Could not find the target _store_messages code to patch")
    print("The code structure might have changed.")