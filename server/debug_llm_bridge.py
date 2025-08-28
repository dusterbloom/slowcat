#!/usr/bin/env python3
"""
Debug LLM Bridge - Check what's actually being sent to qwen models
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consciousness.core import Consciousness

async def debug_llm_integration():
    print("🔍 DEBUGGING LLM BRIDGE INTEGRATION")
    print("=" * 50)
    
    # Test with qwen3-1.7b to see what's happening
    ghost = Consciousness()
    
    try:
        from consciousness.llm_bridge import get_llm_bridge
        llm_bridge = get_llm_bridge()
        
        # Set to qwen model
        llm_bridge.model = "qwen3-1.7b-mlx"
        print(f"✅ Model set to: {llm_bridge.model}")
        print(f"📝 System prompt preview: {llm_bridge.system_prompt[:100]}...")
        
        # Test message building
        test_memories = [
            {'role': 'user', 'content': 'Hello, I am Alex'},
            {'role': 'assistant', 'content': 'Nice to meet you Alex!'}
        ]
        
        messages = llm_bridge.build_messages(
            input_text="What's my name?",
            memories=test_memories,
            symbols=["◯"],
            importance=0.7
        )
        
        print(f"\n📨 MESSAGES BEING SENT TO LLM:")
        for i, msg in enumerate(messages):
            print(f"{i+1}. {msg['role']}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
        
        # Try direct LLM call
        print(f"\n🧪 TESTING DIRECT LLM CALL...")
        
        payload = {
            "model": llm_bridge.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": True
        }
        
        print(f"📤 Payload: {json.dumps(payload, indent=2)[:300]}...")
        
        # Make the actual request
        from urllib.request import Request, urlopen
        import time
        
        url = f"{llm_bridge.base_url}/chat/completions"
        req = Request(url)
        req.add_header('Content-Type', 'application/json')
        data = json.dumps(payload).encode('utf-8')
        
        start_time = time.time()
        print(f"📡 Making request to: {url}")
        
        try:
            with urlopen(req, data=data, timeout=10) as response:
                response_time = time.time() - start_time
                print(f"⚡ Response received in: {response_time:.3f}s")
                print(f"📊 Status code: {response.status}")
                
                full_response = ""
                line_count = 0
                for line in response:
                    line = line.decode('utf-8').strip()
                    line_count += 1
                    
                    print(f"📋 Line {line_count}: {line[:150]}{'...' if len(line) > 150 else ''}")
                    
                    if line.startswith('data: ') and not line.endswith('[DONE]'):
                        try:
                            chunk_data = json.loads(line[6:])  # Remove 'data: '
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_response += content
                                    print(f"   ✅ Got content: '{content}'")
                                else:
                                    print(f"   ⚠️  Delta with no content: {delta}")
                            else:
                                print(f"   ⚠️  No choices in chunk: {chunk_data}")
                        except json.JSONDecodeError as e:
                            print(f"   ❌ JSON decode error: {e}")
                    elif line.endswith('[DONE]'):
                        print(f"   🏁 Stream finished")
                        break
                    elif line.startswith('data: '):
                        print(f"   ⚠️  Unknown data line format")
                    else:
                        print(f"   ℹ️  Non-data line")
                
                print(f"\n📝 FINAL RESPONSE: '{full_response}'")
                print(f"📏 Response length: {len(full_response)} characters")
                
                if not full_response:
                    print(f"❌ NO RESPONSE CONTENT RECEIVED!")
                else:
                    print(f"✅ Response received successfully")
                    
        except Exception as e:
            print(f"❌ Request failed: {e}")
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_llm_integration())