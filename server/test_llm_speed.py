#!/usr/bin/env python3
"""
Test LLM response speed directly
"""

import json
import time
import asyncio
from urllib.request import Request, urlopen

async def test_llm_speed():
    print("🚀 TESTING LLM RESPONSE SPEED")
    print("=" * 40)
    
    # Test different models for speed
    models_to_test = [
        "qwen/qwen3-4b",           # Current model
        "qwen3-1.7b-mlx",          # Smaller, should be faster
        "google/gemma-3-270m",     # Tiny model for ultra-fast
        "qwen2.5-0.5b-instruct-mlx"  # Very small
    ]
    
    test_message = "Hi, how are you?"
    
    for model in models_to_test:
        print(f"\n--- Testing {model} ---")
        
        # Test non-streaming first
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": test_message}
            ],
            "temperature": 0.7,
            "max_tokens": 50,  # Keep short for speed test
            "stream": False
        }
        
        try:
            start_time = time.time()
            
            url = "http://localhost:1234/v1/chat/completions"
            req = Request(url)
            req.add_header('Content-Type', 'application/json')
            data = json.dumps(payload).encode('utf-8')
            
            with urlopen(req, data=data, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            response_time = time.time() - start_time
            response_text = result['choices'][0]['message']['content']
            
            print(f"⚡ Non-streaming: {response_time:.3f}s")
            print(f"📝 Response: {response_text[:60]}...")
            
            # Test streaming version
            payload["stream"] = True
            start_time = time.time()
            first_token_time = None
            complete_time = None
            
            req = Request(url)
            req.add_header('Content-Type', 'application/json')
            data = json.dumps(payload).encode('utf-8')
            
            with urlopen(req, data=data, timeout=10) as response:
                full_response = ""
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: ') and not line.endswith('[DONE]'):
                        try:
                            chunk_data = json.loads(line[6:])  # Remove 'data: '
                            if 'choices' in chunk_data and chunk_data['choices']:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content and first_token_time is None:
                                    first_token_time = time.time() - start_time
                                full_response += content
                        except:
                            continue
                    elif line.endswith('[DONE]'):
                        complete_time = time.time() - start_time
                        break
            
            print(f"⚡ First token: {first_token_time:.3f}s")
            print(f"⚡ Complete: {complete_time:.3f}s")
            print(f"📝 Streamed: {full_response[:60]}...")
            
            # Speed assessment
            if first_token_time and first_token_time < 0.3:
                print("✅ EXCELLENT: First token <300ms")
            elif first_token_time and first_token_time < 0.5:
                print("✅ GOOD: First token <500ms") 
            else:
                print(f"⚠️  SLOW: First token {first_token_time:.3f}s")
                
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("For ultra-low latency voice agent:")
    print("- Use smallest compatible model (270m or 0.5b)")
    print("- Always stream responses") 
    print("- Target <300ms first token for natural conversation")

if __name__ == "__main__":
    asyncio.run(test_llm_speed())