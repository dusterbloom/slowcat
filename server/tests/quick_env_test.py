#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv()

print(f"OPENAI_API_KEY = '{os.getenv('OPENAI_API_KEY')}'")
print(f"OPENAI_BASE_URL = '{os.getenv('OPENAI_BASE_URL')}'")

# Test direct OpenAI client
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL')
)

try:
    response = client.chat.completions.create(
        model="qwen3-4b-instruct-2507",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )
    print("✅ OpenAI client works!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ OpenAI client failed: {e}")