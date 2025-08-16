#!/usr/bin/env python3
"""
Test Ollama memory configuration for the --ollama-memo flag.
"""

import sys
from pathlib import Path

# Add server directory to path
sys.path.append(str(Path(__file__).parent))

from config import config

def test_ollama_config():
    """Test if Ollama configuration is properly set for memory operations."""
    print("🔍 Testing Ollama memory configuration...")
    
    print(f"📊 MemoBase enabled: {config.memobase.enabled}")
    print(f"🔧 Memory LLM base URL: {config.memobase.memory_llm.base_url}")
    print(f"🤖 Memory LLM model: {config.memobase.memory_llm.model}")
    print(f"🏢 Memory LLM provider: {config.memobase.memory_llm.provider_name}")
    print(f"🔑 Memory LLM API key: {config.memobase.memory_llm.api_key}")
    print()
    
    print(f"🔍 Embedding base URL: {config.memobase.embedding.base_url}")
    print(f"🧲 Embedding model: {config.memobase.embedding.model}")
    print(f"🏢 Embedding provider: {config.memobase.embedding.provider_name}")
    print()
    
    print(f"🌐 Main LLM base URL: {config.network.llm_base_url}")
    print(f"🤖 Main LLM model: {config.models.default_llm_model}")
    print()
    
    # Check if we're using separate providers
    using_separate_memory_llm = (
        config.memobase.memory_llm.base_url != config.network.llm_base_url or
        config.memobase.memory_llm.model != config.models.default_llm_model
    )
    
    print(f"🔄 Using separate Memory LLM: {using_separate_memory_llm}")
    
    if using_separate_memory_llm:
        print("✅ Ollama configuration detected for memory operations")
    else:
        print("ℹ️ Using same LLM for main and memory operations")

if __name__ == "__main__":
    test_ollama_config()