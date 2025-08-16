#!/usr/bin/env python3
"""
Test the new Ollama + MemoBase integration
"""
import os
import sys
import requests
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config

def test_ollama_availability():
    """Test if Ollama is available and has required models"""
    print("🦙 Testing Ollama availability...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            print("📦 Available models:")
            for name in model_names:
                print(f"   - {name}")
            
            # Check for required models
            memory_models = [name for name in model_names if 'llama3.2:1b' in name or 'llama3.2:3b' in name]
            embedding_models = [name for name in model_names if 'nomic-embed-text' in name]
            
            print("\n🔍 Checking required models:")
            if memory_models:
                print(f"✅ Memory LLM found: {memory_models[0]}")
            else:
                print("⚠️ No suitable memory LLM found (need llama3.2:1b or llama3.2:3b)")
                
            if embedding_models:
                print(f"✅ Embedding model found: {embedding_models[0]}")
            else:
                print("⚠️ Embedding model not found (need nomic-embed-text)")
                
            return len(memory_models) > 0 and len(embedding_models) > 0
        else:
            print("❌ Ollama is not responding correctly")
            return False
    except requests.RequestException:
        print("❌ Ollama is not running")
        return False

def test_lmstudio_availability():
    """Test if LMStudio is available"""
    print("\n🏭 Testing LMStudio availability...")
    
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json().get('data', [])
            if models:
                print("✅ LMStudio is running")
                print("🤖 Available models:")
                for model in models:
                    print(f"   - {model.get('id', 'unknown')}")
                return True
            else:
                print("⚠️ LMStudio is running but no models loaded")
                return False
        else:
            print("❌ LMStudio is not responding correctly")
            return False
    except requests.RequestException:
        print("❌ LMStudio is not running")
        return False

def test_configuration():
    """Test the MemoBase configuration"""
    print("\n⚙️ Testing MemoBase configuration...")
    
    # Test current config
    print(f"📋 Current configuration:")
    print(f"   Main LLM URL: {config.network.llm_base_url}")
    print(f"   Main LLM Model: {config.models.default_llm_model}")
    print(f"   MemoBase enabled: {config.memobase.enabled}")
    
    # Test provider configs if available
    if hasattr(config.memobase, 'memory_llm'):
        print(f"   Memory LLM URL: {config.memobase.memory_llm.base_url}")
        print(f"   Memory LLM Model: {config.memobase.memory_llm.model}")
        print(f"   Embedding URL: {config.memobase.embedding.base_url}")
        print(f"   Embedding Model: {config.memobase.embedding.model}")
    
    return True

def test_environment_override():
    """Test environment variable override for Ollama"""
    print("\n🌍 Testing environment variable override...")
    
    # Set Ollama environment variables
    os.environ['MEMOBASE_LLM_BASE_URL'] = "http://localhost:11434/v1"
    os.environ['MEMOBASE_LLM_MODEL'] = "llama3.2:1b"
    os.environ['MEMOBASE_EMBEDDING_BASE_URL'] = "http://localhost:11434/v1"
    os.environ['MEMOBASE_EMBEDDING_MODEL'] = "nomic-embed-text"
    
    print("✅ Environment variables set for Ollama mode:")
    print(f"   MEMOBASE_LLM_BASE_URL={os.environ['MEMOBASE_LLM_BASE_URL']}")
    print(f"   MEMOBASE_LLM_MODEL={os.environ['MEMOBASE_LLM_MODEL']}")
    print(f"   MEMOBASE_EMBEDDING_BASE_URL={os.environ['MEMOBASE_EMBEDDING_BASE_URL']}")
    print(f"   MEMOBASE_EMBEDDING_MODEL={os.environ['MEMOBASE_EMBEDDING_MODEL']}")
    
    # Reload config to pick up environment variables
    from config import Config
    test_config = Config()
    
    print("\n📋 Reloaded configuration:")
    print(f"   Memory LLM URL: {test_config.memobase.memory_llm.base_url}")
    print(f"   Memory LLM Model: {test_config.memobase.memory_llm.model}")
    print(f"   Embedding URL: {test_config.memobase.embedding.base_url}")
    print(f"   Embedding Model: {test_config.memobase.embedding.model}")
    
    return True

def main():
    print("🚀 Testing Ollama + MemoBase Integration\n")
    
    # Run tests
    ollama_ok = test_ollama_availability()
    lmstudio_ok = test_lmstudio_availability()
    config_ok = test_configuration()
    env_ok = test_environment_override()
    
    print("\n📊 Test Results:")
    print(f"   Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"   LMStudio: {'✅' if lmstudio_ok else '❌'}")
    print(f"   Configuration: {'✅' if config_ok else '❌'}")
    print(f"   Environment Override: {'✅' if env_ok else '❌'}")
    
    if ollama_ok and lmstudio_ok:
        print("\n🎉 All systems ready for hybrid MemoBase!")
        print("\n💡 Usage:")
        print("   Normal mode:     python bot_v2.py")
        print("   Ollama mode:     python bot_v2.py --ollama-memo")
        print("   With MemoBase:   ENABLE_MEMOBASE=true python bot_v2.py --ollama-memo")
        return True
    else:
        print("\n⚠️ Some systems not ready. Check the issues above.")
        if not lmstudio_ok:
            print("   Start LMStudio and load a model")
        if not ollama_ok:
            print("   Start Ollama: ollama serve")
            print("   Pull models: ollama pull llama3.2:1b && ollama pull nomic-embed-text")
        return False

if __name__ == "__main__":
    main()