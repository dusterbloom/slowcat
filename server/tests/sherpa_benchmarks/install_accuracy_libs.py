#!/usr/bin/env python3
"""
Install required libraries for Advanced Accuracy Enhancement

This script installs the necessary NLP and phonetic libraries
for the accuracy enhancement system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def check_library(lib_name, import_name=None):
    """Check if a library is installed"""
    if import_name is None:
        import_name = lib_name
    
    try:
        __import__(import_name)
        print(f"✅ {lib_name} is installed")
        return True
    except ImportError:
        print(f"❌ {lib_name} is not installed")
        return False

def main():
    print("🚀 Installing Advanced Accuracy Enhancement Libraries")
    print("="*60)
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("⚠️  Warning: Not in a virtual environment. Consider activating venv first.")
    
    # Core NLP libraries
    libraries = [
        ("spacy", "pip install spacy"),
        ("jellyfish", "pip install jellyfish"),
        ("fuzzywuzzy", "pip install fuzzywuzzy"),
        ("python-Levenshtein", "pip install python-Levenshtein"),
        ("editdistance", "pip install editdistance"),
    ]
    
    # Optional phonetic libraries
    optional_libraries = [
        ("metaphone", "pip install metaphone"),
        ("phonetics", "pip install phonetics"),
        ("ollama", "pip install ollama"),
    ]
    
    print("Installing core libraries...")
    failed_installs = []
    
    for lib_name, install_cmd in libraries:
        if not check_library(lib_name):
            if not run_command(install_cmd, f"Installing {lib_name}"):
                failed_installs.append(lib_name)
    
    print("\nInstalling optional libraries...")
    for lib_name, install_cmd in optional_libraries:
        if not check_library(lib_name):
            if not run_command(install_cmd, f"Installing {lib_name}"):
                print(f"⚠️  Optional library {lib_name} failed to install (not critical)")
    
    # Install spaCy model
    print("\nInstalling spaCy English model...")
    spacy_models = [
        "en_core_web_sm",
        "en_core_web_md"  # Fallback
    ]
    
    model_installed = False
    for model in spacy_models:
        if run_command(f"python -m spacy download {model}", f"Installing spaCy model {model}"):
            model_installed = True
            break
    
    if not model_installed:
        print("⚠️  Could not install any spaCy models. NER functionality will be limited.")
    
    # Final check
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    
    print("\nCore libraries:")
    all_core_installed = True
    for lib_name, _ in libraries:
        if check_library(lib_name):
            continue
        else:
            all_core_installed = False
    
    print("\nOptional libraries:")
    for lib_name, _ in optional_libraries:
        check_library(lib_name)
    
    if failed_installs:
        print(f"\n❌ Failed to install: {', '.join(failed_installs)}")
        print("You may need to install these manually or check for system dependencies.")
    
    if all_core_installed:
        print("\n🎉 Core installation successful! You can now run the accuracy enhancement tests.")
        print("\nNext steps:")
        print("1. python test_accuracy_enhancement.py")
        print("2. Test with your own text:")
        print("   python advanced_accuracy_enhancer.py --text 'your test text here'")
    else:
        print("\n⚠️  Some core libraries failed to install. The system may have limited functionality.")
    
    print("\n💡 Tips:")
    print("- Make sure you're in the activated virtual environment")
    print("- On macOS, you may need: xcode-select --install")
    print("- For Ollama integration, install Ollama separately from https://ollama.ai")

if __name__ == "__main__":
    main()