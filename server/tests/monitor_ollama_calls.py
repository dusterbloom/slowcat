#!/usr/bin/env python3
"""
Monitor Ollama API calls to see if MemoBase is using it.
"""

import time
import requests
import json
from datetime import datetime

def monitor_ollama_activity():
    """Monitor Ollama for API activity"""
    print("🔍 Monitoring Ollama API activity...")
    print("   Checking http://localhost:11434")
    print("   Press Ctrl+C to stop\n")
    
    last_stats = None
    
    try:
        while True:
            try:
                # Get current running models
                response = requests.get("http://localhost:11434/api/ps", timeout=2)
                if response.status_code == 200:
                    current_stats = response.json()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Check for changes in running models
                    if current_stats != last_stats:
                        print(f"⏰ {timestamp} - Model activity detected:")
                        
                        models = current_stats.get('models', [])
                        if models:
                            for model in models:
                                name = model.get('name', 'unknown')
                                size_mb = model.get('size', 0) // (1024*1024)
                                expires = model.get('expires_at', 'unknown')
                                print(f"   🤖 {name} ({size_mb}MB) - expires: {expires}")
                        else:
                            print(f"   💤 No models currently loaded")
                        
                        last_stats = current_stats
                        print()
                else:
                    print(f"⚠️ {timestamp} - Ollama API not responding")
                    
            except requests.RequestException as e:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"❌ {timestamp} - Error connecting to Ollama: {e}")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_ollama_activity()