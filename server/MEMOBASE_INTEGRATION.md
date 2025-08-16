# MemoBase Integration Guide

## Overview

MemoBase is now seamlessly integrated into Slowcat via the `run_bot.sh` script. The integration provides advanced semantic memory capabilities with automatic fallback to local memory when MemoBase is unavailable.

## Quick Start

### 1. Enable MemoBase in `.env`
```bash
ENABLE_MEMOBASE=true
MEMOBASE_PROJECT_URL="http://localhost:8019"
MEMOBASE_API_KEY="secret"
MEMOBASE_FALLBACK_TO_LOCAL=true
```

### 2. Run Slowcat
```bash
./run_bot.sh
```

The script will automatically:
- ✅ Install MemoBase package if needed
- ✅ Test connection to MemoBase server
- ✅ Configure fallback to local memory
- ✅ Start the voice agent with enhanced memory

## Features

### 🧠 **Semantic Memory**
- Advanced context understanding via MemoBase
- OpenAI client automatically patched for memory injection
- Conversation context preserved across sessions

### 🎙️ **Speaker Recognition Integration** 
- Per-user memory isolation
- Automatic user switching with voice recognition
- Memory follows the identified speaker

### 🔄 **Graceful Fallback**
- Works even if MemoBase server is down
- Automatic fallback to local memory mode
- No interruption to voice agent functionality

### ⚡ **Zero-Config Operation**
- Automatic package installation
- Connection testing and status reporting
- Seamless integration with existing workflow

## Memory System Priority

The `run_bot.sh` script chooses memory systems in this order:

1. **MemoBase** (if `ENABLE_MEMOBASE=true`)
   - External semantic memory service
   - Requires MemoBase server running
   - Falls back to local mode if unavailable

2. **Mem0** (if `ENABLE_MEM0=true`)
   - Local vector database with Chroma
   - Fully offline operation
   - No external dependencies

3. **Local Memory** (default)
   - Basic conversation history
   - SQLite-based storage
   - Minimal memory features

## Configuration Options

### Required Settings
```bash
ENABLE_MEMOBASE=true                    # Enable MemoBase integration
```

### Optional Settings  
```bash
MEMOBASE_PROJECT_URL="http://localhost:8019"  # MemoBase server URL
MEMOBASE_API_KEY="secret"                      # API key for authentication  
MEMOBASE_FALLBACK_TO_LOCAL=true               # Fallback when unavailable
MEMOBASE_MAX_CONTEXT_SIZE=500                 # Maximum context tokens
MEMOBASE_FLUSH_ON_SESSION_END=true            # Auto-flush on session end
```

## Script Output Example

```bash
$ ./run_bot.sh

📄 Loading environment variables from .env
🧠 MemoBase external memory system enabled
✅ MemoBase package already installed
🌐 MemoBase Configuration:
   Project URL: http://localhost:8019
   API Key: secret
   Fallback to local: true
🔍 Testing MemoBase server connection...
✅ MemoBase server is running and accessible
🧠 Using external semantic memory via MemoBase
📊 MemoBase features enabled:
   • Semantic memory with OpenAI client patching
   • Per-user memory isolation with voice recognition
   • Automatic session management and context injection

📊 Memory System Summary:
   🧠 Primary: MemoBase (external semantic memory)
   ✅ Status: Connected and ready

Starting bot with environment fixes...
```

## Starting MemoBase Server

If you see "MemoBase server not accessible", start the MemoBase server:

```bash
# Follow MemoBase documentation
# https://docs.memobase.io/quickstart
```

The voice agent will automatically detect when the server becomes available.

## Troubleshooting

### MemoBase Package Installation Issues
```bash
# Manual installation
pip install memobase>=0.1.0
```

### Connection Issues
- Check MemoBase server is running on configured URL
- Verify firewall/network settings
- The agent will fallback gracefully to local memory

### Integration Issues
```bash
# Test integration
python test_memobase_integration.py
```

## Benefits

✅ **Enhanced Memory**: Semantic understanding beyond basic history  
✅ **Zero Downtime**: Graceful fallback ensures continuous operation  
✅ **User Isolation**: Per-speaker memory with voice recognition  
✅ **Automatic Setup**: Script handles installation and configuration  
✅ **Production Ready**: Robust error handling and status reporting  

The MemoBase integration transforms Slowcat into a truly intelligent voice agent with persistent, contextual memory capabilities while maintaining the reliability and speed you expect.