# Memory Search Troubleshooting Guide

## ✅ What's Working
1. **Database Storage** - All conversations are being saved to SQLite
2. **Search Functions** - The search_conversations tool works when called directly
3. **Tool Registration** - The tools are properly registered and available

## ❌ The Problem
The LLM is **not calling the search_conversations tool** when asked to recall information.

## 🔍 Root Causes

### 1. Local LLM Function Calling
Many local LLMs have poor function calling support. Check your LM Studio model:
- Some models like Llama don't support function calling well
- Models like Mistral or Hermes often work better
- GPT-based models have the best function calling

### 2. System Prompt Not Loaded
The updated system prompt with memory search instructions won't take effect until you restart the bot.

## 🛠️ Solutions

### Option 1: Restart the Bot
```bash
# Stop the bot (Ctrl+C) and restart it
./run_bot.sh
```

### Option 2: Use a Better Model
In LM Studio, try models like:
- `mistral-7b-instruct`
- `openhermes-2.5-mistral-7b`
- `functionary-small-v2.5`
- Any model that mentions "function calling" support

### Option 3: Test with Direct Commands
Try very explicit commands:
- "Use the search_conversations tool to find my name"
- "Search conversations for Becpe"
- "Call search_conversations with query name"

### Option 4: Check Model Temperature
Lower temperature (0.1-0.3) often helps with tool use.

## 🧪 Testing Memory Search

Run this test to verify everything works:
```bash
python test_memory_search.py
```

## 📝 Manual Database Check
```bash
# See what's stored
sqlite3 data/memory/memory.sqlite "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 10;"

# Search for specific content
sqlite3 data/memory/memory.sqlite "SELECT * FROM conversations WHERE content LIKE '%Becpe%';"
```

## 🔧 Debug the LLM Response
Add this to config.py to see what the LLM is thinking:
```python
llm_params = {
    # ... existing params ...
    "debug": True,  # This will show tool calls in logs
}
```