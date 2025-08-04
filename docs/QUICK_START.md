# Quick Start Guide for New Contributors

Welcome to Slowcat! This guide will get you up and running with the latest features in under 10 minutes.

## 🚀 5-Minute Setup

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- Node.js 18+ (for web client)
- LM Studio (for local LLM)

### Step 1: Clone and Setup
```bash
git clone <repository-url>
cd slowcat
```

### Step 2: Start the Server
```bash
cd server/
./run_bot.sh
```

### Step 3: Start the Web Client
```bash
cd client/
npm install
npm run dev
```

### Step 4: Connect
- Open the URL shown in terminal (usually http://localhost:5173)
- Allow microphone access
- Start talking!

## 🎯 New Features to Try

### 🎵 Music Mode
1. Say "music mode" to activate
2. Try commands:
   - "Play some jazz"
   - "Turn up the volume"
   - "What's playing?"
   - "Skip to next song"
3. Say "stop music mode" to exit

### 📝 Dictation Mode
1. Say "dictation mode" to start
2. Speak naturally - no bot responses
3. Say "stop dictation" to get full transcript

## 🏗️ Architecture Overview

### New Structure (Post-Refactoring)
```
slowcat/
├── server/
│   ├── bot.py           # Original (still works)
│   ├── bot_v2.py        # New modular version
│   ├── core/            # Core architecture
│   │   ├── service_factory.py
│   │   ├── pipeline_builder.py
│   │   └── service_interfaces.py
│   ├── server/          # Web server components
│   └── processors/      # Voice processing modules
├── client/              # React web interface
└── docs/               # Documentation
```

### Key Components
- **Service Factory**: Manages all AI services (STT, LLM, TTS)
- **Pipeline Builder**: Constructs processing pipelines
- **Processors**: Handle specific tasks (music, dictation, voice recognition)

## 🔧 Development Setup

### For Core Development
```bash
cd server/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run with debugging
python bot_v2.py --debug --language en
```

### For Web Client Development
```bash
cd client/
npm install
npm run dev
```

## 🧪 Testing New Features

### Test Music Mode
```python
# Test music library scanning
from server.music.library_scanner import MusicLibraryScanner
scanner = MusicLibraryScanner("/path/to/music")
songs = scanner.scan_library()
print(f"Found {len(songs)} songs")
```

### Test Dictation Mode
```python
# Test dictation processor
from server.processors.dictation_mode import DictationModeProcessor
processor = DictationModeProcessor()
# See docs/REFACTORING_GUIDE.md for full examples
```

## 📚 Essential Documentation

### For New Contributors
1. **[Architecture Guide](REFACTORING_GUIDE.md)** - Understanding the new structure
2. **[MCP Setup](MCP_SETUP.md)** - Adding new capabilities
3. **[Memory System](MEMORY_TROUBLESHOOTING.md)** - Working with conversation memory
4. **[Performance Guide](QUICK_FIX.md)** - Optimization tips

### API References
- **Service Factory**: `core/service_factory.py`
- **Pipeline Builder**: `core/pipeline_builder.py`
- **Music Tools**: `tools/music_tools.py`
- **Dictation Tools**: `tools/time_tools.py`

## 🐛 Common Issues

### "Module not found" errors
```bash
cd server/
pip install -r requirements.txt
```

### LM Studio not connecting
1. Open LM Studio
2. Go to Developer tab
3. Start Local Server
4. Check port in server/config.py

### Microphone permissions
- System Preferences → Security & Privacy → Microphone → Allow Terminal

## 🎤 Voice Commands Reference

### General Commands
- "Hello" - Basic greeting
- "What can you do?" - Feature overview
- "Music mode" - Start music control
- "Dictation mode" - Start transcription

### Music Commands (in music mode)
- "Play [artist/song/genre]"
- "Pause"
- "Resume"
- "Skip"
- "Volume up/down"
- "What's playing?"

### Dictation Commands (in dictation mode)
- Just speak naturally
- "Stop dictation" to finish

## 🔄 Contributing

### Quick Contribution Flow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Test your changes: `python -m pytest tests/`
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Code Style
- Follow existing patterns in `core/`
- Add tests for new features
- Update documentation
- Use type hints

## 📞 Getting Help

### Quick Debug
```bash
# Check if everything works
python server/tests/test_integration.py

# Test voice recognition
python server/tests/test_voice_recognition.py

# Check music system
python server/tests/test_tools.py
```

### Community
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Real-time: Check README for community links

## 🎉 Next Steps

1. **Try the features**: Test music and dictation modes
2. **Read the architecture**: Understand the new modular design
3. **Explore the code**: Start with `bot_v2.py` and `core/`
4. **Join the community**: Contribute improvements and features

Welcome to the Slowcat community! 🐱