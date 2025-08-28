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

### Step 2: Configure and Start the Server
```bash
cd server/
cp .env.example .env   # Surreal-first defaults; edit if needed
./run_bot.sh
```

### Optional: Faster Responses with Speculative Decoding

If you use LM Studio, you can pair a fast draft model with your main model to speed up responses while maintaining quality.

Examples:
- Main: `qwen2.5-7b-instruct-1ms-dynamic-dwq`
- Draft: `qwen2.5-0.5b-instruct-mlx`

Run:
```bash
cd server
./run_bot.sh --llm qwen2.5-7b-instruct-1ms-dynamic-dwq --draft qwen2.5-0.5b-instruct-mlx
```

Environment shortcuts:
- `DRAFT_MODEL=qwen2.5-0.5b-instruct-mlx` (auto-passed as `--draft`)
- Or set `ENABLE_SPECULATIVE=true` and `SPECULATIVE_DRAFT_MODEL=qwen2.5-0.5b-instruct-mlx`
- Optional tuning: `SPECULATIVE_MAX_DRAFT_TOKENS=64`, `SPECULATIVE_MIN_DRAFT_TOKENS=16`

Notes:
- Intended for LM Studio at `http://localhost:1234/v1` (default).
- Slowcat automatically skips speculative decoding on non-local endpoints to avoid provider errors.
- Memory backend defaults to SurrealDB graph. To switch, set `MEMORY_BACKEND=sqlite` or `stateless` in `.env`.

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
- **Graph Memory (SurrealDB)**: Default memory system (sessions/messages/concepts)
- **Smart Context Manager**: Read-only context assembly; Graph Writer handles persistence

See also: `docs/SURREAL_GRAPH.md` for graph schema, verified query idioms, and troubleshooting.

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

### 🎚️ Voice Mood Capture (experimental)

Capture lightweight mood features (energy, pitch, arousal) from voice and store them alongside tape entries.

How to wire:
```python
from processors.audio_tee import AudioTeeProcessor
from processors.vad_event_bridge import VADEventBridge
from processors.mood_analyzer import attach_mood_analyzer
from memory import create_smart_memory_system

# Assuming you already created tee and vad_bridge in your pipeline
memory = create_smart_memory_system()
analyzer = attach_mood_analyzer(tee, vad_bridge, memory.tape_store, sample_rate=16000)
```

Results are stored in SQLite `entry_meta` (or in SurrealDB `tape.metadata` when using that backend).

### Dynamic Tape Head (optional)

- Enable intelligent memory selection in the context manager:

```bash
export ENABLE_DTH=true
```

- Semantic recall (KNN) via SurrealDB
  - SurrealDB memory is the default — no flag required.
  - Tape embeddings are enabled by default. To be explicit or change models:

```bash
export SURREAL_EMBED_TAPE=true
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

- Optional cross‑encoder reranker (if installed):

```bash
export USE_CROSS_ENCODER=true
export CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Tuning knobs live in `server/config/tape_head_policy.json` (`knn_k`, `knn_scan_recent`, `min_time_gap_s`, etc.).

Reranker model availability
- We include `sentence-transformers` in `server/requirements.txt`. On first use, models download from HuggingFace to the local cache (`~/.cache/huggingface`).
- To pre‑download for offline runs:

```bash
python - << 'PY'
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('Cross‑encoder cached')
PY
# Optional: control cache path
# export HF_HOME=/path/to/hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME/hub
```

Note: `server/run_bot.sh` will automatically attempt to fetch the embedding and (if enabled) cross‑encoder models on first run when `ENABLE_DTH=true`, so it works out of the box. If there is no network access, it continues without the reranker and logs a gentle warning; you can pre‑cache models as shown above.

Alternatively, use the helper script to pre-cache both models:

```bash
cd server
python scripts/precache_models.py --enable-cross-encoder
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

## ⚙️ Surreal-first Environment (Recommended)

To run with SurrealDB memory by default and minimal configuration:

1) Copy the minimal template and adjust if needed

```bash
cp server/.env.example.surreal server/.env
```

2) Defaults
- SurrealDB memory is the default (no `USE_SURREALDB` flag required)
- Tape embeddings are enabled (`SURREAL_EMBED_TAPE=true`) for semantic recall
- Non-user facts are allowed by default; cap with `FACTS_MAX_NONUSER` (default 6)
- Quiet STT/TTS streaming logs by default (see `STT_*` and `TTS_*` flags)

Compatibility:
- Old flags like `USE_SURREALDB` are deprecated and ignored (Surreal is already the default)
- To revert to user-only facts, set `FACTS_ONLY_USER_SUBJECT=true` (not recommended)

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
