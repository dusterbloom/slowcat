# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slowcat is a local voice agent for macOS achieving sub-800ms voice-to-voice latency using Apple Silicon. It uses Pipecat framework with MLX for optimal performance on M-series chips. Features include multi-language support, voice recognition, music control, dictation mode, and 100% offline operation.

## Architecture

### Core Pipeline Flow
```
WebRTC Input → VAD → Audio Tee → STT → Memory/Context → LLM → TTS → WebRTC Output
                ↓
        Voice Recognition → Speaker Context
                ↓
        Music Player (with ducking)
                ↓
        Tool Execution (MCP/Local)
```

### Server Architecture (Python)

**Entry Points:**
- `server/bot_v2.py` - Main entry using modular service factory pattern
- `server/bot.py` - Legacy monolithic entry point
- `server/app.py` - FastAPI server with WebRTC endpoints
- `server/run_bot.sh` - Automated setup and launch script

**Key Components:**
- **Service Factory** (`core/service_factory.py`) - Dependency injection with lazy loading
- **Pipeline Builder** (`core/pipeline_builder.py`) - Modular pipeline construction
- **Config** (`config.py`) - Centralized configuration with dataclasses

**Services:**
- **STT**: Sherpa-ONNX streaming (`services/sherpa_streaming_stt_v2.py`), MLX Whisper (`services/whisper_stt_with_lock.py`)
- **TTS**: Kokoro MLX-based (`kokoro_tts.py`) with multi-language voices
- **LLM**: OpenAI-compatible with MCP tools (`services/llm_with_tools.py`)
- **VAD**: Silero through Pipecat framework

**Processors** (`processors/`):
- Audio tee for multi-consumer processing
- Speaker context and name management
- Voice recognition with automatic enrollment
- Local memory persistence
- Video sampling for webcam

### Client Architecture (Next.js)

- Main UI: `client/src/app/page.tsx`
- Components: `client/src/components/VoiceApp.tsx`, `StreamingText.tsx`
- Uses Pipecat client libraries for WebRTC transport

## Development Commands

### Server

```bash
# Quick start
cd server/
./run_bot.sh

# Manual setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server
python bot_v2.py                    # New modular architecture
python bot.py                        # Legacy monolithic
python bot.py --language es          # Multi-language (es, fr, ja, it, zh, pt, de)

# Testing
python -m pytest tests/              # Unit tests
python test_integration.py           # Integration tests
python test_voice_recognition.py     # Voice recognition tests
```

### Client

```bash
cd client/
npm install
npm run dev      # Development server on http://localhost:3000
npm run build    # Production build
npm run lint     # ESLint validation
```

## Important Constraints

1. **Python 3.12 or earlier** - MLX dependency requirement
2. **macOS with Apple Silicon** - Optimized for M-series chips
3. **LM Studio or OpenAI-compatible server** - For LLM responses (default: http://localhost:1234/v1)
4. **Port 7860** - Default server port
5. **Multiprocessing spawn method** - Required for Metal GPU safety

## Environment Variables

- `ENABLE_VOICE_RECOGNITION` - Enable speaker recognition (default: true)
- `ENABLE_MEMORY` - Enable conversation memory (default: true)
- `OPENAI_BASE_URL` - LLM endpoint URL
- `USE_MINIMAL_PROMPTS` - A/B test for system prompts
- `HF_HUB_OFFLINE` - Offline mode for HuggingFace
- `TRANSFORMERS_OFFLINE` - Offline mode for transformers

## Key Files

### Server
- `server/bot_v2.py` - Modular architecture entry point
- `server/core/service_factory.py` - Service dependency injection
- `server/core/pipeline_builder.py` - Pipeline construction
- `server/config.py` - Configuration management
- `server/mcp.json` - MCP tool configurations

### Client
- `client/src/app/page.tsx` - Main UI component
- `client/src/components/VoiceApp.tsx` - Voice interaction component

## Data Storage

- Speaker profiles: `server/data/speaker_profiles/`
- Conversation memory: `server/data/memory/`
- Sherpa models: `server/models/`

## Language/Voice Mapping

- English (en): af_heart
- Spanish (es): ef_dora
- French (fr): ff_siwis
- Japanese (ja): jf_alpha
- Italian (it): im_nicola
- Chinese (zh): zf_xiaobei
- Portuguese (pt): pf_dora
- German (de): am_kristina

## Special Modes

- **Music Mode**: Voice-controlled music playback with ducking
- **Dictation Mode**: Silent transcription to file
- **Video Processing**: Webcam frame sampling for visual context