# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slowcat is a local voice agent for macOS achieving sub-800ms voice-to-voice latency using Apple Silicon. It uses Pipecat framework with MLX for optimal performance on M-series chips.

## Development Commands

### Server (Python)

**Setup and Run**:
```bash
cd server/
./run_bot.sh  # Automated setup and run

# Manual setup:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py
```

**Multi-language Support**:
```bash
python bot.py --language es  # Spanish, French (fr), Japanese (ja), etc.
```

**Environment Variables**:
- `ENABLE_VOICE_RECOGNITION`: Enable/disable speaker recognition (default: true)
- `ENABLE_MEMORY`: Enable/disable local conversation memory (default: true)
- `ENABLE_MEM0`: Enable/disable advanced Mem0 semantic memory (default: false)
- `OPENAI_BASE_URL`: LLM endpoint (default: http://localhost:1234/v1)

### Client (Next.js)

**Development**:
```bash
cd client/
npm install
npm run dev    # Development server on http://localhost:3000
npm run build  # Production build
npm run lint   # Run ESLint
```

## Architecture

### Core Pipeline (server/bot.py)
1. **WebRTC Transport** → Audio/Video streams
2. **Silero VAD** → Voice activity detection
3. **MLX Whisper** → Speech-to-text
4. **LLM Service** → Response generation
5. **Kokoro TTS** → Text-to-speech

### Custom Processors (server/processors/)
- `vad_event_bridge.py`: Bridges VAD events to speaker recognition
- `audio_tee.py`: Multi-consumer audio processing
- `speaker_context_manager.py`: Manages speaker identification
- `video_sampler.py`: Webcam frame sampling
- `speaker_name_manager.py`: Speaker name persistence

### Voice Recognition (server/voice_recognition/)
- Automatic speaker enrollment
- Real-time identification using Resemblyzer
- Profiles stored in `server/data/speaker_profiles/`
- Adjusted thresholds for single-speaker scenarios

### Local Memory (server/processors/)
- Persistent conversation history without cloud services
- Per-speaker memory when voice recognition is enabled
- Stores last 200 conversations, includes last 10 in context
- Memory files in `server/data/memory/`

## Important Constraints

1. **Python 3.12 or earlier required** - MLX dependency
2. **macOS with Apple Silicon required** - Optimized for M-series chips
3. **LM Studio or OpenAI-compatible server required** - For LLM responses
4. **Port 7860** - Server default port

## Testing

```bash
cd server/
python test_integration.py      # Integration tests
python test_voice_recognition.py # Voice recognition tests
```

## Key Files

- `server/bot.py`: Main pipeline implementation
- `server/processors/`: Custom processor implementations
- `client/src/app/page.tsx`: Main UI component
- `server/voice_recognition/voice_identifier.py`: Speaker recognition logic

## Language/Voice Mapping

- English (en): af_heart
- Spanish (es): ef_dora
- French (fr): ff_siwis
- Japanese (ja): jf_alpha
- Italian (it): im_nicola
- Chinese (zh): zf_xiaobei
- Portuguese (pt): pf_dora