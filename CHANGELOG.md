# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-08-21

### 🚀 SurrealDB Memory System (Revolutionary Upgrade)
- **Multi-model Database** - Single SurrealDB replaces 3 separate SQLite databases for unified memory management
- **Drop-in Compatibility** - Seamless replacement maintaining exact same interfaces as existing memory system
- **JWT Authentication** - Secure authentication with persistent credentials and proper error handling
- **Time-travel Queries** - Natural language date parsing for conversational history ("what did we discuss last Tuesday?")
- **Graph Relationships** - Advanced fact storage with natural decay and relationship modeling
- **Apple Silicon Optimized** - Rust-based SurrealDB provides superior performance on M-series chips
- **Environment Toggle** - Instant rollback capability via `USE_SURREALDB=true/false` environment variable
- **Automated Lifecycle** - SurrealDB server automatically started/stopped by `run_bot.sh` with proper cleanup
- **Comprehensive Testing** - Full test suite covering authentication, data persistence, and performance
- **Real-time Operations** - Live async connections with concurrent operations and connection pooling

### 🎵 Music Mode (Major Feature)
- **Local AI DJ functionality** - Complete voice-controlled music system with local file library integration
- **Smart filtering** - Only music commands processed in music mode, ignoring general conversation
- **Minimal interaction** - Bot stays quiet during music mode, brief emoji responses only
- **Voice commands** - Play, pause, skip, volume control, song identification, and playlist management
- **Library integration** - Automatic scanning and indexing of local music files with metadata extraction
- **Multiple audio players** - Both simple and advanced audio player implementations with real device control
- **DJ voice processing** - Specialized voice processing for music-related commands

### 📝 Dictation Mode (Major Feature)
- **Professional transcription** - Voice-to-text without bot interruptions or responses
- **Silent operation** - No bot responses during dictation sessions
- **Multi-language support** - Works with all supported languages for transcription
- **Session-based control** - Say "dictation mode" to start, "stop dictation" to exit
- **Time-aware processing** - Intelligent handling of dictation timing and context

### 🏗️ Architecture Refactoring (Major)
- **Modular design** - Complete separation into `core/`, `server/`, `processors/` directories
- **Service Factory** - Dependency injection system with service management
- **Pipeline Builder** - Reusable pipeline construction framework
- **Service Interfaces** - Clean abstraction layers for all major components
- **Backward compatibility** - `bot_v2.py` maintains identical interface while using new architecture
- **Test organization** - Proper test structure with unit and integration tests

### 🎯 Voice Recognition Improvements
- **Enhanced auto-enrollment** - Faster speaker identification requiring only 3 utterances
- **Improved accuracy** - Better audio preprocessing and quality detection algorithms
- **Persistent profiles** - Speaker data saved across sessions with profile management
- **Performance optimization** - Reduced memory usage and faster processing times
- **Audio quality detection** - Automatic assessment of audio input quality
- **Advanced preprocessing** - Sophisticated audio preprocessing pipeline

### 🔧 Tool Integration (Major)
- **MCP Server Support** - Complete Model Context Protocol integration
- **Custom tool definitions** - Extensible tool system with definitions and handlers
- **File system tools** - Desktop file reading, listing, and management capabilities
- **Search integration** - Brave Search API integration for web queries
- **Memory tools** - Advanced memory search and context injection
- **Music tools** - Specialized tools for music library management and playback
- **Time tools** - Time-aware processing and scheduling capabilities

### 📁 File Structure Reorganization
- **Documentation reorganization** - All documentation moved to dedicated `docs/` folder
- **Test reorganization** - Tests moved to `tests/` with proper structure and organization
- **Core modules** - New `core/` directory for architecture components
- **Server modules** - Dedicated `server/` directory for web components
- **Processor modules** - Organized processors for different functionalities

### ⚡ Performance Optimizations
- **Sub-800ms latency** - Optimized for Apple Silicon with MLX integration
- **Lazy loading** - Models loaded on-demand with warmup capabilities
- **Streaming TTS** - Kokoro TTS with streaming support for faster response
- **Memory management** - Efficient resource usage and garbage collection
- **Background loading** - ML modules load asynchronously without blocking

### 🐛 Bug Fixes
- **Kokoro threading** - Fixed threading issues with Kokoro TTS
- **Offline mode** - Hugging Face models work completely offline
- **Auto-enrollment** - Improved speaker auto-enrollment accuracy
- **Voice recognition** - Enhanced voice recognition reliability
- **Memory search** - Fixed memory search functionality
- **Tool integration** - Resolved various tool integration issues

## [2025-08-03] - Voice Recognition & Multi-language Support

### 🎯 Voice Recognition System
- **Automatic speaker enrollment** - Learns speakers from 3+ utterances with auto-enrollment
- **Real-time identification** - Identifies speakers during conversation flow
- **Local processing** - Complete offline operation with privacy-first approach
- **Profile persistence** - Remembers speakers across sessions with profile management
- **Enhanced accuracy** - Improved recognition algorithms and preprocessing

### 🌍 Multi-language Support
- **8 languages supported** - English, Spanish, French, German, Japanese, Italian, Chinese, Portuguese
- **Automatic voice selection** - Appropriate voices selected per language automatically
- **Whisper integration** - Multi-language speech recognition with Whisper
- **Kokoro TTS** - Multi-language speech synthesis with Kokoro
- **Language detection** - Automatic language detection and switching

### 🔧 Configuration Improvements
- **Enhanced config system** - Comprehensive configuration management with validation
- **Environment variables** - Full environment variable support for configuration
- **Flexible setup** - Easy configuration for different deployment scenarios
- **Documentation** - Complete configuration documentation and examples

## [2025-08-02] - Core Features & Tool Integration

### 🚀 Core Features
- **Local voice agent** - Complete offline operation without cloud dependencies
- **WebRTC transport** - Low-latency peer-to-peer communication system
- **Pipecat framework** - Modular voice AI pipeline architecture
- **React client** - Web-based user interface with real-time updates
- **Tool integration** - MCP server support for external tool integration

### 🛠️ Development Tools
- **Custom WhisperMLX** - Custom Whisper implementation with MLX acceleration
- **File tools** - Desktop file reading, listing, and management tools
- **Search tools** - Brave Search API integration for web queries
- **Memory tools** - Advanced memory search and context injection
- **Testing framework** - Comprehensive testing suite with unit and integration tests

### 📊 Performance Monitoring
- **UI metrics** - Real-time performance monitoring and metrics
- **Custom STT** - Custom speech-to-text with performance optimization
- **Auto-enrollment** - Automatic speaker enrollment with performance tuning
- **Integration tests** - Complete integration testing framework

## [2025-08-20] - Resilient Reconnects, Fixed Context Memory, and Noise Guards

### 🏗️ Architecture & Pipeline
- Robust RTVI handling: supports both `client_ready` and `on_client_ready`, plus `client_disconnected`; adds 2s auto-ready fallback to prevent stalls after fast reconnects.
- Initial context delivery is idempotent and sent once per session (via event or fallback), eliminating “silent” pipelines.
- Idle timeout is env-configurable (`PIPELINE_IDLE_TIMEOUT_SECS`); default disabled to avoid mid-session cancellation.
- GreetingFilter wired into the pipeline to strip verbose LLM introductions and keep replies concise.

### 🎤 STT / 🔊 TTS
- Sherpa OnlineRecognizer (true streaming) with optional punctuation; normalized final transcripts (spaces around punctuation, collapsed repeats).
- Short-final debounce prevents premature replies when punctuation causes early endpoints (`SHERPA_MIN_FINAL_WORDS`, `SHERPA_ENDPOINT_DEBOUNCE_MS`).
- Sensible defaults for endpoint detection (`SHERPA_RULE2_MIN_TRAILING_SILENCE=1.0`).
- Streaming Kokoro TTS preserved; subsecond TTFB warm once initialized.

### 🧠 Memory & Context (Fixed Budget)
- SmartContextManager replaces unbounded context with a fixed token budget and dynamic allocation.
- Compact system prompt + tone guidance; added Interaction Rules for small models (no verbatim echo, join spelled letters, at most one follow-up, no unsolicited capability lists).
- Previous session summary injected on connect; running summary updates mid-session.
- Relevant Facts strictly structured; Conversation Snippets separated and budgeted.
- Recent-turn pairing tracked; dynamic recent window sized by leftover budget.
- Session metadata always present; session counting corrected with `start_session()` (increments once per session) and `update_session()` (no longer bumps session count).

### 🗂️ TapeStore & Summaries
- Noise guard enabled by default: persist a turn only if it’s a question, or has ≥3 alpha words, or length ≥15 chars (tunable via `TAPE_MIN_USEFUL_WORDS`, `TAPE_MIN_USEFUL_LEN`).
- Disconnect summaries improved: filters boilerplate intros and very low-signal lines for compact orientation on next session.

### 🔌 LLM Streaming & Response Dedup
- ResponseTap buffers streaming and commits exactly one final assistant message; prevents fragment leaks into context.
- Dedup OpenAI LLM adapter maintained; no streaming-frame pollution of conversation context.

### 🔧 Configuration Flags & Docs
- New flags documented in AGENTS.md and added to `server/env.example`:
  - Identity/Persistence: `USER_ID`, `FACTS_DB_PATH`, `PIPELINE_IDLE_TIMEOUT_SECS`.
  - Spelling hints (opt-in, generic): `ENABLE_SPELLING_HINTS`, `ENABLE_LOCATION_SPELLING_HINTS`.
  - TapeStore noise thresholds: `TAPE_MIN_USEFUL_WORDS`, `TAPE_MIN_USEFUL_LEN`.
  - STT tuning: `SHERPA_RULE2_MIN_TRAILING_SILENCE`, `SHERPA_MIN_FINAL_WORDS`, `SHERPA_ENDPOINT_DEBOUNCE_MS`.

### ⚡ Performance & Stability
- Eliminated dead-pipeline scenarios after reconnects via RTVI auto-ready.
- Reduced token usage with compact prompts and separated snippets.
- Suppressed noisy sklearn matmul runtime warnings for cleaner logs.

### 🐛 Bug Fixes
- Fixed context duplication (no duplicate “[Session Info]”, no double-inserting current user message).
- Prevented assistant fragment accumulation in recent context.
- Corrected session counting (increments once per session).
- Prevented premature finals and punctuation artifacts from triggering mid-thought replies.

## [2025-07-26] - Initial Setup & Kokoro Integration

### 🎤 Initial Setup
- **Project structure** - Complete project structure with client and server components
- **Dependencies** - All required dependencies and package management
- **Configuration** - Basic configuration setup and environment variables
- **Documentation** - Initial README and setup documentation

### 🔊 Kokoro TTS Integration
- **Kokoro TTS** - Integration with Kokoro text-to-speech system
- **macOS optimization** - Optimized for macOS with Apple Silicon support
- **Audio processing** - Advanced audio processing and quality control
- **Streaming support** - Real-time audio streaming capabilities

## [2025-07-23] - Foundation & Core Architecture

### 🏗️ Foundation
- **Initial commit** - Project foundation with basic structure
- **Git setup** - Complete git repository setup with .gitignore
- **Package management** - npm and pip package management setup
- **Basic architecture** - Core client-server architecture

### 🔧 Core Components
- **React client** - Next.js client application with TypeScript
- **Python server** - FastAPI server with bot functionality
- **WebRTC integration** - Real-time communication setup
- **Basic bot functionality** - Core bot features and commands

### 📋 Initial Features
- **Basic voice processing** - Simple voice input/output handling
- **System instructions** - Basic system prompt and instruction handling
- **Error handling** - Basic error handling and logging
- **Development setup** - Complete development environment setup

---

### Legend
- **🎵** Music-related features
- **📝** Dictation/transcription features
- **🏗️** Architecture/infrastructure changes
- **🎯** Voice recognition improvements
- **🔧** Tool integration
- **📁** File structure changes
- **⚡** Performance improvements
- **🐛** Bug fixes
- **🚀** New features
- **🌍** Internationalization
- **📊** Analytics/monitoring
