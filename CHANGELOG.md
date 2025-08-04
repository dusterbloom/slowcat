# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-08-04

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