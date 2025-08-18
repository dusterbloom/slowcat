# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Slowcat is a local voice agent for macOS achieving sub-800ms voice-to-voice latency using Apple Silicon. It uses Pipecat framework with MLX for optimal performance on M-series chips.

## Development Commands

### Server (Python)

**🚨 CRITICAL: Always activate .venv first!**
```bash
cd server/
source .venv/bin/activate  # ⚠️ REQUIRED - Must activate .venv first!
```

**Setup and Run**:
```bash
cd server/
./run_bot.sh  # Automated setup with MCP integration

# Manual setup:
python -m venv .venv
source .venv/bin/activate  # ⚠️ MUST activate .venv before any Python commands!
pip install -r requirements.txt
python bot_v2.py  # Use refactored main entry point
```

**🔴 VIRTUAL ENVIRONMENT ACTIVATION RULES:**
- **ALWAYS** run `source .venv/bin/activate` first
- **NEVER** run Python commands without activating .venv
- **CHECK** your prompt shows `(.venv)` before proceeding
- If commands fail, first thing to check: is .venv activated?

**Multi-language Support**:
```bash
python bot_v2.py --language es  # Spanish, French (fr), Japanese (ja), etc.
```

**Advanced Options**:
```bash
python bot_v2.py --help                    # Show all options
ENABLE_MCP=false ./run_bot.sh              # Disable tool integration
HF_HUB_OFFLINE=1 ./run_bot.sh              # Force offline mode
```

**Environment Variables**:
- `ENABLE_VOICE_RECOGNITION`: Enable/disable speaker recognition (default: true)
- `ENABLE_MEMORY`: Enable/disable local conversation memory (default: true)
- `ENABLE_MCP`: Enable/disable MCP tool integration (default: true)
- `OPENAI_BASE_URL`: LLM endpoint (default: http://localhost:1234/v1)
- `MCPO_PORT`: MCP proxy port (default: 3001)
- `HF_HUB_OFFLINE`: Force HuggingFace offline mode
- `USE_MINIMAL_PROMPTS`: Use minimal system prompts for A/B testing

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
- `music_mode.py`: Voice-controlled music playback
- `dictation_mode.py`: Silent transcription mode
- `local_memory.py`: Conversation history management
- `smart_turn_manager.py`: Advanced conversation flow control

### Voice Recognition (server/voice_recognition/)
- Automatic speaker enrollment after 3 utterances
- Real-time identification using Resemblyzer
- Profiles stored in `server/data/speaker_profiles/`
- Adjusted thresholds for single-speaker scenarios
- `lightweight.py`: Optimized recognition for performance

### Local Memory System
- **File-based**: `server/data/memory/` - JSON conversation storage
- **Database-based**: `server/data/tool_memory/` - SQLite for tool interactions
- Per-speaker memory when voice recognition is enabled
- Stores last 200 conversations, includes last 10 in context
- Memory search and retrieval capabilities

### Tool Integration (MCP)
- **MCP Proxy**: Automatic MCPO server startup via `run_bot.sh`
- **Built-in Tools**: Web search, file operations, music control, time/date
- **Memory Tools**: Conversation history search and persistence
- **Configuration**: `server/mcp.json` defines available tools

## Important Constraints

1. **Python 3.12 or earlier required** - MLX dependency compatibility
2. **macOS with Apple Silicon required** - Optimized for M-series chips
3. **LM Studio or OpenAI-compatible server required** - For LLM responses
4. **Ports**: 7860 (WebRTC server), 3001 (MCP proxy)
5. **Multiprocessing**: Uses 'spawn' method for macOS Metal GPU safety
6. **Dependencies**: Requires specific MLX, Pipecat, and ONNX versions (see requirements.txt)

## Testing

**Integration and Unit Tests**:
```bash
cd server/
source .venv/bin/activate            # ⚠️ MUST activate .venv first!
python test_integration.py           # Main integration tests
python test_voice_recognition.py     # Voice recognition tests
python tests/test_llm_tools.py       # LLM and tool integration tests
python tests/test_memory.py          # Memory system tests
python tests/test_mcp_e2e.py         # MCP end-to-end tests
python -m pytest tests/unit/         # Unit tests (pipeline builder, service factory)
```

**Performance and Component Tests**:
```bash
cd server/
source .venv/bin/activate            # ⚠️ MUST activate .venv first!
python test_sherpa_api.py             # STT performance benchmarks
python tests/test_performance_optimizations.py  # Performance tests
python tests/test_file_tools.py       # File operation tests
python test_tts_sanitization.py       # TTS text processing tests
```

## Debugging and Troubleshooting

**Debug Scripts**:
```bash
cd server/
source .venv/bin/activate            # ⚠️ MUST activate .venv first!
python debug_sherpa_model.py         # Test STT model loading
python tests/debug_bot.py            # Debug pipeline components
python tests/debug_voice_recognition.py  # Test voice recognition
python tests/debug_memory_tools.py   # Debug memory system
```

**Logs and Monitoring**:
- `server/mcpo.log`: MCP proxy server logs
- `server/bot_debug.log`: Main application debug output
- Environment variable `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` required for macOS

**Common Issues**:
- **MLX/Metal GPU**: Use Python 3.12 or earlier, ensure proper multiprocessing
- **Model Loading**: Check `server/models/` directory and HuggingFace connectivity
- **MCP Tools**: Verify `mcp.json` configuration and MCPO server startup
- **Audio Issues**: Confirm sample rates (24kHz TTS, 16kHz STT)

## Key Files

**Main Entry Points**:
- `server/bot_v2.py`: Current main pipeline (refactored architecture)
- `server/run_bot.sh`: Automated setup and launcher script
- `client/src/app/page.tsx`: Main UI component

**Core Architecture**:
- `server/core/pipeline_builder.py`: Pipeline construction with dependency injection
- `server/core/service_factory.py`: Service creation and configuration
- `server/config.py`: Centralized configuration management
- `server/core/service_interfaces.py`: Service abstractions

**Processing Pipeline**:
- `server/processors/`: Custom processor implementations
- `server/services/`: Core services (STT, TTS, LLM integration)
- `server/voice_recognition/`: Speaker identification system
- `server/tools/`: Built-in tools (web search, file operations, music control)

**Integration Systems**:
- `server/mcp.json`: MCP (Model Context Protocol) tool configurations
- `server/services/simple_mcp_tool_manager.py`: MCP tool integration
- `server/processors/local_memory.py`: Persistent conversation memory

## Language/Voice Mapping

- English (en): af_heart
- Spanish (es): ef_dora
- French (fr): ff_siwis
- Japanese (ja): jf_alpha
- Italian (it): im_nicola
- Chinese (zh): zf_xiaobei
- Portuguese (pt): pf_dora

## Special Features and Modes

**Music Mode**:
- Voice control: *"music mode"* → *"play jazz"* → *"skip song"* → *"what's playing?"*
- Automatic music library indexing from common macOS locations
- Integration with system audio controls

**Dictation Mode**: 
- Silent transcription: *"dictation mode"* → speak → *"stop dictation"*
- Outputs to timestamped files in `server/data/dictation/`
- Professional-quality transcription without AI responses

**Speaker Recognition**:
- Automatic enrollment after 3 utterances
- Per-speaker conversation memory and preferences
- Names stored in `server/data/speaker_profiles/speaker_names.json`

**Conversation Memory**:
- Local storage without cloud dependencies
- Searchable conversation history
- Context injection for natural follow-up conversations

## Pipecat Frame Processing Patterns

### Critical FrameProcessor Implementation Rules

**NEVER ignore these patterns when creating custom processors:**

1. **Mandatory Parent Method Calls**:
```python
class CustomProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # REQUIRED - sets up _FrameProcessor__input_queue
        # Your initialization here
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)  # REQUIRED - handles initialization state
        # Your frame processing logic here
```

2. **StartFrame Handling Pattern**:
```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    await super().process_frame(frame, direction)  # ALWAYS call parent first
    
    if isinstance(frame, StartFrame):
        # Push StartFrame downstream IMMEDIATELY
        await self.push_frame(frame, direction)
        # Then do processor-specific initialization
        self._your_initialization_logic()
        return
    
    # Handle other frames
    if isinstance(frame, YourFrameType):
        # Process your frames
        pass
    
    # ALWAYS forward frames to prevent pipeline blocking
    await self.push_frame(frame, direction)
```

3. **Frame Forwarding Rule**:
   - **MUST** forward ALL frames with `await self.push_frame(frame, direction)`
   - Failing to forward frames **WILL** block the entire pipeline
   - This is the #1 cause of "start frameprocessor push frame" errors

### Common Initialization Errors

**Error: `RTVIProcessor#0 Trying to process SpeechControlParamsFrame#0 but StartFrame not received yet`**
- **Cause**: Processor receives frames before StartFrame due to pipeline timing
- **Solution**: Implement proper StartFrame checking in process_frame
- **Prevention**: Use the exact patterns above

**Error: `AttributeError: '_FrameProcessor__input_queue'`**
- **Cause**: Missing `super().__init__()` call in custom processor
- **Solution**: Always call parent init with proper kwargs
- **Prevention**: Follow mandatory parent method calls pattern

**Error: Pipeline hangs or "start frameprocessor push frame" issues**
- **Cause**: Processor not forwarding frames, blocking pipeline flow
- **Solution**: Ensure every process_frame method calls `await self.push_frame(frame, direction)`
- **Prevention**: Follow frame forwarding rule religiously

### StartFrame Initialization Lifecycle

1. **Pipeline Creation**: Pipeline creates all processors
2. **StartFrame Propagation**: StartFrame flows through pipeline in order
3. **Processor Initialization**: Each processor receives StartFrame and initializes
4. **Frame Processing Begins**: Normal frame processing starts
5. **Frame Flow**: All frames must be forwarded to maintain pipeline flow

### RTVIProcessor Specific Issues

RTVIProcessor requires proper initialization state before processing frames. If using RTVIProcessor in your pipeline:

- Ensure StartFrame reaches RTVIProcessor before any other frames
- RTVIProcessor is automatically added when metrics are enabled in transport
- Use proper initialization checking in custom processors that interact with RTVIProcessor

### Debugging Frame Issues

1. **Enable Pipeline Logging**: Set debug level to see frame flow
2. **Check Frame Forwarding**: Verify every processor calls `push_frame`
3. **Verify Parent Calls**: Ensure `super().__init__()` and `super().process_frame()` are called
4. **Monitor StartFrame Propagation**: Trace StartFrame through pipeline
5. **Use Pipeline Observers**: Implement observers to monitor frame lifecycle

### When Adding New Processors

**CHECKLIST - Use this for EVERY new processor:**
- [ ] Inherits from FrameProcessor
- [ ] Calls `super().__init__(**kwargs)` in __init__
- [ ] Calls `await super().process_frame(frame, direction)` in process_frame
- [ ] Handles StartFrame by pushing it downstream immediately
- [ ] Forwards ALL frames with `await self.push_frame(frame, direction)`
- [ ] Does not block frame flow under any circumstances
- [ ] Tested in isolation and in full pipeline
- [ ] Handles frame processing errors gracefully

## 🚨 Critical LLM Streaming & Context Management 

### The LLM Context Corruption Problem

**Problem**: LLM services send streaming TextFrames that can corrupt conversation context, causing responses like:
```
"I'm I'm Slowcat, I'm Slowcat, your I'm Slowcat, your friendly..."
```

**Root Cause**: LLM streaming frames reach context aggregators and accumulate corrupted text that gets sent back to the LLM on the next request.

### LLM Frame Types & Behavior

**CRITICAL INSIGHT**: OpenAI-compatible LLMs (including LM Studio) send **REPLACEMENT frames**, not **DELTA frames**:

- ❌ **Delta behavior (expected)**: Frame 1: "How", Frame 2: " can", Frame 3: " I help"
- ✅ **Replacement behavior (actual)**: Frame 1: "How", Frame 2: "How can", Frame 3: "How can I help"

Each `TextFrame` during streaming contains the **complete response so far**, not just new content.

### LLM Pipeline Architecture

```
User Input → STT → LLM → [Streaming TextFrames] → Context Aggregator → Memory
                  ↓
               TTS ← [Clean TextFrames] ← Context Filter ← [Blocked/Filtered]
```

**The Fix**: Use `ContextFilter` processor to:
1. **Block ALL** streaming LLM TextFrames from reaching context aggregator
2. **Accumulate** the latest complete response (replacement, not delta)
3. **Send ONE** clean TextFrame to context aggregator at response end

### ContextFilter Implementation Pattern

```python
class ContextFilter(FrameProcessor):
    """Prevents LLM streaming frames from corrupting context"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_response = False
        self._accumulated_response = ""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._accumulated_response = ""
            await self.push_frame(frame, direction)  # Allow start frame
            return
            
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            
            # Send ONLY our clean accumulated response
            if self._accumulated_response.strip():
                clean_frame = TextFrame(self._accumulated_response.strip())
                await self.push_frame(clean_frame, direction)
            
            # Block original end frame (might be corrupted)
            self._accumulated_response = ""
            return
            
        elif isinstance(frame, TextFrame) and self._in_response:
            # LLM sends REPLACEMENT frames - just replace our accumulated response
            if frame.text:
                self._accumulated_response = frame.text
            
            # Block ALL streaming frames from reaching context
            return
        
        # Forward all other frames normally
        await self.push_frame(frame, direction)
```

### Pipeline Component Ordering

**CRITICAL**: ContextFilter must be positioned correctly in pipeline:

```python
# CORRECT order:
services['llm'],                    # Generates streaming responses
services['tts'],                    # Consumes TextFrames for speech
transport.output(),                 # Sends audio to user
processors['context_filter'],       # Filters LLM frames BEFORE context
context_aggregator.assistant(),     # Receives only clean frames
```

**WRONG**: Placing context aggregator before ContextFilter will cause corruption.

### Context Aggregator Types

1. **Standard**: `OpenAIAssistantContextAggregator` - accumulates ALL TextFrames
2. **Dedup**: `DedupAssistantContextAggregator` - blocks streaming LLMTextFrames
3. **ContextFilter**: Blocks ALL LLM frames, sends only final clean response

### Debugging Context Issues

**Symptoms**:
- Stuttering responses with repeated phrases
- "I'm I'm Slowcat" type corruption
- Context growing with duplicated content

**Debug Steps**:
1. Check if ContextFilter is in pipeline and positioned correctly
2. Verify LLMFullResponseStartFrame/EndFrame are being handled
3. Monitor what frames reach context aggregator
4. Examine conversation context sent to LLM (should be clean)

**Log Analysis**:
```
✅ GOOD: "Context filter: Sending ONLY our clean response: 'I'm Slowcat...'"
❌ BAD: Multiple similar frames reaching context aggregator without filtering
```

### Frame Direction Understanding

- **UPSTREAM**: Toward LLM (user input, context)
- **DOWNSTREAM**: From LLM (responses, TTS)

Context corruption typically happens on **DOWNSTREAM** TextFrames that should be filtered.

### Key Files for Context Management

**Core Implementation**:
- `server/processors/context_filter.py`: Blocks streaming LLM frames from context
- `server/services/dedup_openai_llm.py`: Custom LLM service with frame filtering
- `server/core/pipeline_builder.py`: Pipeline component ordering and configuration

**Pipeline Integration**:
- Context filter positioned BEFORE context aggregator in pipeline
- DedupOpenAILLMService used instead of standard OpenAILLMService
- Memory processors positioned after context filtering

**Testing & Validation**:
- Monitor logs for "Context filter: Sending ONLY our clean response"
- Verify no streaming TextFrames reach context aggregator
- Check conversation context sent to LLM is clean (no stuttering)

### Quick Fix Checklist for Context Corruption

If you see stuttering responses like "I'm I'm Slowcat":

1. **Check Pipeline Order**: Ensure ContextFilter comes BEFORE context aggregator
2. **Verify Frame Blocking**: Look for logs showing streaming frames being blocked
3. **Examine LLM Context**: Check what conversation history is sent to LLM
4. **Test Clean Response**: Confirm only one clean TextFrame reaches context per response
5. **Validate Memory**: Ensure conversation memory stores clean responses only

This architecture prevents LLM streaming corruption while maintaining conversation continuity and memory functionality.