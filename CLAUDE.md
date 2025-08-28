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
- `music_mode.py`: Voice-controlled music playbook
- `dictation_mode.py`: Silent transcription mode
- `local_memory.py`: Conversation history management
- `smart_turn_manager.py`: Advanced conversation flow control
- `smart_context_manager.py`: **NEW** - Fixed 4096 token context with fact extraction
- `token_counter.py`: Token counting utilities for context management
- `context_filter.py`: Filters LLM streaming frames to prevent context corruption
- `response_tap.py`: **NEW** - Taps assistant responses for smart context integration

### Voice Recognition (server/voice_recognition/)
- Automatic speaker enrollment after 3 utterances
- Real-time identification using Resemblyzer
- Profiles stored in `server/data/speaker_profiles/`
- Adjusted thresholds for single-speaker scenarios
- `lightweight.py`: Optimized recognition for performance

### Smart Memory System (NEW)
- **Facts Graph**: `server/memory/facts_graph.py` - Structured fact storage with natural decay
- **Query Router**: `server/memory/query_router.py` - Intelligent routing to appropriate memory stores
- **Query Classifier**: `server/memory/query_classifier.py` - Language-agnostic intent classification
- **Tape Store**: `server/memory/tape_store.py` - Verbatim conversation storage
- **Smart Context Manager**: `server/processors/smart_context_manager.py` - Fixed 4096 token context
- **Fact Extractor**: `server/memory/spacy_fact_extractor.py` - SpaCy-based fact extraction

### Legacy Memory System
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
python test_smart_memory.py          # Smart memory system tests
python test_smart_memory_integration.py  # Smart memory integration with LM Studio
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

## Critical Code Quality Rules

**NEVER use hardcoded values in search or logic:**
- No hardcoded search terms like "Potola" 
- No hardcoded user data in algorithms
- All searches must be generic and data-driven
- Fix root causes, not symptoms with hardcoded workarounds

**Coding Standards:**
- Python: Follow PEP 8, 4-space indent, prefer type hints and docstrings
- Files use `snake_case.py`; classes use `PascalCase`
- Keep imports local to feature areas, avoid circular dependencies across `core/`, `processors/`, `services/`
- TypeScript/React: Components `PascalCase.tsx`; hooks/utilities `camelCase.ts`

**Commit Guidelines:**
- Follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`
- PRs must include clear description, rationale, test coverage
- Ensure `pytest` passes and `npm run lint` is clean before review

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

## 🧠 Smart Memory System Architecture

### Fixed Context Problem Solution

**Problem**: The original `context_aggregator.user()` accumulates ALL user messages indefinitely, causing context to grow to 50,000+ tokens and making responses increasingly slow.

**Solution**: Replace with `SmartContextManager` that maintains EXACTLY 4096 tokens regardless of conversation length.

### Memory Component Overview

```
User Input → Smart Context Manager → Fixed 4096 tokens → LLM (always fast)
                  ↓
            Facts Extraction → Facts Graph (structured storage)
                  ↓
            Query Router → Intelligent retrieval from appropriate stores
```

### Token Budget Allocation (4096 total)
- **System Prompt**: 500 tokens (dynamic, includes session info)
- **Facts Context**: 800 tokens (structured knowledge from Facts Graph)
- **Recent Conversation**: 2000 tokens (sliding window of last exchanges)
- **Current Input**: 696 tokens (user's current message)
- **Buffer**: 100 tokens (safety margin)

### Facts Graph Storage

Implements structured memory with natural decay:
- **S4 (Verbatim)**: "my dog name is Potola" (full text)
- **S3 (Structured)**: "user::pet[name=Potola, species=dog]" (parsed)
- **S2 (Tuple)**: "(user, pet, Potola)" (essential facts)
- **S1 (Edge)**: "(user —has_pet→ dog)" (relationship only)
- **S0 (Forgotten)**: fact naturally decays and is removed

### Query Classification (Language-Agnostic)

Uses multi-signal approach instead of hardcoded English patterns:
- **Semantic Vectors**: Multilingual embeddings with intent clustering
- **Universal POS**: Cross-language grammatical patterns
- **NER Entities**: Language-independent named entity recognition
- **LLM Fallback**: Small local model for uncertain cases

### Smart Context Integration

**CRITICAL**: Replace line 528 in `server/core/pipeline_builder.py`:
```python
# OLD (broken - accumulates forever):
context_aggregator.user(),

# NEW (fixed - always 4096 tokens):
SmartContextManager(
    context=context,
    facts_db_path=config.memory.facts_db_path,
    max_tokens=4096
),
```

### Performance Guarantees

- **Turn 1**: 4096 tokens → <100ms response
- **Turn 100**: 4096 tokens → <100ms response  
- **Turn 1000**: 4096 tokens → <100ms response (FOREVER)
- **Memory Usage**: <200MB regardless of conversation length
- **Fact Retrieval**: >90% accuracy for important personal facts

### Environment Configuration

Smart memory system can be tuned via environment variables:
```bash
# Token budget allocation
SC_BUDGET_SYSTEM=500        # System prompt tokens
SC_BUDGET_FACTS=800         # Facts context tokens  
SC_BUDGET_RECENT=2000       # Recent conversation tokens
SC_BUDGET_INPUT=696         # Current input tokens
SC_BUDGET_BUFFER=100        # Safety buffer tokens

# Query router thresholds
ROUTER_THRESHOLD_HIGH=0.8   # Direct routing confidence
ROUTER_THRESHOLD_MED=0.6    # Fallback routing confidence
ROUTER_THRESHOLD_LOW=0.4    # Hybrid search confidence
```

### Key Files for Smart Memory

**Core Implementation**:
- `server/memory/facts_graph.py`: Structured fact storage with decay
- `server/memory/query_router.py`: Multi-store intelligent routing
- `server/memory/query_classifier.py`: Language-agnostic intent classification
- `server/processors/smart_context_manager.py`: Fixed context management

**Pipeline Integration**:
- `server/core/pipeline_builder.py`: Replace context_aggregator.user()
- `server/processors/token_counter.py`: Token counting utilities
- `server/processors/response_tap.py`: Assistant response integration

**Testing & Validation**:
- `test_smart_memory.py`: Component testing
- `test_smart_memory_integration.py`: End-to-end testing with LM Studio
- `docs/TASK_99_FIXED_CONTEXT_SMART_MEMORY.md`: Complete implementation guide

### SurrealDB Integration (Current Branch: feature/surrealdb-memory)

**New Memory Backend**: Enhanced memory system using SurrealDB multi-model database:
- `server/memory/surreal_memory.py`: SurrealDB memory implementation
- `server/scripts/reflection_daemon.py`: Background reflection process
- `server/utils/private_reflector.py`: Private fact reflection utilities

**Configuration**:
- Copy `server/env.example` to `server/.env` and set required variables
- Key environment variables: `USER_ID`, `FACTS_DB_PATH`, `PIPELINE_IDLE_TIMEOUT_SECS`
- Memory creation via `create_smart_memory_system()` in query router

**Security Notes**:
- Never commit `.env` files
- Grant microphone permission on macOS for voice input
- All processing remains local - no cloud dependencies

<!-- BACKLOG.MD GUIDELINES START -->
# Instructions for the usage of Backlog.md CLI Tool

## What is Backlog.md?

**Backlog.md is the complete project management system for this codebase.** It provides everything needed to manage tasks, track progress, and collaborate on development - all through a powerful CLI that operates on markdown files.

### Core Capabilities

✅ **Task Management**: Create, edit, assign, prioritize, and track tasks with full metadata
✅ **Acceptance Criteria**: Granular control with add/remove/check/uncheck by index
✅ **Board Visualization**: Terminal-based Kanban board (`backlog board`) and web UI (`backlog browser`)
✅ **Git Integration**: Automatic tracking of task states across branches
✅ **Dependencies**: Task relationships and subtask hierarchies
✅ **Documentation & Decisions**: Structured docs and architectural decision records
✅ **Export & Reporting**: Generate markdown reports and board snapshots
✅ **AI-Optimized**: `--plain` flag provides clean text output for AI processing

### Why This Matters to You (AI Agent)

1. **Comprehensive system** - Full project management capabilities through CLI
2. **The CLI is the interface** - All operations go through `backlog` commands
3. **Unified interaction model** - You can use CLI for both reading (`backlog task 1 --plain`) and writing (`backlog task edit 1`)
4. **Metadata stays synchronized** - The CLI handles all the complex relationships

### Key Understanding

- **Tasks** live in `backlog/tasks/` as `task-<id> - <title>.md` files
- **You interact via CLI only**: `backlog task create`, `backlog task edit`, etc.
- **Use `--plain` flag** for AI-friendly output when viewing/listing
- **Never bypass the CLI** - It handles Git, metadata, file naming, and relationships

---

# ⚠️ CRITICAL: NEVER EDIT TASK FILES DIRECTLY

**ALL task operations MUST use the Backlog.md CLI commands**
- ✅ **DO**: Use `backlog task edit` and other CLI commands
- ✅ **DO**: Use `backlog task create` to create new tasks
- ✅ **DO**: Use `backlog task edit <id> --check-ac <index>` to mark acceptance criteria
- ❌ **DON'T**: Edit markdown files directly
- ❌ **DON'T**: Manually change checkboxes in files
- ❌ **DON'T**: Add or modify text in task files without using CLI

**Why?** Direct file editing breaks metadata synchronization, Git tracking, and task relationships.

---

## 1. Source of Truth & File Structure

### 📖 **UNDERSTANDING** (What you'll see when reading)
- Markdown task files live under **`backlog/tasks/`** (drafts under **`backlog/drafts/`**)
- Files are named: `task-<id> - <title>.md` (e.g., `task-42 - Add GraphQL resolver.md`)
- Project documentation is in **`backlog/docs/`**
- Project decisions are in **`backlog/decisions/`**

### 🔧 **ACTING** (How to change things)
- **All task operations MUST use the Backlog.md CLI tool**
- This ensures metadata is correctly updated and the project stays in sync
- **Always use `--plain` flag** when listing or viewing tasks for AI-friendly text output

---

## 2. Common Mistakes to Avoid

### ❌ **WRONG: Direct File Editing**
```markdown
# DON'T DO THIS:
1. Open backlog/tasks/task-7 - Feature.md in editor
2. Change "- [ ]" to "- [x]" manually
3. Add notes directly to the file
4. Save the file
```

### ✅ **CORRECT: Using CLI Commands**
```bash
# DO THIS INSTEAD:
backlog task edit 7 --check-ac 1  # Mark AC #1 as complete
backlog task edit 7 --notes "Implementation complete"  # Add notes
backlog task edit 7 -s "In Progress" -a @agent-k  # Multiple commands: change status and assign the task
```

---

## 3. Understanding Task Format (Read-Only Reference)

⚠️ **FORMAT REFERENCE ONLY** - The following sections show what you'll SEE in task files.
**Never edit these directly! Use CLI commands to make changes.**

### Task Structure You'll See

```markdown
---
id: task-42
title: Add GraphQL resolver
status: To Do
assignee: [@sara]
labels: [backend, api]
---

## Description
Brief explanation of the task purpose.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First criterion
- [x] #2 Second criterion (completed)
- [ ] #3 Third criterion
<!-- AC:END -->

## Implementation Plan
1. Research approach
2. Implement solution

## Implementation Notes
Summary of what was done.
```

### How to Modify Each Section

| What You Want to Change | CLI Command to Use |
|------------------------|-------------------|
| Title | `backlog task edit 42 -t "New Title"` |
| Status | `backlog task edit 42 -s "In Progress"` |
| Assignee | `backlog task edit 42 -a @sara` |
| Labels | `backlog task edit 42 -l backend,api` |
| Description | `backlog task edit 42 -d "New description"` |
| Add AC | `backlog task edit 42 --ac "New criterion"` |
| Check AC #1 | `backlog task edit 42 --check-ac 1` |
| Uncheck AC #2 | `backlog task edit 42 --uncheck-ac 2` |
| Remove AC #3 | `backlog task edit 42 --remove-ac 3` |
| Add Plan | `backlog task edit 42 --plan "1. Step one\n2. Step two"` |
| Add Notes | `backlog task edit 42 --notes "What I did"` |

---

## 4. Defining Tasks

### Creating New Tasks

**Always use CLI to create tasks:**
```bash
backlog task create "Task title" -d "Description" --ac "First criterion" --ac "Second criterion"
```

### Title (one liner)
Use a clear brief title that summarizes the task.

### Description (The "why")
Provide a concise summary of the task purpose and its goal. Explains the context without implementation details.

### Acceptance Criteria (The "what")

**Understanding the Format:**
- Acceptance criteria appear as numbered checkboxes in the markdown files
- Format: `- [ ] #1 Criterion text` (unchecked) or `- [x] #1 Criterion text` (checked)

**Managing Acceptance Criteria via CLI:**

⚠️ **IMPORTANT: How AC Commands Work**
- **Adding criteria (`--ac`)** accepts multiple flags: `--ac "First" --ac "Second"` ✅
- **Checking/unchecking/removing** accept multiple flags too: `--check-ac 1 --check-ac 2` ✅
- **Mixed operations** work in a single command: `--check-ac 1 --uncheck-ac 2 --remove-ac 3` ✅

```bash
# Add new criteria (MULTIPLE values allowed)
backlog task edit 42 --ac "User can login" --ac "Session persists"

# Check specific criteria by index (MULTIPLE values supported)
backlog task edit 42 --check-ac 1 --check-ac 2 --check-ac 3  # Check multiple ACs
# Or check them individually if you prefer:
backlog task edit 42 --check-ac 1    # Mark #1 as complete
backlog task edit 42 --check-ac 2    # Mark #2 as complete

# Mixed operations in single command
backlog task edit 42 --check-ac 1 --uncheck-ac 2 --remove-ac 3

# ❌ STILL WRONG - These formats don't work:
# backlog task edit 42 --check-ac 1,2,3  # No comma-separated values
# backlog task edit 42 --check-ac 1-3    # No ranges
# backlog task edit 42 --check 1         # Wrong flag name

# Multiple operations of same type
backlog task edit 42 --uncheck-ac 1 --uncheck-ac 2  # Uncheck multiple ACs
backlog task edit 42 --remove-ac 2 --remove-ac 4    # Remove multiple ACs (processed high-to-low)
```

**Key Principles for Good ACs:**
- **Outcome-Oriented:** Focus on the result, not the method
- **Testable/Verifiable:** Each criterion should be objectively testable
- **Clear and Concise:** Unambiguous language
- **Complete:** Collectively cover the task scope
- **User-Focused:** Frame from end-user or system behavior perspective

Good Examples:
- "User can successfully log in with valid credentials"
- "System processes 1000 requests per second without errors"

Bad Example (Implementation Step):
- "Add a new function handleLogin() in auth.ts"

### Task Breakdown Strategy

1. Identify foundational components first
2. Create tasks in dependency order (foundations before features)
3. Ensure each task delivers value independently
4. Avoid creating tasks that block each other

### Task Requirements

- Tasks must be **atomic** and **testable** or **verifiable**
- Each task should represent a single unit of work for one PR
- **Never** reference future tasks (only tasks with id < current task id)
- Ensure tasks are **independent** and don't depend on future work

---

## 5. Implementing Tasks

### Implementation Plan (The "how") (only after starting work)
```bash
backlog task edit 42 -s "In Progress" -a @{myself}
backlog task edit 42 --plan "1. Research patterns\n2. Implement\n3. Test"
```

### Implementation Notes (Imagine you need to copy paste this into a PR description)
```bash
backlog task edit 42 --notes "Implemented using pattern X, modified files Y and Z"
```

**IMPORTANT**: Do NOT include an Implementation Plan when creating a task. The plan is added only after you start implementation.
- Creation phase: provide Title, Description, Acceptance Criteria, and optionally labels/priority/assignee.
- When you begin work, switch to edit and add the plan: `backlog task edit <id> --plan "..."`.
- Add Implementation Notes only after completing the work: `backlog task edit <id> --notes "..."`.

Phase discipline: What goes where
- Creation: Title, Description, Acceptance Criteria, labels/priority/assignee.
- Implementation: Implementation Plan (after moving to In Progress).
- Wrap-up: Implementation Notes, AC and Definition of Done checks.

**IMPORTANT**: Only implement what's in the Acceptance Criteria. If you need to do more, either:
1. Update the AC first: `backlog task edit 42 --ac "New requirement"`
2. Or create a new task: `backlog task create "Additional feature"`

---

## 6. Typical Workflow

```bash
# 1. Identify work
backlog task list -s "To Do" --plain

# 2. Read task details
backlog task 42 --plain

# 3. Start work: assign yourself & change status
backlog task edit 42 -a @myself -s "In Progress"

# 4. Add implementation plan
backlog task edit 42 --plan "1. Analyze\n2. Refactor\n3. Test"

# 5. Work on the task (write code, test, etc.)

# 6. Mark acceptance criteria as complete (supports multiple in one command)
backlog task edit 42 --check-ac 1 --check-ac 2 --check-ac 3  # Check all at once
# Or check them individually if preferred:
# backlog task edit 42 --check-ac 1
# backlog task edit 42 --check-ac 2
# backlog task edit 42 --check-ac 3

# 7. Add implementation notes
backlog task edit 42 --notes "Refactored using strategy pattern, updated tests"

# 8. Mark task as done
backlog task edit 42 -s Done
```

---

## 7. Definition of Done (DoD)

A task is **Done** only when **ALL** of the following are complete:

### ✅ Via CLI Commands:
1. **All acceptance criteria checked**: Use `backlog task edit <id> --check-ac <index>` for each
2. **Implementation notes added**: Use `backlog task edit <id> --notes "..."`
3. **Status set to Done**: Use `backlog task edit <id> -s Done`

### ✅ Via Code/Testing:
4. **Tests pass**: Run test suite and linting
5. **Documentation updated**: Update relevant docs if needed
6. **Code reviewed**: Self-review your changes
7. **No regressions**: Performance, security checks pass

⚠️ **NEVER mark a task as Done without completing ALL items above**

---

## 8. Quick Reference: DO vs DON'T

### Viewing Tasks
| Task | ✅ DO | ❌ DON'T |
|------|-------|----------|
| View task | `backlog task 42 --plain` | Open and read .md file directly |
| List tasks | `backlog task list --plain` | Browse backlog/tasks folder |
| Check status | `backlog task 42 --plain` | Look at file content |

### Modifying Tasks
| Task | ✅ DO | ❌ DON'T |
|------|-------|----------|
| Check AC | `backlog task edit 42 --check-ac 1` | Change `- [ ]` to `- [x]` in file |
| Add notes | `backlog task edit 42 --notes "..."` | Type notes into .md file |
| Change status | `backlog task edit 42 -s Done` | Edit status in frontmatter |
| Add AC | `backlog task edit 42 --ac "New"` | Add `- [ ] New` to file |

---

## 9. Complete CLI Command Reference

### Task Creation
| Action | Command |
|--------|---------|
| Create task | `backlog task create "Title"` |
| With description | `backlog task create "Title" -d "Description"` |
| With AC | `backlog task create "Title" --ac "Criterion 1" --ac "Criterion 2"` |
| With all options | `backlog task create "Title" -d "Desc" -a @sara -s "To Do" -l auth --priority high` |
| Create draft | `backlog task create "Title" --draft` |
| Create subtask | `backlog task create "Title" -p 42` |

### Task Modification
| Action | Command |
|--------|---------|
| Edit title | `backlog task edit 42 -t "New Title"` |
| Edit description | `backlog task edit 42 -d "New description"` |
| Change status | `backlog task edit 42 -s "In Progress"` |
| Assign | `backlog task edit 42 -a @sara` |
| Add labels | `backlog task edit 42 -l backend,api` |
| Set priority | `backlog task edit 42 --priority high` |

### Acceptance Criteria Management
| Action | Command |
|--------|---------|
| Add AC | `backlog task edit 42 --ac "New criterion" --ac "Another"` |
| Remove AC #2 | `backlog task edit 42 --remove-ac 2` |
| Remove multiple ACs | `backlog task edit 42 --remove-ac 2 --remove-ac 4` |
| Check AC #1 | `backlog task edit 42 --check-ac 1` |
| Check multiple ACs | `backlog task edit 42 --check-ac 1 --check-ac 3` |
| Uncheck AC #3 | `backlog task edit 42 --uncheck-ac 3` |
| Mixed operations | `backlog task edit 42 --check-ac 1 --uncheck-ac 2 --remove-ac 3 --ac "New"` |

### Task Content
| Action | Command |
|--------|---------|
| Add plan | `backlog task edit 42 --plan "1. Step one\n2. Step two"` |
| Add notes | `backlog task edit 42 --notes "Implementation details"` |
| Add dependencies | `backlog task edit 42 --dep task-1 --dep task-2` |

### Task Operations
| Action | Command |
|--------|---------|
| View task | `backlog task 42 --plain` |
| List tasks | `backlog task list --plain` |
| Filter by status | `backlog task list -s "In Progress" --plain` |
| Filter by assignee | `backlog task list -a @sara --plain` |
| Archive task | `backlog task archive 42` |
| Demote to draft | `backlog task demote 42` |

---

## 10. Troubleshooting

### If You Accidentally Edited a File Directly

1. **DON'T PANIC** - But don't save or commit
2. Revert the changes
3. Make changes properly via CLI
4. If already saved, the metadata might be out of sync - use `backlog task edit` to fix

### Common Issues

| Problem | Solution |
|---------|----------|
| "Task not found" | Check task ID with `backlog task list --plain` |
| AC won't check | Use correct index: `backlog task 42 --plain` to see AC numbers |
| Changes not saving | Ensure you're using CLI, not editing files |
| Metadata out of sync | Re-edit via CLI to fix: `backlog task edit 42 -s <current-status>` |

---

## Remember: The Golden Rule

**🎯 If you want to change ANYTHING in a task, use the `backlog task edit` command.**
**📖 Only READ task files directly, never WRITE to them.**

Full help available: `backlog --help`

<!-- BACKLOG.MD GUIDELINES END -->
