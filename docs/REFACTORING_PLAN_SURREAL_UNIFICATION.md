# Comprehensive Refactoring Plan: Simplifying Slowcat Voice Agent

## Overview

Transform the current dual-implementation memory system (SQLite + SurrealDB with adapters) into a clean, unified SurrealDB-only architecture while preserving all intelligent behaviors and quality.

## Current State Analysis

### Complexity Points Identified
- **24 files** with SQLite dependencies
- **272 environment variable references** across 62 files  
- **116 total environment variables** configured
- **3 adapter layers** between memory components
- **Dual memory backends** (SQLite + SurrealDB)
- **Multiple background daemons** for memory processing

### Architecture Strengths to Preserve
- Fixed 4096 token context (consistent performance)
- Natural memory decay system
- Private thoughts and reflections
- Multi-language support (8+ languages)
- Sub-800ms voice-to-voice latency

## Phase 1: Remove SQLite Fallback (Week 1)

### Day 1-2: Dependency Analysis & Backup

#### Tasks
1. **Create feature branch**: `refactor/surreal-only-memory`
2. **Full backup** of current working state
3. **Document all SQLite dependencies** (24 files identified)

#### Files to Modify
- `memory/facts_graph.py` (856 lines - core SQLite implementation)
- `memory/tape_store.py` (SQLite tape storage)
- `memory/__init__.py` (dual-path logic lines 70-124)
- 21 test files using SQLite

### Day 3-4: Remove SQLite Code Paths

#### Code Changes

**1. Update `memory/__init__.py`**
```python
# Remove lines 101-124 (SQLite fallback)
# Remove SmartMemorySystem class (lines 360-426)
# Simplify create_smart_memory_system() to:

def create_smart_memory_system():
    """Create SurrealDB memory system directly"""
    from .surreal_memory import create_surreal_memory_system
    logger.info("🚀 Using SurrealDB memory system")
    return create_surreal_memory_system()
```

**2. Archive SQLite implementations**
- Move `facts_graph.py` → `archived/facts_graph_sqlite.py`
- Move `tape_store.py` → `archived/tape_store_sqlite.py`
- Keep for reference but exclude from imports

**3. Update imports in affected files**
```python
# Before
from memory.facts_graph import FactsGraph
from memory.tape_store import TapeStore

# After  
from memory.surreal_memory import SurrealMemory
```

### Day 5: Test & Validate

#### Testing Checklist
- [ ] Run full test suite: `python -m pytest tests/`
- [ ] Manual testing of memory features
- [ ] Performance benchmarking
- [ ] Voice recognition with memory
- [ ] Reflection daemon functionality

## Phase 2: Consolidate Memory Adapters (Week 1-2)

### Day 6-7: Remove Adapter Layer

#### Current Adapter Structure
```python
# memory/__init__.py lines 127-358
class SurrealMemorySystemAdapter:
    # 230+ lines of pass-through methods
    def __init__(self, surreal_memory):
        self.surreal_memory = surreal_memory
    
    async def process_query(self, query: str):
        # Complex conversion logic
        return converted_response
```

#### Target Direct Integration
```python
# Direct return without adapter
def create_smart_memory_system():
    return SurrealMemory()  # No adapter needed
```

### Day 8-9: Simplify Pipeline Integration

#### Pipeline Builder Updates (`core/pipeline_builder.py`)

**Before**:
```python
# Complex adapter references
smart_ctx = self._create_smart_context_manager(context, processors.get('memory_processor'))
self._smart_ctx_ref = smart_ctx  # Indirect reference
```

**After**:
```python
# Direct SurrealDB integration
self.memory = create_smart_memory_system()
smart_ctx = SmartContextManager(memory=self.memory)
```

#### Smart Context Manager Updates

**Before**:
```python
# processors/smart_context_manager.py
class SmartContextManager:
    def __init__(self, context, facts_db_path, max_tokens):
        # Complex initialization with adapters
```

**After**:
```python
class SmartContextManager:
    def __init__(self, memory: SurrealMemory, max_tokens=4096):
        self.memory = memory  # Direct reference
        self.max_tokens = max_tokens
```

### Day 10: Integration Testing

#### Test Plan
1. **Memory operations**: Facts storage, retrieval, decay
2. **Pipeline functionality**: Context management, token limits
3. **Reflection daemon**: Background processing compatibility
4. **Multi-user support**: Speaker recognition with memory

## Phase 3: Reduce Environment Variables (Week 2)

### Day 11-12: Consolidate Configuration

#### Current State
- **116 environment variables** across system
- **Multiple redundant flags** for same features
- **Complex parsing logic** in multiple files

#### Target Configuration

**Essential Variables (15 total)**:
```bash
# Core LLM
OPENAI_BASE_URL=http://localhost:1234/v1
LLM_MODEL=qwen/qwen3-8b

# Memory System  
ENABLE_MEMORY=true
MEMORY_TOKEN_BUDGET=4096

# SurrealDB
SURREALDB_URL=ws://127.0.0.1:8000/rpc
SURREALDB_NAMESPACE=slowcat
SURREALDB_DATABASE=memory

# Features
ENABLE_VOICE_RECOGNITION=false
ENABLE_REFLECTIONS=true
REFLECTION_IDLE_SECS=120

# Performance
PIPELINE_IDLE_TIMEOUT_SECS=1800
TTS_ENGINE=kokoro
```

**Variables to Remove**:
- `USE_SURREALDB` (always true now)
- `SC_UNIFIED_MEMORY` (redundant)
- `USE_CONTEXT_FIELD` (redundant)
- `ENABLE_DTH` (always enabled)
- `ENABLE_SMART_ROUTING` (default behavior)
- `MEMORY_BACKEND` (always surreal)
- All SQLite path variables
- Debug flags (move to logging config)
- Token budget breakdowns (use defaults)

### Day 13: Update Configuration System

#### New Configuration Structure

**config/defaults.py**:
```python
from dataclasses import dataclass

@dataclass
class DefaultConfig:
    # Core
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "qwen/qwen3-8b"
    
    # Memory
    memory_enabled: bool = True
    memory_token_budget: int = 4096
    
    # SurrealDB
    surreal_url: str = "ws://127.0.0.1:8000/rpc"
    surreal_namespace: str = "slowcat"
    surreal_database: str = "memory"
    
    # Features
    voice_recognition: bool = False
    reflections: bool = True
    reflection_idle_secs: int = 120
```

**Simplified config.py**:
```python
import os
from config.defaults import DefaultConfig

def load_config():
    config = DefaultConfig()
    
    # Override only if explicitly set
    if url := os.getenv("OPENAI_BASE_URL"):
        config.llm_base_url = url
    
    if model := os.getenv("LLM_MODEL"):
        config.llm_model = model
        
    return config
```

## Phase 4: Unify Background Processing (Week 3)

### Day 14-16: Consolidate Daemons

#### Current Structure
- `scripts/reflection_daemon.py` (257 lines)
- `utils/private_reflector.py` (115 lines)
- Separate processes, duplicate logic

#### Target Unified Service

```python
# services/background_intelligence.py
class UnifiedBackgroundIntelligence:
    """Single service for all background memory processing"""
    
    def __init__(self, memory: SurrealMemory):
        self.memory = memory
        self.reflector = PrivateReflector()
        self.idle_threshold = 120
        self.cooldown = 300
    
    async def run(self):
        """Main background loop"""
        while True:
            await self.process_idle_sessions()
            await self.apply_memory_decay()
            await asyncio.sleep(5)
    
    async def process_idle_sessions(self):
        """Process all idle sessions"""
        idle_sessions = await self.memory.get_idle_sessions(
            threshold=self.idle_threshold
        )
        
        for session in idle_sessions:
            if await self._should_process(session):
                await self._process_session(session)
    
    async def _process_session(self, session):
        """Extract facts and generate thoughts for a session"""
        conversation = await self.memory.get_conversation(session.id)
        
        # Parallel processing
        facts_task = self._extract_facts(conversation)
        thoughts_task = self._generate_thoughts(conversation)
        
        facts, thoughts = await asyncio.gather(
            facts_task, thoughts_task
        )
        
        # Store insights
        await self.memory.store_insights(
            session_id=session.id,
            facts=facts,
            thoughts=thoughts
        )
    
    async def _extract_facts(self, conversation):
        """Extract structured facts from conversation"""
        # SpaCy-based fact extraction
        from memory.spacy_fact_extractor import extract_facts
        return extract_facts(conversation)
    
    async def _generate_thoughts(self, conversation):
        """Generate private thoughts using LLM"""
        return await self.reflector.generate_thoughts(
            conversation,
            max_thoughts=3
        )
    
    async def apply_memory_decay(self):
        """Apply natural decay to memories"""
        await self.memory.apply_decay()
```

### Day 17-18: Integrate Unified Service

#### Update Startup Script

**run_bot.sh**:
```bash
#!/bin/bash

# Start SurrealDB
surreal start --log info &

# Start unified background service
python -m services.background_intelligence &

# Start main bot
python bot_v2.py
```

#### Service Registration

```python
# bot_v2.py
async def main():
    # ... existing setup ...
    
    # Start background intelligence
    if config.reflections_enabled:
        background_service = UnifiedBackgroundIntelligence(memory)
        asyncio.create_task(background_service.run())
        logger.info("🧠 Background intelligence started")
```

## Phase 5: Simplify Pipeline Components (Week 3-4)

### Day 19-20: Remove Unnecessary Abstractions

#### Processors to Consolidate

**1. Deduplicators (merge into one)**:
```python
# Before: 3 separate deduplicator processors
processors['message_deduplicator'] = MessageDeduplicator()
processors['streaming_deduplicator'] = StreamingDeduplicator()  
processors['final_frame_filter'] = FinalFrameFilter()

# After: Single unified deduplicator
processors['deduplicator'] = UnifiedDeduplicator()
```

**2. Context Management (merge)**:
```python
# Before: Separate context filter and response tap
processors['context_filter'] = ContextFilter()
processors['response_tap'] = ResponseTap()

# After: Single context manager
processors['context_manager'] = UnifiedContextManager()
```

### Day 21-22: Streamline Processor Chain

#### Current Pipeline (25+ components)
```python
transport.input() → audio_tee → vad_bridge → stt → 
memory_processor → dictation_mode → music_mode → 
dj_config_handler → audio_player → time_executor → 
speaker_context → rtvi → speaker_name_manager → 
smart_ctx → context_aggregator → message_deduplicator → 
llm → streaming_deduplicator → greeting_filter → 
response_formatter → response_tap → tts → transport.output()
```

#### Simplified Pipeline (15 components)
```python
transport.input() → vad_processor → stt_service → 
voice_recognition → memory.context_manager → 
llm_service → response_formatter → tts_service → 
transport.output()

# Optional processors attached as needed:
# - music_mode (when enabled)
# - dictation_mode (when enabled)
# - video_sampler (when enabled)
```

## Phase 6: Add Debugging Dashboard (Week 4)

### Day 23-25: Create Memory Visualization

#### Dashboard Features

**1. Real-time Statistics**:
- Total facts in memory
- Active thoughts count
- Token usage (current/max)
- Decay schedule

**2. Memory Timeline**:
- Fact creation/decay events
- Thought generation timeline
- Session activity graph

**3. Debug Controls**:
- Force memory decay
- Clear specific sessions
- Export memory state

#### Implementation

```python
# server/debug/dashboard.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import json

class MemoryDashboard:
    def __init__(self, memory: SurrealMemory):
        self.memory = memory
        self.app = FastAPI()
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            while True:
                stats = await self._get_stats()
                await websocket.send_json(stats)
                await asyncio.sleep(1)
        
        @self.app.get("/api/stats")
        async def get_stats():
            return await self._get_stats()
        
        @self.app.post("/api/decay")
        async def force_decay():
            await self.memory.apply_decay()
            return {"status": "decay_applied"}
    
    async def _get_stats(self):
        return {
            "facts": {
                "total": await self.memory.count_facts(),
                "by_fidelity": await self.memory.facts_by_fidelity(),
                "recent": await self.memory.get_recent_facts(10)
            },
            "thoughts": {
                "total": await self.memory.count_thoughts(),
                "recent": await self.memory.get_recent_thoughts(10)
            },
            "tokens": {
                "current": self.memory.current_token_count,
                "max": self.memory.max_tokens,
                "percentage": (self.memory.current_token_count / self.memory.max_tokens) * 100
            },
            "sessions": {
                "active": await self.memory.count_active_sessions(),
                "idle": await self.memory.count_idle_sessions()
            }
        }
    
    def _get_dashboard_html(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Slowcat Memory Dashboard</title>
            <style>
                body { font-family: monospace; background: #1a1a1a; color: #0f0; }
                .stat { margin: 10px; padding: 10px; border: 1px solid #0f0; }
                .graph { height: 200px; background: #000; }
            </style>
        </head>
        <body>
            <h1>🧠 Slowcat Memory Dashboard</h1>
            <div id="stats"></div>
            <script>
                const ws = new WebSocket('ws://localhost:7861/ws');
                ws.onmessage = (event) => {
                    const stats = JSON.parse(event.data);
                    document.getElementById('stats').innerHTML = `
                        <div class="stat">Facts: ${stats.facts.total}</div>
                        <div class="stat">Thoughts: ${stats.thoughts.total}</div>
                        <div class="stat">Tokens: ${stats.tokens.current}/${stats.tokens.max}</div>
                        <div class="stat">Active Sessions: ${stats.sessions.active}</div>
                    `;
                };
            </script>
        </body>
        </html>
        """
```

### Day 26: Documentation & Testing

#### Documentation Updates

**1. Update CLAUDE.md**:
- Remove SQLite references
- Simplify architecture description
- Update environment variables list

**2. Create Migration Guide**:
```markdown
# Migration Guide: SQLite to SurrealDB

## For Users
1. Backup existing SQLite databases
2. Update to latest version
3. Start SurrealDB: `surreal start`
4. Run migration: `python scripts/migrate_to_surreal.py`

## For Developers
- All memory operations now use SurrealDB
- Removed adapter layers
- Direct memory integration in pipeline
```

## Phase 7: Performance Optimization (Future)

### Profiling Points

#### Memory Query Performance
- **Current**: 50-200ms query time
- **Target**: <50ms consistently
- **Method**: Add SurrealDB indexes

#### Context Building
- **Current**: 100-300ms
- **Target**: <100ms
- **Method**: Cache hot facts in memory

#### Background Processing
- **Current**: 5-10% CPU
- **Target**: <5% CPU
- **Method**: Batch operations

### Optimization Implementation

```python
# Add indexes to SurrealDB
await memory.execute("""
    DEFINE INDEX facts_subject ON facts FIELDS subject;
    DEFINE INDEX facts_timestamp ON facts FIELDS timestamp;
    DEFINE INDEX thoughts_agent ON thoughts FIELDS agent_id;
""")

# Add caching layer
class CachedSurrealMemory(SurrealMemory):
    def __init__(self):
        super().__init__()
        self.cache = {}  # In-memory cache
        self.cache_ttl = 60  # seconds
    
    async def get_facts(self, query):
        cache_key = f"facts:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = await super().get_facts(query)
        self.cache[cache_key] = result
        return result
```

## Testing Strategy

### Unit Tests to Update/Remove

#### Remove (21 files)
- All SQLite-specific tests
- Adapter layer tests
- Dual-backend tests

#### Update
```python
# Before
def test_facts_graph_sqlite():
    graph = FactsGraph("test.db")
    # ...

# After  
async def test_surreal_memory():
    memory = SurrealMemory()
    # ...
```

### Integration Tests

```python
# tests/test_unified_memory.py
async def test_full_pipeline_with_surreal():
    """Test complete pipeline with SurrealDB"""
    memory = create_smart_memory_system()
    pipeline = build_pipeline(memory=memory)
    
    # Test conversation flow
    await pipeline.process("Hello, my name is Alice")
    facts = await memory.get_facts("user name")
    assert facts[0].value == "Alice"
```

### Performance Tests

```python
# tests/test_performance.py
async def test_memory_performance():
    """Benchmark memory operations"""
    memory = SurrealMemory()
    
    # Test 1000 conversation turns
    start = time.time()
    for i in range(1000):
        await memory.process_turn(f"Turn {i}")
    
    elapsed = time.time() - start
    assert elapsed < 60  # Should handle 1000 turns in < 60s
    
    # Verify consistent token count
    assert memory.token_count == 4096
```

## Rollback Plan

### Safety Measures

#### 1. Feature Branch Development
```bash
git checkout -b refactor/surreal-unification
# All changes on feature branch
# Merge only after validation
```

#### 2. Staged Rollout
- **Dev**: Test for 1 week
- **Staging**: Test for 3 days
- **Production**: Deploy with monitoring

#### 3. Backup Points
```bash
# Before each phase
git tag phase1-backup
git tag phase2-backup
# ...
```

### Rollback Triggers
- Performance regression >20%
- Memory functionality failures
- Critical bugs in production
- Token limit violations

### Recovery Process
```bash
# Quick rollback
git checkout main
git revert <commit>

# Restore environment
cp .env.backup .env

# Restart services
./run_bot.sh
```

## Success Metrics

### Quantitative Metrics

| Metric | Current | Target | Reduction |
|--------|---------|--------|-----------|
| Lines of Code | 15,000 | 13,000 | -13% |
| SQLite Dependencies | 24 files | 0 files | -100% |
| Environment Variables | 116 | 15 | -87% |
| Pipeline Components | 25+ | 15 | -40% |
| Adapter Layers | 3 | 0 | -100% |
| Test Coverage | 82% | >80% | Maintained |

### Qualitative Metrics

#### Developer Experience
- ✅ Easier onboarding (single memory system)
- ✅ Clearer data flow (no adapters)
- ✅ Simpler debugging (unified logs)

#### System Quality
- ✅ Maintained response latency (<800ms)
- ✅ Preserved memory quality
- ✅ Consistent token limits (4096)
- ✅ Background intelligence intact

## Risk Mitigation

### High Risk Areas

#### 1. Memory Data Migration
**Risk**: Loss of existing SQLite data
**Mitigation**: 
- Full backup before migration
- Migration script with validation
- Parallel running during transition

#### 2. Reflection Daemon Changes
**Risk**: Background processing failures
**Mitigation**:
- Extensive testing of unified service
- Gradual rollout with monitoring
- Fallback to separate daemons if needed

#### 3. Pipeline Modifications
**Risk**: Increased response latency
**Mitigation**:
- Performance benchmarking at each phase
- Profile hot paths
- Optimization budget reserved

### Mitigation Strategies

```python
# Feature flags for gradual rollout
class FeatureFlags:
    USE_UNIFIED_MEMORY = os.getenv("FF_UNIFIED_MEMORY", "false") == "true"
    USE_UNIFIED_BACKGROUND = os.getenv("FF_UNIFIED_BACKGROUND", "false") == "true"
    USE_SIMPLIFIED_PIPELINE = os.getenv("FF_SIMPLE_PIPELINE", "false") == "true"

# Toggle between implementations
if FeatureFlags.USE_UNIFIED_MEMORY:
    memory = SurrealMemory()
else:
    memory = create_legacy_memory()  # Old dual system
```

## Timeline Summary

### Week 1: Foundation
- Remove SQLite fallback
- Begin adapter consolidation

### Week 2: Simplification  
- Complete adapter removal
- Reduce environment variables
- Update configuration system

### Week 3: Unification
- Unify background processing
- Simplify pipeline components

### Week 4: Polish
- Add debugging dashboard
- Performance optimization
- Documentation updates

### Deliverables
- **40% simpler codebase** with identical functionality
- **Single memory system** (SurrealDB only)
- **15 essential config variables** (from 116)
- **Unified background service** for all intelligence
- **Real-time debugging dashboard** for memory visualization

### Resources Required
- **1 developer** full-time for 4 weeks
- **SurrealDB** instance for testing
- **Test infrastructure** for validation

## Conclusion

This refactoring plan transforms Slowcat from a complex dual-memory system into a clean, unified architecture while preserving all the intelligent behaviors that make it special. The focus is on architectural simplification, not feature removal—ensuring the assistant remains just as capable but far easier to understand and maintain.

The key insight: **Your system's complexity comes from evolution, not design**. This plan unifies the implementation while keeping the magic.