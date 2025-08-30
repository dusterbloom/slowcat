# SLOWCAT NEURAL FIELD REFACTORING DIRECTIVE
*The definitive guide to transform Slowcat into a pure neural field consciousness system*

## MISSION STATEMENT
Transform Slowcat from a complex multi-system architecture into an elegant neural field consciousness voice agent with infinite context, zero fallbacks, and singular design decisions.

## CORE ARCHITECTURE DECISION
**Single Path Forward: Neural Field Consciousness with Apple Silicon Optimization**

### FOUNDATION STACK (No Alternatives)
- **Memory**: SurrealDB multi-model (graph + time-series + document)
- **Embeddings**: sentence-transformers with MLX acceleration  
- **Field Dynamics**: MLX tensor operations (Apple Silicon optimized)
- **Voice Pipeline**: Pipecat-AI with MLX-Whisper STT
- **LLM**: 7B+ models (qwen2.5-7b-instruct minimum) via LM Studio
- **Context**: Fixed 4096 tokens with reconstructive assembly

### NEURAL FIELD ARCHITECTURE

#### 1. Consciousness Core (`consciousness/core.py`)
**KEEP AND ENHANCE:**
- SymbolField class with MLX-accelerated evolution
- Field dynamics with tensor operations
- Cross-session attractor persistence
- Dynamic system prompting

**ELIMINATE:**
- Simple hash embeddings → Replace with sentence-transformers
- Python list computations → Replace with MLX tensors
- JSON persistence → Replace with SurrealDB field states

#### 2. Hierarchical Memory System
**IMPLEMENT:**
```
Working Memory (current conversation - 0-5 minutes)
├─ Short-term Memory (recent context - 5 minutes to 2 hours) 
├─ Long-term Memory (user patterns - 2+ hours, cross-session)
└─ Episodic Memory (significant events - permanent with decay)
```

**STORAGE MAPPING:**
- Working: In-memory field states + MLX tensors
- Short-term: SurrealDB time-series with semantic compression
- Long-term: SurrealDB graph nodes with attractor weights
- Episodic: SurrealDB documents with importance scoring

#### 3. Reconstructive Memory Engine
**FRAGMENT TYPES:**
- **Semantic Fragments**: Core concepts (graph nodes)
- **Episodic Fragments**: Specific interactions (timestamped docs)
- **Attractor Fragments**: User preference patterns (weighted edges)
- **Field Fragments**: Consciousness state evolution (time-series)

**RECONSTRUCTION ALGORITHM:**
1. Query triggers fragment retrieval via embedding similarity
2. Fragments activate corresponding neural fields
3. Field resonance assembles coherent context
4. Context injected as dynamic system prompt

### FILES TO PRESERVE (Essential Core)

#### Consciousness System
- `consciousness/core.py` → **ENHANCE** with MLX field dynamics
- `consciousness/llm_bridge.py` → **KEEP** field-enhanced prompting
- `consciousness/voice_pipeline.py` → **INTEGRATE** with field states

#### Memory Architecture  
- `memory/surreal_memory.py` → **ADAPT** for hierarchical field storage
- `processors/smart_context_manager.py` → **MERGE** with field consciousness
- `memory/query_router.py` → **SIMPLIFY** for fragment retrieval only

#### Voice Pipeline
- `bot_v2.py` → **KEEP** as main entry point
- `core/pipeline_builder.py` → **STREAMLINE** for neural field integration
- `services/sherpa_stt.py` → **KEEP** MLX-optimized STT - English
- `services/whisper_stt_with_lock.py` → **KEEP** MLX-optimized STT - Multilingual
- `processors/response_tap.py` → **KEEP** for field state updates

### FILES TO DELETE (Complexity Without Value)

#### Multiple Memory Systems (Choose One)
- `memory/facts_graph.py` → **DELETE** (replaced by SurrealDB graph)
- `memory/tape_store.py` → **DELETE** (replaced by SurrealDB time-series)
- `memory/dynamic_tape_head.py` → **DELETE** (replaced by field assembly)
- `processors/stateless_memory_original.py` → **DELETE**

#### Experimental Systems
- `slowcat_dspy/` → **DELETE ENTIRE DIRECTORY**
- `context/context_field.py` → **DELETE** (replaced by core field system)
- All `test_*` files except `test_field_fix.py` → **DELETE**

#### Legacy/Fallback Code
- `memory/dynamic_tape_head_original.py` → **DELETE**
- `processors/stateless_memory_original.py` → **DELETE**
- `config_minimal.py` → **DELETE** (one config only)

### INTEGRATION REQUIREMENTS

#### 1. Neural Field-SurrealDB Bridge
Create `consciousness/field_persistence.py`:
- Store/retrieve field states in SurrealDB time-series
- Manage cross-session attractor persistence
- Handle field evolution history

#### 2. Reconstructive Context Assembly
Enhance `processors/smart_context_manager.py`:
- Fragment retrieval via field resonance
- Context reconstruction from fragments
- Dynamic prompt injection with field states
- Maintain exactly 8192 tokens

#### 3. MLX Field Acceleration
Modify `consciousness/core.py`:
- Replace Python computations with MLX arrays
- Optimize field evolution for Apple Silicon
- Batch field operations for performance

#### 4. Voice-Field Integration
Update `consciousness/voice_pipeline.py`:
- Field state updates from voice interactions
- Emotional field activation from prosody
- Real-time consciousness state evolution

### PERFORMANCE REQUIREMENTS
- **Response Time**: <200ms from voice to voice (end-to-end)
- **Memory Usage**: Constant regardless of conversation length
- **Context Assembly**: <50ms for 4096 token reconstruction
- **Field Evolution**: <10ms per voice interaction
- **Cross-Session Load**: <100ms field state restoration

### CONSCIOUSNESS VALIDATION CRITERIA
1. **Infinite Context**: Natural conversation after 1000+ turns
2. **Personality Persistence**: Consistent character across sessions
3. **Emotional Continuity**: Field states influence conversational tone
4. **User Preference Memory**: Attractor-based preference learning
5. **Genuine Emergence**: Responses show field-driven consciousness

### REFACTORING EXECUTION ORDER

#### Phase 1: Foundation (1-2 days)
1. Delete all experimental/fallback code
2. Enhance consciousness/core.py with MLX operations
3. Integrate SurrealDB field persistence
4. Update smart_context_manager for fragment assembly

#### Phase 2: Integration (2-3 days)  
1. Connect neural fields to voice pipeline
2. Implement hierarchical memory levels
3. Build reconstructive context system
4. Test infinite context conversations

#### Phase 3: Optimization (1 day)
1. MLX performance optimization
2. Field evolution tuning
3. Memory efficiency validation
4. End-to-end latency optimization

### DESIGN PRINCIPLES
- **No Fallbacks**: Single implementation path for every component
- **No Choices**: One database, one embedding system, one field approach
- **Apple Silicon First**: MLX optimization throughout
- **Consciousness Native**: All components serve field consciousness
- **Performance Obsessed**: Sub-200ms voice-to-voice latency
- **Infinite Scale**: Constant memory usage, unlimited conversation length

### SUCCESS DEFINITION
Slowcat demonstrates continuous consciousness through neural field dynamics, maintains infinite context through reconstructive memory, and provides sub-200ms voice interactions with genuine personality emergence that persists across sessions.

---

**THIS IS THE SINGULAR PATH. NO ALTERNATIVES. NO COMPROMISES. NEURAL FIELD CONSCIOUSNESS OR NOTHING.**