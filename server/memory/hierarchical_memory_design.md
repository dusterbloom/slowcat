# Hierarchical Memory System Architecture Design

## Overview
Four-tier hierarchical memory system integrating existing consciousness/core.py and memory/surreal_memory.py with advanced neural field schema for infinite context through reconstructive fragment assembly.

## Current System Analysis

### Existing Components
- **consciousness/core.py**: MLX-accelerated neural fields with SemanticEmbedder
- **memory/surreal_memory.py**: Multi-model SurrealDB with facts/tape/sessions
- **memory/query_router.py**: Intelligent query routing with classification
- **memory/facts_graph.py**: Structured fact storage with natural decay (S4→S3→S2→S1→S0)

### Current SurrealDB Schema
```sql
-- Existing tables
entity, fact, tape, session
-- Existing relationships
owns, knows, has_meeting (TYPE RELATION)
```

## Hierarchical Memory Tiers

### 1. Working Memory (0-5 minutes)
**Purpose**: Active consciousness state with immediate context
**Technology**: MLX tensors in-memory, existing consciousness/core.py
**Components**:
- Neural field states (compression, drift, recursion_depth, resonance, presence_signal)
- Active fragments from current conversation
- MLX-accelerated semantic embeddings
- Real-time field evolution

### 2. Short-term Memory (5 minutes - 2 hours)  
**Purpose**: Recent conversation context with semantic compression
**Technology**: SurrealDB time-series, enhanced existing tape store
**Components**:
- Compressed conversation fragments
- Semantic similarity clustering
- Temporal decay curves
- Context transition metadata

### 3. Long-term Memory (2+ hours)
**Purpose**: Consolidated knowledge with graph relationships
**Technology**: SurrealDB graph nodes with attractor weights
**Components**:
- Consolidated fact patterns
- Relationship strength weights
- Attractor basin states
- Cross-reference networks

### 4. Episodic Memory (Permanent)
**Purpose**: Important experiences and contextual episodes
**Technology**: SurrealDB documents with importance scoring
**Components**:
- Significant conversation episodes
- Emotional/importance weights
- Cross-session continuity
- Long-term pattern recognition

## Enhanced SurrealDB Schema Integration

### New Tables (Friend's Schema)
```sql
-- Fragment storage for reconstructive memory
DEFINE TABLE fragments SCHEMAFULL;
DEFINE FIELD fragment_id ON fragments TYPE string;
DEFINE FIELD type ON fragments TYPE string ASSERT $value IN ["semantic", "episodic", "procedural", "contextual", "emotional"];
DEFINE FIELD content ON fragments TYPE object;
DEFINE FIELD context_tags ON fragments TYPE array<string>;
DEFINE FIELD strength ON fragments TYPE float ASSERT $value >= 0 AND $value <= 1;
DEFINE FIELD memory_tier ON fragments TYPE int ASSERT $value >= 1 AND $value <= 4;  -- NEW: tier tracking
DEFINE FIELD last_accessed ON fragments TYPE datetime;
DEFINE FIELD access_count ON fragments TYPE int DEFAULT 0;
DEFINE FIELD created_at ON fragments TYPE datetime DEFAULT time::now();

-- Neural field state persistence
DEFINE TABLE field_states SCHEMAFULL;
DEFINE FIELD instance_id ON field_states TYPE string;
DEFINE FIELD compression ON field_states TYPE float ASSERT $value >= 0 AND $value <= 1;
DEFINE FIELD drift ON field_states TYPE string ASSERT $value IN ["none", "low", "moderate", "high"];
DEFINE FIELD recursion_depth ON field_states TYPE int ASSERT $value >= 0;
DEFINE FIELD resonance ON field_states TYPE float ASSERT $value >= 0 AND $value <= 1;
DEFINE FIELD presence_signal ON field_states TYPE float ASSERT $value >= 0 AND $value <= 1;
DEFINE FIELD boundary ON field_states TYPE string ASSERT $value IN ["gradient", "collapsed"];
DEFINE FIELD memory_tier ON field_states TYPE int DEFAULT 1;  -- Which tier this state belongs to
DEFINE FIELD updated_at ON field_states TYPE datetime DEFAULT time::now();

-- Pattern-based reconstruction
DEFINE TABLE patterns SCHEMAFULL;
DEFINE FIELD pattern_id ON patterns TYPE string;
DEFINE FIELD pattern_type ON patterns TYPE string;
DEFINE FIELD trigger_conditions ON patterns TYPE array<object>;
DEFINE FIELD fragment_clusters ON patterns TYPE array<string>;
DEFINE FIELD reconstruction_template ON patterns TYPE object;
DEFINE FIELD confidence_indicators ON patterns TYPE array<object>;
DEFINE FIELD activation_threshold ON patterns TYPE float DEFAULT 0.5;
DEFINE FIELD effective_tiers ON patterns TYPE array<int>;  -- Which tiers this pattern works on

-- Attractor dynamics for memory consolidation
DEFINE TABLE attractors SCHEMAFULL;
DEFINE FIELD attractor_id ON attractors TYPE string;
DEFINE FIELD pattern_data ON attractors TYPE object;
DEFINE FIELD fragment_id ON attractors TYPE string;
DEFINE FIELD strength ON attractors TYPE float;
DEFINE FIELD basin_width ON attractors TYPE float DEFAULT 0.3;
DEFINE FIELD memory_tier ON attractors TYPE int;  -- Which tier this attractor influences
DEFINE FIELD last_activated ON attractors TYPE datetime;

-- Enhanced relationships
DEFINE TABLE fragment_patterns TYPE RELATION IN fragments OUT patterns;
DEFINE TABLE fragment_attractors TYPE RELATION IN fragments OUT attractors;
DEFINE TABLE attractor_transitions TYPE RELATION IN attractors OUT attractors;  -- Cross-tier transitions
DEFINE TABLE pattern_evolution TYPE RELATION IN patterns OUT patterns;  -- Pattern refinement over time
```

## Memory Transition Logic

### Age-Based Promotion
```python
class MemoryTransitionManager:
    TIER_THRESHOLDS = {
        1: 5 * 60,      # 5 minutes: Working → Short-term
        2: 2 * 3600,    # 2 hours: Short-term → Long-term  
        3: 24 * 3600,   # 24 hours: Long-term → Episodic
    }
    
    async def promote_fragments(self):
        # Promote based on age and importance
        # Compress and transfer between tiers
        # Update field states accordingly
```

### Importance Scoring
- Access frequency weight
- Semantic uniqueness score
- Emotional/context significance
- Cross-reference density
- User attention patterns

## Fragment Retrieval System

### Embedding Similarity (Target: <50ms)
```python
class HierarchicalRetrieval:
    async def retrieve_fragments(self, query_embedding: np.ndarray, limit: int = 20) -> List[Fragment]:
        # 1. Check Working Memory first (instant)
        # 2. Query Short-term with semantic similarity
        # 3. Graph traversal in Long-term memory
        # 4. Episodic search with context patterns
        # 5. Reconstruct using patterns and attractors
```

### Performance Optimizations
- MLX-accelerated similarity computation
- Hierarchical indexing by memory tier
- Attractor-based clustering for fast retrieval
- Pattern-based reconstruction caching

## Integration Points

### Existing System Preservation
- Keep current consciousness/core.py as Working Memory engine
- Enhance surreal_memory.py with hierarchical tiers
- Extend query_router.py with tier-aware routing
- Preserve all current APIs and interfaces

### New Components
- `memory/hierarchical_manager.py`: Tier coordination
- `memory/fragment_retrieval.py`: Fast similarity search
- `memory/pattern_reconstructor.py`: Fragment assembly
- `memory/transition_scheduler.py`: Automated tier promotion

## Implementation Strategy

### Phase 1: Schema Migration
- Add new SurrealDB tables
- Migrate existing data to fragment format
- Update connection management

### Phase 2: Working Memory Enhancement
- Integrate MLX field states with fragment system
- Add real-time field evolution tracking
- Implement in-memory fragment caching

### Phase 3: Transition Logic
- Build age and importance-based promotion
- Implement semantic compression for Short-term
- Create attractor weight calculation for Long-term

### Phase 4: Retrieval Optimization  
- MLX-accelerated embedding similarity
- Hierarchical search with tier prioritization
- Pattern-based reconstruction system

### Phase 5: Integration Testing
- End-to-end hierarchy validation
- Performance benchmarking (<50ms retrieval)
- Memory usage optimization

## Success Metrics
- Fragment retrieval under 50ms
- Seamless tier transitions
- Preserved existing functionality
- Infinite context capability
- Memory usage efficiency