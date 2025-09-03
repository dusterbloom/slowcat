# Task 17: M3 Memory Persistence Implementation Plan

## Context & Analysis
- **Goal**: Integrate M3-inspired memory system with SurrealDB for Apple Silicon optimization
- **Current State**: Existing SurrealDB schema with messages, sessions, knowledge tables
- **Approach**: Keep good existing components, remove legacy, add M3-specific tables

## Schema Analysis

### ✅ KEEP (Already Good):
- `messages` table - conversation storage with embeddings
- `sessions` table - conversation session tracking  
- `speakers` table - speaker identification and voice profiles
- `entity` table - entity recognition and canonical names
- Relations: `message_belongs_to`, `session_involves`, `entity_mentioned_in`

### ❌ REMOVE (Legacy/Redundant):
- `memory_fragments` table - replaced by M3 nodes
- `field_states` table - consciousness field not needed
- `engrams` and all engram-related tables (`engram_appears_in`, `engram_contains`, etc.)
- `knowledge` table - replaced by M3 semantic nodes
- Complex engram functions in schema

### ➕ ADD (M3-Specific Components):

#### 1. M3 Core Tables

**m3_nodes** - Core memory nodes with content and embeddings
```sql
DEFINE TABLE m3_nodes SCHEMAFULL;
DEFINE FIELD node_id ON m3_nodes TYPE int;
DEFINE FIELD node_type ON m3_nodes TYPE string ASSERT $value IN ['voice', 'episodic', 'semantic'];
DEFINE FIELD contents ON m3_nodes TYPE array<string>;  -- Actual content!
DEFINE FIELD embeddings ON m3_nodes TYPE array<array<float>>;  -- Multiple embeddings per node
DEFINE FIELD clip_id ON m3_nodes TYPE int;  -- Temporal reference
DEFINE FIELD source_message ON m3_nodes TYPE option<record<messages>>;  -- Link to origin message
DEFINE FIELD metadata ON m3_nodes TYPE object {
    speaker_id: string,
    confidence: float,
    extraction_method: string,
    created_at: datetime,
    last_accessed: datetime,
    access_count: int
};

-- Indexes
DEFINE INDEX m3_nodes_type_idx ON m3_nodes FIELDS node_type;
DEFINE INDEX m3_nodes_clip_idx ON m3_nodes FIELDS clip_id;
DEFINE INDEX m3_nodes_speaker_idx ON m3_nodes FIELDS metadata.speaker_id;
DEFINE INDEX m3_nodes_id_idx ON m3_nodes FIELDS node_id UNIQUE;
```

**m3_edges** - Graph relationships between nodes
```sql
DEFINE TABLE m3_edges SCHEMAFULL;
DEFINE FIELD source ON m3_edges TYPE record<m3_nodes>;
DEFINE FIELD target ON m3_edges TYPE record<m3_nodes>;
DEFINE FIELD weight ON m3_edges TYPE float DEFAULT 1.0;
DEFINE FIELD edge_type ON m3_edges TYPE string DEFAULT 'similarity';
DEFINE FIELD created_at ON m3_edges TYPE datetime DEFAULT time::now();
DEFINE FIELD last_reinforced ON m3_edges TYPE datetime DEFAULT time::now();

-- Indexes
DEFINE INDEX m3_edges_source_idx ON m3_edges FIELDS source;
DEFINE INDEX m3_edges_target_idx ON m3_edges FIELDS target;
DEFINE INDEX m3_edges_weight_idx ON m3_edges FIELDS weight;
```

**m3_equivalences** - Identity resolution for entities/speakers
```sql
DEFINE TABLE m3_equivalences SCHEMAFULL;
DEFINE FIELD canonical_id ON m3_equivalences TYPE string;
DEFINE FIELD node_ids ON m3_equivalences TYPE array<int>;
DEFINE FIELD entity_type ON m3_equivalences TYPE string DEFAULT 'speaker';
DEFINE FIELD confidence ON m3_equivalences TYPE float DEFAULT 0.8;
DEFINE FIELD created_at ON m3_equivalences TYPE datetime DEFAULT time::now();
DEFINE FIELD last_updated ON m3_equivalences TYPE datetime DEFAULT time::now();

-- Indexes
DEFINE INDEX m3_equiv_canonical_idx ON m3_equivalences FIELDS canonical_id UNIQUE;
DEFINE INDEX m3_equiv_type_idx ON m3_equivalences FIELDS entity_type;
```

**m3_clips** - Temporal organization of memory
```sql
DEFINE TABLE m3_clips SCHEMAFULL;
DEFINE FIELD clip_id ON m3_clips TYPE int;
DEFINE FIELD session_id ON m3_clips TYPE string;  -- Links to sessions table
DEFINE FIELD start_time ON m3_clips TYPE datetime;
DEFINE FIELD end_time ON m3_clips TYPE option<datetime>;
DEFINE FIELD node_count ON m3_clips TYPE int DEFAULT 0;
DEFINE FIELD dominant_speaker ON m3_clips TYPE option<string>;

-- Indexes
DEFINE INDEX m3_clips_id_idx ON m3_clips FIELDS clip_id UNIQUE;
DEFINE INDEX m3_clips_session_idx ON m3_clips FIELDS session_id;
DEFINE INDEX m3_clips_time_idx ON m3_clips FIELDS start_time;
```

#### 2. Bridge Relations (Connect M3 to Existing)

**m3_node_from_message** - Links M3 nodes to their source messages
```sql
DEFINE TABLE m3_node_from_message TYPE RELATION IN m3_nodes OUT messages;
DEFINE FIELD extraction_confidence ON m3_node_from_message TYPE float DEFAULT 0.8;
DEFINE FIELD extraction_method ON m3_node_from_message TYPE string;
DEFINE FIELD created_at ON m3_node_from_message TYPE datetime DEFAULT time::now();
```

**m3_node_about_entity** - Links M3 nodes to entities they describe
```sql
DEFINE TABLE m3_node_about_entity TYPE RELATION IN m3_nodes OUT entity;
DEFINE FIELD relevance_score ON m3_node_about_entity TYPE float DEFAULT 1.0;
DEFINE FIELD relationship_type ON m3_node_about_entity TYPE string;
DEFINE FIELD discovered_at ON m3_node_about_entity TYPE datetime DEFAULT time::now();
```

## Implementation Files

### 1. Schema Migration (`server/schema/m3_migration.surql`)
- Remove deprecated tables (engrams, memory_fragments, field_states, knowledge)
- Add M3 tables
- Create bridge relations
- Add M3-specific functions

### 2. M3 Integration Layer (`server/memory/m3_surreal_integration.py`)
```python
class M3SurrealIntegration:
    """M3 memory integration with SurrealDB"""
    
    async def store_m3_node(self, node_type: str, contents: List[str], 
                           embeddings: List[List[float]], clip_id: int,
                           source_message_id: str = None) -> int:
        """Store M3 node with content and embeddings"""
    
    async def create_m3_edge(self, source_node_id: int, target_node_id: int,
                            weight: float, edge_type: str = 'similarity') -> bool:
        """Create edge between M3 nodes"""
    
    async def search_similar_nodes(self, query_embedding: List[float],
                                  node_type: str = None, limit: int = 10) -> List[Dict]:
        """Vector similarity search across M3 nodes"""
    
    async def resolve_equivalence(self, entity_name: str, node_ids: List[int]) -> str:
        """Create or update equivalence relation"""
    
    async def get_clip_nodes(self, clip_id: int) -> List[Dict]:
        """Get all nodes in a temporal clip"""
    
    async def create_new_clip(self, session_id: str) -> int:
        """Start new temporal clip"""
```

### 3. M3 Memory Processor (`server/processors/m3_memory_processor.py`)
```python
class M3MemoryProcessor(FrameProcessor):
    """Process frames into M3 memory nodes"""
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process different frame types into M3 nodes"""
        
        if isinstance(frame, TranscriptionFrame):
            await self._create_voice_node(frame)
        elif isinstance(frame, TextFrame) and self._is_assistant_response:
            await self._create_semantic_nodes(frame)
    
    async def _create_voice_node(self, frame: TranscriptionFrame):
        """Create voice node from audio transcription"""
    
    async def _create_semantic_nodes(self, frame: TextFrame):
        """Extract and create semantic nodes from text"""
    
    async def _infer_edges(self, new_node_id: int):
        """Infer edges to related nodes based on similarity"""
```

### 4. Migration Script (`server/scripts/migrate_to_m3.py`)
```python
class M3Migration:
    """Safely migrate existing schema to M3"""
    
    async def backup_existing_data(self):
        """Backup current messages, sessions, entities"""
    
    async def apply_schema_migration(self):
        """Apply M3 schema changes"""
    
    async def migrate_existing_messages(self, limit: int = 1000):
        """Convert existing messages to M3 nodes (optional)"""
    
    async def verify_migration(self):
        """Test M3 operations work correctly"""
```

### 5. Configuration Updates (`server/run_bot.sh`)
```bash
# M3 Memory Configuration
export M3_MEMORY_ENABLED=${M3_MEMORY_ENABLED:-false}
export M3_NAMESPACE=${M3_NAMESPACE:-slowcat}
export M3_DATABASE=${M3_DATABASE:-m3_memory}
export M3_EMBEDDING_MODEL=${M3_EMBEDDING_MODEL:-all-MiniLM-L6-v2}
export M3_CLIP_DURATION_SECONDS=${M3_CLIP_DURATION_SECONDS:-30}
export M3_EDGE_THRESHOLD=${M3_EDGE_THRESHOLD:-0.7}
export M3_DECAY_RATE=${M3_DECAY_RATE:-0.01}
```

### 6. Integration Tests (`server/tests/test_m3_integration.py`)
```python
class TestM3Integration:
    """Test M3 memory system integration"""
    
    async def test_node_creation_with_embeddings(self):
        """Test creating nodes with content and embeddings"""
    
    async def test_edge_inference(self):
        """Test automatic edge creation between similar nodes"""
    
    async def test_equivalence_resolution(self):
        """Test speaker/entity equivalence tracking"""
    
    async def test_clip_based_retrieval(self):
        """Test temporal clip organization"""
    
    async def test_similarity_search(self):
        """Test vector similarity search across nodes"""
```

## Key Relationships

### Message → M3 Node Flow:
1. **User Message** → `messages` table
2. **Audio Processing** → Voice M3 node (with audio embeddings)
3. **Fact Extraction** → Semantic M3 nodes (with semantic embeddings)  
4. **Edge Inference** → Connect related nodes
5. **Entity Linking** → Link nodes to entities via relations

### Query Flow:
1. **User Query** → Generate embedding
2. **Similarity Search** → Find similar M3 nodes
3. **Graph Traversal** → Follow edges to related nodes
4. **Temporal Filter** → Filter by clips/sessions
5. **Entity Resolution** → Resolve equivalences

## Performance Targets
- Node creation: <5ms
- Edge inference: <10ms
- Similarity search: <20ms for 10k nodes
- Memory footprint: <100MB for 100k nodes
- Clip organization: Real-time during conversation

## Migration Safety
- **Non-destructive**: Keeps existing messages/sessions
- **Optional**: Can run alongside current system
- **Reversible**: Backup before migration
- **Gradual**: Can migrate data incrementally

## Next Steps When Resuming:
1. Start with `server/schema/m3_migration.surql` - the schema file
2. Then `server/memory/m3_surreal_integration.py` - the integration layer
3. Test schema migration with `server/scripts/migrate_to_m3.py`
4. Create processor and update run_bot.sh
5. Write comprehensive tests

## Key Design Principles:
- **M3 nodes hold content AND embeddings** (not empty containers)
- **Multiple embeddings per node** (audio + semantic)
- **Temporal organization via clips**
- **Graph relationships for memory connections**
- **Equivalence resolution for identity tracking**
- **Bridge relations to existing schema components**