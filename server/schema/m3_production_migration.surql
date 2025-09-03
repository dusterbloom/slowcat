-- M3 Production Migration Schema for slowcat/memory_graph
-- Integrates M3 memory system with existing sessions/messages structure
-- This migration preserves existing data and adds M3 components

-- ============================================================================
-- M3 CORE TABLES
-- ============================================================================

-- Users/Characters table (central identity management)
DEFINE TABLE users SCHEMAFULL;
DEFINE FIELD user_id ON users TYPE string;
DEFINE FIELD username ON users TYPE string;
DEFINE FIELD display_name ON users TYPE option<string>;
DEFINE FIELD voice_id ON users TYPE option<string>;  -- For voice recognition
DEFINE FIELD face_id ON users TYPE option<string>;   -- For future face recognition
DEFINE FIELD created_at ON users TYPE datetime DEFAULT time::now();
DEFINE FIELD last_seen ON users TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata ON users TYPE object DEFAULT {};

-- Indexes for users
DEFINE INDEX users_id_idx ON users FIELDS user_id UNIQUE;
DEFINE INDEX users_username_idx ON users FIELDS username UNIQUE;
DEFINE INDEX users_voice_idx ON users FIELDS voice_id;

-- Characters table (assistant personas, NPCs, etc.)  
DEFINE TABLE characters SCHEMAFULL;
DEFINE FIELD character_id ON characters TYPE string;
DEFINE FIELD name ON characters TYPE string;
DEFINE FIELD role ON characters TYPE string DEFAULT 'assistant';
DEFINE FIELD personality ON characters TYPE option<string>;
DEFINE FIELD voice_id ON characters TYPE option<string>;
DEFINE FIELD created_at ON characters TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata ON characters TYPE object DEFAULT {};

-- Indexes for characters
DEFINE INDEX characters_id_idx ON characters FIELDS character_id UNIQUE;
DEFINE INDEX characters_name_idx ON characters FIELDS name;

-- M3 Clips: Temporal organization of memory (tied to sessions)
DEFINE TABLE m3_clips SCHEMAFULL;
DEFINE FIELD clip_id ON m3_clips TYPE int;
DEFINE FIELD session_id ON m3_clips TYPE string;  -- Links to existing sessions table
DEFINE FIELD start_time ON m3_clips TYPE datetime;
DEFINE FIELD end_time ON m3_clips TYPE option<datetime>;
DEFINE FIELD node_count ON m3_clips TYPE int DEFAULT 0;
DEFINE FIELD dominant_speaker ON m3_clips TYPE option<string>;
DEFINE FIELD metadata ON m3_clips TYPE object DEFAULT {};

-- Indexes for M3 clips
DEFINE INDEX m3_clips_id_idx ON m3_clips FIELDS clip_id UNIQUE;
DEFINE INDEX m3_clips_session_idx ON m3_clips FIELDS session_id;
DEFINE INDEX m3_clips_time_idx ON m3_clips FIELDS start_time;

-- M3 Nodes: Core memory nodes with content and embeddings
DEFINE TABLE m3_nodes SCHEMAFULL;
DEFINE FIELD node_id ON m3_nodes TYPE int;
DEFINE FIELD node_type ON m3_nodes TYPE string ASSERT $value IN ['voice', 'episodic', 'semantic'];
DEFINE FIELD contents ON m3_nodes TYPE array<string>;  -- Actual content stored in node
DEFINE FIELD embeddings ON m3_nodes TYPE array<array<float>>;  -- Multiple embeddings per node
DEFINE FIELD clip_id ON m3_nodes TYPE int;  -- Links to m3_clips
DEFINE FIELD source_message_id ON m3_nodes TYPE option<record<messages>>;  -- Link to origin message
DEFINE FIELD speaker_id ON m3_nodes TYPE string;  -- Links to users or characters
DEFINE FIELD created_at ON m3_nodes TYPE datetime DEFAULT time::now();
DEFINE FIELD last_accessed ON m3_nodes TYPE datetime DEFAULT time::now();
DEFINE FIELD access_count ON m3_nodes TYPE int DEFAULT 0;
DEFINE FIELD confidence ON m3_nodes TYPE float DEFAULT 0.8;
DEFINE FIELD metadata ON m3_nodes TYPE object DEFAULT {};

-- Indexes for M3 nodes
DEFINE INDEX m3_nodes_id_idx ON m3_nodes FIELDS node_id UNIQUE;
DEFINE INDEX m3_nodes_type_idx ON m3_nodes FIELDS node_type;
DEFINE INDEX m3_nodes_clip_idx ON m3_nodes FIELDS clip_id;
DEFINE INDEX m3_nodes_speaker_idx ON m3_nodes FIELDS speaker_id;
DEFINE INDEX m3_nodes_created_idx ON m3_nodes FIELDS created_at;

-- M3 Edges: Graph relationships between nodes
DEFINE TABLE m3_edges SCHEMAFULL;
DEFINE FIELD source ON m3_edges TYPE record<m3_nodes>;
DEFINE FIELD target ON m3_edges TYPE record<m3_nodes>;
DEFINE FIELD weight ON m3_edges TYPE float DEFAULT 1.0;
DEFINE FIELD edge_type ON m3_edges TYPE string DEFAULT 'similarity';
DEFINE FIELD created_at ON m3_edges TYPE datetime DEFAULT time::now();
DEFINE FIELD last_reinforced ON m3_edges TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata ON m3_edges TYPE object DEFAULT {};

-- Indexes for M3 edges
DEFINE INDEX m3_edges_source_idx ON m3_edges FIELDS source;
DEFINE INDEX m3_edges_target_idx ON m3_edges FIELDS target;
DEFINE INDEX m3_edges_weight_idx ON m3_edges FIELDS weight;
DEFINE INDEX m3_edges_type_idx ON m3_edges FIELDS edge_type;

-- M3 Equivalences: Identity resolution for entities/speakers
DEFINE TABLE m3_equivalences SCHEMAFULL;
DEFINE FIELD canonical_id ON m3_equivalences TYPE string;
DEFINE FIELD node_ids ON m3_equivalences TYPE array<int>;
DEFINE FIELD entity_type ON m3_equivalences TYPE string DEFAULT 'speaker';
DEFINE FIELD confidence ON m3_equivalences TYPE float DEFAULT 0.8;
DEFINE FIELD created_at ON m3_equivalences TYPE datetime DEFAULT time::now();
DEFINE FIELD last_updated ON m3_equivalences TYPE datetime DEFAULT time::now();
DEFINE FIELD metadata ON m3_equivalences TYPE object DEFAULT {};

-- Indexes for M3 equivalences
DEFINE INDEX m3_equiv_canonical_idx ON m3_equivalences FIELDS canonical_id UNIQUE;
DEFINE INDEX m3_equiv_type_idx ON m3_equivalences FIELDS entity_type;

-- ============================================================================
-- BRIDGE RELATIONS (Connect M3 to Existing Schema)
-- ============================================================================

-- Link M3 nodes to their source messages
DEFINE TABLE m3_node_from_message TYPE RELATION IN m3_nodes OUT messages;
DEFINE FIELD extraction_confidence ON m3_node_from_message TYPE float DEFAULT 0.8;
DEFINE FIELD extraction_method ON m3_node_from_message TYPE string DEFAULT 'automatic';
DEFINE FIELD created_at ON m3_node_from_message TYPE datetime DEFAULT time::now();

-- Link M3 clips to sessions (many clips per session possible)
DEFINE TABLE m3_clip_in_session TYPE RELATION IN m3_clips OUT sessions;
DEFINE FIELD created_at ON m3_clip_in_session TYPE datetime DEFAULT time::now();

-- ============================================================================
-- M3 FUNCTIONS
-- ============================================================================

-- Get next available clip ID
DEFINE FUNCTION fn::next_clip_id() {
    LET $max_id = math::max((SELECT VALUE clip_id FROM m3_clips));
    RETURN IF $max_id != NONE THEN $max_id + 1 ELSE 1 END;
};

-- Get next available node ID
DEFINE FUNCTION fn::next_node_id() {
    LET $max_id = math::max((SELECT VALUE node_id FROM m3_nodes));
    RETURN IF $max_id != NONE THEN $max_id + 1 ELSE 1 END;
};

-- Create M3 clip with proper session linking
DEFINE FUNCTION fn::create_m3_clip($session_id: string) {
    LET $clip_id = fn::next_clip_id();
    
    LET $clip = CREATE m3_clips CONTENT {
        clip_id: $clip_id,
        session_id: $session_id,
        start_time: time::now(),
        end_time: NONE,
        node_count: 0,
        dominant_speaker: NONE,
        metadata: {}
    };
    
    -- Link clip to session
    LET $session_record = (SELECT * FROM sessions WHERE session_id = $session_id LIMIT 1)[0];
    IF $session_record {
        LET $clip_record = $clip[0];
        RELATE $clip_record->m3_clip_in_session->$session_record;
    };
    
    RETURN $clip_id;
};

-- Create M3 node with automatic ID assignment
DEFINE FUNCTION fn::create_m3_node($node_type: string, $contents: array<string>, $embeddings: array<array<float>>, $clip_id: int, $speaker_id: string, $source_message_id: option<record<messages>>) {
    LET $node_id = fn::next_node_id();
    
    LET $node = CREATE m3_nodes CONTENT {
        node_id: $node_id,
        node_type: $node_type,
        contents: $contents,
        embeddings: $embeddings,
        clip_id: $clip_id,
        source_message_id: $source_message_id,
        speaker_id: $speaker_id,
        created_at: time::now(),
        last_accessed: time::now(),
        access_count: 0,
        confidence: 0.8,
        metadata: {}
    };
    
    -- Update clip node count
    UPDATE m3_clips SET node_count += 1 WHERE clip_id = $clip_id;
    
    -- Link to source message if provided
    IF $source_message_id != NONE {
        LET $node_record = $node[0];
        RELATE $node_record->m3_node_from_message->$source_message_id;
    };
    
    RETURN $node_id;
};

-- Create edge between M3 nodes
DEFINE FUNCTION fn::create_m3_edge($source_node_id: int, $target_node_id: int, $weight: float, $edge_type: string) {
    LET $source_node = (SELECT * FROM m3_nodes WHERE node_id = $source_node_id LIMIT 1)[0];
    LET $target_node = (SELECT * FROM m3_nodes WHERE node_id = $target_node_id LIMIT 1)[0];
    
    IF $source_node AND $target_node {
        LET $edge = CREATE m3_edges CONTENT {
            source: $source_node.id,
            target: $target_node.id,
            weight: $weight,
            edge_type: $edge_type,
            created_at: time::now(),
            last_reinforced: time::now(),
            metadata: {}
        };
        RETURN true;
    } ELSE {
        RETURN false;
    };
};

-- Vector similarity search across M3 nodes
DEFINE FUNCTION fn::search_similar_m3_nodes($query_embedding: array<float>, $node_type: option<string>, $limit: int, $threshold: float) {
    RETURN SELECT *,
        vector::similarity::cosine($query_embedding, embeddings[0]) AS similarity
    FROM m3_nodes
    WHERE IF $node_type != NONE THEN node_type = $node_type ELSE true END
      AND vector::similarity::cosine($query_embedding, embeddings[0]) >= $threshold
    ORDER BY similarity DESC 
    LIMIT $limit;
};

-- Get all nodes in a temporal clip
DEFINE FUNCTION fn::get_clip_nodes($clip_id: int) {
    RETURN SELECT * FROM m3_nodes WHERE clip_id = $clip_id ORDER BY created_at ASC;
};

-- Close current clip and optionally create new one
DEFINE FUNCTION fn::close_m3_clip($clip_id: int, $create_new: bool, $session_id: option<string>) {
    UPDATE m3_clips SET end_time = time::now() WHERE clip_id = $clip_id;
    
    IF $create_new AND $session_id != NONE {
        RETURN fn::create_m3_clip($session_id);
    } ELSE {
        RETURN $clip_id;
    };
};

-- Resolve or create equivalence relation
DEFINE FUNCTION fn::resolve_m3_equivalence($entity_name: string, $node_ids: array<int>, $entity_type: string) {
    LET $existing = (SELECT * FROM m3_equivalences WHERE canonical_id = $entity_name LIMIT 1)[0];
    
    IF $existing {
        -- Update existing equivalence
        UPDATE m3_equivalences SET 
            node_ids = array::union($existing.node_ids, $node_ids),
            last_updated = time::now()
        WHERE canonical_id = $entity_name;
    } ELSE {
        -- Create new equivalence
        CREATE m3_equivalences CONTENT {
            canonical_id: $entity_name,
            node_ids: $node_ids,
            entity_type: $entity_type,
            confidence: 0.8,
            created_at: time::now(),
            last_updated: time::now(),
            metadata: {}
        };
    };
    
    RETURN $entity_name;
};

-- Edge weight decay function (for memory aging)
DEFINE FUNCTION fn::decay_m3_edges($decay_rate: float) {
    UPDATE m3_edges SET weight = weight * (1 - $decay_rate) 
    WHERE last_reinforced < time::now() - 1d;
    
    -- Remove very weak edges
    DELETE m3_edges WHERE weight < 0.1;
    
    RETURN "Edge decay completed";
};

-- ============================================================================
-- INITIALIZATION DATA
-- ============================================================================

-- Create default system user
INSERT INTO users {
    user_id: "system",
    username: "system", 
    display_name: "System",
    created_at: time::now(),
    metadata: {"type": "system"}
};

-- Create Slowcat assistant character
INSERT INTO characters {
    character_id: "slowcat",
    name: "Slowcat",
    role: "assistant",
    personality: "Helpful voice assistant",
    created_at: time::now(),
    metadata: {"version": "m3"}
};

-- Migration complete marker
CREATE migration_status:m3_production CONTENT {
    version: "1.0.0",
    applied_at: time::now(),
    status: "completed",
    namespace: "slowcat",
    database: "memory_graph", 
    description: "M3 Production Integration - Adds M3 tables while preserving existing sessions/messages"
};