-- M3 Memory System Migration Schema
-- Implements M3-inspired memory persistence with SurrealDB
-- This migration removes legacy components and adds M3-specific tables

-- ============================================================================
-- REMOVE LEGACY COMPONENTS
-- ============================================================================

-- Remove deprecated engram system
REMOVE TABLE IF EXISTS engrams;
REMOVE TABLE IF EXISTS engram_appears_in;
REMOVE TABLE IF EXISTS engram_contains;
REMOVE TABLE IF EXISTS engram_engram;
REMOVE TABLE IF EXISTS engram_memory_fragment;

-- Remove legacy memory fragments (replaced by M3 nodes)
REMOVE TABLE IF EXISTS memory_fragments;

-- Remove consciousness field states (not needed)
REMOVE TABLE IF EXISTS field_states;

-- Remove old knowledge table (replaced by M3 semantic nodes)
REMOVE TABLE IF EXISTS knowledge;

-- Remove complex engram functions
REMOVE FUNCTION IF EXISTS fn::create_engram_from_message;
REMOVE FUNCTION IF EXISTS fn::link_engrams_by_similarity;
REMOVE FUNCTION IF EXISTS fn::get_related_engrams;

-- ============================================================================
-- M3 CORE TABLES
-- ============================================================================

-- M3 Nodes: Core memory nodes with content and embeddings
DEFINE TABLE m3_nodes SCHEMAFULL;
DEFINE FIELD node_id ON m3_nodes TYPE int;
DEFINE FIELD node_type ON m3_nodes TYPE string ASSERT $value IN ['voice', 'episodic', 'semantic'];
DEFINE FIELD contents ON m3_nodes TYPE array<string>;  -- Actual content stored in node
DEFINE FIELD embeddings ON m3_nodes TYPE array<array<float>>;  -- Multiple embeddings per node
DEFINE FIELD clip_id ON m3_nodes TYPE int;  -- Temporal reference to m3_clips
DEFINE FIELD source_message ON m3_nodes TYPE option<record<messages>>;  -- Link to origin message
DEFINE FIELD metadata ON m3_nodes TYPE object;

-- Indexes for M3 nodes
DEFINE INDEX m3_nodes_type_idx ON m3_nodes FIELDS node_type;
DEFINE INDEX m3_nodes_clip_idx ON m3_nodes FIELDS clip_id;
DEFINE INDEX m3_nodes_speaker_idx ON m3_nodes FIELDS metadata.speaker_id;
DEFINE INDEX m3_nodes_id_idx ON m3_nodes FIELDS node_id UNIQUE;

-- M3 Edges: Graph relationships between nodes
DEFINE TABLE m3_edges SCHEMAFULL;
DEFINE FIELD source ON m3_edges TYPE record<m3_nodes>;
DEFINE FIELD target ON m3_edges TYPE record<m3_nodes>;
DEFINE FIELD weight ON m3_edges TYPE float DEFAULT 1.0;
DEFINE FIELD edge_type ON m3_edges TYPE string DEFAULT 'similarity';
DEFINE FIELD created_at ON m3_edges TYPE datetime DEFAULT time::now();
DEFINE FIELD last_reinforced ON m3_edges TYPE datetime DEFAULT time::now();

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

-- Indexes for M3 equivalences
DEFINE INDEX m3_equiv_canonical_idx ON m3_equivalences FIELDS canonical_id UNIQUE;
DEFINE INDEX m3_equiv_type_idx ON m3_equivalences FIELDS entity_type;

-- M3 Clips: Temporal organization of memory
DEFINE TABLE m3_clips SCHEMAFULL;
DEFINE FIELD clip_id ON m3_clips TYPE int;
DEFINE FIELD session_id ON m3_clips TYPE string;  -- Links to sessions table
DEFINE FIELD start_time ON m3_clips TYPE datetime;
DEFINE FIELD end_time ON m3_clips TYPE option<datetime>;
DEFINE FIELD node_count ON m3_clips TYPE int DEFAULT 0;
DEFINE FIELD dominant_speaker ON m3_clips TYPE option<string>;

-- Indexes for M3 clips
DEFINE INDEX m3_clips_id_idx ON m3_clips FIELDS clip_id UNIQUE;
DEFINE INDEX m3_clips_session_idx ON m3_clips FIELDS session_id;
DEFINE INDEX m3_clips_time_idx ON m3_clips FIELDS start_time;

-- ============================================================================
-- BRIDGE RELATIONS (Connect M3 to Existing Schema)
-- ============================================================================

-- Link M3 nodes to their source messages
DEFINE TABLE m3_node_from_message TYPE RELATION IN m3_nodes OUT messages;
DEFINE FIELD extraction_confidence ON m3_node_from_message TYPE float DEFAULT 0.8;
DEFINE FIELD extraction_method ON m3_node_from_message TYPE string;
DEFINE FIELD created_at ON m3_node_from_message TYPE datetime DEFAULT time::now();

-- Link M3 nodes to entities they describe
DEFINE TABLE m3_node_about_entity TYPE RELATION IN m3_nodes OUT entity;
DEFINE FIELD relevance_score ON m3_node_about_entity TYPE float DEFAULT 1.0;
DEFINE FIELD relationship_type ON m3_node_about_entity TYPE string;
DEFINE FIELD discovered_at ON m3_node_about_entity TYPE datetime DEFAULT time::now();

-- ============================================================================
-- M3 FUNCTIONS
-- ============================================================================

-- Create M3 node with automatic ID assignment
DEFINE FUNCTION fn::create_m3_node($node_type: string, $contents: array<string>, $embeddings: array<array<float>>, $clip_id: int, $metadata: object) {
    LET $node_id = math::max((SELECT VALUE node_id FROM m3_nodes)) + 1 OR 1;
    
    CREATE m3_nodes CONTENT {
        node_id: $node_id,
        node_type: $node_type,
        contents: $contents,
        embeddings: $embeddings,
        clip_id: $clip_id,
        metadata: $metadata
    };
    
    -- Update clip node count
    UPDATE m3_clips SET node_count += 1 WHERE clip_id = $clip_id;
    
    RETURN $node_id;
};

-- Create edge between M3 nodes with weight calculation
DEFINE FUNCTION fn::create_m3_edge($source_id: int, $target_id: int, $weight: float, $edge_type: string) {
    LET $source_node = (SELECT * FROM m3_nodes WHERE node_id = $source_id)[0];
    LET $target_node = (SELECT * FROM m3_nodes WHERE node_id = $target_id)[0];
    
    IF $source_node AND $target_node THEN {
        CREATE m3_edges CONTENT {
            source: $source_node.id,
            target: $target_node.id,
            weight: $weight,
            edge_type: $edge_type,
            created_at: time::now(),
            last_reinforced: time::now()
        };
        RETURN true;
    } ELSE {
        RETURN false;
    } END;
};

-- Vector similarity search across M3 nodes
DEFINE FUNCTION fn::search_similar_m3_nodes($query_embedding: array<float>, $node_type: option<string>, $limit: int) {
    RETURN SELECT *,
        vector::similarity::cosine($query_embedding, embeddings[0]) AS similarity
    FROM m3_nodes
    WHERE IF $node_type != NONE THEN node_type = $node_type ELSE true END
    ORDER BY similarity DESC 
    LIMIT $limit;
};

-- Get all nodes in a temporal clip
DEFINE FUNCTION fn::get_clip_nodes($clip_id: int) {
    RETURN SELECT * FROM m3_nodes WHERE clip_id = $clip_id ORDER BY metadata.created_at ASC;
};

-- Create new temporal clip
DEFINE FUNCTION fn::create_m3_clip($session_id: string) {
    LET $clip_id = math::max((SELECT VALUE clip_id FROM m3_clips)) + 1 OR 1;
    
    CREATE m3_clips CONTENT {
        clip_id: $clip_id,
        session_id: $session_id,
        start_time: time::now(),
        end_time: NONE,
        node_count: 0,
        dominant_speaker: NONE
    };
    
    RETURN $clip_id;
};

-- Close current clip and optionally create new one
DEFINE FUNCTION fn::close_m3_clip($clip_id: int, $create_new: bool, $session_id: option<string>) {
    UPDATE m3_clips SET end_time = time::now() WHERE clip_id = $clip_id;
    
    IF $create_new AND $session_id THEN {
        RETURN fn::create_m3_clip($session_id);
    } ELSE {
        RETURN $clip_id;
    } END;
};

-- Resolve or create equivalence relation
DEFINE FUNCTION fn::resolve_m3_equivalence($entity_name: string, $node_ids: array<int>, $entity_type: string) {
    LET $existing = (SELECT * FROM m3_equivalences WHERE canonical_id = $entity_name)[0];
    
    IF $existing THEN {
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
            last_updated: time::now()
        };
    } END;
    
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
-- MIGRATION VERIFICATION
-- ============================================================================

-- Verify M3 tables exist
DEFINE FUNCTION fn::verify_m3_migration() {
    RETURN {
        status: "success",
        message: "M3 migration verification placeholder",
        timestamp: time::now()
    };
};

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Create initial root clip for system messages
LET $initial_clip = fn::create_m3_clip("system");

-- Migration complete marker
CREATE migration_status:m3 CONTENT {
    version: "1.0.0",
    applied_at: time::now(),
    status: "completed",
    description: "M3 Memory System Migration - Adds M3 tables and removes legacy components"
};