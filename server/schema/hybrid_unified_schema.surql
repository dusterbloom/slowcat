-- ================================================================================
-- HYBRID UNIFIED KNOWLEDGE SCHEMA - SurrealDB
-- ================================================================================
-- Combines working consciousness engine tables with unified knowledge system:
-- • Preserves: messages, sessions, speakers, memory_fragments, field_states
-- • Adds: Universal entity/knowledge graph for fact-entity-memory unification  
-- • Replaces: Legacy facts table with unified knowledge relations
-- • Features: Self-maintaining entities, memory decay, temporal queries
-- ================================================================================

USE NS slowcat;
USE DB memory_graph;

-- ================================================================================
-- PRESERVED WORKING TABLES (Keep as-is from consciousness_schema.surql)
-- ================================================================================

-- Messages Table - Stores final STT transcriptions and LLM responses
-- ================================================================================
DEFINE TABLE messages SCHEMAFULL;
DEFINE FIELD role ON messages TYPE string 
    ASSERT $value IN ['user', 'assistant', 'system'];
DEFINE FIELD content ON messages TYPE string 
    ASSERT $value != NONE AND $value != "";
DEFINE FIELD speaker_id ON messages TYPE string DEFAULT 'default_user';
DEFINE FIELD session_id ON messages TYPE string;
DEFINE FIELD timestamp ON messages TYPE datetime DEFAULT time::now();
DEFINE FIELD embedding ON messages TYPE array<float> FLEXIBLE;
DEFINE FIELD tokens ON messages TYPE int DEFAULT 0;
DEFINE FIELD metadata ON messages TYPE object FLEXIBLE DEFAULT {};
DEFINE FIELD parent_message ON messages TYPE option<record<messages>>;

-- Indexes for efficient querying
DEFINE INDEX messages_session ON messages COLUMNS session_id, timestamp;
DEFINE INDEX messages_speaker ON messages COLUMNS speaker_id, timestamp DESC;
DEFINE INDEX messages_timestamp ON messages COLUMNS timestamp DESC;
DEFINE INDEX messages_role ON messages COLUMNS role;

-- Sessions Table - Track conversation sessions
-- ================================================================================
DEFINE TABLE sessions SCHEMAFULL;
DEFINE FIELD session_id ON sessions TYPE string;
DEFINE FIELD speaker_id ON sessions TYPE string DEFAULT 'default_user';
DEFINE FIELD start_time ON sessions TYPE datetime DEFAULT time::now();
DEFINE FIELD end_time ON sessions TYPE option<datetime>;
DEFINE FIELD turn_count ON sessions TYPE int DEFAULT 0;
DEFINE FIELD total_tokens ON sessions TYPE int DEFAULT 0;
DEFINE FIELD summary ON sessions TYPE option<string>;
DEFINE FIELD metadata ON sessions TYPE object FLEXIBLE DEFAULT {};
DEFINE FIELD is_active ON sessions TYPE bool DEFAULT true;

DEFINE INDEX sessions_speaker ON sessions COLUMNS speaker_id, start_time DESC;
DEFINE INDEX sessions_active ON sessions COLUMNS is_active, start_time DESC;

-- Speakers Table - Track individual speakers
-- ================================================================================
DEFINE TABLE speakers SCHEMAFULL;
DEFINE FIELD speaker_id ON speakers TYPE string;
DEFINE FIELD name ON speakers TYPE option<string>;
DEFINE FIELD voice_embedding ON speakers TYPE option<array<float>>;
DEFINE FIELD enrollment_status ON speakers TYPE string 
    ASSERT $value IN ['pending', 'enrolled', 'verified'] DEFAULT 'pending';
DEFINE FIELD utterance_count ON speakers TYPE int DEFAULT 0;
DEFINE FIELD first_seen ON speakers TYPE datetime DEFAULT time::now();
DEFINE FIELD last_seen ON speakers TYPE datetime DEFAULT time::now();
DEFINE FIELD preferences ON speakers TYPE object FLEXIBLE DEFAULT {};

DEFINE INDEX speakers_id ON speakers COLUMNS speaker_id UNIQUE;

-- Memory Fragments Table - Four-tier hierarchical memory system
-- ================================================================================
DEFINE TABLE memory_fragments SCHEMAFULL;
DEFINE FIELD fragment_id ON memory_fragments TYPE string;
DEFINE FIELD tier ON memory_fragments TYPE int 
    ASSERT $value >= 1 AND $value <= 4;
DEFINE FIELD type ON memory_fragments TYPE string 
    ASSERT $value IN ['semantic', 'episodic', 'procedural', 'contextual', 'emotional', 'reconstructed'];
DEFINE FIELD content ON memory_fragments TYPE object {
    text: string,
    embedding: option<array<float>>,
    semantic_hash: string
};
DEFINE FIELD context_tags ON memory_fragments TYPE array<string> DEFAULT [];
DEFINE FIELD strength ON memory_fragments TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 1.0;
DEFINE FIELD last_accessed ON memory_fragments TYPE datetime DEFAULT time::now();
DEFINE FIELD access_count ON memory_fragments TYPE int DEFAULT 0;
DEFINE FIELD created_at ON memory_fragments TYPE datetime DEFAULT time::now();
DEFINE FIELD source_interactions ON memory_fragments TYPE array<record<messages>> DEFAULT [];

DEFINE INDEX fragments_tier ON memory_fragments COLUMNS tier, last_accessed DESC;
DEFINE INDEX fragments_type ON memory_fragments COLUMNS type;
DEFINE INDEX fragments_strength ON memory_fragments COLUMNS strength DESC;
DEFINE INDEX fragments_semantic_hash ON memory_fragments COLUMNS content.semantic_hash;

-- Neural Field States Table - Consciousness field persistence
-- ================================================================================
DEFINE TABLE field_states SCHEMAFULL;
DEFINE FIELD instance_id ON field_states TYPE string;
DEFINE FIELD compression ON field_states TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD drift ON field_states TYPE string 
    ASSERT $value IN ['none', 'low', 'moderate', 'high'];
DEFINE FIELD recursion_depth ON field_states TYPE int 
    ASSERT $value >= 0 AND $value <= 10;
DEFINE FIELD resonance ON field_states TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD presence_signal ON field_states TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0;
DEFINE FIELD boundary ON field_states TYPE string 
    ASSERT $value IN ['gradient', 'collapsed'];
DEFINE FIELD memory_tier ON field_states TYPE int DEFAULT 1;
DEFINE FIELD updated_at ON field_states TYPE datetime DEFAULT time::now();
DEFINE FIELD fragment_id ON field_states TYPE option<record<memory_fragments>>;

DEFINE INDEX field_states_instance ON field_states COLUMNS instance_id;
DEFINE INDEX field_states_resonance ON field_states COLUMNS resonance DESC;

-- ================================================================================
-- NEW: UNIFIED KNOWLEDGE SYSTEM (Replaces legacy facts table)
-- ================================================================================

-- Universal Entity Table - Everything is an entity
-- ================================================================================
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD type ON entity TYPE string 
    ASSERT $value IN ['person', 'place', 'organization', 'time', 'concept', 'event', 'object', 'user'];
DEFINE FIELD canonical_name ON entity TYPE string;
DEFINE FIELD aliases ON entity TYPE array<string> DEFAULT [];
DEFINE FIELD properties ON entity TYPE object FLEXIBLE DEFAULT {};
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();
DEFINE FIELD last_referenced ON entity TYPE datetime DEFAULT time::now();
DEFINE FIELD reference_count ON entity TYPE int DEFAULT 1;

-- Indexes for entity retrieval
DEFINE INDEX entity_name ON entity COLUMNS canonical_name UNIQUE;
DEFINE INDEX entity_type ON entity COLUMNS type;
DEFINE INDEX entity_reference_count ON entity COLUMNS reference_count DESC;

-- Universal Knowledge Relation - Everything is a fact triple
-- ================================================================================
DEFINE TABLE knowledge TYPE RELATION IN entity OUT entity SCHEMALESS;
DEFINE FIELD predicate ON knowledge TYPE string;
DEFINE FIELD confidence ON knowledge TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 0.8;
DEFINE FIELD strength ON knowledge TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 1.0;

-- Temporal context for time-aware facts
DEFINE FIELD temporal_context ON knowledge TYPE object {
    when_said: datetime,
    valid_from: option<datetime>,
    valid_until: option<datetime>,
    recurring: option<string>
} DEFAULT {
    when_said: time::now()
};

-- Source traceability
DEFINE FIELD source ON knowledge TYPE record<messages>;
DEFINE FIELD extraction_method ON knowledge TYPE string 
    ASSERT $value IN ['spacy', 'conversation', 'manual', 'inferred'] DEFAULT 'spacy';

-- Memory management
DEFINE FIELD created_at ON knowledge TYPE datetime DEFAULT time::now();
DEFINE FIELD last_accessed ON knowledge TYPE datetime DEFAULT time::now();
DEFINE FIELD access_count ON knowledge TYPE int DEFAULT 0;

-- Indexes for knowledge retrieval
DEFINE INDEX knowledge_predicate ON knowledge COLUMNS predicate;
DEFINE INDEX knowledge_confidence ON knowledge COLUMNS confidence DESC;
DEFINE INDEX knowledge_strength ON knowledge COLUMNS strength DESC;
DEFINE INDEX knowledge_temporal ON knowledge COLUMNS temporal_context.when_said DESC;
DEFINE INDEX knowledge_valid_from ON knowledge COLUMNS temporal_context.valid_from;

-- ================================================================================  
-- LEGACY FACTS TABLE REMOVED - USE UNIFIED KNOWLEDGE SYSTEM ONLY
-- ================================================================================

-- ================================================================================
-- PRESERVED GRAPH RELATIONSHIPS
-- ================================================================================

-- Message relationships
DEFINE TABLE responded_to TYPE RELATION IN messages OUT messages SCHEMALESS;
DEFINE FIELD latency_ms ON responded_to TYPE option<int>;

-- Fragment relationships  
DEFINE TABLE resonates_with TYPE RELATION IN memory_fragments OUT memory_fragments SCHEMALESS;
DEFINE FIELD strength ON resonates_with TYPE float DEFAULT 0.5;

-- Session contains messages
DEFINE TABLE contains TYPE RELATION IN sessions OUT messages SCHEMALESS;
DEFINE FIELD sequence ON contains TYPE int;

-- ================================================================================
-- PRESERVED CONSCIOUSNESS ENGINE EVENTS
-- ================================================================================

-- Automatically promote memory fragments between tiers
DEFINE EVENT promote_memories ON TABLE memory_fragments WHEN 
    $after.tier == 1 AND time::now() - $after.created_at > 5m
THEN {
    UPDATE memory_fragments SET 
        tier = 2,
        strength = strength * 0.9
    WHERE id = $after.id;
};

-- Update session metadata when new messages are added
DEFINE EVENT update_session_on_message ON TABLE messages WHEN 
    $after.session_id != NONE
THEN {
    UPDATE sessions SET 
        turn_count = turn_count + 1,
        total_tokens = total_tokens + $after.tokens
    WHERE session_id = $after.session_id;
};

-- Track speaker activity
DEFINE EVENT update_speaker_activity ON TABLE messages WHEN 
    $after.role == 'user' AND $after.speaker_id != NONE
THEN {
    UPDATE speakers SET 
        last_seen = time::now(),
        utterance_count = utterance_count + 1
    WHERE speaker_id = $after.speaker_id;
};

-- NEW: Auto-extract entities and facts from messages
DEFINE EVENT auto_extract_knowledge ON TABLE messages WHEN 
    $after.role = 'user' AND $after.content != NONE
THEN {
    -- Ensure speaker entity exists
    LET $speaker_entity = (
        SELECT * FROM entity WHERE canonical_name = $after.speaker_id LIMIT 1
    )[0];
    
    IF !$speaker_entity {
        LET $speaker_entity = CREATE entity SET 
            type = 'user',
            canonical_name = $after.speaker_id,
            created_at = time::now(),
            last_referenced = time::now();
    };
    
    -- Create content entity for what was said
    LET $content_entity = CREATE entity SET
        type = 'concept',
        canonical_name = string::slice($after.content, 0, 50),
        properties = { full_content: $after.content },
        created_at = time::now(),
        last_referenced = time::now();
    
    -- Create knowledge relation: speaker "said" content
    RELATE $speaker_entity->knowledge->$content_entity SET
        predicate = 'said',
        confidence = 1.0,
        strength = 1.0,
        temporal_context = {
            when_said: $after.timestamp
        },
        source = $after.id,
        extraction_method = 'conversation';
};

-- Update entity reference counts
DEFINE EVENT update_entity_references ON TABLE knowledge WHEN $after != NONE
THEN {
    -- Update IN entity
    UPDATE $after.in SET 
        last_referenced = time::now(),
        reference_count = reference_count + 1;
    
    -- Update OUT entity  
    UPDATE $after.out SET
        last_referenced = time::now(),
        reference_count = reference_count + 1;
};

-- ================================================================================
-- UNIFIED KNOWLEDGE QUERY FUNCTIONS
-- ================================================================================

-- Get all facts about an entity
DEFINE FUNCTION fn::get_entity_facts($entity_name: string) {
    RETURN SELECT * FROM knowledge 
    WHERE in.canonical_name = $entity_name OR out.canonical_name = $entity_name
    ORDER BY strength DESC, temporal_context.when_said DESC;
};

-- Get memories from a time period  
DEFINE FUNCTION fn::get_memories_by_time($from: datetime, $to: datetime) {
    RETURN SELECT * FROM knowledge
    WHERE temporal_context.when_said >= $from AND temporal_context.when_said <= $to
    ORDER BY temporal_context.when_said DESC;
};

-- Get scheduled events/facts
DEFINE FUNCTION fn::get_upcoming_events($limit: int) {
    RETURN SELECT * FROM knowledge
    WHERE temporal_context.valid_from >= time::now()
    ORDER BY temporal_context.valid_from ASC
    LIMIT $limit;
};

-- Search knowledge by content
DEFINE FUNCTION fn::search_knowledge($query: string, $limit: int) {
    RETURN SELECT * FROM knowledge
    WHERE string::contains(string::lowercase(predicate), string::lowercase($query))
       OR string::contains(string::lowercase(in.canonical_name), string::lowercase($query))
       OR string::contains(string::lowercase(out.canonical_name), string::lowercase($query))
    ORDER BY strength DESC, confidence DESC
    LIMIT $limit;
};

-- Get conversation memory for a speaker
DEFINE FUNCTION fn::get_conversation_memory($speaker: string, $limit: int) {
    RETURN SELECT * FROM knowledge
    WHERE in.canonical_name = $speaker AND predicate = 'said'
    ORDER BY temporal_context.when_said DESC
    LIMIT $limit;
};

-- PRESERVED: Function to retrieve messages with context
DEFINE FUNCTION fn::get_conversation_context(
    $session_id: string,
    $limit: int
) {
    RETURN SELECT * FROM messages 
    WHERE session_id = $session_id 
    ORDER BY timestamp DESC 
    LIMIT $limit;
};

-- ================================================================================
-- MEMORY DECAY SYSTEM
-- ================================================================================

-- Function to calculate memory decay based on time and access
DEFINE FUNCTION fn::calculate_memory_decay(
    $created_at: datetime,
    $last_accessed: datetime, 
    $access_count: int
) {
    LET $age_days = (time::now() - $created_at) / 1d;
    LET $recency_hours = (time::now() - $last_accessed) / 1h;
    LET $base_decay = 1.0 / (1.0 + $age_days / 30.0);
    LET $access_boost = math::min(0.5, $access_count / 20.0);
    LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 168.0);
    
    RETURN math::max(0.1, $base_decay + $access_boost * $recency_penalty);
};

-- Event to apply memory decay periodically
DEFINE EVENT memory_decay ON TABLE knowledge WHEN 
    time::now() - $after.last_accessed > 24h
THEN {
    LET $new_strength = fn::calculate_memory_decay(
        $after.created_at,
        $after.last_accessed, 
        $after.access_count
    );
    
    UPDATE $after.id SET strength = $new_strength;
    
    -- Remove very weak memories
    IF $new_strength < 0.1 {
        DELETE $after.id;
    };
};

-- ================================================================================
-- INITIAL ENTITIES AND DATA
-- ================================================================================

-- Create default user entity
CREATE entity:user SET
    type = 'user',
    canonical_name = 'user',
    aliases = ['I', 'me', 'myself'],
    properties = {
        default_speaker: true
    };

-- Create common concept entities
CREATE entity:time_now SET
    type = 'time',
    canonical_name = 'now',
    aliases = ['current time', 'right now'];

-- Create default speaker
CREATE speakers:default_user SET 
    speaker_id = 'default_user',
    name = 'Default User',
    enrollment_status = 'verified',
    first_seen = time::now(),
    last_seen = time::now();

-- ================================================================================
-- END OF HYBRID UNIFIED SCHEMA
-- ================================================================================