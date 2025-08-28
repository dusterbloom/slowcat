-- ================================================================================
-- SLOWCAT CONSCIOUSNESS ENGINE - SURREALDB SCHEMA
-- ================================================================================
-- Multi-model database schema supporting:
-- 1. Conversation messages (user/assistant)
-- 2. Four-tier memory hierarchy
-- 3. Neural field states persistence
-- 4. Graph relationships and semantic search
-- 5. Session and speaker management
-- ================================================================================

-- Namespace and Database Setup
-- ================================================================================
USE NS slowcat;
USE DB consciousness;

-- ================================================================================
-- CORE TABLES
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
DEFINE FIELD metadata ON messages TYPE object FLEXIBLE;
DEFINE FIELD parent_message ON messages TYPE option<record<messages>>;

-- Indexes for efficient querying
DEFINE INDEX messages_session ON messages COLUMNS session_id, timestamp;
DEFINE INDEX messages_speaker ON messages COLUMNS speaker_id, timestamp DESC;
DEFINE INDEX messages_timestamp ON messages COLUMNS timestamp DESC;
DEFINE INDEX messages_role ON messages COLUMNS role;

-- Memory Fragments Table - Four-tier hierarchical memory system
-- ================================================================================
DEFINE TABLE memory_fragments SCHEMAFULL;
DEFINE FIELD fragment_id ON memory_fragments TYPE string;
DEFINE FIELD tier ON memory_fragments TYPE int 
    ASSERT $value >= 1 AND $value <= 4;
-- Tier 1: Working Memory (0-5 min)
-- Tier 2: Short-term Memory (5min-2hr)  
-- Tier 3: Long-term Memory (2hr+)
-- Tier 4: Episodic Memory (permanent)
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

-- Facts Graph Table - Structured knowledge storage
-- ================================================================================
DEFINE TABLE facts SCHEMAFULL;
DEFINE FIELD subject ON facts TYPE string;
DEFINE FIELD predicate ON facts TYPE string;
DEFINE FIELD object ON facts TYPE string;
DEFINE FIELD confidence ON facts TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 0.8;
DEFINE FIELD source ON facts TYPE record<messages>;
DEFINE FIELD extracted_at ON facts TYPE datetime DEFAULT time::now();
DEFINE FIELD last_verified ON facts TYPE datetime DEFAULT time::now();
DEFINE FIELD decay_rate ON facts TYPE float DEFAULT 0.1;
DEFINE FIELD importance ON facts TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 0.5;

DEFINE INDEX facts_subject ON facts COLUMNS subject;
DEFINE INDEX facts_predicate ON facts COLUMNS predicate;
DEFINE INDEX facts_confidence ON facts COLUMNS confidence DESC;

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

-- ================================================================================
-- GRAPH EDGES (RELATIONSHIPS)
-- ================================================================================

-- Message relationships
DEFINE TABLE responded_to SCHEMALESS;
DEFINE FIELD in ON responded_to TYPE record<messages>;
DEFINE FIELD out ON responded_to TYPE record<messages>;
DEFINE FIELD latency_ms ON responded_to TYPE option<int>;

-- Fragment relationships  
DEFINE TABLE resonates_with SCHEMALESS;
DEFINE FIELD in ON resonates_with TYPE record<memory_fragments>;
DEFINE FIELD out ON resonates_with TYPE record<memory_fragments>;
DEFINE FIELD strength ON resonates_with TYPE float DEFAULT 0.5;

-- Fact relationships
DEFINE TABLE related_to SCHEMALESS;
DEFINE FIELD in ON related_to TYPE record<facts>;
DEFINE FIELD out ON related_to TYPE record<facts>;
DEFINE FIELD relation_type ON related_to TYPE string;

-- Session-Message relationships
DEFINE TABLE contains_message SCHEMALESS;
DEFINE FIELD in ON contains_message TYPE record<sessions>;
DEFINE FIELD out ON contains_message TYPE record<messages>;
DEFINE FIELD sequence ON contains_message TYPE int;

-- Speaker-Session relationships
DEFINE TABLE participated_in SCHEMALESS;
DEFINE FIELD in ON participated_in TYPE record<speakers>;
DEFINE FIELD out ON participated_in TYPE record<sessions>;
DEFINE FIELD role ON participated_in TYPE string DEFAULT 'user';

-- ================================================================================
-- EVENTS AND TRIGGERS
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

-- ================================================================================
-- FUNCTIONS
-- ================================================================================

-- Function to retrieve messages with context
DEFINE FUNCTION fn::get_conversation_context(
    $session_id: string,
    $limit: int
) {
    RETURN SELECT * FROM messages 
    WHERE session_id = $session_id 
    ORDER BY timestamp DESC 
    LIMIT $limit;
};

-- Function to calculate memory decay
DEFINE FUNCTION fn::calculate_decay(
    $created_at: datetime,
    $last_accessed: datetime,
    $access_count: int
) {
    LET $age_hours = (time::now() - $created_at) / 1h;
    LET $recency_hours = (time::now() - $last_accessed) / 1h;
    LET $base_decay = 1.0 / (1.0 + $age_hours / 24.0);
    LET $access_boost = math::min(1.0, $access_count / 10.0);
    LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 12.0);
    
    RETURN $base_decay * (0.5 + 0.3 * $access_boost + 0.2 * $recency_penalty);
};

-- Function to find similar messages using embeddings
DEFINE FUNCTION fn::find_similar_messages(
    $embedding: array<float>,
    $limit: int,
    $threshold: float
) {
    -- This would use vector similarity in production
    -- Placeholder for now - would integrate with vector index
    RETURN SELECT * FROM messages 
    WHERE embedding != NONE
    LIMIT $limit;
};

-- ================================================================================
-- PERMISSIONS (Adjust based on deployment needs)
-- ================================================================================

-- Create access for the application
DEFINE ACCESS application ON DATABASE TYPE RECORD
    SIGNUP ( CREATE user SET email = $email, pass = crypto::argon2::generate($pass) )
    SIGNIN ( SELECT * FROM user WHERE email = $email AND crypto::argon2::compare(pass, $pass) )
    DURATION FOR TOKEN 30d, FOR SESSION 12h;

-- Define scopes for different access levels
DEFINE SCOPE readwrite FOR application
    WHERE role = 'admin' OR role = 'user';

DEFINE SCOPE readonly FOR application  
    WHERE role = 'viewer';

-- ================================================================================
-- INITIAL DATA AND SYSTEM RECORDS
-- ================================================================================

-- Create default speaker
CREATE speakers:default_user SET 
    speaker_id = 'default_user',
    name = 'Default User',
    enrollment_status = 'verified',
    first_seen = time::now(),
    last_seen = time::now();

-- Create system session for initialization
CREATE sessions:system SET
    session_id = 'system',
    speaker_id = 'system',
    start_time = time::now(),
    is_active = false;

-- ================================================================================
-- INDEXES FOR VECTOR SEARCH (When supported)
-- ================================================================================
-- Note: SurrealDB vector search is evolving. These may need adjustment
-- based on the version you're using.

-- Vector index for message embeddings (placeholder syntax)
-- DEFINE INDEX messages_vector ON messages FIELDS embedding VECTOR DIMENSION 768 DISTANCE COSINE;

-- Vector index for memory fragment embeddings  
-- DEFINE INDEX fragments_vector ON memory_fragments FIELDS content.embedding VECTOR DIMENSION 768 DISTANCE COSINE;

-- ================================================================================
-- END OF SCHEMA
-- ================================================================================ 