-- ================================================================================
-- UNIFIED KNOWLEDGE SCHEMA - SurrealDB
-- ================================================================================
-- Universal fact-entity-memory structure that handles:
-- • Conversation memory (what was said)
-- • Extracted facts (who, what, where) 
-- • Temporal references (when, scheduling)
-- • Entity relationships (connections between people, places, concepts)
-- • Memory strength and decay over time
-- ================================================================================

USE NS slowcat;
USE DB memory_graph;

-- ================================================================================
-- CORE TABLES
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
DEFINE TABLE knowledge TYPE RELATION IN entity OUT entity;
DEFINE FIELD predicate ON knowledge TYPE string;  -- "works_at", "said", "scheduled_for", "is_a", "owns"
DEFINE FIELD confidence ON knowledge TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 0.8;
DEFINE FIELD strength ON knowledge TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 1.0;  -- Memory strength (decays over time)

-- Temporal context for time-aware facts
DEFINE FIELD temporal_context ON knowledge TYPE object {
    when_said: datetime,           -- When this fact was mentioned
    valid_from: option<datetime>,  -- When this fact becomes valid
    valid_until: option<datetime>, -- When this fact expires
    recurring: option<string>      -- "daily", "weekly", "monthly" for recurring events
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

-- Messages and Sessions (keep existing working tables)
-- ================================================================================
DEFINE TABLE messages SCHEMAFULL;
DEFINE FIELD role ON messages TYPE string 
    ASSERT $value IN ['user', 'assistant', 'system'];
DEFINE FIELD content ON messages TYPE string 
    ASSERT $value != NONE AND $value != "";
DEFINE FIELD speaker_id ON messages TYPE string DEFAULT 'default_user';
DEFINE FIELD session_id ON messages TYPE string;
DEFINE FIELD timestamp ON messages TYPE datetime DEFAULT time::now();
DEFINE FIELD tokens ON messages TYPE int DEFAULT 0;
DEFINE FIELD metadata ON messages TYPE object FLEXIBLE DEFAULT {};

DEFINE INDEX messages_session ON messages COLUMNS session_id, timestamp;
DEFINE INDEX messages_speaker ON messages COLUMNS speaker_id, timestamp DESC;

DEFINE TABLE sessions SCHEMAFULL;
DEFINE FIELD session_id ON sessions TYPE string;
DEFINE FIELD speaker_id ON sessions TYPE string DEFAULT 'default_user';
DEFINE FIELD start_time ON sessions TYPE datetime DEFAULT time::now();
DEFINE FIELD turn_count ON sessions TYPE int DEFAULT 0;
DEFINE FIELD total_tokens ON sessions TYPE int DEFAULT 0;
DEFINE FIELD is_active ON sessions TYPE bool DEFAULT true;
DEFINE FIELD metadata ON sessions TYPE object FLEXIBLE DEFAULT {};

DEFINE INDEX sessions_speaker ON sessions COLUMNS speaker_id, start_time DESC;

-- ================================================================================
-- ESSENTIAL RELATIONS
-- ================================================================================

-- Session contains messages
DEFINE TABLE contains TYPE RELATION IN sessions OUT messages;
DEFINE FIELD sequence ON contains TYPE int;  -- Message order in session

-- Message responses (conversation threading)
DEFINE TABLE responds_to TYPE RELATION IN messages OUT messages;
DEFINE FIELD latency_ms ON responds_to TYPE option<int>;

-- ================================================================================
-- SELF-MAINTAINING EVENTS
-- ================================================================================

-- Auto-link messages to sessions
DEFINE EVENT auto_link_message ON TABLE messages WHEN $after.session_id != NONE
THEN {
    -- Create session->message relation
    LET $session = (SELECT * FROM sessions WHERE session_id = $after.session_id LIMIT 1)[0];
    IF $session {
        RELATE $session->contains->$after.id SET sequence = $session.turn_count;
    };
};

-- Auto-extract entities and facts from messages
DEFINE EVENT auto_extract_knowledge ON TABLE messages WHEN 
    $after.role = 'user' AND $after.content != NONE
THEN {
    -- This would trigger fact extraction
    -- Implementation: Call external fact extraction service or function
    -- For now, create a basic "said" relationship
    
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
-- KNOWLEDGE QUERY FUNCTIONS
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
    LET $base_decay = 1.0 / (1.0 + $age_days / 30.0);  -- 30-day half-life
    LET $access_boost = math::min(0.5, $access_count / 20.0);  -- Access boost
    LET $recency_penalty = 1.0 / (1.0 + $recency_hours / 168.0);  -- Weekly recency
    
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
-- INITIAL ENTITIES
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

-- ================================================================================
-- EXAMPLE QUERIES FOR TESTING
-- ================================================================================

-- Examples of how to query this system:

-- 1. What do we know about the user?
-- SELECT * FROM fn::get_entity_facts('user');

-- 2. What was said in the last hour?
-- SELECT * FROM fn::get_memories_by_time(time::now() - 1h, time::now());

-- 3. What's scheduled for tomorrow?
-- SELECT * FROM knowledge WHERE temporal_context.valid_from >= date::tomorrow() 
--   AND temporal_context.valid_from < date::tomorrow() + 1d;

-- 4. Search for mentions of "coffee"
-- SELECT * FROM fn::search_knowledge('coffee', 10);

-- 5. Get recent conversation memory
-- SELECT * FROM fn::get_conversation_memory('user', 20);

-- ================================================================================
-- END OF SCHEMA
-- ================================================================================