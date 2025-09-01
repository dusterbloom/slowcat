-- ------------------------------
-- OPTION
-- ------------------------------

OPTION IMPORT;

-- ------------------------------
-- FUNCTIONS
-- ------------------------------

DEFINE FUNCTION fn::calculate_memory_decay($created_at: datetime, $last_accessed: datetime, $access_count: int) {
LET $age_seconds = time::now() - $created_at;
LET $hours_old = $age_seconds / 3600;
LET $simple_decay = 1f - ($hours_old * 0.01f);
LET $access_boost = $access_count * 0.05f;
LET $capped_boost = IF $access_boost > 0.3f THEN 0.3f ELSE $access_boost END;
LET $final_strength = $simple_decay + $capped_boost;
RETURN IF $final_strength < 0.1f THEN 0.1f ELSE (IF $final_strength > 1f THEN 1f ELSE $final_strength END) END;
} PERMISSIONS FULL;
DEFINE FUNCTION fn::cleanup_fragments($limit: int) {
LET $fragments = (SELECT id FROM knowledge WHERE fn::calculate_memory_decay(created_at, last_accessed, access_count) < 0.1f LIMIT $limit);
FOR $fragment IN $fragments { DELETE $fragment.id; };
RETURN count($fragments);
} PERMISSIONS FULL;
DEFINE FUNCTION fn::decay_background_memories($batch_size: int) {
LET $batch = (SELECT * FROM knowledge WHERE last_accessed < (time::now() - 1h) ORDER BY last_accessed
 LIMIT $batch_size);
FOR $fact IN $batch {
LET $new_strength = fn::calculate_memory_decay($fact.created_at, $fact.last_accessed, $fact.access_count);
UPDATE $fact.id SET strength = $new_strength;
};
RETURN count($batch);
} PERMISSIONS FULL;
DEFINE FUNCTION fn::detect_engrams($session_id: string, $min_confidence: float, $min_cluster_size: int) {
LET $session_knowledge = (SELECT *, in.canonical_name AS subject, out.canonical_name AS object FROM knowledge WHERE session_id = $session_id AND confidence >= $min_confidence);
IF count($session_knowledge) >= $min_cluster_size {
LET $subjects = array::group((SELECT VALUE subject FROM $session_knowledge WHERE subject != NONE));
LET $objects = array::group((SELECT VALUE object FROM $session_knowledge WHERE object != NONE));
LET $predicates = array::group((SELECT VALUE predicate FROM $session_knowledge WHERE predicate != NONE));
LET $all_symbols = array::union($subjects, array::union($objects, $predicates));
LET $dominant_symbols = array::slice($all_symbols, 0, 6);
IF count($dominant_symbols) >= 3 {
LET $knowledge_refs = array::group((SELECT VALUE id FROM $session_knowledge));
LET $avg_confidence = math::mean((SELECT VALUE confidence FROM $session_knowledge));
CREATE engrams SET dominant_symbols = $dominant_symbols, narrative_summary = 'Attractor state: ' + string::join($dominant_symbols, ', '), knowledge_ids = $knowledge_refs, session_ids = [$session_id], coherence_score = $avg_confidence, activation_count = 1, created_at = time::now(), last_activated = time::now();
RETURN { coherence_score: $avg_confidence, dominant_symbols: $dominant_symbols, engram_created: true, knowledge_count: count($knowledge_refs) };
} ELSE { RETURN { engram_created: false, reason: 'insufficient_symbol_coherence' }; };
} ELSE { RETURN { engram_created: false, reason: 'insufficient_knowledge_cluster_size' }; };
} PERMISSIONS FULL;
DEFINE FUNCTION fn::detect_memory_patterns($query: string, $similarity_threshold: float) {
LET $query_fragments = (SELECT * FROM knowledge WHERE (predicate != NONE AND string::contains(string::lowercase(predicate), string::lowercase($query))) OR (in.canonical_name != NONE AND string::contains(string::lowercase(in.canonical_name), string::lowercase($query))) OR (out.canonical_name != NONE AND string::contains(string::lowercase(out.canonical_name), string::lowercase($query))) AND fn::calculate_memory_decay(created_at, last_accessed, access_count) >= 0.1f AND fn::calculate_memory_decay(created_at, last_accessed, access_count) <= 0.5f);
RETURN $query_fragments;
} PERMISSIONS FULL;
DEFINE FUNCTION fn::get_conversation_context($session_id: string, $limit: int) { RETURN (SELECT * FROM messages WHERE session_id = $session_id ORDER BY timestamp DESC
 LIMIT $limit); } PERMISSIONS FULL;
DEFINE FUNCTION fn::get_conversation_memory($speaker: string, $limit: int) { RETURN (SELECT * FROM knowledge WHERE in.canonical_name = $speaker AND predicate = 'said' ORDER BY temporal_context.when_said DESC
 LIMIT $limit); } PERMISSIONS FULL;
DEFINE FUNCTION fn::get_entity_facts($entity_name: string) { RETURN (SELECT * FROM knowledge WHERE in.canonical_name = $entity_name OR out.canonical_name = $entity_name ORDER BY strength DESC, temporal_context.when_said DESC
); } PERMISSIONS FULL;
DEFINE FUNCTION fn::get_memories_by_time($from: datetime, $to: datetime) { RETURN (SELECT * FROM knowledge WHERE temporal_context.when_said >= $from AND temporal_context.when_said <= $to ORDER BY temporal_context.when_said DESC
); } PERMISSIONS FULL;
DEFINE FUNCTION fn::reconstruct_fragments($entity_name: string, $boost_factor: float) {
LET $fragments = (SELECT * FROM knowledge WHERE (in.canonical_name = $entity_name OR out.canonical_name = $entity_name) AND fn::calculate_memory_decay(created_at, last_accessed, access_count) < 0.5f AND fn::calculate_memory_decay(created_at, last_accessed, access_count) > 0.1f);
IF count($fragments) > 1 {
FOR $fragment IN $fragments {
LET $current_strength = fn::calculate_memory_decay($fragment.created_at, $fragment.last_accessed, $fragment.access_count);
LET $boosted_strength = IF $current_strength + $boost_factor < 1f THEN $current_strength + $boost_factor ELSE 1f END;
UPDATE $fragment.id SET strength = $boosted_strength, last_accessed = time::now(), access_count = access_count + 1, reconstruction_boost = (reconstruction_boost OR 0) + $boost_factor;
};
RETURN count($fragments);
} ELSE { RETURN 0; };
} PERMISSIONS FULL;
DEFINE FUNCTION fn::search_knowledge($query: string, $limit: int) { RETURN (SELECT * FROM knowledge WHERE (predicate != NONE AND string::contains(string::lowercase(predicate), string::lowercase($query))) OR (in.canonical_name != NONE AND string::contains(string::lowercase(in.canonical_name), string::lowercase($query))) OR (out.canonical_name != NONE AND string::contains(string::lowercase(out.canonical_name), string::lowercase($query))) ORDER BY strength DESC, confidence DESC
 LIMIT $limit); } PERMISSIONS FULL;
DEFINE FUNCTION fn::search_knowledge_advanced($query: string, $query_embedding: array<float>, $limit: int) {
LET $vector_results = (SELECT *, vector::similarity::cosine(embedding, $query_embedding) AS vector_score, 0.3f AS text_score FROM knowledge WHERE embedding != NONE AND vector::similarity::cosine(embedding, $query_embedding) > 0.1f ORDER BY vector_score DESC
 LIMIT $limit);
LET $text_results = (SELECT *, 0f AS vector_score, 1f AS text_score FROM knowledge WHERE (predicate != NONE AND string::contains(string::lowercase(predicate), string::lowercase($query))) OR (in.canonical_name != NONE AND string::contains(string::lowercase(in.canonical_name), string::lowercase($query))) OR (out.canonical_name != NONE AND string::contains(string::lowercase(out.canonical_name), string::lowercase($query))) ORDER BY strength DESC, confidence DESC
 LIMIT $limit);
LET $combined = array::union($vector_results, $text_results);
RETURN (SELECT *, (vector_score * 0.7f + text_score * 0.3f) AS combined_score FROM $combined ORDER BY combined_score DESC
 LIMIT $limit);
} PERMISSIONS FULL;

-- ------------------------------
-- TABLE: engrams
-- ------------------------------

DEFINE TABLE engrams TYPE NORMAL SCHEMAFULL PERMISSIONS NONE;

DEFINE FIELD activation_count ON engrams TYPE int DEFAULT 0 PERMISSIONS FULL;
DEFINE FIELD coherence_score ON engrams TYPE float DEFAULT 0.5f PERMISSIONS FULL;
DEFINE FIELD created_at ON engrams TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD dominant_symbols ON engrams TYPE array<string> PERMISSIONS FULL;
DEFINE FIELD dominant_symbols[*] ON engrams TYPE string PERMISSIONS FULL;
DEFINE FIELD knowledge_ids ON engrams TYPE array<record<knowledge>> PERMISSIONS FULL;
DEFINE FIELD knowledge_ids[*] ON engrams TYPE record<knowledge> PERMISSIONS FULL;
DEFINE FIELD last_activated ON engrams TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD narrative_summary ON engrams TYPE string PERMISSIONS FULL;
DEFINE FIELD session_ids ON engrams TYPE array<string> PERMISSIONS FULL;
DEFINE FIELD session_ids[*] ON engrams TYPE string PERMISSIONS FULL;



-- ------------------------------
-- TABLE: entity
-- ------------------------------

DEFINE TABLE entity TYPE ANY SCHEMALESS PERMISSIONS NONE;

DEFINE FIELD aliases ON entity TYPE array<string> DEFAULT [] PERMISSIONS FULL;
DEFINE FIELD aliases[*] ON entity TYPE string PERMISSIONS FULL;
DEFINE FIELD canonical_name ON entity TYPE string PERMISSIONS FULL;
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD global_salience ON entity TYPE float DEFAULT 0.1f PERMISSIONS FULL;
DEFINE FIELD last_activation ON entity TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD last_referenced ON entity TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD properties ON entity FLEXIBLE TYPE object DEFAULT {  } PERMISSIONS FULL;
DEFINE FIELD reference_count ON entity TYPE int DEFAULT 1 PERMISSIONS FULL;
DEFINE FIELD type ON entity TYPE string ASSERT $value INSIDE ['person', 'place', 'organization', 'time', 'concept', 'event', 'object', 'user'] PERMISSIONS FULL;

DEFINE INDEX entity_type ON entity FIELDS type;


-- ------------------------------
-- TABLE: field_states
-- ------------------------------

DEFINE TABLE field_states TYPE ANY SCHEMALESS PERMISSIONS NONE;

DEFINE FIELD boundary ON field_states TYPE string ASSERT $value INSIDE ['gradient', 'collapsed'] PERMISSIONS FULL;
DEFINE FIELD compression ON field_states TYPE float ASSERT $value >= 0f AND $value <= 1f PERMISSIONS FULL;
DEFINE FIELD drift ON field_states TYPE string ASSERT $value INSIDE ['none', 'low', 'moderate', 'high'] PERMISSIONS FULL;
DEFINE FIELD fragment_id ON field_states TYPE option<record<memory_fragments>> PERMISSIONS FULL;
DEFINE FIELD instance_id ON field_states TYPE string PERMISSIONS FULL;
DEFINE FIELD memory_tier ON field_states TYPE int DEFAULT 1 PERMISSIONS FULL;
DEFINE FIELD presence_signal ON field_states TYPE float ASSERT $value >= 0f AND $value <= 1f PERMISSIONS FULL;
DEFINE FIELD recursion_depth ON field_states TYPE int ASSERT $value >= 0 AND $value <= 10 PERMISSIONS FULL;
DEFINE FIELD resonance ON field_states TYPE float ASSERT $value >= 0f AND $value <= 1f PERMISSIONS FULL;
DEFINE FIELD updated_at ON field_states TYPE datetime DEFAULT time::now() PERMISSIONS FULL;

DEFINE INDEX field_states_instance ON field_states FIELDS instance_id;


-- ------------------------------
-- TABLE: knowledge
-- ------------------------------

DEFINE TABLE knowledge TYPE ANY SCHEMALESS PERMISSIONS NONE;

DEFINE FIELD access_count ON knowledge TYPE int DEFAULT 0 PERMISSIONS FULL;
DEFINE FIELD confidence ON knowledge TYPE float DEFAULT 0.8f ASSERT $value >= 0f AND $value <= 1f PERMISSIONS FULL;
DEFINE FIELD embedding ON knowledge FLEXIBLE TYPE option<array<float>> PERMISSIONS FULL;
DEFINE FIELD embedding[*] ON knowledge FLEXIBLE TYPE float PERMISSIONS FULL;
DEFINE FIELD extraction_method ON knowledge TYPE string DEFAULT 'spacy' ASSERT $value INSIDE ['spacy', 'conversation', 'manual', 'inferred'] PERMISSIONS FULL;
DEFINE FIELD last_accessed ON knowledge TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD predicate ON knowledge TYPE string PERMISSIONS FULL;
DEFINE FIELD strength ON knowledge TYPE float DEFAULT 1f ASSERT $value >= 0f AND $value <= 1f PERMISSIONS FULL;

DEFINE INDEX knowledge_valid_from ON knowledge FIELDS temporal_context.valid_from;


-- ------------------------------
-- TABLE: memory_fragments
-- ------------------------------

DEFINE TABLE memory_fragments TYPE ANY SCHEMALESS PERMISSIONS NONE;

DEFINE FIELD access_count ON memory_fragments TYPE int DEFAULT 0 PERMISSIONS FULL;
DEFINE FIELD context_tags ON memory_fragments TYPE array<string> DEFAULT [] PERMISSIONS FULL;
DEFINE FIELD context_tags[*] ON memory_fragments TYPE string PERMISSIONS FULL;
DEFINE FIELD created_at ON memory_fragments TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD fragment_id ON memory_fragments TYPE string PERMISSIONS FULL;
DEFINE FIELD last_accessed ON memory_fragments TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD source_interactions ON memory_fragments TYPE array<record<messages>> DEFAULT [] PERMISSIONS FULL;
DEFINE FIELD source_interactions[*] ON memory_fragments TYPE record<messages> PERMISSIONS FULL;
DEFINE FIELD strength ON memory_fragments TYPE float DEFAULT 1f ASSERT $value >= 0f AND $value <= 1f PERMISSIONS FULL;
DEFINE FIELD tier ON memory_fragments TYPE int ASSERT $value >= 1 AND $value <= 4 PERMISSIONS FULL;
DEFINE FIELD type ON memory_fragments TYPE string ASSERT $value INSIDE ['semantic', 'episodic', 'procedural', 'contextual', 'emotional', 'reconstructed'] PERMISSIONS FULL;

DEFINE INDEX fragments_semantic_hash ON memory_fragments FIELDS content.semantic_hash;
DEFINE INDEX fragments_type ON memory_fragments FIELDS type;


-- ------------------------------
-- TABLE: messages
-- ------------------------------

DEFINE TABLE messages TYPE ANY SCHEMALESS PERMISSIONS NONE;

DEFINE FIELD content ON messages TYPE string ASSERT $value != NONE AND $value != '' PERMISSIONS FULL;
DEFINE FIELD embedding ON messages FLEXIBLE TYPE array<float> PERMISSIONS FULL;
DEFINE FIELD embedding[*] ON messages FLEXIBLE TYPE float PERMISSIONS FULL;
DEFINE FIELD metadata ON messages FLEXIBLE TYPE object PERMISSIONS FULL;
DEFINE FIELD parent_message ON messages TYPE option<record<messages>> PERMISSIONS FULL;
DEFINE FIELD role ON messages TYPE string ASSERT $value INSIDE ['user', 'assistant', 'system'] PERMISSIONS FULL;
DEFINE FIELD session_id ON messages TYPE string PERMISSIONS FULL;
DEFINE FIELD speaker_id ON messages TYPE string DEFAULT 'default_user' PERMISSIONS FULL;
DEFINE FIELD timestamp ON messages TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD tokens ON messages TYPE int DEFAULT 0 PERMISSIONS FULL;

DEFINE INDEX idx_messages_role ON messages FIELDS role;
DEFINE INDEX idx_messages_session_id ON messages FIELDS session_id;
DEFINE INDEX idx_messages_session_timestamp ON messages FIELDS session_id, timestamp;
DEFINE INDEX idx_messages_speaker_id ON messages FIELDS speaker_id;
DEFINE INDEX idx_messages_speaker_role ON messages FIELDS speaker_id, role;
DEFINE INDEX idx_messages_timestamp ON messages FIELDS timestamp;
DEFINE INDEX messages_role ON messages FIELDS role;


-- ------------------------------
-- TABLE: sessions
-- ------------------------------

DEFINE TABLE sessions TYPE ANY SCHEMALESS PERMISSIONS NONE;

DEFINE FIELD end_time ON sessions TYPE option<datetime> PERMISSIONS FULL;
DEFINE FIELD is_active ON sessions TYPE bool DEFAULT true PERMISSIONS FULL;
DEFINE FIELD metadata ON sessions FLEXIBLE TYPE object DEFAULT {  } PERMISSIONS FULL;
DEFINE FIELD session_id ON sessions TYPE string PERMISSIONS FULL;
DEFINE FIELD speaker_id ON sessions TYPE string DEFAULT 'default_user' PERMISSIONS FULL;
DEFINE FIELD start_time ON sessions TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD summary ON sessions TYPE option<string> PERMISSIONS FULL;
DEFINE FIELD total_tokens ON sessions TYPE int DEFAULT 0 PERMISSIONS FULL;
DEFINE FIELD turn_count ON sessions TYPE int DEFAULT 0 PERMISSIONS FULL;

DEFINE INDEX idx_sessions_created_at ON sessions FIELDS created_at;
DEFINE INDEX idx_sessions_speaker_id ON sessions FIELDS speaker_id;
DEFINE INDEX idx_sessions_speaker_status ON sessions FIELDS speaker_id, status;
DEFINE INDEX idx_sessions_status ON sessions FIELDS status;


-- ------------------------------
-- TABLE: speakers
-- ------------------------------

DEFINE TABLE speakers TYPE ANY SCHEMALESS PERMISSIONS NONE;

DEFINE FIELD enrollment_status ON speakers TYPE string DEFAULT 'pending' ASSERT $value INSIDE ['pending', 'enrolled', 'verified'] PERMISSIONS FULL;
DEFINE FIELD first_seen ON speakers TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD last_seen ON speakers TYPE datetime DEFAULT time::now() PERMISSIONS FULL;
DEFINE FIELD name ON speakers TYPE option<string> PERMISSIONS FULL;
DEFINE FIELD preferences ON speakers FLEXIBLE TYPE object DEFAULT {  } PERMISSIONS FULL;
DEFINE FIELD speaker_id ON speakers TYPE string PERMISSIONS FULL;
DEFINE FIELD utterance_count ON speakers TYPE int DEFAULT 0 PERMISSIONS FULL;
DEFINE FIELD voice_embedding ON speakers TYPE option<array<float>> PERMISSIONS FULL;
DEFINE FIELD voice_embedding[*] ON speakers TYPE float PERMISSIONS FULL;

DEFINE INDEX speakers_id ON speakers FIELDS speaker_id UNIQUE;


