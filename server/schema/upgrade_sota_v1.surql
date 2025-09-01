-- ================================================================================
-- SLOWCAT COGNITIVE ARCHITECTURE - SOTA UPGRADE v1
-- ================================================================================
-- Three Pillars Implementation:
-- Pillar 1: The Immutable Guardian (Database Layer)
--   - Ultimate source of truth with semantic integrity
--   - Impossible to store malformed facts
--   - Self-healing through inference and validation
-- ================================================================================

USE NS slowcat;
USE DB memory_graph;

-- ================================================================================
-- RELATION TYPE TAXONOMY - The Foundation of Semantic Intelligence
-- ================================================================================

-- Define relation types with properties for inference engine
DEFINE TABLE relation_types SCHEMAFULL;
DEFINE FIELD name ON relation_types TYPE string;
DEFINE FIELD category ON relation_types TYPE string 
    ASSERT $value IN ['spatial', 'temporal', 'causal', 'logical', 'social', 'ownership', 'attribute', 'identity'];
DEFINE FIELD properties ON relation_types TYPE object {
    transitive: bool,           -- If A→B and B→C then A→C
    symmetric: bool,            -- If A→B then B→A  
    reflexive: bool,            -- A→A is valid
    functional: bool,           -- A can only relate to one B (prevents contradictions)
    inverse_of: option<string>, -- owns/owned_by pairs
    confidence_decay: float     -- How much confidence degrades over inference hops
} DEFAULT {
    transitive: false,
    symmetric: false,
    reflexive: false,
    functional: false,
    confidence_decay: 0.1
};

DEFINE INDEX relation_types_name ON relation_types COLUMNS name UNIQUE;
DEFINE INDEX relation_types_category ON relation_types COLUMNS category;

-- Core relation types for cognitive reasoning
CREATE relation_types:is_a SET 
    name = 'is_a',
    category = 'logical', 
    properties = {
        transitive: true,
        reflexive: true,
        functional: false,
        confidence_decay: 0.05
    };

CREATE relation_types:part_of SET 
    name = 'part_of',
    category = 'spatial',
    properties = {
        transitive: true,
        functional: false,
        inverse_of: 'contains',
        confidence_decay: 0.08
    };

CREATE relation_types:located_in SET
    name = 'located_in',
    category = 'spatial',
    properties = {
        transitive: true,
        functional: true,
        inverse_of: 'contains',
        confidence_decay: 0.05
    };

CREATE relation_types:causes SET
    name = 'causes',
    category = 'causal',
    properties = {
        transitive: false,  -- Causality chains need careful handling
        functional: false,
        confidence_decay: 0.15
    };

CREATE relation_types:before SET
    name = 'before',
    category = 'temporal',
    properties = {
        transitive: true,
        functional: false,
        confidence_decay: 0.1
    };

CREATE relation_types:owns SET
    name = 'owns',
    category = 'ownership',
    properties = {
        functional: false,
        inverse_of: 'owned_by',
        confidence_decay: 0.05
    };

CREATE relation_types:works_at SET
    name = 'works_at',
    category = 'social',
    properties = {
        functional: true,  -- Person typically works at one primary place
        confidence_decay: 0.1
    };

-- ================================================================================
-- SOURCE RELIABILITY TRACKING - Trust Metric for Facts
-- ================================================================================

DEFINE TABLE source_reliability SCHEMAFULL;
DEFINE FIELD source_type ON source_reliability TYPE string;
DEFINE FIELD accuracy_score ON source_reliability TYPE float 
    ASSERT $value >= 0.0 AND $value <= 1.0 DEFAULT 0.8;
DEFINE FIELD total_facts ON source_reliability TYPE int DEFAULT 0;
DEFINE FIELD verified_facts ON source_reliability TYPE int DEFAULT 0;
DEFINE FIELD contradiction_count ON source_reliability TYPE int DEFAULT 0;
DEFINE FIELD last_updated ON source_reliability TYPE datetime DEFAULT time::now();

DEFINE INDEX source_reliability_type ON source_reliability COLUMNS source_type UNIQUE;

-- Initialize source reliability scores
CREATE source_reliability:spacy SET 
    source_type = 'spacy',
    accuracy_score = 0.85,
    total_facts = 0;

CREATE source_reliability:conversation SET
    source_type = 'conversation', 
    accuracy_score = 0.95,
    total_facts = 0;

CREATE source_reliability:inferred SET
    source_type = 'inferred',
    accuracy_score = 0.7,
    total_facts = 0;

CREATE source_reliability:manual SET
    source_type = 'manual',
    accuracy_score = 1.0,
    total_facts = 0;

-- ================================================================================
-- ENHANCED ENTITY VALIDATION - The Guardian's First Line of Defense  
-- ================================================================================

-- Prevent malformed entities at creation time
DEFINE EVENT normalize_entity BEFORE CREATE ON entity
THEN {
    -- Clean canonical name (language-agnostic)
    LET $clean = string::trim($value.canonical_name);
    
    -- Remove possessive markers (English-specific but harmless for others)
    IF string::ends_with($clean, "'s") {
        LET $clean = string::slice($clean, 0, string::length($clean) - 2);
    } ELSE IF string::ends_with($clean, "s'") {
        LET $clean = string::slice($clean, 0, string::length($clean) - 1);
    };
    
    -- Remove leading articles (English but harmless to check)
    LET $words = string::words($clean);
    IF array::len($words) > 1 AND array::at($words, 0) IN ['the', 'a', 'an', 'The', 'A', 'An'] {
        LET $clean = string::join(array::slice($words, 1), ' ');
    };
    
    -- Convert to lowercase for canonicalization
    LET $clean = string::lowercase($clean);
    
    -- Reject empty or too-short entities
    IF string::length($clean) < 2 {
        THROW "Entity name too short after cleaning";
    };
    
    -- Check for existing entity (exact match or alias)
    LET $existing = (SELECT * FROM entity WHERE 
        canonical_name = $clean OR 
        $clean IN aliases
    LIMIT 1)[0];
    
    IF $existing != NONE {
        -- Merge into existing entity instead of duplicating
        UPDATE $existing.id SET 
            aliases = array::union(aliases, [$value.canonical_name]),
            reference_count = reference_count + 1,
            last_referenced = time::now();
        THROW "Entity merged into existing: " + <string>$existing.id;
    };
    
    -- Set the cleaned canonical name
    SET $value.canonical_name = $clean;
};

-- ================================================================================
-- FACT VALIDATION - The Guardian's Core Intelligence
-- ================================================================================

-- Prevent malformed, circular, and duplicate facts
DEFINE EVENT validate_knowledge BEFORE CREATE ON knowledge
THEN {
    -- CRITICAL: Prevent self-referential relations
    IF $value.in == $value.out {
        THROW "Self-referential relation not allowed: " + <string>$value.in;
    };
    
    -- Validate predicate is meaningful
    IF string::length($value.predicate) < 2 {
        THROW "Predicate too short: " + $value.predicate;
    };
    
    -- Check for generic predicates without semantic value
    IF $value.predicate IN ['is', 'has', 'was'] AND string::length(<string>$value.out) < 3 {
        THROW "Generic predicate requires meaningful object: " + $value.predicate + " -> " + <string>$value.out;
    };
    
    -- Validate against relation type properties if defined
    LET $rel_type = (SELECT * FROM relation_types WHERE name = $value.predicate LIMIT 1)[0];
    IF $rel_type != NONE {
        -- Check functional constraint (one-to-one relationships)
        IF $rel_type.properties.functional {
            LET $existing = (SELECT * FROM knowledge WHERE 
                in = $value.in AND 
                predicate = $value.predicate
            LIMIT 1)[0];
            
            IF $existing != NONE AND $existing.out != $value.out {
                THROW "Functional relation violation: " + <string>$value.in + " can only " + $value.predicate + " one entity";
            };
        };
    };
    
    -- Check for exact duplicates
    LET $duplicate = (SELECT * FROM knowledge WHERE 
        in = $value.in AND 
        out = $value.out AND 
        predicate = $value.predicate
    LIMIT 1)[0];
    
    IF $duplicate != NONE {
        -- Strengthen existing fact instead of duplicating
        UPDATE $duplicate.id SET 
            confidence = math::min(1.0, confidence + 0.1),
            access_count = access_count + 1,
            last_accessed = time::now(),
            strength = math::min(1.0, strength + 0.05);
        
        -- Update source reliability
        LET $source_rel = (SELECT * FROM source_reliability 
            WHERE source_type = $value.extraction_method LIMIT 1)[0];
        IF $source_rel != NONE {
            UPDATE $source_rel.id SET 
                total_facts = total_facts + 1,
                verified_facts = verified_facts + 1;
        };
        
        THROW "Fact strengthened instead of duplicated: " + <string>$duplicate.id;
    };
};

-- Adjust confidence based on source reliability after creation
DEFINE EVENT adjust_confidence AFTER CREATE ON knowledge
THEN {
    LET $source_rel = (SELECT * FROM source_reliability 
        WHERE source_type = $value.extraction_method LIMIT 1)[0];
    
    IF $source_rel != NONE {
        -- Apply source reliability to confidence
        UPDATE $value.id SET 
            confidence = confidence * $source_rel.accuracy_score;
            
        -- Update source stats
        UPDATE $source_rel.id SET 
            total_facts = total_facts + 1;
    };
};

-- ================================================================================
-- INFERENCE ENGINE - The Guardian's Reasoning Mind
-- ================================================================================

-- Transitive inference for supported relations
DEFINE FUNCTION fn::infer_transitive_facts($predicate: string) {
    LET $rel_type = (SELECT * FROM relation_types WHERE name = $predicate LIMIT 1)[0];
    
    IF $rel_type != NONE AND $rel_type.properties.transitive {
        LET $new_facts = 0;
        
        -- Find transitive chains A→B→C to create A→C
        FOR $fact1 IN (SELECT * FROM knowledge WHERE predicate = $predicate AND confidence > 0.5) {
            FOR $fact2 IN (SELECT * FROM knowledge WHERE 
                predicate = $predicate AND 
                in = $fact1.out AND 
                confidence > 0.5) {
                
                -- Calculate inferred confidence with decay
                LET $confidence = $fact1.confidence * $fact2.confidence * (1.0 - $rel_type.properties.confidence_decay);
                
                -- Only create fact if confidence is meaningful
                IF $confidence > 0.3 {
                    -- Check if inferred fact already exists
                    LET $existing = (SELECT * FROM knowledge WHERE 
                        in = $fact1.in AND 
                        out = $fact2.out AND 
                        predicate = $predicate
                    LIMIT 1)[0];
                    
                    IF $existing == NONE {
                        CREATE knowledge SET
                            in = $fact1.in,
                            out = $fact2.out,
                            predicate = $predicate,
                            confidence = $confidence,
                            strength = $confidence,
                            extraction_method = 'inferred',
                            created_at = time::now(),
                            last_accessed = time::now(),
                            access_count = 0,
                            temporal_context = {
                                when_said: time::now(),
                                inferred_from: [$fact1.id, $fact2.id]
                            };
                        
                        LET $new_facts = $new_facts + 1;
                    };
                };
            };
        };
        
        RETURN $new_facts;
    };
    
    RETURN 0;
};

-- Detect contradictory facts
DEFINE FUNCTION fn::detect_contradictions() {
    RETURN SELECT 
        k1.*,
        k2.* as contradicts_with,
        (k1.confidence + k2.confidence) as combined_confidence
    FROM knowledge k1, knowledge k2
    WHERE k1.in = k2.in
      AND k1.predicate = k2.predicate  
      AND k1.out != k2.out
      AND k1.id != k2.id
      AND k1.predicate IN (
          SELECT name FROM relation_types WHERE properties.functional = true
      )
    ORDER BY combined_confidence DESC;
};

-- Resolve contradictions by lowering confidence of weaker fact
DEFINE FUNCTION fn::resolve_contradictions() {
    LET $contradictions = fn::detect_contradictions();
    LET $resolved = 0;
    
    FOR $contradiction IN $contradictions {
        IF $contradiction.confidence > $contradiction.contradicts_with.confidence {
            -- Lower confidence of weaker fact
            UPDATE $contradiction.contradicts_with.id SET
                confidence = confidence * 0.7,
                strength = strength * 0.8;
        } ELSE {
            -- Lower confidence of current fact
            UPDATE $contradiction.id SET
                confidence = confidence * 0.7,
                strength = strength * 0.8;
        };
        
        LET $resolved = $resolved + 1;
    };
    
    RETURN $resolved;
};

-- ================================================================================
-- SEMANTIC SEARCH WITH VECTOR SIMILARITY
-- ================================================================================

-- Enhanced search combining text and vector similarity
DEFINE FUNCTION fn::semantic_search($query: string, $query_embedding: option<array<float>>, $limit: int) {
    -- Text-based search
    LET $text_results = SELECT *, 
        ('text' as search_type),
        (1.0 as text_score),
        (0.0 as vector_score)
    FROM knowledge
    WHERE string::contains(string::lowercase(predicate), string::lowercase($query))
       OR string::contains(string::lowercase(in.canonical_name), string::lowercase($query))
       OR string::contains(string::lowercase(out.canonical_name), string::lowercase($query))
    ORDER BY strength DESC, confidence DESC
    LIMIT $limit;
    
    -- Vector-based search (when embeddings available)
    LET $vector_results = [];
    IF $query_embedding != NONE {
        LET $vector_results = SELECT *, 
            ('vector' as search_type),
            (0.0 as text_score),
            (vector::similarity::cosine(embedding, $query_embedding) as vector_score)
        FROM knowledge
        WHERE embedding != NONE AND vector::similarity::cosine(embedding, $query_embedding) > 0.7
        ORDER BY vector_score DESC
        LIMIT $limit;
    };
    
    -- Combine and deduplicate results
    RETURN array::union($text_results, $vector_results);
};

-- ================================================================================
-- MEMORY DECAY AND MAINTENANCE
-- ================================================================================

-- Advanced decay calculation considering access patterns
DEFINE FUNCTION fn::calculate_advanced_decay(
    $created_at: datetime,
    $last_accessed: datetime, 
    $access_count: int,
    $confidence: float
) {
    LET $age_days = (time::now() - $created_at) / 1d;
    LET $recency_hours = (time::now() - $last_accessed) / 1h;
    
    -- Base decay over time (slower for high confidence)
    LET $base_decay = 1.0 / (1.0 + ($age_days / (30.0 * $confidence)));
    
    -- Access frequency boost
    LET $access_boost = math::min(0.3, $access_count / 50.0);
    
    -- Recency penalty (unused memories fade faster)
    LET $recency_penalty = 1.0 / (1.0 + $recency_hours / (168.0 * $confidence));
    
    -- Combine factors
    LET $final_strength = math::max(0.05, $base_decay + $access_boost * $recency_penalty);
    
    RETURN $final_strength;
};

-- Periodic memory maintenance
DEFINE EVENT advanced_memory_decay ON TABLE knowledge WHEN 
    time::now() - $after.last_accessed > 72h  -- 3 days
THEN {
    LET $new_strength = fn::calculate_advanced_decay(
        $after.created_at,
        $after.last_accessed, 
        $after.access_count,
        $after.confidence
    );
    
    UPDATE $after.id SET strength = $new_strength;
    
    -- Archive very weak memories instead of deleting
    IF $new_strength < 0.1 AND $after.extraction_method != 'manual' {
        UPDATE $after.id SET 
            strength = 0.05,  -- Archive marker
            metadata = object::set(metadata, 'archived_at', time::now());
    };
};

-- ================================================================================
-- GUARDIAN INITIALIZATION
-- ================================================================================

-- Create guardian entity to track system knowledge
CREATE entity:guardian SET
    type = 'concept',
    canonical_name = 'system_guardian',
    properties = {
        role: 'knowledge_validation',
        version: 'sota_v1',
        initialized_at: time::now()
    };

-- Track schema upgrade
CREATE knowledge SET
    in = entity:guardian,
    out = entity:guardian,
    predicate = 'upgraded_to',
    confidence = 1.0,
    strength = 1.0,
    extraction_method = 'manual',
    temporal_context = {
        when_said: time::now(),
        upgrade_version: 'sota_v1'
    };

-- ================================================================================
-- END OF SOTA UPGRADE v1
-- ================================================================================