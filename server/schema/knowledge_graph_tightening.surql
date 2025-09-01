-- ================================================================================
-- KNOWLEDGE GRAPH TIGHTENING - Ontology and Semantic Structure  
-- ================================================================================
-- Adds semantic structure to the existing knowledge graph without breaking it.
-- Creates predicate ontology, inference rules, and consistency constraints.
-- ================================================================================

USE NS slowcat;
USE DB memory_graph;

-- ================================================================================
-- PREDICATE ONTOLOGY - The Core of Graph Tightness
-- ================================================================================

-- Predicate types define semantic categories and properties
DEFINE TABLE predicate_types SCHEMAFULL;
DEFINE FIELD name ON predicate_types TYPE string;
DEFINE FIELD category ON predicate_types TYPE string 
    ASSERT $value IN ['identity', 'spatial', 'temporal', 'social', 'attribute', 'ownership', 'causality'];
DEFINE FIELD parent ON predicate_types TYPE option<record<predicate_types>>;
DEFINE FIELD properties ON predicate_types TYPE object DEFAULT {
    transitive: false,
    symmetric: false,
    functional: false,
    domain_types: [],
    range_types: []
};
DEFINE FIELD properties.transitive ON predicate_types TYPE bool DEFAULT false;
DEFINE FIELD properties.symmetric ON predicate_types TYPE bool DEFAULT false;
DEFINE FIELD properties.functional ON predicate_types TYPE bool DEFAULT false;
DEFINE FIELD properties.inverse_of ON predicate_types TYPE option<string>;
DEFINE FIELD properties.domain_types ON predicate_types TYPE array<string> DEFAULT [];
DEFINE FIELD properties.range_types ON predicate_types TYPE array<string> DEFAULT [];

DEFINE INDEX predicate_types_name ON predicate_types COLUMNS name UNIQUE;
DEFINE INDEX predicate_types_category ON predicate_types COLUMNS category;
DEFINE INDEX predicate_types_parent ON predicate_types COLUMNS parent;

-- ================================================================================
-- CORE PREDICATE ONTOLOGY HIERARCHY
-- ================================================================================

-- Identity Relations
CREATE predicate_types:is_a SET 
    name = 'is_a',
    category = 'identity',
    properties = {
        transitive: true,
        domain_types: ['person', 'concept', 'object'],
        range_types: ['concept']
    };

CREATE predicate_types:instance_of SET
    name = 'instance_of', 
    category = 'identity',
    parent = predicate_types:is_a,
    properties = {
        transitive: false,
        functional: false
    };

-- Naming Relations (Hierarchical)
CREATE predicate_types:has_name SET
    name = 'has_name',
    category = 'identity',
    properties = {
        functional: true,
        domain_types: ['person', 'object', 'place'],
        range_types: ['concept']
    };

CREATE predicate_types:pet_name SET
    name = 'pet_name',
    category = 'identity', 
    parent = predicate_types:has_name,
    properties = {
        functional: true,
        domain_types: ['user'],
        range_types: ['concept']
    };

-- Spatial Relations
CREATE predicate_types:located_in SET
    name = 'located_in',
    category = 'spatial',
    properties = {
        transitive: true,
        functional: false,
        inverse_of: 'contains',
        domain_types: ['person', 'place', 'object'],
        range_types: ['place']
    };

CREATE predicate_types:part_of SET
    name = 'part_of',
    category = 'spatial',
    properties = {
        transitive: true,
        functional: false,
        inverse_of: 'contains',
        domain_types: ['place', 'object', 'concept'],
        range_types: ['place', 'object', 'concept']
    };

-- Social Relations
CREATE predicate_types:works_at SET
    name = 'works_at',
    category = 'social',
    properties = {
        functional: true,  -- Person primarily works at one place
        domain_types: ['person', 'user'],
        range_types: ['organization', 'place']
    };

CREATE predicate_types:knows SET
    name = 'knows',
    category = 'social',
    properties = {
        symmetric: true,
        domain_types: ['person', 'user'],
        range_types: ['person', 'user']
    };

-- Attribute Relations  
CREATE predicate_types:has_attribute SET
    name = 'has_attribute',
    category = 'attribute',
    properties = {
        functional: false,
        domain_types: ['person', 'object', 'concept'],
        range_types: ['concept']
    };

CREATE predicate_types:age SET
    name = 'age',
    category = 'attribute',
    parent = predicate_types:has_attribute,
    properties = {
        functional: true,
        domain_types: ['person', 'user'],
        range_types: ['concept']
    };

CREATE predicate_types:job SET
    name = 'job',
    category = 'attribute',
    parent = predicate_types:has_attribute,
    properties = {
        functional: true,
        domain_types: ['person', 'user'],
        range_types: ['concept']
    };

-- Ownership Relations
CREATE predicate_types:owns SET
    name = 'owns',
    category = 'ownership',
    properties = {
        functional: false,
        inverse_of: 'owned_by',
        domain_types: ['person', 'user'],
        range_types: ['object', 'concept']
    };

-- ================================================================================
-- PREDICATE NORMALIZATION - Map Loose Predicates to Ontology
-- ================================================================================

DEFINE TABLE predicate_mappings SCHEMAFULL;
DEFINE FIELD input_predicate ON predicate_mappings TYPE string;
DEFINE FIELD canonical_predicate ON predicate_mappings TYPE record<predicate_types>;
DEFINE FIELD confidence ON predicate_mappings TYPE float DEFAULT 1.0;
DEFINE FIELD notes ON predicate_mappings TYPE option<string>;

DEFINE INDEX predicate_mappings_input ON predicate_mappings COLUMNS input_predicate UNIQUE;

-- Map current loose predicates to canonical forms
CREATE predicate_mappings:location_map SET
    input_predicate = 'location',
    canonical_predicate = predicate_types:located_in,
    confidence = 0.9,
    notes = 'Generic location mapped to located_in';

CREATE predicate_mappings:works_at_map SET
    input_predicate = 'works_at',
    canonical_predicate = predicate_types:works_at,
    confidence = 1.0;

CREATE predicate_mappings:job_map SET
    input_predicate = 'job',
    canonical_predicate = predicate_types:job,
    confidence = 1.0;

CREATE predicate_mappings:age_map SET
    input_predicate = 'age', 
    canonical_predicate = predicate_types:age,
    confidence = 1.0;

-- Dynamic name mappings (dog_name, cat_name, etc. → pet_name)
CREATE predicate_mappings:dog_name_map SET
    input_predicate = 'dog_name',
    canonical_predicate = predicate_types:pet_name,
    confidence = 1.0,
    notes = 'Specific pet name mapped to generic pet_name';

CREATE predicate_mappings:cat_name_map SET
    input_predicate = 'cat_name',
    canonical_predicate = predicate_types:pet_name,
    confidence = 1.0,
    notes = 'Specific pet name mapped to generic pet_name';

-- ================================================================================
-- SEMANTIC VALIDATION FUNCTIONS
-- ================================================================================

-- Function to normalize predicates using the ontology
DEFINE FUNCTION fn::normalize_predicate($input_predicate: string) {
    -- Check for exact mapping
    LET $mapping = (SELECT * FROM predicate_mappings WHERE input_predicate = $input_predicate LIMIT 1)[0];
    IF $mapping != NONE {
        RETURN $mapping.canonical_predicate.name;
    };
    
    -- Check for pattern-based mappings (*_name → pet_name)
    IF string::ends_with($input_predicate, '_name') {
        LET $pet_mapping = (SELECT * FROM predicate_mappings WHERE input_predicate = 'pet_name_pattern' LIMIT 1)[0];
        IF $pet_mapping != NONE {
            RETURN 'pet_name';
        };
    };
    
    -- Return as-is if no mapping found (for backward compatibility)
    RETURN $input_predicate;
};

-- Function to validate predicate usage
DEFINE FUNCTION fn::validate_predicate_usage($predicate: string, $subject_type: string, $object_type: string) {
    LET $pred_type = (SELECT * FROM predicate_types WHERE name = $predicate LIMIT 1)[0];
    
    IF $pred_type == NONE {
        RETURN { valid: false, reason: 'unknown_predicate' };
    };
    
    -- Check domain constraints (subject type)
    IF array::len($pred_type.properties.domain_types) > 0 AND 
       $subject_type NOT IN $pred_type.properties.domain_types {
        RETURN { valid: false, reason: 'invalid_subject_type', expected: $pred_type.properties.domain_types };
    };
    
    -- Check range constraints (object type)  
    IF array::len($pred_type.properties.range_types) > 0 AND
       $object_type NOT IN $pred_type.properties.range_types {
        RETURN { valid: false, reason: 'invalid_object_type', expected: $pred_type.properties.range_types };
    };
    
    RETURN { valid: true };
};

-- ================================================================================
-- ENHANCED INFERENCE ENGINE
-- ================================================================================

-- Transitive closure with ontology awareness
DEFINE FUNCTION fn::infer_transitive_relations($predicate: string, $max_depth: int) {
    LET $pred_type = (SELECT * FROM predicate_types WHERE name = $predicate LIMIT 1)[0];
    
    IF $pred_type == NONE OR !$pred_type.properties.transitive {
        RETURN { inferred: 0, reason: 'not_transitive' };
    };
    
    LET $new_facts = 0;
    LET $depth = 1;
    
    WHILE $depth <= $max_depth {
        -- Find A→B and B→C to create A→C
        LET $chains = (
            SELECT k1.in as subject, k2.out as object, 
                   (k1.confidence * k2.confidence * 0.9) as confidence
            FROM knowledge k1, knowledge k2
            WHERE k1.predicate = $predicate 
              AND k2.predicate = $predicate
              AND k1.out = k2.in
              AND k1.in != k2.out  -- Prevent cycles
              AND NOT EXISTS (
                  SELECT * FROM knowledge k3 
                  WHERE k3.predicate = $predicate 
                    AND k3.in = k1.in 
                    AND k3.out = k2.out
              )
        );
        
        LET $batch_count = 0;
        FOR $chain IN $chains {
            IF $chain.confidence > 0.3 {
                RELATE $chain.subject->knowledge->$chain.object SET
                    predicate = $predicate,
                    confidence = $chain.confidence,
                    strength = $chain.confidence,
                    extraction_method = 'inferred',
                    created_at = time::now(),
                    temporal_context = { when_said: time::now() };
                    
                LET $batch_count = $batch_count + 1;
            };
        };
        
        LET $new_facts = $new_facts + $batch_count;
        LET $depth = $depth + 1;
        
        -- Stop if no new facts were created
        IF $batch_count == 0 { BREAK; };
    };
    
    RETURN { inferred: $new_facts, depth_reached: $depth };
};

-- Function to create inverse relations
DEFINE FUNCTION fn::create_inverse_relations($predicate: string) {
    LET $pred_type = (SELECT * FROM predicate_types WHERE name = $predicate LIMIT 1)[0];
    
    IF $pred_type == NONE OR $pred_type.properties.inverse_of == NONE {
        RETURN { created: 0, reason: 'no_inverse' };
    };
    
    LET $inverse_predicate = $pred_type.properties.inverse_of;
    LET $created = 0;
    
    -- For each A→B relation, create B→A inverse
    FOR $fact IN (SELECT * FROM knowledge WHERE predicate = $predicate) {
        -- Check if inverse already exists
        LET $existing = (SELECT * FROM knowledge 
                        WHERE in = $fact.out 
                          AND out = $fact.in 
                          AND predicate = $inverse_predicate 
                        LIMIT 1)[0];
                        
        IF $existing == NONE {
            RELATE $fact.out->knowledge->$fact.in SET
                predicate = $inverse_predicate,
                confidence = $fact.confidence * 0.95,
                strength = $fact.strength * 0.95,
                extraction_method = 'inferred',
                created_at = time::now(),
                temporal_context = { when_said: time::now() },
                inverse_of = $fact.id;
                
            LET $created = $created + 1;
        };
    };
    
    RETURN { created: $created };
};

-- ================================================================================
-- CONSISTENCY CHECKING
-- ================================================================================

-- Function to detect functional constraint violations
DEFINE FUNCTION fn::detect_functional_violations() {
    LET $violations = [];
    
    -- Check each functional predicate
    FOR $pred_type IN (SELECT * FROM predicate_types WHERE properties.functional = true) {
        -- Find entities with multiple values for this predicate
        LET $multi_value_entities = (
            SELECT in, count() as violation_count, array::group(out) as conflicting_values
            FROM knowledge 
            WHERE predicate = $pred_type.name
            GROUP BY in
            HAVING count() > 1
        );
        
        IF array::len($multi_value_entities) > 0 {
            LET $violations = array::push($violations, {
                predicate: $pred_type.name,
                violation_type: 'functional_constraint',
                violations: $multi_value_entities
            });
        };
    };
    
    RETURN $violations;
};

-- ================================================================================
-- MIGRATION FUNCTION - Upgrade Existing Knowledge
-- ================================================================================

-- Function to migrate existing knowledge to use canonical predicates
DEFINE FUNCTION fn::migrate_to_canonical_predicates() {
    LET $migrated = 0;
    
    FOR $mapping IN (SELECT * FROM predicate_mappings) {
        -- Find facts using the old predicate
        LET $old_facts = (SELECT * FROM knowledge WHERE predicate = $mapping.input_predicate);
        
        FOR $fact IN $old_facts {
            -- Update to canonical predicate
            UPDATE $fact.id SET 
                predicate = $mapping.canonical_predicate.name,
                confidence = $fact.confidence * $mapping.confidence,
                metadata.migrated_from = $mapping.input_predicate,
                metadata.migration_date = time::now();
                
            LET $migrated = $migrated + 1;
        };
    };
    
    RETURN { migrated: $migrated };
};

-- ================================================================================
-- IMMEDIATE IMPROVEMENTS
-- ================================================================================

-- Create default pet_name pattern mapping for dynamic predicates
CREATE predicate_mappings:pet_pattern SET
    input_predicate = 'pet_name_pattern',
    canonical_predicate = predicate_types:pet_name,
    confidence = 0.9,
    notes = 'Pattern for *_name → pet_name mappings';

-- Track ontology version
CREATE entity:ontology_version SET
    type = 'concept',
    canonical_name = 'knowledge_graph_ontology_v1',
    properties = {
        version: '1.0',
        created: time::now(),
        predicate_types_count: (SELECT count() FROM predicate_types)[0].count,
        mappings_count: (SELECT count() FROM predicate_mappings)[0].count
    };

-- ================================================================================
-- END KNOWLEDGE GRAPH TIGHTENING
-- ================================================================================