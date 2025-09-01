# Three Pillars Cognitive Architecture - Integration Guide

## Overview

The Three Pillars Cognitive Architecture provides a state-of-the-art knowledge management system with strict separation of concerns:

1. **Pillar 1: The Immutable Guardian** (Database Layer) - Ultimate source of truth with semantic integrity
2. **Pillar 2: The Intelligent Scribe** (Application Layer) - Translates chaotic input into clean requests  
3. **Pillar 3: The Coherent Narrator** (Retrieval/Display Layer) - Formats clean data for users

## Integration Steps

### Step 1: Apply the Schema Migration

First, apply the SOTA schema upgrade to enable the Guardian's protective capabilities:

```bash
cd server
source .venv/bin/activate

# Connect to SurrealDB and apply schema
surreal start --log trace file:memory_graph.db
surreal sql --conn http://localhost:8000 --user root --pass root --ns slowcat --db memory_graph --file schema/upgrade_sota_v1.surql
```

This creates:
- ✅ **Relation type taxonomy** with inference properties
- ✅ **Source reliability tracking** for trust metrics
- ✅ **Entity normalization** preventing duplicates
- ✅ **Fact validation** blocking malformed data
- ✅ **Inference engine** functions for reasoning
- ✅ **Memory decay** with advanced algorithms

### Step 2: Initialize the Background Inference Engine

Add this to your main application startup (e.g., in `bot_v2.py`):

```python
from memory.inference_worker import initialize_cognitive_background
from memory.surreal_connection import SurrealDBConnectionManager

async def startup_sequence():
    # ... existing startup code ...
    
    # Initialize connection manager
    connection_manager = SurrealDBConnectionManager()
    
    # Start cognitive background processes
    await initialize_cognitive_background(connection_manager)
    
    logger.info("🧠 Three Pillars Cognitive Architecture initialized")
```

### Step 3: Update Fact Storage to Use the Scribe

The fact storage system is already updated to use the Three Pillars architecture. The flow is:

```
Raw Facts → Cognitive Scribe (normalization) → Guardian (validation) → Storage
```

No changes needed - the existing `store_facts()` method now uses:
- **Scribe**: Cleans and prepares facts
- **Guardian**: Validates and stores only clean data

### Step 4: Enhanced Display (Optional)

Update your display formatters to trust the clean data from Guardian:

```python
# The DTH formatter is already fixed, but you can simplify it further
# since the Guardian guarantees clean data

def format_facts_context(facts):
    """Format facts - simplified because Guardian ensures data quality"""
    lines = []
    for fact in facts:
        # No need for complex validation - Guardian already did it
        if fact.subject == 'user':
            lines.append(f"- you {fact.predicate} {fact.value}")
        else:
            lines.append(f"- {fact.subject} {fact.predicate} {fact.value}")
    return "\n".join(lines)
```

## Verification & Testing

### Test the Guardian Protection

Try to manually insert bad data to verify the Guardian rejects it:

```sql
-- These should all be rejected by the Guardian
CREATE knowledge SET in = entity:test, out = entity:test, predicate = 'self_relation';  -- Circular
CREATE knowledge SET in = entity:user, out = entity:empty, predicate = 'is';  -- Generic
CREATE entity SET canonical_name = '';  -- Empty name
CREATE entity SET canonical_name = 'Test Entity';  -- Should merge with existing test_entity
```

Expected results:
```
"Self-referential relation not allowed"
"Generic predicate requires meaningful object"
"Entity name too short after cleaning"  
"Entity merged into existing: entity:test_entity"
```

### Test the Scribe Processing

```python
from memory.cognitive_scribe import get_cognitive_scribe

# Test fact preparation
scribe = get_cognitive_scribe()

# This should be cleaned and accepted
clean_fact = scribe.prepare_fact("My dog's name", "is", "Potola")
# Result: CleanFact(subject='my dog', predicate='is', object='potola')

# This should be rejected in pre-validation
bad_fact = scribe.prepare_fact("test", "is", "test")  # Circular
# Result: None (rejected)
```

### Test the Inference Engine

Check that background inference is working:

```python
from memory.inference_worker import get_inference_stats

# Check inference statistics
stats = get_inference_stats()
print(f"Facts inferred: {stats['facts_inferred']}")
print(f"Contradictions resolved: {stats['contradictions_resolved']}")
```

## Monitoring & Observability

### Key Metrics to Track

1. **Guardian Rejection Rate**: How often the database rejects facts
   ```python
   scribe_stats = get_cognitive_scribe().get_scribe_stats()
   guardian_rejection_rate = 1 - scribe_stats['guardian_acceptance_rate']
   ```

2. **Inference Productivity**: New facts generated through reasoning
   ```sql
   SELECT count() FROM knowledge WHERE extraction_method = 'inferred';
   ```

3. **Memory Health**: Distribution of fact strengths
   ```sql
   SELECT 
     count() as total_facts,
     (SELECT count() FROM knowledge WHERE strength > 0.8) as strong_facts,
     (SELECT count() FROM knowledge WHERE strength < 0.3) as weak_facts
   FROM knowledge;
   ```

4. **Source Reliability**: How reliable each source is
   ```sql
   SELECT * FROM source_reliability ORDER BY accuracy_score DESC;
   ```

### Dashboard Queries

Monitor system health with these SurrealQL queries:

```sql
-- System Overview
SELECT 
  (SELECT count() FROM entity) as total_entities,
  (SELECT count() FROM knowledge) as total_facts,
  (SELECT count() FROM knowledge WHERE extraction_method = 'inferred') as inferred_facts,
  (SELECT math::mean(confidence) FROM knowledge) as avg_confidence
;

-- Recent Activity (last 24 hours)  
SELECT count() as new_facts
FROM knowledge 
WHERE created_at > time::now() - 24h;

-- Top Relations
SELECT predicate, count() as usage
FROM knowledge 
GROUP BY predicate 
ORDER BY usage DESC 
LIMIT 10;

-- Contradiction Report
SELECT * FROM fn::detect_contradictions();
```

## Performance Optimization

### Database Indexes

The schema includes optimized indexes, but monitor these queries for performance:

```sql
-- Most common queries that should be fast:
SELECT * FROM knowledge WHERE in.canonical_name = 'user';  -- User facts
SELECT * FROM knowledge WHERE predicate = 'located_in';    -- Spatial queries
SELECT * FROM entity WHERE canonical_name = 'test';        -- Entity lookup
```

### Memory Management

The system automatically manages memory through:
- **Decay**: Unused facts fade over time
- **Archival**: Very weak facts are archived, not deleted
- **Consolidation**: Duplicate facts are merged with higher confidence

### Background Processing

The inference engine runs these cycles:
- **Quick inference**: Every 5 minutes (lightweight)
- **Deep inference**: Every 1 hour (comprehensive)
- **Contradiction check**: Every 30 minutes
- **Memory maintenance**: Every 2 hours
- **Health reports**: Every 24 hours

## Troubleshooting

### Common Issues

1. **High Guardian Rejection Rate**
   - Check the scribe's pre-validation logic
   - Review LLM fact extraction quality
   - Monitor logs for rejection reasons

2. **No Facts Being Inferred**
   - Verify relation_types are defined with `transitive: true`
   - Check that base facts exist for inference chains
   - Ensure inference engine is running

3. **Memory Growing Too Fast**
   - Tune decay parameters in the schema
   - Check for facts with `extraction_method = 'manual'` (never decay)
   - Review source reliability scores

4. **Contradictions Not Resolving**
   - Check that functional relations are defined in relation_types
   - Verify contradiction detection is finding conflicts
   - Monitor confidence adjustment in resolution

### Debug Logging

Enable detailed logging to troubleshoot:

```python
import logging
logging.getLogger('memory.cognitive_scribe').setLevel(logging.DEBUG)
logging.getLogger('memory.inference_worker').setLevel(logging.DEBUG)
```

### Manual Fixes

If the system gets into a bad state:

```sql
-- Clean up circular facts
DELETE FROM knowledge WHERE in = out;

-- Reset source reliability
UPDATE source_reliability SET accuracy_score = 0.8 WHERE source_type = 'spacy';

-- Archive very weak facts
UPDATE knowledge SET 
  strength = 0.05,
  metadata.archived_at = time::now()
WHERE strength < 0.1 AND extraction_method != 'manual';
```

## Future Enhancements

### Vector Search Integration

When SurrealDB vector support is fully available:

```sql
-- Enhanced semantic search
DEFINE INDEX knowledge_embedding ON knowledge 
FIELDS embedding VECTOR DIMENSION 384 DISTANCE COSINE;

-- Semantic similarity queries
SELECT *, vector::similarity::cosine(embedding, $query_embedding) as similarity
FROM knowledge 
WHERE vector::similarity::cosine(embedding, $query_embedding) > 0.7
ORDER BY similarity DESC;
```

### Advanced Reasoning

Future inference capabilities:
- **Causal reasoning**: If A causes B and B causes C, then A indirectly causes C
- **Temporal reasoning**: Event ordering and duration calculations
- **Probabilistic inference**: Uncertainty propagation through reasoning chains
- **Contradiction ranking**: Prioritize resolution based on source reliability

### Multi-Language Support

Extend the normalization for other languages:
- **Romance languages**: Handle accented characters and gendered articles
- **Germanic languages**: Compound word splitting and case handling
- **Asian languages**: Word segmentation and character normalization
- **Arabic/Hebrew**: Right-to-left text and script-specific cleaning

This architecture provides the foundation for a truly intelligent, self-maintaining knowledge system that grows more accurate and useful over time.