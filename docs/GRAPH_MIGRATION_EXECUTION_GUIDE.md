# Graph Migration Execution Guide

## 🎯 Overview

This guide walks you through executing the complete migration from flat tables to graph-native schema. All scripts have been created and are ready to run.

## 📁 Created Files

### Scripts
- `server/scripts/extract_backup_data.py` - Parse SURQL backup file 
- `server/scripts/migrate_to_graph.py` - Transform data to graph schema
- `server/scripts/validate_migration.py` - Verify migration success

### Memory System  
- `server/memory/graph_surreal_memory.py` - New graph-native memory class

## 🚀 Execution Steps

### Step 1: Prepare Environment

```bash
cd server/
source .venv/bin/activate

# Ensure SurrealDB is running and reachable
# Backup file present at: server/memory/localslowcat-2025-08-27.surql

# Use a clean graph database
export SURREALDB_URL=${SURREALDB_URL:-ws://127.0.0.1:8000/rpc}
export SURREALDB_USER=${SURREALDB_USER:-root}
export SURREALDB_PASS=${SURREALDB_PASS:-slowcat_secure_2024}
export SURREALDB_NAMESPACE=${SURREALDB_NAMESPACE:-slowcat}
export SURREALDB_DATABASE=${SURREALDB_DATABASE:-memory_graph}

# Prevent legacy flat tables from being created by any legacy code paths
export SC_SKIP_LEGACY_SCHEMA_INIT=true
export SC_SCHEMA_MODE=graph

If you previously started the bot and it recreated legacy flat tables in this DB, remove them now to keep the graph DB clean (in Surrealist):

```sql
REMOVE TABLE fact;
REMOVE TABLE tape;
REMOVE TABLE sessions;
REMOVE TABLE session_summary;
REMOVE TABLE thought;        -- legacy flat version
REMOVE TABLE emergent_event;  -- legacy flat version
```
```

### Step 2: Apply Graph Schema

Apply the idempotent graph schema to the clean `memory_graph` DB:

```bash
python scripts/apply_graph_schema.py --ns "$SURREALDB_NAMESPACE" --db "$SURREALDB_DATABASE"
```

Verify core tables and relations exist (in Surrealist): `INFO FOR TABLE user;`, `session;`, `message;`, `concept;`, `knows;`, `contains;`.

### Step 3: Extract Data from Backup

```bash
cd scripts/
python extract_backup_data.py --backup "../memory/localslowcat-2025-08-27.surql"
```

**Expected Output:**
- Parses your 15MB+ SURQL backup
- Extracts all INSERT statements  
- Creates JSON files in `extracted_data/` for inspection
- Shows count of records per table

### Step 4: Run Migration to Graph Schema

```bash
python migrate_to_graph.py --backup "../memory/localslowcat-2025-08-27.surql"
```

**What This Does:**
- Creates `user` nodes from speakers (peppi, etc.)
- Creates `session` nodes linked to users
- Converts `tape` → `message` nodes with session relationships
- Transforms `fact` → `user->knows->concept` relationships  
- Links `thought`/`emergent_event` to sessions
- Creates all graph relationships using `RELATE` statements

**Expected Output:**
```
Users created: 1
Sessions created: 25+  
Messages created: 1000+
Concepts created: 4
Knowledge relations: 4
Contains relations: 1000+
Reflects relations: 18
```

### Step 5: Validate Migration Success

```bash
python validate_migration.py
```

**What This Tests:**
- ✅ All users created from speakers
- ✅ Sessions linked to users  
- ✅ Messages linked to sessions
- ✅ Facts converted to knowledge relationships
- ✅ Graph traversal queries work
- ✅ Search functionality preserved
- ✅ Data integrity maintained

### Step 6: Update Memory Factory

Update `server/memory/__init__.py` to use the new graph memory:

```python
# Replace in create_smart_memory_system()
from .graph_surreal_memory import GraphSurrealMemory
return GraphSurrealMemory()  # Direct graph memory
```

### Step 7: Test Integration

Run existing tests to ensure compatibility:

```bash
cd ../  # Back to server/
python test_phase1_improvements.py  # Should still work
python -m pytest tests/ -v  # Run test suite
```

## 🔍 Verification Queries

After migration, you can run these in Surrealist to verify the graph structure:

### Check Users and Their Knowledge
```sql
-- See what user knows
SELECT ->knows->concept.* FROM user:peppi;

-- User's knowledge with relationship details
SELECT 
    relationship,
    strength,  
    fidelity,
    out.name as concept_name
FROM user:peppi->knows;
```

### Check Session Conversations
```sql  
-- Get session messages
SELECT ->contains->message.* FROM session:specific_id;

-- User's conversation history
SELECT 
    id,
    summary,
    turn_count,
    (->contains->message.content)[0..3] as sample_messages
FROM session 
WHERE user_id = user:peppi;
```

### Graph Traversals
```sql
-- Multi-hop: User -> Sessions -> Messages
SELECT content 
FROM user:peppi<-session->contains->message 
ORDER BY timestamp DESC 
LIMIT 10;

-- Find sessions where user mentioned specific concepts
SELECT session.*
FROM user:peppi->knows->concept<-mentions<-message<-contains<-session
WHERE concept.name = 'Potola';
```

## 🎯 Key Benefits Achieved

### 1. True Graph Relationships
**Before**: Isolated facts with no context
```json
{"subject": "user", "predicate": "dog_name", "value": "Potola"}
```

**After**: Connected knowledge graph
```
user:peppi -> knows -> concept:potola
  {relationship: "has_pet", strength: 0.92, learned_from_session: "peppi_20326"}
```

### 2. Conversation Context  
**Before**: Flat messages with session_id strings
```json
{"content": "Hello", "session_id": "peppi_20326"}
```

**After**: Linked conversation flow
```
session:abc123 -> contains -> message:xyz789
  {sequence_num: 1, created_at: "2025-08-26T17:12:16Z"}
```

### 3. Graph Analytics
```sql
-- Find related concepts user knows
SELECT DISTINCT out.name 
FROM user:peppi->knows 
WHERE out.kind = 'pet';

-- Cross-session knowledge patterns  
SELECT session.summary, knowledge.relationship
FROM user:peppi->knows as knowledge,
     user:peppi<-session as session
WHERE knowledge.learned_from_session = session.id;
```

## 🚨 Troubleshooting

### Common Issues

**1. "SurrealDB connection failed"**
- Ensure SurrealDB is running: `surreal start`
- Check credentials in `.env`

**2. "No data found in backup"** 
- Verify backup file path in scripts
- Check backup file format

**3. "Migration script errors"**
- Check SurrealDB logs for constraint violations
- Ensure graph schema was created first

**4. "Validation tests fail"**
- Run migration script again
- Check for orphaned relationships

### Recovery Plan

If migration fails, you can restore:
```bash
# Restore from your backup
surreal import --conn ws://localhost:8000 \
  --user root --pass your_password \
  --ns slowcat --db memory \
  localslowcat-2025-08-27.surql
```

## 🎉 Success Criteria

✅ **All validation tests pass**  
✅ **Graph traversals work**: `user:peppi->knows->concept.*`  
✅ **Conversations linked**: `session->contains->message`  
✅ **Search preserved**: FTS + relationship filtering  
✅ **Performance improved**: Indexed graph queries  
✅ **Data integrity**: No orphaned relationships  

After successful migration, you'll have a true graph database with:
- **Connected knowledge** instead of isolated facts
- **Relationship-based queries** instead of complex JOINs  
- **Graph traversals** for multi-hop analytics
- **Native SurrealDB features** fully leveraged

## 🚀 Next Steps After Migration

1. **Update pipeline integration** to use graph memory
2. **Implement advanced graph queries** for smart retrieval  
3. **Add real-time subscriptions** with `LIVE SELECT`
4. **Performance optimization** with specialized indexes
5. **Graph visualization** for debugging and analytics

Your voice agent will now have true **relationship intelligence** instead of just isolated data storage!
