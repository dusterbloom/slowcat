# Clean Slate SurrealDB Migration Plan

## 🎯 Strategy: Delete & Rebuild with Proper Graph Schema

**Backup Status**: ✅ Done  
**Risk Level**: Low (backup available)  
**Benefit**: Native graph relationships from day one

## Phase 1: Clean Slate (Run in Surrealist)

### Step 1: Export Current Data (Optional Safety)
```sql
-- Export key data before deletion
SELECT * FROM fact;
SELECT * FROM tape; 
SELECT * FROM sessions;
SELECT * FROM session_summary;
SELECT * FROM thought;
SELECT * FROM emergent_event;
```

### Step 2: Drop All Tables
```sql
-- Clean slate - remove all existing tables
REMOVE TABLE fact;
REMOVE TABLE tape;
REMOVE TABLE session;     -- duplicate table
REMOVE TABLE sessions;
REMOVE TABLE session_summary;
REMOVE TABLE thought;
REMOVE TABLE emergent_event;
```

### Step 3: Create Graph-Native Schema

#### Core Entities (Nodes)
```sql
-- Users/Speakers as entities
DEFINE TABLE user SCHEMAFULL;
DEFINE FIELD name ON user TYPE string;
DEFINE FIELD first_seen ON user TYPE datetime VALUE time::now();
DEFINE FIELD last_seen ON user TYPE datetime VALUE time::now();
DEFINE FIELD total_interactions ON user TYPE number DEFAULT 0;
DEFINE FIELD metadata ON user TYPE object DEFAULT {};

-- Sessions as conversation containers
DEFINE TABLE session SCHEMAFULL;
DEFINE FIELD user_id ON session TYPE record<user>;
DEFINE FIELD agent_id ON session TYPE string DEFAULT 'slowcat';
DEFINE FIELD started_at ON session TYPE datetime VALUE time::now();
DEFINE FIELD ended_at ON session TYPE datetime;
DEFINE FIELD turn_count ON session TYPE number DEFAULT 0;
DEFINE FIELD duration_secs ON session TYPE number DEFAULT 0;
DEFINE FIELD summary ON session TYPE string DEFAULT '';
DEFINE FIELD keywords ON session TYPE array<string> DEFAULT [];
DEFINE FIELD status ON session TYPE string DEFAULT 'active'; -- 'active', 'ended', 'archived'

-- Concepts as knowledge entities
DEFINE TABLE concept SCHEMAFULL;
DEFINE FIELD name ON concept TYPE string;
DEFINE FIELD kind ON concept TYPE string;  -- 'pet', 'person', 'place', 'thing', etc.
DEFINE FIELD properties ON concept TYPE object DEFAULT {};
DEFINE FIELD mentioned_count ON concept TYPE number DEFAULT 0;
DEFINE FIELD first_mentioned ON concept TYPE datetime VALUE time::now();
DEFINE FIELD last_mentioned ON concept TYPE datetime VALUE time::now();

-- Messages as conversation turns
DEFINE TABLE message SCHEMAFULL;
DEFINE FIELD session_id ON message TYPE record<session>;
DEFINE FIELD speaker_type ON message TYPE string; -- 'user' or 'assistant'
DEFINE FIELD content ON message TYPE string;
DEFINE FIELD timestamp ON message TYPE datetime VALUE time::now();
DEFINE FIELD sequence_num ON message TYPE number;
DEFINE FIELD embedding ON message TYPE array<number>;
DEFINE FIELD metadata ON message TYPE object DEFAULT {};

-- Thoughts as private agent reflections
DEFINE TABLE thought SCHEMAFULL;
DEFINE FIELD agent_id ON thought TYPE string;
DEFINE FIELD session_id ON thought TYPE record<session>;
DEFINE FIELD thought_type ON thought TYPE string; -- 'observation', 'hypothesis', 'followup_seed'
DEFINE FIELD content ON thought TYPE string;
DEFINE FIELD timestamp ON thought TYPE datetime VALUE time::now();
DEFINE FIELD visibility ON thought TYPE string DEFAULT 'private';
DEFINE FIELD confidence ON thought TYPE number DEFAULT 0.5;
```

#### Relationships (Edges)
```sql
-- User knows concepts (facts)
DEFINE TABLE knows TYPE RELATION IN user OUT concept SCHEMAFULL;
DEFINE FIELD relationship ON knows TYPE string;  -- 'has_pet', 'likes', 'lives_in', etc.
DEFINE FIELD fidelity ON knows TYPE number DEFAULT 3;      -- S4=verbatim to S0=forgotten
DEFINE FIELD strength ON knows TYPE number DEFAULT 0.6;     -- confidence/importance
DEFINE FIELD learned_at ON knows TYPE datetime VALUE time::now();
DEFINE FIELD reinforced_at ON knows TYPE datetime VALUE time::now();
DEFINE FIELD source_message ON knows TYPE record<message>;   -- where this fact came from
DEFINE FIELD access_count ON knows TYPE number DEFAULT 0;
DEFINE FIELD decay_rate ON knows TYPE number DEFAULT 1.0;

-- Sessions contain messages (conversation flow)
DEFINE TABLE contains TYPE RELATION IN session OUT message SCHEMAFULL;
DEFINE FIELD sequence_num ON contains TYPE number;
DEFINE FIELD created_at ON contains TYPE datetime VALUE time::now();

-- Messages mention concepts (semantic links)
DEFINE TABLE mentions TYPE RELATION IN message OUT concept SCHEMAFULL;
DEFINE FIELD relevance ON mentions TYPE number DEFAULT 0.5;
DEFINE FIELD context ON mentions TYPE string DEFAULT '';
DEFINE FIELD extracted_at ON mentions TYPE datetime VALUE time::now();

-- Sessions generate thoughts (reflection links)
DEFINE TABLE reflects TYPE RELATION IN session OUT thought SCHEMAFULL;
DEFINE FIELD generated_at ON reflects TYPE datetime VALUE time::now();
DEFINE FIELD trigger_event ON reflects TYPE string DEFAULT '';
```

#### Indexes for Performance
```sql
-- Core lookups
DEFINE INDEX idx_user_name ON user FIELDS name UNIQUE;
DEFINE INDEX idx_session_user ON session FIELDS user_id;
DEFINE INDEX idx_session_agent ON session FIELDS agent_id;
DEFINE INDEX idx_concept_name ON concept FIELDS name;
DEFINE INDEX idx_concept_kind ON concept FIELDS kind;

-- Message queries (hot path)
DEFINE INDEX idx_message_session ON message FIELDS session_id;
DEFINE INDEX idx_message_timestamp ON message FIELDS timestamp;
DEFINE INDEX idx_message_sequence ON message FIELDS session_id, sequence_num;

-- Relationship traversal
DEFINE INDEX idx_knows_user ON knows FIELDS in;
DEFINE INDEX idx_knows_concept ON knows FIELDS out;
DEFINE INDEX idx_knows_strength ON knows FIELDS strength;
DEFINE INDEX idx_contains_session ON contains FIELDS in;
DEFINE INDEX idx_mentions_message ON mentions FIELDS in;

-- Full-text search
DEFINE ANALYZER simple TOKENIZERS blank FILTERS lowercase;
DEFINE INDEX idx_message_content ON message FIELDS content SEARCH ANALYZER simple;

-- Thoughts
DEFINE INDEX idx_thought_session ON thought FIELDS session_id;
DEFINE INDEX idx_thought_agent ON thought FIELDS agent_id;
DEFINE INDEX idx_thought_type ON thought FIELDS thought_type;
```

#### Events and Triggers
```sql
-- Auto-update session stats when messages added
DEFINE EVENT update_session_stats ON message WHEN $after != NONE THEN (
    UPDATE session SET 
        turn_count = (
            SELECT count() FROM message WHERE session_id = $after.session_id
        )[0],
        ended_at = time::now()
    WHERE id = $after.session_id
);

-- Auto-update user interaction counts
DEFINE EVENT update_user_stats ON message WHEN $after != NONE THEN (
    UPDATE user SET 
        total_interactions = total_interactions + 1,
        last_seen = time::now()
    WHERE id = (
        SELECT user_id FROM session WHERE id = $after.session_id
    )[0]
);

-- Auto-update concept mention counts
DEFINE EVENT update_concept_stats ON mentions WHEN $after != NONE THEN (
    UPDATE concept SET 
        mentioned_count = mentioned_count + 1,
        last_mentioned = time::now()
    WHERE id = $after.out;
);
```

## Phase 2: Data Migration Script

Create `scripts/migrate_to_graph.py`:

```python
import asyncio
import json
from surrealdb import AsyncSurreal
from loguru import logger

async def migrate_backup_to_graph():
    """Migrate backed up data to new graph schema"""
    
    db = AsyncSurreal("ws://127.0.0.1:8000/rpc")
    await db.connect()
    await db.signin({"username": "root", "password": "your_password"})
    await db.use("slowcat", "memory")
    
    # Load backup data (adjust paths as needed)
    with open('backup_facts.json') as f:
        old_facts = json.load(f)
    
    # Create users from speaker_ids
    speakers = set()
    for fact in old_facts:
        speakers.add(fact.get('speaker_id', 'unknown'))
    
    for speaker in speakers:
        if speaker != 'unknown':
            await db.query("""
                CREATE user SET
                    name = $speaker,
                    first_seen = time::now()
            """, {"speaker": speaker})
    
    # Convert facts to user->concept relationships
    for fact in old_facts:
        # Create concept if needed
        concept_id = await db.query("""
            CREATE concept SET
                name = $value,
                kind = $species
            ON DUPLICATE KEY UPDATE mentioned_count = mentioned_count + 1
            RETURN id
        """, {
            "value": fact.get('value', ''),
            "species": fact.get('species', 'unknown')
        })
        
        # Create knowledge relationship
        if concept_id:
            await db.query("""
                RELATE user:$user->knows->concept:$concept SET
                    relationship = $predicate,
                    fidelity = $fidelity,
                    strength = $strength,
                    learned_at = time::from::secs($created),
                    reinforced_at = time::from::secs($last_seen)
            """, {
                "user": fact.get('subject', 'unknown'),
                "concept": concept_id[0]['result'][0]['id'],
                "predicate": fact.get('predicate', 'related_to'),
                "fidelity": fact.get('fidelity', 3),
                "strength": fact.get('strength', 0.6),
                "created": fact.get('created', time.time()),
                "last_seen": fact.get('last_seen', time.time())
            })
    
    logger.info("Migration complete!")

if __name__ == "__main__":
    asyncio.run(migrate_backup_to_graph())
```

## Benefits of Graph Schema

### 1. Natural Relationships
```sql
-- Find what user knows about dogs
SELECT ->knows->concept WHERE concept.kind = 'pet' AND concept.name CONTAINS 'dog'
FROM user:alice;

-- Get conversation where user mentioned jazz
SELECT <-contains<-session<-mentions<-concept 
WHERE concept.name = 'jazz'
FROM message;

-- Find sessions where specific facts were learned
SELECT source_message->session 
FROM knows 
WHERE relationship = 'has_pet';
```

### 2. Graph Traversals
```sql
-- Multi-hop: users who like similar music
SELECT ->knows->concept<-knows<-user 
WHERE concept.kind = 'music_genre'
FROM user:alice;

-- Session context with related facts
SELECT 
    session.*,
    ->contains->message.*,
    ->contains->message->mentions->concept.*
FROM session:specific_session;
```

### 3. Time-Travel & Analytics
```sql
-- How knowledge evolved over time
SELECT relationship, fidelity, strength, reinforced_at
FROM knows 
WHERE in = user:alice AND out.name = 'Potola'
ORDER BY reinforced_at;

-- Session patterns
SELECT count(), date(started_at)
FROM session
WHERE user_id = user:alice
GROUP BY date(started_at);
```

## Migration Steps

1. **Run Phase 1 DDL** in Surrealist (table drops + schema creation)
2. **Export current data** if not already backed up
3. **Run migration script** to populate graph with relationships
4. **Update Python code** to use graph queries
5. **Test & validate** all functionality works

## Risk Mitigation

- ✅ **Backup completed** - can restore if needed
- ✅ **Incremental approach** - migrate table by table
- ✅ **Validation queries** - verify data integrity
- ✅ **Rollback plan** - restore from backup if issues

This gives us **true graph relationships** instead of isolated flat tables!