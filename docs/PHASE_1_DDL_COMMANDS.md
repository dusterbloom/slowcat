# Phase 1 DDL Commands for Surrealist

**Run these commands in Surrealist Desktop App connected to your SurrealDB instance.**

## 1. Remove Duplicate Session Table

```sql
-- Remove the duplicate 'session' table (keep 'sessions')
REMOVE TABLE session;
```

## 2. Add Missing session_summary Table

```sql
-- Create session_summary table for storing conversation summaries
DEFINE TABLE session_summary SCHEMAFULL;
DEFINE FIELD session_id ON session_summary TYPE string;
DEFINE FIELD summary ON session_summary TYPE string;
DEFINE FIELD keywords ON session_summary TYPE array<string> DEFAULT [];
DEFINE FIELD turns ON session_summary TYPE number DEFAULT 0;
DEFINE FIELD duration_s ON session_summary TYPE number DEFAULT 0;
DEFINE FIELD ts ON session_summary TYPE datetime VALUE time::now();
```

## 3. Create Essential Indexes (Idempotent)

```sql
-- Prevent duplicate facts (enforces deduplication at DB level)
DEFINE INDEX idx_fact_unique ON fact FIELDS subject, predicate, value, species, agent_id UNIQUE;

-- Hot tape queries for performance
DEFINE INDEX idx_tape_ts ON tape FIELDS ts;
DEFINE INDEX idx_tape_speaker_ts ON tape FIELDS speaker_id, ts;
DEFINE INDEX idx_tape_agent_ts ON tape FIELDS agent_id, ts;

-- Session lookups
DEFINE INDEX idx_sessions_speaker ON sessions FIELDS speaker_id UNIQUE;

-- Session summary lookups
DEFINE INDEX idx_session_summary_id ON session_summary FIELDS session_id;
```

## 4. Add Full-Text Search

```sql
-- Simple analyzer for English-like text
DEFINE ANALYZER simple TOKENIZERS blank FILTERS lowercase;

-- Full-text index on message content for fast search
DEFINE INDEX idx_tape_search ON tape FIELDS content SEARCH ANALYZER simple;
```

## 5. Create Auto-Increment Event

```sql
-- Automatically increment session counters when new tape entries are added
DEFINE EVENT inc_turn ON tape WHEN $after != NONE THEN (
  UPDATE sessions SET 
    total_turns = (total_turns ?? 0) + 1,
    last_interaction = time::now()
  WHERE speaker_id = $after.speaker_id
);
```

## Verification Queries

After running the above commands, verify with these queries:

```sql
-- Check that session table is removed
INFO FOR TABLE session;

-- Check session_summary table exists
INFO FOR TABLE session_summary;

-- Check indexes are created
INFO FOR TABLE fact;
INFO FOR TABLE tape;
INFO FOR TABLE sessions;

-- Check analyzer exists
INFO FOR ANALYZER simple;

-- Check event exists
INFO FOR TABLE tape;
```

## Expected Output

- `session` table should return "Table not found"
- `session_summary` should show the 6 defined fields
- Tables should show their respective indexes
- `simple` analyzer should be listed
- `tape` table should show the `inc_turn` event

**Note**: All DDL commands are idempotent and safe to run multiple times.