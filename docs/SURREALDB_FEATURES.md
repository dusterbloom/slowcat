# SurrealDB Features To Leverage (Slowcat Memory)

This document summarizes stable SurrealDB capabilities we can (and should) use for the voice‑agent memory system. The focus is on features that are available in current 1.x builds and that map cleanly to our existing schema and code paths.

Status legend: In use = already in `server/memory/surreal_memory.py`; Planned = safe next step.

## Core Capabilities

- Schema and Types: Use `DEFINE TABLE ... SCHEMAFULL` and `DEFINE FIELD ... TYPE ... DEFAULT ...` for strict typing. In use.
- Namespaces/Databases: Isolate environments via `NAMESPACE` and `DATABASE`. In use.
- SurrealQL: Parameterized queries with `SELECT/CREATE/UPDATE/DELETE`, `WHERE`, `ORDER BY`, `LIMIT`. In use.
- Time Functions: `time::now()`, `time::from::secs()` and related helpers for timestamps. In use.
- Graph Relations: Native graph with `RELATE`, traversal via `->` and `<-`. Planned.
- Live Queries: Server‑pushed change streams with `LIVE SELECT ...`. Planned.
- Time‑Travel Reads: Point‑in‑time reads with `AT time::...` for record versions. Planned.
- Indexes: `DEFINE INDEX` for uniqueness and performance on frequent predicates. Planned (partial).
- Events/Triggers: `DEFINE EVENT` to react to row changes atomically. Planned.
- User Functions: `DEFINE FUNCTION` for reusable logic in SurrealQL. Planned.
- Row/Field Permissions: Table‑scoped `PERMISSIONS` with row‑level predicates. Planned.

## Our Current Logical Model (recap)

Tables (see `server/memory/surreal_memory.py`):
- `fact` (structured knowledge): `subject`, `predicate`, `value`, `fidelity`, `strength`, `last_seen`, `created`, `access_count`, `source_text`, `agent_id`.
- `tape` (time‑series dialogue): `ts`, `speaker_id`, `role`, `content`, `session_id`, `agent_id`, `embedding?`, `metadata{}`.
- `thought` (private agent notes): `ts`, `agent_id`, `thought_type`, `content`, `links[]`, `visibility`.
- `emergent_event` (observability only): `ts`, `agent_id`, `kind`, `content_snippet`, `meta_json{}`, `session_id?`, `user_id?`, `confidence`.
- `sessions` (rolled‑up stats): `speaker_id`, `session_count`, `last_interaction`, `first_seen`, `total_turns`.

## High‑Value Additions (safe and incremental)

### 1) Composite and Unique Indexes

Purpose: Enforce deduplication, speed up lookups used in hot paths.

SurrealQL (create once, idempotent):
```sql
-- Prevent duplicate facts for same (subject, predicate, value, species, agent)
DEFINE INDEX idx_fact_unique ON fact FIELDS subject, predicate, value, species, agent_id UNIQUE;

-- Fast session lookups and sorting
DEFINE INDEX idx_sessions_speaker ON sessions FIELDS speaker_id UNIQUE;

-- Hot tape queries
DEFINE INDEX idx_tape_ts ON tape FIELDS ts;
DEFINE INDEX idx_tape_speaker_ts ON tape FIELDS speaker_id, ts;
DEFINE INDEX idx_tape_agent_ts ON tape FIELDS agent_id, ts;
```

Impact:
- Removes need for multi‑step “select then insert” in reinforce paths (enforced by constraint).
- Reduces latency for recent/knn pre‑scan and session counters.

### 2) Full‑Text Search Index on `tape.content`

Purpose: Efficient keyword search and prefix/contains patterns for recall.

SurrealQL (define analyzer + index):
```sql
-- Minimal analyzer suitable for English‑like text
DEFINE ANALYZER simple TOKENIZERS blank FILTERS lowercase;

-- Full‑text index on message content
DEFINE INDEX idx_tape_search ON tape FIELDS content SEARCH ANALYZER simple;
```

Usage:
- Keep current fallback (`string::contains(...)`) for compatibility.
- Prefer the built‑in FTS when present for faster ranked results (consult your server version for the exact search predicate supported by the index).

### 3) Graph Relations for Facts (optional migration)

Purpose: Model facts as edges so we can traverse semantically (who knows what, when).

Pattern:
- Store entities as records: `entity:<type>:<id>` (e.g., `entity:user:peppi`, `entity:pet:potola`).
- Store relations with `RELATE` (e.g., user —[has_pet]→ pet) and additional edge fields (fidelity, strength, last_seen).

SurrealQL example:
```sql
-- Entities
CREATE entity:user:peppi SET kind = 'user', display_name = 'peppi';
CREATE entity:pet:potola SET kind = 'pet', species = 'dog', name = 'Potola';

-- Fact as an edge (with attributes)
RELATE entity:user:peppi->has_pet->entity:pet:potola SET fidelity = 3, strength = 0.8, last_seen = time::now();

-- Traverse
SELECT ->has_pet->entity.* FROM entity:user:peppi;
```

Impact:
- Enables first‑class graph traversal for memory queries and explanations.
- Can be phased in while keeping the existing `fact` table (dual‑write during migration, read from both).

### 4) Live Queries for Real‑Time UI/Agents

Purpose: React instantly to new tape/fact/summary entries.

SurrealQL example:
```sql
-- Stream tape events for a speaker
LIVE SELECT * FROM tape WHERE speaker_id = $speaker_id;

-- Kill a live query by its ID when no longer needed
KILL <live-query-id>;
```

Integration:
- Use in the web client or a background task to update overlays, notify processors, or drive lightweight reflection without polling.

### 5) Time‑Travel Reads

Purpose: Reconstruct memory “as of” a given point for audits or regression analysis.

SurrealQL example:
```sql
-- Point‑in‑time view of a table (record versions)
SELECT * FROM fact AT time::from::secs($ts_cutoff) WHERE subject = 'user';
```

Notes:
- Most useful on slowly‑changing structures (`fact`, `sessions`). For `tape`, we already store `ts` per row, so point‑in‑time slices can be expressed with `WHERE ts <= ...`.

### 6) Events/Triggers to Maintain Counters

Purpose: Move simple rollups from application code into the database atomically.

SurrealQL example:
```sql
-- Increment session counters automatically on new tape rows
DEFINE EVENT inc_turn ON tape WHEN $after != NONE THEN (
  UPDATE sessions SET 
    total_turns = (total_turns ?? 0) + 1,
    last_interaction = time::now()
  WHERE speaker_id = $after.speaker_id;
);
```

Impact:
- Guarantees counters stay in sync even if a writer crashes or backfills.
- Lets the app simplify `update_session(...)` paths.

### 7) Row‑Level Permissions

Purpose: Tenant and agent isolation at the DB layer.

SurrealQL example:
```sql
DEFINE TABLE fact SCHEMAFULL
  PERMISSIONS
    FOR select WHERE agent_id = $auth.agent_id,
    FOR create WHERE agent_id = $auth.agent_id,
    FOR update WHERE agent_id = $auth.agent_id,
    FOR delete WHERE agent_id = $auth.agent_id;
```

Notes:
- Combine with namespace/database separation for stronger isolation.
- Pair with SCOPE‑based auth if you need per‑user JWTs.

### 8) User‑Defined Functions (utility logic)

Purpose: Encapsulate reusable scoring/decay helpers used by queries or events.

SurrealQL example:
```sql
DEFINE FUNCTION fn::decay($strength, $rate) {
  RETURN math::round($strength * $rate, 3);
};
```

Use:
- In events that adjust `strength` or in ad‑hoc analytics queries.

## Suggested Rollout Plan

1) Apply indexes (safe): uniqueness on `fact`, hot indexes on `tape` / `sessions`.
2) Add full‑text analyzer/index on `tape.content`; keep existing code path as fallback.
3) Introduce events to update `sessions` counters on `tape` inserts.
4) Expose a small `LIVE SELECT` subscription in the client for real‑time updates.
5) Pilot graph edges for a subset of facts (dual‑write + read), then migrate.
6) Add table permissions where multi‑agent isolation is required.

## DDL Checklist (can be run in Surrealist)

Run once per database (adjust to your namespace/db):
```sql
-- Indexes
DEFINE INDEX idx_fact_unique ON fact FIELDS subject, predicate, value, species, agent_id UNIQUE;
DEFINE INDEX idx_sessions_speaker ON sessions FIELDS speaker_id UNIQUE;
DEFINE INDEX idx_tape_ts ON tape FIELDS ts;
DEFINE INDEX idx_tape_speaker_ts ON tape FIELDS speaker_id, ts;
DEFINE INDEX idx_tape_agent_ts ON tape FIELDS agent_id, ts;

-- Full‑text (optional but recommended)
DEFINE ANALYZER simple TOKENIZERS blank FILTERS lowercase;
DEFINE INDEX idx_tape_search ON tape FIELDS content SEARCH ANALYZER simple;

-- Event for turn counters
DEFINE EVENT inc_turn ON tape WHEN $after != NONE THEN (
  UPDATE sessions SET 
    total_turns = (total_turns ?? 0) + 1,
    last_interaction = time::now()
  WHERE speaker_id = $after.speaker_id;
);
```

## Notes on Compatibility

- All examples use stable SurrealQL constructs available in current 1.x builds.
- If your server exposes newer search/vector options, you can extend the FTS index to use a richer analyzer and ranking model; the above keeps to conservative, widely‑available features.
- Our Python client paths remain valid; these additions improve performance, correctness, and simplify some app‑side logic.

```
File owner: memory team
Last updated: keep in sync with run‑time schema in `server/memory/surreal_memory.py`.
```

