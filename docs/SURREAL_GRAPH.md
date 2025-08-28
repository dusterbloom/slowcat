# SurrealDB Graph Memory (Slowcat)

This document summarizes how the server uses SurrealDB for sessions, tape, and facts, and how to initialize and troubleshoot locally.

## Schema Overview

- `user` (record)
  - Created as `user:<safe_name>` (e.g., `user:unit_user_smoke`)
  - Fields: `name`, `first_seen`, `last_seen`, `total_interactions`, `metadata`

- `session` (document)
  - ID: `session:<epoch_seconds>`
  - `user_id`: stored as a string record id (e.g., `"user:unit_user_smoke"`) for compatibility
  - Fields: `agent_id`, `started_at`, `turn_count`, `summary`, `keywords`, `status`
  - Indexes: `session_by_user` on `user_id`

- `message` (document)
  - Fields: `session_id` (record<session>), `role`, `speaker_type`, `content`, `raw_content`, `timestamp`, `sequence_num`, `sender_id`
  - Indexes: `message_by_session`, `message_by_time`, `message_unique_seq(session_id,sequence_num)`

- `concept` (document)
  - Fields: `name`, `kind`
  - Indexes: `concept_by_name`

- `knows` (relation)
  - `IN` = `user`, `OUT` = `concept`
  - Fields: `relationship`, `fidelity`, `strength`, `learned_at`, `reinforced_at`, `source_text`, `learned_from_session`, `access_count`, `decay_rate`

- `contains` (relation)
  - `IN` = `session`, `OUT` = `message`
  - Fields: `sequence_num`, `created_at`

- `fact_plain` (document, fallback)
  - SCHEMALESS doc table storing `user_id`, `subject`, `predicate`, `value`, `fidelity`, `strength`, `source_text`, `created`
  - Used as a retrieval fallback when relation edges are sparse or driver traversal differs.

## Initialization

Use the included scripts to initialize and verify:

```bash
cd server
source .venv/bin/activate
export SURREALDB_DATABASE=memory_graph
python scripts/init_surreal_graph_db.py     # applies schema (idempotent)
python scripts/surreal_smoke.py            # end-to-end smoke for sessions/facts/tape
```

If you need raw visibility:

```bash
python scripts/db_probe.py unit_user_smoke
```

This prints the raw `session` rows and `knows` edges so you can confirm shapes.

## Query Idioms (verified)

- Sessions (string id):
  ```sql
  SELECT id, started_at, user_id
  FROM session
  WHERE user_id = 'user:unit_user_smoke'
  ORDER BY started_at DESC
  LIMIT 5;
  ```

- Relation (if edges exist):
  ```sql
  -- traversal
  SELECT relationship, fidelity, strength, out.name AS concept_name
  FROM type::thing('user', 'unit_user_smoke')->knows
  ORDER BY strength DESC
  LIMIT 10;

  -- relation table – record IN
  SELECT relationship, fidelity, strength, out.name AS concept_name
  FROM knows
  WHERE in = type::thing('user', 'unit_user_smoke')
  ORDER BY strength DESC
  LIMIT 10;

  -- relation table – string IN
  SELECT relationship, fidelity, strength, out.name AS concept_name
  FROM knows
  WHERE string::concat('', in) = 'user:unit_user_smoke'
  ORDER BY strength DESC
  LIMIT 10;
  ```

- Fallback doc facts:
  ```sql
  SELECT subject, predicate, value, fidelity, strength
  FROM fact_plain
  WHERE user_id = 'user:unit_user_smoke'
    AND string::contains(string::lowercase(value), 'coffee')
  ORDER BY strength DESC
  LIMIT 10;
  ```

## Behavior in Slowcat

- `create_session` stores `user_id` as a string (e.g., `"user:alice"`) for compatibility.
- `get_user_sessions` pulls and filters sessions in Python (`str(user_id) == 'user:<key>'`) and sorts by `started_at`.
- `store_fact` both RELATEs the edge and writes a `fact_plain` row.
- `search_facts` tries traversal and relation table first; falls back to `fact_plain` if needed; maps `out.name` to the `value` field for consistent results.

This approach ensures correct retrieval even if relation edges are sparse, and avoids driver-specific traversal inconsistencies.

