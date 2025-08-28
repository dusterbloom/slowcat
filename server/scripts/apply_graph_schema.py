#!/usr/bin/env python3
"""
Apply graph-native SurrealDB schema (idempotent) to the target database.

Defaults (override via env or CLI):
- URL: ws://127.0.0.1:8000/rpc
- USER/PASS: root / slowcat_secure_2024
- NAMESPACE: slowcat
- DATABASE: memory_graph

Usage examples:
  python server/scripts/apply_graph_schema.py \
    --url ws://127.0.0.1:8000/rpc --ns slowcat --db memory_graph \
    --user root --pass slowcat_secure_2024

Notes:
- Safe to run multiple times; uses DEFINE/REMOVE which are idempotent in effect.
"""

import argparse
import os
from textwrap import dedent
from loguru import logger

try:
    from surrealdb import AsyncSurreal
except ImportError:
    raise SystemExit("SurrealDB client not available. Install with: pip install surrealdb")


DDL = dedent(
    """
    -- Core entities
    DEFINE TABLE user SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD name ON user TYPE string;
    DEFINE FIELD first_seen ON user TYPE datetime VALUE time::now();
    DEFINE FIELD last_seen ON user TYPE datetime VALUE time::now();
    DEFINE FIELD total_interactions ON user TYPE number DEFAULT 0;
    DEFINE FIELD metadata ON user TYPE object DEFAULT {};

    DEFINE TABLE session SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD user_id ON session TYPE record<user>;
    DEFINE FIELD agent_id ON session TYPE string DEFAULT 'slowcat';
    DEFINE FIELD started_at ON session TYPE datetime VALUE time::now();
    DEFINE FIELD ended_at ON session TYPE datetime;
    DEFINE FIELD turn_count ON session TYPE number DEFAULT 0;
    DEFINE FIELD duration_secs ON session TYPE number DEFAULT 0;
    DEFINE FIELD summary ON session TYPE string DEFAULT '';
    DEFINE FIELD keywords ON session TYPE array<string> DEFAULT [];
    DEFINE FIELD status ON session TYPE string DEFAULT 'active';

    DEFINE TABLE concept SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD name ON concept TYPE string;
    DEFINE FIELD kind ON concept TYPE string;
    DEFINE FIELD properties ON concept TYPE object DEFAULT {};
    DEFINE FIELD mentioned_count ON concept TYPE number DEFAULT 0;
    DEFINE FIELD first_mentioned ON concept TYPE datetime VALUE time::now();
    DEFINE FIELD last_mentioned ON concept TYPE datetime VALUE time::now();

    DEFINE TABLE message SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD session_id ON message TYPE record<session>;
    DEFINE FIELD speaker_type ON message TYPE string; -- 'user' | 'assistant'
    DEFINE FIELD content ON message TYPE string;
    DEFINE FIELD raw_content ON message TYPE option<string>;
    DEFINE FIELD clean_content ON message TYPE option<string>;
    DEFINE FIELD timestamp ON message TYPE datetime VALUE time::now();
    DEFINE FIELD sequence_num ON message TYPE number;
    DEFINE FIELD embedding ON message TYPE array<number>;
    DEFINE FIELD metadata ON message TYPE object DEFAULT {};

    DEFINE TABLE thought SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD agent_id ON thought TYPE string;
    DEFINE FIELD session_id ON thought TYPE record<session>;
    DEFINE FIELD thought_type ON thought TYPE string;
    DEFINE FIELD content ON thought TYPE string;
    DEFINE FIELD timestamp ON thought TYPE datetime VALUE time::now();
    DEFINE FIELD visibility ON thought TYPE string DEFAULT 'private';
    DEFINE FIELD confidence ON thought TYPE number DEFAULT 0.5;

    -- Relations
    DEFINE TABLE knows TYPE RELATION IN user OUT concept SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD relationship ON knows TYPE string;
    DEFINE FIELD fidelity ON knows TYPE number DEFAULT 3;
    DEFINE FIELD strength ON knows TYPE number DEFAULT 0.6;
    DEFINE FIELD learned_at ON knows TYPE datetime VALUE time::now();
    DEFINE FIELD reinforced_at ON knows TYPE datetime VALUE time::now();
    DEFINE FIELD source_message ON knows TYPE record<message>;
    DEFINE FIELD access_count ON knows TYPE number DEFAULT 0;
    DEFINE FIELD decay_rate ON knows TYPE number DEFAULT 1.0;

    DEFINE TABLE contains TYPE RELATION IN session OUT message SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD sequence_num ON contains TYPE number;
    DEFINE FIELD created_at ON contains TYPE datetime VALUE time::now();

    DEFINE TABLE mentions TYPE RELATION IN message OUT concept SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD relevance ON mentions TYPE number DEFAULT 0.5;
    DEFINE FIELD context ON mentions TYPE string DEFAULT '';
    DEFINE FIELD extracted_at ON mentions TYPE datetime VALUE time::now();

    DEFINE TABLE reflects TYPE RELATION IN session OUT thought SCHEMAFULL PERMISSIONS FOR select, create, update, delete WHERE true;
    DEFINE FIELD generated_at ON reflects TYPE datetime VALUE time::now();
    DEFINE FIELD trigger_event ON reflects TYPE string DEFAULT '';

    -- Indexes
    DEFINE INDEX idx_user_name ON user FIELDS name UNIQUE;
    DEFINE INDEX idx_session_user ON session FIELDS user_id;
    DEFINE INDEX idx_session_agent ON session FIELDS agent_id;
    DEFINE INDEX idx_concept_name ON concept FIELDS name;
    DEFINE INDEX idx_concept_kind ON concept FIELDS kind;
    DEFINE INDEX idx_message_session ON message FIELDS session_id;
    DEFINE INDEX idx_message_timestamp ON message FIELDS timestamp;
    DEFINE INDEX idx_message_sequence ON message FIELDS session_id, sequence_num;
    DEFINE INDEX uniq_message_session_seq ON message FIELDS session_id, sequence_num UNIQUE;
    DEFINE INDEX idx_knows_user ON knows FIELDS in;
    DEFINE INDEX idx_knows_concept ON knows FIELDS out;
    DEFINE INDEX idx_knows_strength ON knows FIELDS strength;
    DEFINE INDEX idx_contains_session ON contains FIELDS in;
    DEFINE INDEX idx_mentions_message ON mentions FIELDS in;

    -- FTS analyzer and index
    DEFINE ANALYZER simple TOKENIZERS blank FILTERS lowercase;
    DEFINE INDEX idx_message_content ON message FIELDS content SEARCH ANALYZER simple;

    -- Events
    REMOVE EVENT update_session_stats ON message;
    DEFINE EVENT update_session_stats ON message WHEN $after != NONE THEN (
        UPDATE session SET 
            turn_count = (SELECT count() FROM message WHERE session_id = $after.session_id)[0].count,
            ended_at = time::now()
        WHERE id = $after.session_id
    );

    REMOVE EVENT update_user_stats ON message;
    DEFINE EVENT update_user_stats ON message WHEN $after != NONE THEN (
        UPDATE user SET 
            total_interactions = (total_interactions ?? 0) + 1,
            last_seen = time::now()
        WHERE id = (SELECT user_id FROM session WHERE id = $after.session_id)[0]
    );

    DEFINE EVENT update_concept_stats ON mentions WHEN $after != NONE THEN (
        UPDATE concept SET 
            mentioned_count = (mentioned_count ?? 0) + 1,
            last_mentioned = time::now()
        WHERE id = $after.out
    );
    """
)


async def apply_schema(url: str, ns: str, db: str, user: str, password: str):
    client = AsyncSurreal(url)
    await client.connect()
    # The Python client accepts either username/password or user/pass depending on version
    try:
        await client.signin({"username": user, "password": password})
    except Exception:
        await client.signin({"user": user, "pass": password})
    await client.use(ns, db)
    logger.info(f"Applying graph schema to {ns}.{db}...")
    res = await client.query(DDL)
    await client.close()
    logger.info("Schema application complete.")
    return res


def parse_args():
    p = argparse.ArgumentParser(description="Apply graph schema to SurrealDB")
    p.add_argument('--url', default=os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc'))
    p.add_argument('--ns', default=os.getenv('SURREALDB_NAMESPACE', 'slowcat'))
    p.add_argument('--db', default=os.getenv('SURREALDB_DATABASE', 'memory_graph'))
    p.add_argument('--user', default=os.getenv('SURREALDB_USER', 'root'))
    p.add_argument('--pass', dest='password', default=os.getenv('SURREALDB_PASS', 'slowcat_secure_2024'))
    return p.parse_args()


if __name__ == '__main__':
    import asyncio
    args = parse_args()
    asyncio.run(apply_schema(args.url, args.ns, args.db, args.user, args.password))
