# **LLM → SurrealDB (SurrealQL) in Python — Zero-Time Onboarding Cheat Sheet**

> Targets **SurrealDB v2.x** and the **official Python SDK (`surrealdb`)**. Copy-paste the snippets; swap values, not structure. This guide bakes in **LLM-safe patterns** (parameterization, strict allowlists) and the 20% of SurrealQL that unlocks 80% of use-cases. ([SurrealDB][1])

---

## 0) TL;DR mental model

* SurrealDB is **multi-model** (document + graph + vector + full-text) with a **SQL-like** language (**SurrealQL**). Think “Postgres + Neo4j + a vector DB + FTS” under one roof. ([SurrealDB][2])
* Talk to it via **HTTP** (stateless) or **WebSocket RPC** (stateful; enables live queries & connection-scoped variables). Use the **Python SDK** for both. ([SurrealDB][3])

---

## 1) Install & run

```bash
# Python SDK
pip install surrealdb
```

**Why 8000?** That’s the default HTTP/RPC port. You can talk **HTTP** at `http://localhost:8000` or **WS RPC** at `ws://localhost:8000/rpc`. ([SurrealDB][3])

---

## 2) Minimal Python client (LLM-safe)

```python
# surreal_client.py
import asyncio
from surrealdb import Surreal
from surrealdb.common import RecordID

DB_URL = "ws://localhost:8000/rpc"     # or "http://localhost:8000"
NS = "app"
DB = "demo"

ALLOWED = {"SELECT", "INSERT", "CREATE", "UPDATE", "UPSERT", "DELETE", "RELATE", "LET", "BEGIN", "COMMIT", "CANCEL", "LIVE", "KILL"}

def _allowlisted(sql: str) -> bool:
    head = sql.strip().split(None, 1)[0].upper()
    return head in ALLOWED

class SurrealClient:
    def __init__(self, url=DB_URL):
        self.url = url
        self.db = Surreal(url)

    async def connect(self, user="root", password="root"):
        await self.db.connect()
        await self.db.signin({"username": user, "password": password})
        await self.db.use(namespace=NS, database=DB)

    async def query(self, sql: str, vars: dict | None = None):
        if not _allowlisted(sql):
            raise ValueError("Statement not allowed")
        return await self.db.query(sql, vars or {})

    async def insert(self, table: str, data):
        return await self.db.insert(table, data)

    async def upsert(self, table_or_id, data=None):
        return await self.db.upsert(table_or_id, data or {})

    async def relate(self, edge_table: str, in_id: RecordID, out_id: RecordID, data: dict | None = None):
        return await self.db.insert_relation(edge_table, {"in": in_id, "out": out_id, **(data or {})})

    async def live(self, table: str):
        # returns a UUID; use subscribe_live to consume
        return await self.db.live(table)

    async def subscribe(self, uuid: str):
        async for event in self.db.subscribe_live(uuid):
            yield event

    async def kill(self, uuid: str):
        return await self.db.kill(uuid)

# Quick smoke test
if __name__ == "__main__":
    async def main():
        c = SurrealClient()
        await c.connect()
        await c.query("DEFINE TABLE IF NOT EXISTS person SCHEMALESS;")
        await c.insert("person", [{"name":"Tobie"},{"name":"Jaime"}])
        res = await c.query("SELECT * FROM person WHERE name = $name", {"name": "Tobie"})
        print(res)
    asyncio.run(main())
```

* `signin()` + `use()` are the canonical flow; **WS RPC** supports connection-scoped state.
* `query(sql, vars)` uses **server-side variable binding**—**never** string-concatenate LLM text.
* `insert_relation()` is the cleanest way to create edges from Python.
* `live()` + `subscribe_live()` implement **realtime change streams**. ([SurrealDB][4])

---

## 3) Auth & headers quick-ref (for HTTP calls)

* **HTTP** is stateless. Send `Surreal-NS` and `Surreal-DB` headers (v2) and an `Authorization` token (or `-u user:pass`).
* Endpoints include `/sql`, `/key/:table`, `/signin`, `/signup`, `/status`, etc.
* `POST /signin` yields a token; reuse it as `Authorization: Bearer <token>`. ([SurrealDB][3])

---

## 4) LLM-safe SurrealQL patterns (copy-ready)

**Always bind variables**:

```sql
-- Good: bind $vars (no injection)
SELECT * FROM person WHERE name = $name AND age > $min_age;

-- Python call
await c.query("SELECT * FROM person WHERE name = $name AND age > $min_age",
              {"name": llm_name, "min_age": 21})
```

**Safe dynamic identifiers**:

```sql
-- Dynamic table
SELECT * FROM type::table($tb) WHERE created_at > $since;

-- Dynamic record id (table + id)
LET $rid = type::thing($tb, $id);
SELECT * FROM $rid;
```

> Use `type::table` and `type::thing` whenever the LLM must choose a table or record id dynamically. ([SurrealDB][5])

**Connection-scoped LET** (under WS RPC), or pass a `vars` map per `query`:

```sql
LET $today = time::now();
SELECT * FROM orders WHERE created_at >= $today - 7d;
```

> Prefer `LET` / `vars` over inline literals; much safer for LLMs and enables caching. ([SurrealDB][6])

---

## 5) Core SurrealQL you’ll actually use

### 5.1 Create / Insert / Upsert

```sql
-- Create one record (optional explicit id)
CREATE person CONTENT { name: "Agatha", tags: ["writer"] };

-- Insert many (best for bulk)
INSERT INTO person [
  { name: "Tobie" },
  { name: "Jaime" }
];

-- Upsert (replace or create)
UPSERT person:jaime CONTENT { name: "Jaime", settings: { active: true } };
```

> Prefer `INSERT` for bulk arrays, `CREATE` for single items (explicit id optional), `UPSERT` to overwrite. ([SurrealDB][7])

### 5.2 Update / Merge / Patch

```sql
-- Replace
UPDATE person:jaime CONTENT { name: "Jaime", active: true };

-- Merge (partial update)
UPDATE person:jaime MERGE { settings: { marketing: false } };

-- JSON Patch (RFC 6902)
UPDATE person:jaime PATCH [
  { op: "replace", path: "/settings/active", value: false }
];
```

> `UPDATE CONTENT` replaces; `MERGE` deep-merges; `PATCH` applies JSON Patches. ([SurrealDB][8])

### 5.3 Select + Fetch (de-reference links/edges)

```sql
-- Basic filter + paging
SELECT id, name FROM person WHERE name ~ $prefix LIMIT 25 START 0;

-- Bring linked records inline (record links or edges)
SELECT *, author.email FROM article FETCH author;
```

> `FETCH` replaces record ids with the actual records; great for one-shot queries that cross relations. ([SurrealDB][9])

### 5.4 Graph edges and traversals

```sql
-- Define an *edge* table (optional; or RELATE ad-hoc)
DEFINE TABLE follows TYPE RELATION FROM person TO person;

-- Create an edge
RELATE person:alice -> follows -> person:bob CONTENT { since: time::now() };

-- Traverse: who Alice follows
SELECT ->follows->person.name FROM person:alice;

-- Traverse reverse: who follows Bob
SELECT <-follows<-person.name FROM person:bob;
```

> SurrealDB supports **record links** (store `person:bob` inside a field) *and* **graph edges** (first-class relations with properties). Use arrow `->` / `<-` in queries; prefer edge tables for type-safe schemas. ([SurrealDB][10])

### 5.5 Full-Text Search (FTS)

```sql
-- 1) Define an analyzer (pick tokenizers/filters for your language)
DEFINE ANALYZER en TOKENIZERS class FILTERS lowercase, ascii;

-- 2) Define a FTS index
DEFINE INDEX idx_title ON TABLE article FIELDS title
  SEARCH ANALYZER en HIGHLIGHTS BM25;

-- 3) Query with matches predicate and optional scoring/highlights
SELECT id, title, search::score(1) AS score
FROM article
WHERE title @1@ $q
ORDER BY score DESC;
```

> `DEFINE ANALYZER` + `DEFINE INDEX ... SEARCH ANALYZER ...` unlock FTS. Use `search::score`, `search::highlight`. ([SurrealDB][11])

### 5.6 Vector search (ANN or exact)

```sql
-- HNSW vector index for 384-dim embeddings (cosine distance)
DEFINE INDEX h_vec ON pts FIELDS embedding HNSW DIMENSION 384 DIST COSINE;

-- K-NN query: top 5 nearest to $v (uses index distance)
LET $v = $embedding;  -- pass from Python
SELECT id, vector::distance::knn() AS dist
FROM pts
WHERE embedding <|5|> $v;
```

> Use **HNSW** (approx) or **M-Tree** (exact). Operators like `<|k|>` plus `vector::distance::*` / `vector::similarity::*` make queries terse. ([SurrealDB][12])

### 5.7 Transactions

```sql
BEGIN TRANSACTION;
  CREATE account:one SET balance = 100;
  CREATE account:two SET balance = 100;
  UPDATE account:one SET balance -= 30;
  UPDATE account:two SET balance += 30;
COMMIT TRANSACTION;
```

> Use `BEGIN` / `COMMIT` / `CANCEL`; you can `THROW` to abort conditionally. Just send the multi-statement batch via `db.query(...)`. ([SurrealDB][13])

### 5.8 Introspection

```sql
INFO FOR DB;                 -- overview of DB
SHOW TABLES;                 -- list tables
INFO FOR TABLE person;       -- fields, indexes, permissions
```

> Handy during LLM tool planning & schema discovery. ([SurrealDB][13])

---

## 6) Real-time (LIVE queries) in Python

```python
# subscribe_people.py
import asyncio
from surreal_client import SurrealClient

async def main():
    c = SurrealClient()
    await c.connect()
    uuid = await c.live("person")  # start live feed on table
    async for evt in c.subscribe(uuid):
        print("Live event:", evt)   # {"action":"CREATE","result":{...}} etc.

asyncio.run(main())
```

* Kill a subscription with `await c.kill(uuid)`.
* Set `diff=True` in `.live(table, diff=True)` to receive JSON Patches instead of full records. ([SurrealDB][14])

---

## 7) Permissions (ship secure defaults)

```sql
-- Tables are permissions-aware; default is NONE (good!)
DEFINE TABLE person
  SCHEMALESS
  PERMISSIONS
    FOR select, create, update, delete WHERE $auth.role = "admin";
```

> Start with **`PERMISSIONS NONE`** or tight predicates; avoid “dev-open” tables in production. ([SurrealDB][15])

---

## 8) LLM prompt scaffolds (drop-in)

**System (tool) prompt for SurrealQL generation**

```
You write ONLY SurrealQL covered by the allowlist:
SELECT, INSERT, CREATE, UPDATE, UPSERT, DELETE, RELATE, LET, BEGIN, COMMIT, CANCEL.
Never concatenate user input into SQL. Always use $variables.
For dynamic tables or record ids, use type::table($tb) and type::thing($tb, $id).
Prefer FETCH to expand linked records in one round trip.
For FTS, use @1@ and search::score(1). For vectors, use <|k|>.
Return JSON: {"sql": "...", "vars": {...}, "notes": "..."} with no extra text.
```

**Example user message → model output**

Input:

```
Get top 5 articles with “rust web” in the title, include author emails.
```

Model output:

```json
{
  "sql": "SELECT id, title, author.email, search::score(1) AS score FROM article FETCH author WHERE title @1@ $q ORDER BY score DESC LIMIT 5;",
  "vars": { "q": "rust web" },
  "notes": "Uses FTS with analyzer-backed index, FETCH to inline author."
}
```

Then call:

```python
res = await c.query(payload["sql"], payload["vars"])
```

---

## 9) Common recipes

**Create table & fields (schemafull) with indexes**

```sql
DEFINE TABLE person SCHEMAFULL;
DEFINE FIELD name ON person TYPE string ASSERT $value != NONE;
DEFINE INDEX idx_person_name ON person FIELDS name UNIQUE;
```

**Record links vs edges**

```sql
-- Record link (pointer)
CREATE post CONTENT { author: person:alice };

-- Edge (first-class relation with properties)
RELATE person:alice -> likes -> post:42 CONTENT { at: time::now() };
```

**Bulk insert & then query page 2**

```sql
INSERT INTO product [
  { sku:"A1", price: 10.0 }, { sku:"A2", price: 12.5 }, { sku:"A3", price: 8.0 }
];
SELECT * FROM product ORDER BY price LIMIT 10 START 10;
```

**Upsert many via Python**

```python
await c.upsert("person", {"marketing": False})   # all rows
await c.upsert(RecordID("person","alice"), {"name":"Alice"})
```

**Graph traversal with filter**

```sql
SELECT ->follows->person[name, created_at]
FROM person:alice
WHERE created_at > $since;
```

**Vector hybrid filter**

```sql
LET $v = $embedding;
SELECT id, title
FROM article
WHERE embedding <|10|> $v
  AND published_at >= $since;
```

---

## 10) Troubleshooting fast

* **401s over HTTP** → ensure `Surreal-NS` / `Surreal-DB` headers (v2) + token or basic auth. ([SurrealDB][3])
* **“Nothing happens” on LIVE** → you must **mutate** the table (INSERT/UPDATE/DELETE) to see events; confirm you subscribed to the right table and keep the WS open. ([SurrealDB][14])
* **LLM injections / malformed SQL** → enforce the allowlist + variables; reject unknown statements; require `type::table`/`type::thing` for dynamic identifiers. ([SurrealDB][5])

---

## 11) Bonus: one-file schema scaffold (FTS + vector + graph)

```sql
USE NS app DB demo;

DEFINE ANALYZER en TOKENIZERS class FILTERS lowercase, ascii;

DEFINE TABLE person SCHEMAFULL;
DEFINE FIELD name      ON person TYPE string ASSERT $value != NONE;
DEFINE FIELD embedding ON person TYPE array;  -- e.g., 384 floats
DEFINE INDEX person_name_fts ON person FIELDS name SEARCH ANALYZER en HIGHLIGHTS BM25;
DEFINE INDEX person_vec      ON person FIELDS embedding HNSW DIMENSION 384 DIST COSINE;

DEFINE TABLE follows TYPE RELATION FROM person TO person;

INSERT INTO person [
  { id: person:alice, name: "Alice", embedding: [/* ... */] },
  { id: person:bob,   name: "Bob",   embedding: [/* ... */] }
];

RELATE person:alice -> follows -> person:bob CONTENT { since: time::now() };
```

> Shows analyzers, FTS, vector HNSW, and graph relations in one go. ([SurrealDB][11])

---

## Appendix A — Reference links used

* Python SDK (overview, methods, quick start, queries, live/subscribe) ([SurrealDB][1])
* Connection & authentication (HTTP vs WS; headers; `/signin`) ([SurrealDB][3])
* Core statements: `INSERT`, `UPSERT`, `UPDATE`, `MERGE`, `PATCH`, `RELATE`, `SELECT`, `FETCH` ([SurrealDB][7])
* Graph & links (when/why; patterns) ([SurrealDB][16])
* Vector search (HNSW/M-Tree, syntax `<|k|>`, distance funcs) ([SurrealDB][12])
* Full-Text Search (analyzers, indexes, functions) ([SurrealDB][11])
* Transactions (`BEGIN`/`COMMIT`/`CANCEL`/`THROW`) ([SurrealDB][13])
* Permissions defaults & table definition ([SurrealDB][15])

---

## Appendix B — Minimal LLM tool contract (JSON)

```jsonc
{
  "name": "run_surrealql",
  "description": "Executes SurrealQL against SurrealDB with variable binding.",
  "parameters": {
    "type": "object",
    "properties": {
      "sql": { "type": "string", "description": "SurrealQL; first word must be allowlisted" },
      "vars": { "type": "object", "additionalProperties": true }
    },
    "required": ["sql"]
  }
}
```

Server side:

```python
def run_surrealql(sql: str, vars: dict | None = None):
    if not _allowlisted(sql):
        raise ValueError("Disallowed statement")
    return asyncio.run(client.query(sql, vars or {}))
```

---

**You’re ready.** Point any LLM at this scaffold, enforce the allowlist + `$vars`, and you’ve got safe, production-grade SurrealQL generation with Python—**documents, graphs, vectors, FTS, and realtime**—all in one.

[1]: https://surrealdb.com/docs/sdk/python "Python SDKs | Integration"
[2]: https://surrealdb.com/docs/surrealdb "SurrealDB"
[3]: https://surrealdb.com/docs/surrealdb/integration/http "HTTP Protocol | Integration"
[4]: https://surrealdb.com/docs/sdk/python/methods/subscribelive "Python | SDK | Methods | subscribe_live"
[5]: https://surrealdb.com/docs/surrealql/functions/database/type?utm_source=chatgpt.com "Type functions | SurrealQL"
[6]: https://surrealdb.com/docs/sdk/python/concepts/handling-authentication "Python | SDK | Handle authentication"
[7]: https://surrealdb.com/docs/sdk/python/methods/insert "Python | SDK | Methods | insert"
[8]: https://surrealdb.com/docs/sdk/python/methods/update "Python | SDK | Methods | update"
[9]: https://surrealdb.com/docs/surrealql/statements/select?utm_source=chatgpt.com "SELECT statement | SurrealQL"
[10]: https://surrealdb.com/docs/surrealql/datamodel/records?utm_source=chatgpt.com "Record links | SurrealQL"
[11]: https://surrealdb.com/docs/surrealql/statements/define/analyzer?utm_source=chatgpt.com "DEFINE ANALYZER statement | SurrealQL"
[12]: https://surrealdb.com/docs/surrealdb/reference-guide/vector-search "Vector Search | Reference guides"
[13]: https://surrealdb.com/docs/surrealql/transactions "Transactions | SurrealQL"
[14]: https://surrealdb.com/docs/sdk/python/methods/live "Python | SDK | Methods | live"
[15]: https://surrealdb.com/docs/surrealql/statements/define/table "DEFINE TABLE statement | SurrealQL"
[16]: https://surrealdb.com/docs/surrealdb/reference-guide/graph-relations?utm_source=chatgpt.com "Graph relations | Reference guides"
