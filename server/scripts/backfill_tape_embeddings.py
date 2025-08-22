"""
Backfill SurrealDB tape embeddings so KNN retrieval can use historical data.

Usage:
  cd server && python -m scripts.backfill_tape_embeddings \
      --limit 2000 --batch 100 --model all-MiniLM-L6-v2

Environment:
  SURREALDB_URL (default: ws://localhost:8000/rpc)
  SURREALDB_NAMESPACE (default: slowcat)
  SURREALDB_DATABASE (default: memory)
  SURREALDB_USER (default: root)
  SURREALDB_PASS (default: slowcat_secure_2024)

Notes:
  - Requires sentence-transformers locally (no network).
  - Embeddings are stored into the `embedding` field of `tape` rows.
"""
from __future__ import annotations

import os
import argparse
import asyncio
from typing import List, Dict, Any

from loguru import logger


async def _connect_surreal():
    from memory.surreal_memory import SurrealMemory

    url = os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc")
    ns = os.getenv("SURREALDB_NAMESPACE", "slowcat")
    db = os.getenv("SURREALDB_DATABASE", "memory")
    mem = SurrealMemory(url, ns, db)
    await mem.connect()
    return mem


async def _fetch_missing(mem, limit: int, force: bool = False) -> List[Dict[str, Any]]:
    """Fetch tape rows to embed.

    If force is False: get rows where embedding is missing/empty.
    If force is True:  get oldest rows regardless of current embedding.
    """
    if force:
        sql = """
            SELECT id, content, ts, role, speaker_id
            FROM tape
            ORDER BY ts ASC
            LIMIT $limit
        """
        params = {"limit": limit}
    else:
        # Try to capture both NONE, null, and empty values
        sql = """
            SELECT id, content, ts, role, speaker_id, embedding
            FROM tape
            WHERE embedding = NONE OR embedding = null
            ORDER BY ts ASC
            LIMIT $limit
        """
        params = {"limit": limit}
    res = await mem.db.query(sql, params)
    rows: List[Dict[str, Any]] = []
    try:
        if isinstance(res, list):
            if res and isinstance(res[0], dict) and "result" in res[0]:
                rows = res[0]["result"] or []
            else:
                rows = res  # some clients return flat rows
        elif isinstance(res, dict) and "result" in res:
            rows = res["result"] or []
    except Exception:
        rows = []
    return rows


async def _update_embeddings(mem, ids: List[str], vectors: List[List[float]]):
    # Update each row; batched update is not guaranteed across client versions
    for rid, emb in zip(ids, vectors):
        try:
            await mem.db.query("UPDATE $id SET embedding = $emb", {"id": rid, "emb": emb})
        except Exception as e:
            logger.warning(f"Update failed for {rid}: {e}")


def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore
    logger.info(f"🧠 Loading sentence-transformers model: {model_name}")
    return SentenceTransformer(model_name)


async def backfill(limit: int, batch: int, model_name: str, force: bool = False):
    mem = await _connect_surreal()
    try:
        rows = await _fetch_missing(mem, limit, force=force)
        total = len(rows)
        if total == 0:
            if force:
                logger.info("✅ No tape rows selected under --force (database may be empty).")
            else:
                logger.info("✅ No tape rows missing embeddings (or driver reports none). Try --force to recompute.")
            return
        logger.info(f"📥 Found {total} tape rows missing embeddings (limit={limit}).")

        encoder = _load_encoder(model_name)

        # Process in batches
        start = 0
        done = 0
        while start < total:
            chunk = rows[start : start + batch]
            ids = [r.get("id") for r in chunk if r.get("id")]
            texts = [str(r.get("content") or "") for r in chunk]
            # Compute embeddings
            vectors = encoder.encode(texts, convert_to_numpy=False)  # list[list[float]]
            # Convert numpy arrays to plain lists when present
            safe_vectors: List[List[float]] = []
            for v in vectors:
                try:
                    safe_vectors.append(v.tolist())  # numpy array
                except Exception:
                    safe_vectors.append(list(v))     # already list-like

            await _update_embeddings(mem, ids, safe_vectors)
            done += len(ids)
            start += batch
            logger.info(f"✅ Updated {done}/{total} rows with embeddings")

        logger.info("🎉 Backfill complete.")
    finally:
        try:
            await mem.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Backfill SurrealDB tape embeddings")
    parser.add_argument("--limit", type=int, default=int(os.getenv("BACKFILL_LIMIT", "1000")), help="Max rows to backfill")
    parser.add_argument("--batch", type=int, default=int(os.getenv("BACKFILL_BATCH", "100")), help="Batch size for updates")
    parser.add_argument("--model", type=str, default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), help="SentenceTransformer model name")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even if present (oldest first)")
    args = parser.parse_args()

    asyncio.run(backfill(args.limit, args.batch, args.model, force=args.force))


if __name__ == "__main__":
    main()
