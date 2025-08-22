#!/usr/bin/env python3
"""
Pre-cache embedding and cross-encoder models for DTH.

Usage examples:
  # Preload embedding only (default)
  python server/scripts/precache_models.py

  # Preload both embedding and cross-encoder
  python server/scripts/precache_models.py --enable-cross-encoder

Environment overrides:
  EMBEDDING_MODEL=all-MiniLM-L6-v2
  CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
  HF_HOME=/path/to/hf_cache
  TRANSFORMERS_CACHE=$HF_HOME/hub
"""

from __future__ import annotations

import os
import sys
import argparse


def preload_embedding(model_name: str) -> bool:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        print(f"[precache] sentence-transformers not available: {e}")
        return False
    try:
        SentenceTransformer(model_name)
        print(f"[precache] ✅ Embedding model cached: {model_name}")
        return True
    except Exception as e:
        print(f"[precache] ⚠️  Failed to cache embedding model '{model_name}': {e}")
        return False


def preload_cross_encoder(model_name: str) -> bool:
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        print(f"[precache] sentence-transformers not available: {e}")
        return False
    try:
        CrossEncoder(model_name)
        print(f"[precache] ✅ Cross-encoder cached: {model_name}")
        return True
    except Exception as e:
        print(f"[precache] ⚠️  Failed to cache cross-encoder '{model_name}': {e}")
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-cache DTH models")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformer embedding model name",
    )
    parser.add_argument(
        "--enable-cross-encoder",
        action="store_true",
        default=os.getenv("USE_CROSS_ENCODER", "false").lower() == "true",
        help="Also cache the cross-encoder reranker",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        help="Cross-encoder model name",
    )
    args = parser.parse_args(argv)

    cache_root = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.path.expanduser("~/.cache/huggingface")
    print(f"[precache] Using cache: {cache_root}")

    ok_embed = preload_embedding(args.embedding_model)

    ok_ce = True
    if args.enable_cross_encoder:
        ok_ce = preload_cross_encoder(args.cross_encoder_model)
    else:
        print("[precache] Cross-encoder disabled; skipping")

    print("[precache] Done.")
    return 0 if (ok_embed and ok_ce) else 1


if __name__ == "__main__":
    raise SystemExit(main())

