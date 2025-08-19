#!/usr/bin/env python3
"""
Clean Facts DB: remove noisy/non-canonical facts and optionally normalize subjects.

Canonical rules (kept):
- subject == 'user'
- predicate in {'location','age','job','works_at','likes'} OR predicate endswith '_name'
- for 'age': value must be numeric (0<age<130)
- for 'location': value not empty and not one of {'location','my location'} (case-insensitive)

Normalization:
- subject in {'i','me','my','myself','you','your','yours','yourself'} -> 'user' (when predicate canonical)

Removals:
- predicate in {'is','has'}
- non-canonical predicates (not in allowed and not *_name)
- empty/None values where required

Usage:
  python server/scripts/clean_facts_db.py [--db server/data/facts.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Tuple

ALLOWED = {"location", "age", "job", "works_at", "likes"}
PRONOUNS = {"i", "me", "my", "myself", "you", "your", "yours", "yourself"}
DROP_PREDICATES = {"is", "has"}


def ensure_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"Facts DB not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def normalize_subjects(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Normalize pronoun subjects to 'user' for canonical predicates."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, subject, predicate, value FROM facts
        WHERE lower(subject) IN ({})
          AND (
            lower(predicate) IN ({}) OR predicate LIKE '%_name'
          )
        """.format(
            ",".join(["?"] * len(PRONOUNS)), ",".join(["?"] * len(ALLOWED))
        ),
        tuple(PRONOUNS) + tuple(ALLOWED),
    )
    rows = cur.fetchall()
    if not rows:
        return 0
    updated = 0
    for row in rows:
        if dry_run:
            updated += 1
            continue
        cur.execute("UPDATE facts SET subject='user' WHERE id=?", (row["id"],))
        updated += 1
    if not dry_run:
        conn.commit()
    return updated


def delete_noisy(conn: sqlite3.Connection, dry_run: bool) -> Tuple[int, int]:
    """Delete non-canonical or low-value facts. Returns (deleted_count, kept_count)."""
    cur = conn.cursor()
    # Build conditions for deletion
    # 1) Drop vague predicates 'is'/'has'
    cond_drop_pred = "lower(predicate) IN ({})".format(",".join(["?"] * len(DROP_PREDICATES)))

    # 2) Non-canonical predicates (not allowed and not *_name)
    cond_non_canonical = "(lower(predicate) NOT IN ({}) AND predicate NOT LIKE '%_name')".format(
        ",".join(["?"] * len(ALLOWED))
    )

    # 3) Subject not 'user' (after normalization pass, still non-user)
    cond_bad_subject = "lower(subject) != 'user'"

    # 4) Age must be numeric and sensible
    cond_bad_age = "(lower(predicate)='age' AND (value IS NULL OR TRIM(value)='' OR CAST(value AS INTEGER) <= 0 OR CAST(value AS INTEGER) > 129))"

    # 5) Location must be meaningful
    cond_bad_location = "(lower(predicate)='location' AND (value IS NULL OR TRIM(value)='' OR lower(value) IN ('location','my location')))"

    where = f"({cond_drop_pred}) OR ({cond_non_canonical}) OR ({cond_bad_subject}) OR {cond_bad_age} OR {cond_bad_location}"
    params = tuple(DROP_PREDICATES) + tuple(ALLOWED)

    # Count matches
    cur.execute(f"SELECT COUNT(*) FROM facts WHERE {where}", params)
    to_delete = cur.fetchone()[0]

    kept = 0
    cur.execute("SELECT COUNT(*) FROM facts")
    total = cur.fetchone()[0]
    kept = max(0, total - to_delete)

    if to_delete and not dry_run:
        cur.execute(f"DELETE FROM facts WHERE {where}", params)
        conn.commit()

    return to_delete, kept


def main():
    parser = argparse.ArgumentParser(description="Clean noisy/non-canonical facts from facts.db")
    default_db = None
    try:
        # Prefer config path if importable
        from config import config  # type: ignore

        default_db = Path(config.memory.facts_db_path)
    except Exception:
        default_db = Path(__file__).resolve().parents[1] / "data" / "facts.db"

    parser.add_argument("--db", type=Path, default=default_db, help="Path to facts.db")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()

    conn = ensure_db(args.db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM facts")
    total_before = cur.fetchone()[0]
    print(f"Facts DB: {args.db}")
    print(f"Total facts before: {total_before}")

    updated = normalize_subjects(conn, args.dry_run)
    print(f"Subjects normalized to 'user': {updated}")

    deleted, kept = delete_noisy(conn, args.dry_run)
    print(f"Deleted noisy facts: {deleted}")

    if not args.dry_run:
        cur.execute("SELECT COUNT(*) FROM facts")
        total_after = cur.fetchone()[0]
        print(f"Total facts after: {total_after}")
    else:
        print(f"(dry-run) Estimated facts kept: {kept}")

    conn.close()


if __name__ == "__main__":
    main()

