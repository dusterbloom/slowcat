#!/usr/bin/env python3
"""
Migrate flat Surreal tables to graph schema (robust, idempotent).

- Reads .surql via extractor
- Writes into user/session/message/concept + knows/contains
- Parameterized via env/CLI. Defaults to DB=memory_graph.
"""

import argparse
import os
import uuid
from typing import Dict, List, Any
from dataclasses import dataclass
from loguru import logger

from extract_backup_data import SurrealQLParser

from datetime import datetime, timezone

try:
    from surrealdb import AsyncSurreal
except ImportError:
    raise SystemExit("SurrealDB client not available. pip install surrealdb")

# RecordID may not be present in 1.0.6; detect and fallback
try:
    from surrealdb.types import RecordID  # type: ignore
    HAVE_RECORDID = True
except Exception:
    HAVE_RECORDID = False


@dataclass
class MigrationStats:
    users_created: int = 0
    sessions_created: int = 0
    messages_created: int = 0
    concepts_created: int = 0
    knowledge_relations: int = 0
    contains_relations: int = 0
    thoughts_created: int = 0
    errors: int = 0


class GraphMigrator:
    def __init__(self, url: str, ns: str, db: str, user: str, password: str):
        self.url = url
        self.ns = ns
        self.dbname = db
        self.user = user
        self.password = password
        self.db: AsyncSurreal | None = None
        self.stats = MigrationStats()
        self.speaker_to_user_map: Dict[str, str] = {}
        self.session_id_map: Dict[str, str] = {}
        self.concept_cache: Dict[str, str] = {}
        self.default_user_id: str | None = None

    async def connect(self):
        client = AsyncSurreal(self.url)
        await client.connect()
        try:
            await client.signin({"username": self.user, "password": self.password})
        except Exception:
            await client.signin({"user": self.user, "pass": self.password})
        await client.use(self.ns, self.dbname)
        self.db = client
        # Log intended context (Surreal 1.0.6 lacks db::ns/db::name to introspect)
        logger.info(f"Connected to SurrealDB (intended): ns={self.ns} db={self.dbname}")

    async def close(self):
        if self.db:
            await self.db.close()
            self.db = None

    # ---------------- Migration Phases ----------------
    async def create_users(self, data: Dict[str, List[Dict]]):
        assert self.db
        speakers = set()
        for tbl in ("tape", "sessions"):
            for rec in data.get(tbl, []):
                spk = rec.get("speaker_id")
                if spk and spk not in ("unknown", "slowcat"):
                    speakers.add(spk)
        for speaker in sorted(speakers):
            try:
                uid = f"user:{self._safe_id(speaker)}"
                await self.db.query(
                    f"""
                    CREATE {uid} SET
                        name = $name,
                        first_seen = time::now(),
                        last_seen = time::now(),
                        total_interactions = 0,
                        metadata = {{}}
                    """,
                    {"name": speaker},
                )
                self.speaker_to_user_map[speaker] = uid
                self.stats.users_created += 1
            except Exception as e:
                logger.error(f"user {speaker}: {e}")
                self.stats.errors += 1
        logger.info(f"Users created: {self.stats.users_created}")
        # Choose default user for 'user' facts: prefer 'peppi', otherwise most frequent speaker in tape
        if 'peppi' in self.speaker_to_user_map:
            self.default_user_id = self.speaker_to_user_map['peppi']
        else:
            counts: Dict[str, int] = {}
            for tr in data.get('tape', []):
                spk = tr.get('speaker_id')
                if spk and spk != 'unknown' and spk in self.speaker_to_user_map:
                    counts[spk] = counts.get(spk, 0) + 1
            if counts:
                top = max(counts.items(), key=lambda kv: kv[1])[0]
                self.default_user_id = self.speaker_to_user_map.get(top)
        if self.default_user_id:
            logger.info(f"Default user for 'user' facts: {self.default_user_id}")
        logger.info(f"[users] speakers={len(speakers)} created={self.stats.users_created} errors={self.stats.errors}")

    async def create_sessions(self, data: Dict[str, List[Dict]]):
        assert self.db
        summaries = data.get("session_summary", [])
        # From sessions table
        for rec in data.get("sessions", []):
            try:
                spk = rec.get("speaker_id", "unknown")
                uid = self.speaker_to_user_map.get(spk)
                if not uid:
                    logger.warning(f"No user for speaker {spk}")
                    continue
                sess_id = f"session:{uuid.uuid4().hex[:12]}"
                smry = next((s.get("summary", "") for s in summaries if s.get("session_id") == rec.get("id")), "")
                await self.db.query(
                    f"""
                    CREATE {sess_id} SET
                        user_id = type::thing('user', $user_key),
                        agent_id = 'slowcat',
                        started_at = $started,
                        turn_count = $turns,
                        summary = $summary,
                        keywords = $keywords,
                        status = 'ended'
                    """,
                    {
                        "user_key": uid.split(':',1)[1] if ':' in uid else uid,
                        # use original ISO string if present; else current time
                        "started": rec.get("first_seen", "") or None,
                        "turns": int(rec.get("total_turns", 0) or 0),
                        "summary": smry,
                        "keywords": [],
                    },
                )
                self.session_id_map[str(rec.get("id", ""))] = sess_id
                self.stats.sessions_created += 1
            except Exception as e:
                logger.error(f"session from record: {e}")
                self.stats.errors += 1
        # From tape uniques
        for tr in data.get("tape", []):
            key = tr.get("session_id")
            spk = tr.get("speaker_id", "unknown")
            if not key or key in self.session_id_map:
                continue
            uid = self.speaker_to_user_map.get(spk)
            if not uid:
                continue
            try:
                new_id = f"session:{uuid.uuid4().hex[:12]}"
                await self.db.query(
                    f"CREATE {new_id} SET user_id=type::thing('user',$user_key), agent_id='slowcat', started_at=time::now(), turn_count=0, summary='', keywords=[], status='active'",
                    {"user_key": uid.split(':',1)[1] if ':' in uid else uid},
                )
                self.session_id_map[key] = new_id
                self.stats.sessions_created += 1
            except Exception as e:
                logger.error(f"session from tape: {e}")
                self.stats.errors += 1
        logger.info(f"Sessions created: {self.stats.sessions_created}")
        logger.info(f"[sessions] created={self.stats.sessions_created} total_map={len(self.session_id_map)} errors={self.stats.errors}")

    async def create_concepts(self, data: Dict[str, List[Dict]]):
        assert self.db
        for f in data.get("fact", []):
            val = f.get("value")
            kind = f.get("species", "unknown")
            if not val:
                continue
            cid = f"concept:{self._safe_id(val)}"
            try:
                await self.db.query(
                    f"CREATE {cid} SET name=$name, kind=$kind, properties={{}}, mentioned_count=0, first_mentioned=time::now(), last_mentioned=time::now()",
                    {"name": val, "kind": kind},
                )
                self.concept_cache[val] = cid
                self.stats.concepts_created += 1
            except Exception as e:
                logger.error(f"concept {val}: {e}")
                self.stats.errors += 1
        logger.info(f"Concepts created: {self.stats.concepts_created}")
        logger.info(f"[concepts] cache={len(self.concept_cache)} errors={self.stats.errors}")

    async def create_messages(self, data: Dict[str, List[Dict]]):
        assert self.db
        seq: Dict[str, int] = {}
        created = 0
        def sanitize_text(t: str) -> str:
            if not t:
                return t
            import re
            s = re.sub(r"\s+", " ", str(t)).strip()
            # Drop accidental field dumps like "embedding: [..]" appended into content
            s = re.sub(r"embedding:\s*\[[^\]]*\]", "", s, flags=re.IGNORECASE)
            # Fix triple-fragmented names like "Po to la" -> Potola
            def _fix_fragments(match: re.Match) -> str:
                parts = match.group(0).split()
                combined = ''.join(parts)
                # Capitalize first if first token capitalized
                if parts[0][0].isupper():
                    return combined[0].upper() + combined[1:]
                return combined
            s = re.sub(r"\b([A-Za-z]{1,3}\s){2,4}[A-Za-z]{1,3}\b", _fix_fragments, s)
            # Normalize quotes
            s = s.replace("\u2019", "'").replace("\u2014", "-")
            return re.sub(r"\s+", " ", s).strip()
        for tr in data.get("tape", []):
            try:
                key = tr.get("session_id")
                if not key:
                    continue
                if key not in self.session_id_map:
                    # create on-demand
                    spk = tr.get("speaker_id", "unknown")
                    if spk in self.speaker_to_user_map:
                        new_id = f"session:{uuid.uuid4().hex[:12]}"
                        await self.db.query(
                            f"CREATE {new_id} SET user_id=$uid, agent_id='slowcat', started_at=time::now(), turn_count=0, summary='', keywords=[], status='active'",
                            {"uid": self.speaker_to_user_map[spk]},
                        )
                        self.session_id_map[key] = new_id
                        self.stats.sessions_created += 1
                    else:
                        logger.warning(f"No user for message session {key}")
                        continue
                sid = self.session_id_map[key]
                if HAVE_RECORDID:
                    s_table, s_id = self._split_rid(sid)
                    s_rid = RecordID(s_table, s_id)
                msg_id = f"message:{uuid.uuid4().hex[:12]}"
                if HAVE_RECORDID:
                    m_table, m_id = self._split_rid(msg_id)
                    m_rid = RecordID(m_table, m_id)
                ts = tr.get("ts")
                if isinstance(ts, (int, float)):
                    ts_value = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                else:
                    ts_value = ts or datetime.now(timezone.utc)
                seq[sid] = seq.get(sid, 0) + 1
                order = tr.get("sequence_num") or seq[sid]
                # Ensure embedding is an array
                emb = tr.get('embedding')
                if not isinstance(emb, list):
                    emb = []
                role = tr.get('role', 'user')
                if role not in ('user','assistant'):
                    role = 'user'
                # Use SurrealQL with type::thing for reliability across client versions
                await self.db.query(
                    f"""
                    CREATE {msg_id} SET
                        session_id = type::thing('session', $skey),
                        speaker_type = $role,
                        content = $content,
                        timestamp = $ts,
                        sequence_num = $seq,
                        embedding = $emb,
                        metadata = {{}}
                    """,
                    {
                        'skey': sid.split(':',1)[1] if ':' in sid else sid,
                        'role': role,
                        'content': sanitize_text(tr.get('content', '')),
                        'ts': ts_value,
                        'seq': int(order),
                        'emb': emb,
                    },
                )
                await self.db.query(
                    "RELATE $sid->contains->$mid SET sequence_num=$seq, created_at=time::now()",
                    {'sid': sid, 'mid': msg_id, 'seq': int(order)},
                )
                created += 1
                self.stats.contains_relations += 1
            except Exception as e:
                logger.error(f"message: {e}")
                self.stats.errors += 1
        self.stats.messages_created = created
        logger.info(f"Messages created: {self.stats.messages_created}")
        logger.info(f"[messages] created={self.stats.messages_created} errors={self.stats.errors}")

    async def create_knowledge(self, data: Dict[str, List[Dict]]):
        assert self.db
        default_user = self.default_user_id
        for f in data.get("fact", []):
            try:
                subj = f.get("subject") or "unknown"
                val = f.get("value")
                if not val:
                    continue
                if subj == "user" and default_user:
                    uid = default_user
                else:
                    uid = self.speaker_to_user_map.get(subj)
                cid = self.concept_cache.get(val)
                if not uid or not cid:
                    logger.warning(f"skip fact subj={subj} val={val}")
                    continue
                created = f.get('created'); last_seen = f.get('last_seen')
                learned_at = created or datetime.now(timezone.utc)
                reinforced_at = last_seen or datetime.now(timezone.utc)
                if HAVE_RECORDID:
                    u_table, u_id = self._split_rid(uid)
                    c_table, c_id = self._split_rid(cid)
                    await self.db.relate(RecordID(u_table, u_id), 'knows', RecordID(c_table, c_id), {
                        'relationship': f.get('predicate', 'related_to'),
                        'fidelity': f.get('fidelity', 3),
                        'strength': f.get('strength', 0.6),
                        'learned_at': learned_at,
                        'reinforced_at': reinforced_at,
                        'access_count': f.get('access_count', 0),
                        'decay_rate': 1.0,
                    })
                else:
                    await self.db.query(
                        """
                        RELATE $u->knows->$c SET
                            relationship=$pred,
                            fidelity=$fid,
                            strength=$str,
                            learned_at=$learned,
                            reinforced_at=$reinforced,
                            access_count=$acc,
                            decay_rate=1.0
                        """,
                        {
                            'u': uid,
                            'c': cid,
                            'pred': f.get('predicate', 'related_to'),
                            'fid': f.get('fidelity', 3),
                            'str': f.get('strength', 0.6),
                            'learned': learned_at,
                            'reinforced': reinforced_at,
                            'acc': f.get('access_count', 0),
                        },
                    )
                self.stats.knowledge_relations += 1
            except Exception as e:
                logger.error(f"knows: {e}")
                self.stats.errors += 1
        logger.info(f"Knowledge relations: {self.stats.knowledge_relations}")
        logger.info(f"[knows] created={self.stats.knowledge_relations} errors={self.stats.errors}")

    async def create_mentions(self):
        """Create message->mentions->concept edges with fuzzy matching."""
        assert self.db
        created = 0
        try:
            # Helpers
            def rows(res):
                if not res: return []
                if isinstance(res, list) and res and isinstance(res[0], dict) and 'result' in res[0]:
                    return res[0]['result'] or []
                return res
            import unicodedata, re
            from difflib import SequenceMatcher
            def norm(s: str) -> str:
                if not s: return ''
                s2 = unicodedata.normalize('NFKD', s)
                s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
                s2 = re.sub(r"\s+", " ", s2).strip().lower()
                return s2
            def collapse(s: str) -> str:
                return re.sub(r"[^a-z0-9]", "", norm(s))

            # Load concepts and all messages (small corpora; adjust if needed)
            concepts = rows(await self.db.query("SELECT id, name FROM concept"))
            messages = rows(await self.db.query("SELECT id, content FROM message LIMIT 100000"))
            msg_cache = [(m.get('id'), m.get('content',''), norm(m.get('content','')), collapse(m.get('content',''))) for m in messages]

            for c in concepts:
                cid = c.get('id'); cname = c.get('name')
                if not cid or not cname:
                    continue
                qn = norm(cname); qc = collapse(cname)
                for mid, raw, mn, mc in msg_cache:
                    if not mid: continue
                    hit = False
                    if qn and qn in mn:
                        hit = True
                    elif qc and qc in mc:
                        hit = True
                    else:
                        # fuzzy on tokens
                        tokens = re.findall(r"[a-z0-9]+", mn)
                        tgt_tokens = re.findall(r"[a-z0-9]+", qn)
                        tgt = ' '.join(tgt_tokens) if tgt_tokens else ''
                        if tgt:
                            for tok in tokens:
                                if len(tok) >= 3 and SequenceMatcher(a=tgt, b=tok).ratio() >= 0.87:
                                    hit = True
                                    break
                    if not hit:
                        continue
                    # Exists?
                    ex = rows(await self.db.query("SELECT count() AS c FROM mentions WHERE in=$m AND out=$c", {"m": mid, "c": cid}))
                    exists = False
                    if ex and isinstance(ex[0], dict):
                        for k in ('c','count','count()'):
                            if k in ex[0] and isinstance(ex[0][k], (int,float)) and int(ex[0][k])>0:
                                exists=True; break
                    if exists:
                        continue
                    await self.db.query("RELATE $m->mentions->$c SET relevance=0.6, extracted_at=time::now()", {"m": mid, "c": cid})
                    created += 1
            logger.info(f"[mentions] created={created}")
        except Exception as e:
            logger.warning(f"mentions backfill failed: {e}")

    async def create_thoughts(self, data: Dict[str, List[Dict]]):
        assert self.db
        for t in data.get("thought", []):
            try:
                tid = f"thought:{uuid.uuid4().hex[:12]}"
                ts = t.get("ts")
                texpr = "$ts" if isinstance(ts, str) else "time::now()"
                await self.db.query(
                    f"CREATE {tid} SET agent_id=$agent, session_id=$sid, thought_type=$type, content=$content, timestamp={texpr}, visibility=$vis, confidence=$conf",
                    {
                        "agent": t.get("agent_id", "slowcat"),
                        "sid": None,
                        "type": t.get("thought_type", "observation"),
                        "content": t.get("content", t.get("content_snippet", "")),
                        "vis": t.get("visibility", "private"),
                        "conf": float(t.get("confidence", 0.5) or 0.5),
                        "ts": ts or "",
                    },
                )
                self.stats.thoughts_created += 1
            except Exception as e:
                logger.error(f"thought: {e}")
                self.stats.errors += 1
        logger.info(f"Thoughts created: {self.stats.thoughts_created}")

    # ---------------- Helpers ----------------
    def _safe_id(self, s: str) -> str:
        import re
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(s).lower())[:50]

    def _split_rid(self, rid: str):
        if isinstance(rid, str) and ':' in rid:
            table, rec = rid.split(':', 1)
            return table, rec
        return rid, rid


async def run_migration(backup_path: str, url: str, ns: str, db: str, user: str, password: str):
    logger.info(f"Reading backup: {backup_path}")
    parser = SurrealQLParser(backup_path)
    data = parser.parse_backup()
    logger.info("Extraction stats: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))

    mig = GraphMigrator(url, ns, db, user, password)
    await mig.connect()
    try:
        await mig.create_users(data)
        await mig.create_sessions(data)
        await mig.create_concepts(data)
        await mig.create_messages(data)
        await mig.create_knowledge(data)
        await mig.create_thoughts(data)
        await mig.create_mentions()
        logger.info(
            f"Done: users={mig.stats.users_created}, sessions={mig.stats.sessions_created}, messages={mig.stats.messages_created}, "
            f"concepts={mig.stats.concepts_created}, knows={mig.stats.knowledge_relations}, contains={mig.stats.contains_relations}, "
            f"thoughts={mig.stats.thoughts_created}, errors={mig.stats.errors}"
        )
        # Cross-check counts from DB
        try:
            def _rows(res):
                if not res: return []
                if isinstance(res, list) and res and isinstance(res[0], dict) and 'result' in res[0]:
                    return res[0]['result'] or []
                return res
            counts = {}
            for tbl in ("user","session","message","concept","knows","contains","mentions","reflects"):
                rr = _rows(await mig.db.query(f"SELECT count() AS c FROM {tbl}"))
                val = 0
                if rr:
                    first = rr[0]
                    if isinstance(first, dict):
                        for k in ('c','count','count()'):
                            if k in first and isinstance(first[k], (int,float)):
                                val = int(first[k]); break
                        if val == 0:
                            for v in first.values():
                                if isinstance(v, (int,float)):
                                    val = int(v); break
                    elif isinstance(first, (int,float)):
                        val = int(first)
                counts[tbl] = val
            logger.info("DB counts: " + ", ".join(f"{k}={v}" for k,v in counts.items()))
        except Exception as e:
            logger.warning(f"DB count cross-check failed: {e}")
    finally:
        await mig.close()


def parse_args():
    p = argparse.ArgumentParser(description="Migrate .surql backup to graph schema")
    p.add_argument('--backup', default=os.getenv('BACKUP_FILE'))
    p.add_argument('--url', default=os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc'))
    p.add_argument('--ns', default=os.getenv('SURREALDB_NAMESPACE', 'slowcat'))
    p.add_argument('--db', default=os.getenv('SURREALDB_DATABASE', 'memory_graph'))
    p.add_argument('--user', default=os.getenv('SURREALDB_USER', 'root'))
    p.add_argument('--pass', dest='password', default=os.getenv('SURREALDB_PASS', 'slowcat_secure_2024'))
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    backup = args.backup or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory', 'localslowcat-2025-08-27.surql'))
    import asyncio
    asyncio.run(run_migration(backup, args.url, args.ns, args.db, args.user, args.password))
