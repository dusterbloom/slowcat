"""
Facts Graph - Structured knowledge storage with natural decay

Implements the friend's proposal for memory as predictive compression:
- Facts stored with fidelity levels S4 → S0
- Natural decay over time unless reinforced
- Compression from verbatim to structured to forgotten
- Fast retrieval for personal facts

Fidelity Levels:
- S4 (verbatim): "my dog name is Potola"
- S3 (structured): user::pet[name=Potola, species=dog]  
- S2 (tuple): (user, pet, Potola)
- S1 (edge): (user —has_pet→ dog) [name forgotten]
- S0 (forgotten): completely removed
"""

import sqlite3
import time
import json
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger


@dataclass
class Fact:
    """A single fact with decay properties"""
    subject: str         # "user"
    predicate: str       # "pet"  
    value: Optional[str] # "Potola" (can be None at S1)
    species: Optional[str] = None  # Additional qualifier
    fidelity: int = 3         # 4..0 (current fidelity level)
    strength: float = 0.6     # 0..1 (decay resistance)
    last_seen: float = 0      # Unix timestamp
    created: float = 0        # When first created
    access_count: int = 0     # How often accessed
    source_text: str = ""     # Original text that created this fact


# Constants (tunable)
DECAY_HALF_LIFE_S = 6 * 3600    # 6 hours half-life
PROMOTE_THRESH = 0.75           # Strength threshold to promote fidelity
DEMOTE_THRESH = 0.25           # Strength threshold to demote fidelity  
EMA_ALPHA = 0.35               # Exponential moving average factor
MAX_FACTS = 512                # Hard cap on total facts
FIDELITY_NAMES = {
    4: "verbatim",
    3: "structured", 
    2: "tuple",
    1: "edge",
    0: "forgotten"
}


class FactsGraph:
    """
    Graph-based fact storage with natural decay
    
    Features:
    - SQLite backend for persistence
    - Fidelity-based compression (S4→S0)
    - Time-based exponential decay
    - Reinforcement learning on access
    - Fast lookup by subject/predicate
    """
    
    def __init__(self, db_path: str = "data/facts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        self._init_tables()
        logger.info(f"🧠 Facts graph initialized: {self.db_path}")
        
        # Performance metrics
        self.reinforcements = 0
        self.decays_applied = 0
        self.facts_forgotten = 0
        
    def _init_tables(self):
        """Initialize database tables"""
        
        # Main facts table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,  
            value TEXT,                     -- Nullable at S1
            species TEXT,                   -- Optional qualifier
            fidelity INTEGER NOT NULL DEFAULT 3,
            strength REAL NOT NULL DEFAULT 0.6,
            last_seen REAL NOT NULL,
            created REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            source_text TEXT DEFAULT ''
        )""")
        
        # Indexes for fast lookup
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_subject_predicate 
        ON facts(subject, predicate)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fidelity_strength
        ON facts(fidelity DESC, strength DESC)
        """)
        
        # Session metadata table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            speaker_id TEXT PRIMARY KEY,
            session_count INTEGER DEFAULT 0,
            last_interaction REAL,
            first_seen REAL,
            total_turns INTEGER DEFAULT 0
        )""")
        
        # Synopsis table (rolling summary)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS synopsis (
            id INTEGER PRIMARY KEY CHECK (id=1),
            text TEXT NOT NULL DEFAULT '',
            updated_ts REAL NOT NULL DEFAULT 0,
            token_count INTEGER DEFAULT 0
        )""")
        
        self.conn.commit()
        
    def reinforce_or_insert(self, fact_data: Dict) -> bool:
        """
        Store new fact or reinforce existing one
        
        Args:
            fact_data: Dict with subject, predicate, value, etc.
            
        Returns:
            bool: True if fact was reinforced, False if newly inserted
        """
        cur = self.conn.cursor()
        now = time.time()
        
        # Check if fact exists
        cur.execute("""
        SELECT id, fidelity, strength, access_count
        FROM facts 
        WHERE subject=? AND predicate=? 
        AND COALESCE(value,'') = COALESCE(?,'')
        AND COALESCE(species,'') = COALESCE(?,'')
        """, (
            fact_data['subject'],
            fact_data['predicate'], 
            fact_data.get('value'),
            fact_data.get('species')
        ))
        
        row = cur.fetchone()
        
        if row:
            # Existing fact - reinforce it
            fact_id, old_fidelity, old_strength, access_count = row
            
            # Apply EMA to strengthen
            new_strength = self._ema(old_strength, 1.0)
            new_fidelity = max(old_fidelity, fact_data.get('fidelity', 3))
            
            cur.execute("""
            UPDATE facts 
            SET fidelity=?, strength=?, last_seen=?, access_count=access_count+1
            WHERE id=?
            """, (new_fidelity, new_strength, now, fact_id))
            
            self.reinforcements += 1
            logger.debug(f"✨ Reinforced fact: {fact_data['subject']}.{fact_data['predicate']} "
                        f"(S{old_fidelity}→S{new_fidelity}, {old_strength:.2f}→{new_strength:.2f})")
            
            self.conn.commit()
            return True
            
        else:
            # New fact - insert it
            cur.execute("""
            INSERT INTO facts(
                subject, predicate, value, species, fidelity, 
                strength, last_seen, created, source_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_data['subject'],
                fact_data['predicate'],
                fact_data.get('value'),
                fact_data.get('species'),
                fact_data.get('fidelity', 3),
                fact_data.get('strength', 0.6),
                now,
                now,
                fact_data.get('source_text', '')
            ))
            
            logger.debug(f"➕ New fact: {fact_data['subject']}.{fact_data['predicate']} = "
                        f"{fact_data.get('value', '?')} (S{fact_data.get('fidelity', 3)})")
            
            self.conn.commit()
            return False
    
    def get_facts(self, 
                  subject: str = None,
                  predicate: str = None, 
                  min_fidelity: int = 1,
                  limit: int = 50) -> List[Fact]:
        """
        Retrieve facts with optional filtering
        
        Args:
            subject: Filter by subject (e.g., "user")
            predicate: Filter by predicate (e.g., "pet")
            min_fidelity: Minimum fidelity level (default 1)
            limit: Maximum facts to return
            
        Returns:
            List of Fact objects sorted by relevance
        """
        cur = self.conn.cursor()
        
        query = """
        SELECT subject, predicate, value, species, fidelity, strength,
               last_seen, created, access_count, source_text
        FROM facts 
        WHERE fidelity >= ?
        """
        params = [min_fidelity]
        
        if subject:
            query += " AND subject = ?"
            params.append(subject)
            
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)
            
        # Order by fidelity, then strength, then recency
        query += """ 
        ORDER BY fidelity DESC, strength DESC, last_seen DESC
        LIMIT ?
        """
        params.append(limit)
        
        cur.execute(query, params)
        
        facts = []
        for row in cur.fetchall():
            fact = Fact(
                subject=row['subject'],
                predicate=row['predicate'],
                value=row['value'],
                species=row['species'],
                fidelity=row['fidelity'],
                strength=row['strength'],
                last_seen=row['last_seen'],
                created=row['created'],
                access_count=row['access_count'],
                source_text=row['source_text']
            )
            facts.append(fact)
            
        return facts
    
    def search_facts(self, query: str, limit: int = 10) -> List[Fact]:
        """
        Search facts by text content
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching facts
        """
        # Simple keyword search for now
        keywords = query.lower().split()
        
        # Get all facts with reasonable fidelity
        all_facts = self.get_facts(min_fidelity=2, limit=100)
        
        # Score facts by keyword matches
        scored_facts = []
        for fact in all_facts:
            score = 0
            
            # Create searchable text
            search_text = f"{fact.subject} {fact.predicate} {fact.value or ''} {fact.species or ''}".lower()
            
            # Count keyword matches
            for keyword in keywords:
                if keyword in search_text:
                    score += 1
                    
            # Boost by fidelity and strength
            score = score * (fact.fidelity / 4) * fact.strength
            
            if score > 0:
                scored_facts.append((score, fact))
        
        # Sort by score and return top results
        scored_facts.sort(key=lambda x: x[0], reverse=True)
        return [fact for score, fact in scored_facts[:limit]]
    
    def decay_facts(self):
        """
        Apply time-based decay to all facts
        Should be called periodically (e.g., every 100 conversation turns)
        """
        cur = self.conn.cursor()
        now = time.time()
        
        # Get all facts for decay processing
        cur.execute("""
        SELECT id, fidelity, strength, last_seen, value
        FROM facts
        """)
        
        facts_processed = 0
        facts_changed = 0
        facts_deleted = 0
        
        for row in cur.fetchall():
            fact_id, fidelity, strength, last_seen, value = row
            
            # Calculate time-based decay
            dt = max(0.0, now - last_seen)
            half_lives = dt / DECAY_HALF_LIFE_S
            decayed_strength = strength * (0.5 ** half_lives)
            
            # Determine new fidelity based on strength
            new_fidelity = fidelity
            new_value = value
            
            # Demote if strength too low
            if decayed_strength < DEMOTE_THRESH and fidelity > 0:
                new_fidelity = fidelity - 1
                
                # Drop value when crossing S2→S1 (edge only)
                if new_fidelity <= 1 and value is not None:
                    new_value = None
                    
            # Promote if strength very high (rare, but possible)
            elif decayed_strength > PROMOTE_THRESH and fidelity < 4:
                new_fidelity = fidelity + 1
            
            # Delete if reached S0 (forgotten)
            if new_fidelity <= 0:
                cur.execute("DELETE FROM facts WHERE id=?", (fact_id,))
                facts_deleted += 1
                
            # Update if changed
            elif (new_fidelity != fidelity or 
                  abs(decayed_strength - strength) > 1e-4 or
                  new_value != value):
                
                cur.execute("""
                UPDATE facts 
                SET fidelity=?, strength=?, value=?
                WHERE id=?
                """, (new_fidelity, decayed_strength, new_value, fact_id))
                facts_changed += 1
                
            facts_processed += 1
        
        self.conn.commit()
        self.decays_applied += 1
        self.facts_forgotten += facts_deleted
        
        logger.info(f"🕰️ Decay applied: {facts_processed} facts processed, "
                   f"{facts_changed} changed, {facts_deleted} forgotten")
    
    def get_top_facts(self, limit: int = 10) -> List[Fact]:
        """
        Get the most important facts (highest fidelity + strength)
        """
        return self.get_facts(min_fidelity=2, limit=limit)
    
    def update_session(self, speaker_id: str):
        """Update session metadata for speaker"""
        cur = self.conn.cursor()
        now = time.time()

        cur.execute("""
        INSERT INTO sessions(speaker_id, session_count, last_interaction, first_seen, total_turns)
        VALUES (?, 1, ?, ?, 1)
        ON CONFLICT(speaker_id) DO UPDATE SET
            last_interaction = excluded.last_interaction,
            total_turns = total_turns + 1
        """, (speaker_id, now, now))
        
        self.conn.commit()

    def start_session(self, speaker_id: str):
        """Mark the start of a new session for a speaker.

        Increments session_count exactly once per session without affecting
        total_turns. Also updates last_interaction and first_seen if needed.
        """
        cur = self.conn.cursor()
        now = time.time()
        cur.execute(
            """
            INSERT INTO sessions(speaker_id, session_count, last_interaction, first_seen, total_turns)
            VALUES (?, 1, ?, ?, 0)
            ON CONFLICT(speaker_id) DO UPDATE SET
                session_count = session_count + 1,
                last_interaction = excluded.last_interaction
            """,
            (speaker_id, now, now)
        )
        self.conn.commit()
    
    def get_session_info(self, speaker_id: str) -> Dict:
        """Get session metadata for dynamic prompts"""
        cur = self.conn.cursor()
        
        cur.execute("""
        SELECT session_count, last_interaction, first_seen, total_turns
        FROM sessions
        WHERE speaker_id = ?
        """, (speaker_id,))
        
        row = cur.fetchone()
        if not row:
            return {
                'session_count': 0,
                'last_interaction': None,
                'first_seen': time.time(),
                'total_turns': 0
            }
            
        return {
            'session_count': row['session_count'],
            'last_interaction': row['last_interaction'],
            'first_seen': row['first_seen'],
            'total_turns': row['total_turns']
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        cur = self.conn.cursor()
        
        # Count facts by fidelity
        cur.execute("""
        SELECT fidelity, COUNT(*) as count
        FROM facts
        GROUP BY fidelity
        ORDER BY fidelity DESC
        """)
        
        fidelity_counts = {f"S{row['fidelity']}": row['count'] for row in cur.fetchall()}
        
        # Total facts
        cur.execute("SELECT COUNT(*) FROM facts")
        total_facts = cur.fetchone()[0]
        
        # Performance metrics
        return {
            'total_facts': total_facts,
            'fidelity_distribution': fidelity_counts,
            'reinforcements': self.reinforcements,
            'decays_applied': self.decays_applied,
            'facts_forgotten': self.facts_forgotten,
            'db_size_kb': self.db_path.stat().st_size // 1024 if self.db_path.exists() else 0
        }
    
    def _ema(self, prev: float, obs: float, alpha: float = EMA_ALPHA) -> float:
        """Exponential moving average"""
        return alpha * obs + (1 - alpha) * prev
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Utility functions
def extract_facts_from_text(text: str) -> List[Dict]:
    """
    Extract facts from natural language text using spaCy dependency parsing
    
    Uses pure linguistic analysis - NO LLM calls, NO hardcoded patterns.
    Returns structured facts in the format expected by FactsGraph.
    """
    try:
        # Import here to avoid circular dependencies
        from memory.spacy_fact_extractor import extract_facts_from_text as spacy_extract
        
        facts = spacy_extract(text)
        logger.debug(f"🔍 spaCy extracted {len(facts)} facts from text")
        return facts
        
    except Exception as e:
        logger.error(f"spaCy fact extraction failed: {e}")
        return []


# Self-test
if __name__ == "__main__":
    import tempfile
    import shutil
    
    def test_facts_graph():
        """Test Facts Graph functionality"""
        logger.info("🧠 Testing Facts Graph")
        
        # Use temporary database
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/test_facts.db"
            graph = FactsGraph(db_path)
            
            # Test fact extraction
            text = "My dog name is Potola and my cat is Whiskers. I live in San Francisco."
            facts = extract_facts_from_text(text)
            logger.info(f"Extracted {len(facts)} facts: {facts}")
            
            # Test fact insertion
            for fact in facts:
                graph.reinforce_or_insert(fact)
            
            # Test retrieval
            all_facts = graph.get_facts()
            logger.info(f"Stored {len(all_facts)} facts")
            
            # Test search
            dog_facts = graph.search_facts("dog name")
            logger.info(f"Dog facts: {dog_facts}")
            
            # Test reinforcement
            graph.reinforce_or_insert(facts[0])  # Reinforce first fact
            
            # Test decay
            graph.decay_facts()
            
            # Test stats
            stats = graph.get_stats()
            logger.info(f"Graph stats: {stats}")
            
            graph.close()
            logger.info("✅ Facts Graph test complete")
    
    test_facts_graph()
