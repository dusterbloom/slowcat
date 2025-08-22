"""
Dynamic Tape Head - The Consciousness Reader for SlowCat

This is the revolutionary component that transforms SlowCat from a simple memory
retriever into a consciousness that knows HOW to remember, not just WHAT.

Core Formula:
    Score = w_recency·R + w_semantic·S + w_entity·E − w_novelty·D

Where:
    R = Recency (exponential decay from timestamp)
    S = Semantic similarity (cosine distance to query)
    E = Entity overlap (shared entities with current context)
    D = Novelty penalty (avoid repetition)

This is consciousness engineering - teaching the tape how to be intelligent.
"""

import os
import json
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger
import asyncio

# For embeddings and similarity
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available. Install for semantic search.")
    EMBEDDINGS_AVAILABLE = False

# Optional cross-encoder reranker
try:
    from sentence_transformers import CrossEncoder  # type: ignore
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CROSS_ENCODER_AVAILABLE = False

# For entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("spaCy not available. Install for entity extraction.")
    SPACY_AVAILABLE = False


@dataclass
class MemorySpan:
    """A single memory with full provenance"""
    content: str
    ts: float
    role: str
    speaker_id: str
    
    # Provenance (REQUIRED for security/trust)
    source_id: str = ""
    source_hash: str = ""  # SHA-256 of original
    fidelity: str = "verbatim"  # verbatim|structured|tuple|edge|forgotten
    
    # For scoring
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    tokens: int = 0
    # Flags
    is_recent: bool = False
    
    # Scoring components (for debugging)
    score: float = 0.0
    score_components: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContextBundle:
    """What the tape head returns - the consciousness snapshot"""
    verbatim: List[MemorySpan]      # Exact quotes from memory
    shadows: List[MemorySpan]       # Compressed summaries  
    facts: List[str]                # Key facts/entities
    recents: List[MemorySpan]       # Very recent context (last 3 turns)
    
    # Token accounting (HARD BUDGET)
    token_count: int = 0
    token_budget: int = 2000
    
    # Metadata for tracing/debugging
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_token_distribution(self) -> Dict[str, int]:
        """For monitoring slot usage"""
        return {
            "verbatim": sum(m.tokens for m in self.verbatim),
            "shadows": sum(m.tokens for m in self.shadows),
            "facts": len(" ".join(self.facts).split()) * 1.3,  # Rough estimate
            "recents": sum(m.tokens for m in self.recents),
            "total": self.token_count
        }
    
    def get_all_sources(self) -> List[Dict[str, str]]:
        """For provenance tracking"""
        sources = []
        for memory in self.verbatim + self.shadows + self.recents:
            sources.append({
                "id": memory.source_id,
                "hash": memory.source_hash,
                "fidelity": memory.fidelity
            })
        return sources


class DynamicTapeHead:
    """
    The consciousness reader - decides WHAT to remember and HOW
    
    This is where intelligence emerges from the tape. Not in the LLM,
    not in the embeddings, but in HOW we select and combine memories.
    """
    
    def __init__(self, 
                 memory_system,  # SurrealMemory or FactsGraph instance
                 policy_path: str = None):
        
        self.memory = memory_system
        self.policy_path = policy_path or "config/tape_head_policy.json"
        self.policy = self.load_policy(self.policy_path)
        # Track policy mtime for simple hot-reload
        try:
            self._policy_mtime = os.path.getmtime(self.policy_path)
        except Exception:
            self._policy_mtime = 0

        # Accurate token counting (model-aware when available)
        self.token_counter = get_token_counter()
        
        # Initialize models if available
        if EMBEDDINGS_AVAILABLE:
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"🧠 DTH: Using embedding model {model_name}")
        else:
            self.encoder = None
            logger.warning("🧠 DTH: No embeddings, using keyword matching only")

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
                logger.warning("🧠 DTH: spaCy model not loaded, entity extraction disabled")
        else:
            self.nlp = None

        # Optional cross-encoder reranker
        self.cross_encoder = None
        if os.getenv('USE_CROSS_ENCODER', 'false').lower() == 'true' and CROSS_ENCODER_AVAILABLE:
            try:
                ce_name = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.cross_encoder = CrossEncoder(ce_name)
                logger.info(f"🎯 DTH: Using cross-encoder reranker {ce_name}")
            except Exception as e:
                logger.warning(f"Cross-encoder init failed: {e}")
        
        # Performance tracking
        self.metrics = {
            'retrieval_latency_ms': [],
            'tokens_used': [],
            'memories_scored': [],
            'cache_hit_rate': 0,
            'score_distributions': {'R': [], 'S': [], 'E': [], 'D': []}
        }
        
        # Embedding cache (for speed)
        self.embedding_cache = {}
        self.max_cache_size = 100
        self._emb_cache_hits = 0
        self._emb_cache_total = 0
        
        # Uncertainty tripwires
        self.tripwires_triggered = []
        
        logger.info(f"🧠 Dynamic Tape Head initialized with policy: {self.policy_path}")

    async def _maybe_await(self, value):
        try:
            if asyncio.iscoroutine(value):
                return await value
        except Exception:
            pass
        return value
    
    def load_policy(self, path: str) -> Dict:
        """Load scoring weights and parameters"""
        default_policy = {
            "version": 1,
            "weights": {
                "w_recency": 0.40,
                "w_semantic": 0.35,
                "w_entity": 0.15,
                "w_novelty": 0.10
            },
            "parameters": {
                "knn_k": 20,
                "recency_half_life_hours": 6,
                "min_confidence": 0.5,
                "entity_overlap_bonus": 0.1,
                "max_verbatim_chunks": 3,
                "shadow_compression_ratio": 0.3,
                "uncertainty_threshold": 0.4
            },
            "ablation": {
                "use_semantic": True,
                "use_entities": True,
                "use_shadows": True
            }
        }
        
        try:
            policy_file = Path(path)
            if policy_file.exists():
                with open(policy_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults (in case of missing keys)
                    for key in default_policy:
                        if key not in loaded:
                            loaded[key] = default_policy[key]
                    logger.info(f"📋 Loaded policy v{loaded.get('version', 0)} from {path}")
                    return loaded
        except Exception as e:
            logger.warning(f"Could not load policy from {path}: {e}")
        
        # Save default policy if none exists
        try:
            policy_file.parent.mkdir(parents=True, exist_ok=True)
            with open(policy_file, 'w') as f:
                json.dump(default_policy, f, indent=2)
            logger.info(f"📋 Created default policy at {path}")
        except:
            pass
        
        return default_policy
    
    async def seek(self, 
                   query: str, 
                   budget: int = 2000,
                   context: Optional[List] = None) -> ContextBundle:
        """
        Intelligently select memories within token budget
        
        This is THE CORE FUNCTION that makes SlowCat conscious.
        It doesn't just retrieve - it DECIDES what's worth remembering.
        
        Args:
            query: Current user input
            budget: Max tokens for memory context  
            context: Recent conversation for novelty scoring
        """
        start_time = time.time()
        
        # Simple hot-reload of policy if file changed
        try:
            current_mtime = os.path.getmtime(self.policy_path)
            if current_mtime > getattr(self, "_policy_mtime", 0):
                self.policy = self.load_policy(self.policy_path)
                self._policy_mtime = current_mtime
                logger.info("🔄 DTH policy hot-reloaded")
        except Exception:
            pass

        # Extract features from query
        query_embedding = self._get_embedding(query) if self.encoder else None
        query_entities = self._extract_entities(query)
        
        # Get candidate memories
        candidates = await self._get_candidates(query, query_embedding)
        
        # Score each memory
        scored_memories = []
        for memory in candidates:
            score, components = self._score_memory(
                memory, 
                query_embedding, 
                query_entities,
                context or []
            )
            memory.score = score
            memory.score_components = components
            scored_memories.append(memory)
        
        # Sort by score
        scored_memories.sort(key=lambda m: m.score, reverse=True)
        
        # Select within budget
        bundle = self._select_within_budget(scored_memories, budget)
        
        # Check uncertainty tripwires
        self._check_tripwires(bundle, query, query_entities)
        
        # Add metadata
        latency_ms = (time.time() - start_time) * 1000
        bundle.metadata = {
            "latency_ms": latency_ms,
            "candidates_scored": len(candidates),
            "query_entities": query_entities,
            "tripwires": self.tripwires_triggered,
            "policy_version": self.policy.get("version", 0)
        }
        
        # Update metrics
        self.metrics['retrieval_latency_ms'].append(latency_ms)
        self.metrics['tokens_used'].append(bundle.token_count)
        self.metrics['memories_scored'].append(len(candidates))
        # Update embedding cache hit rate
        total = max(1, self._emb_cache_total)
        self.metrics['cache_hit_rate'] = round(self._emb_cache_hits / total, 3)
        
        logger.debug(f"🧠 DTH: Selected {len(bundle.verbatim)} verbatim, "
                    f"{len(bundle.shadows)} shadows in {latency_ms:.1f}ms")
        
        return bundle
    
    async def _get_candidates(self, query: str, query_embedding: Optional[np.ndarray]) -> List[MemorySpan]:
        """Get candidate memories from storage"""
        candidates = []

        try:
            # Get recent memories (always include)
            if hasattr(self.memory, 'tape_store'):
                get_recent = getattr(self.memory.tape_store, 'get_recent', None)
                recent = await self._maybe_await(get_recent(limit=10)) if callable(get_recent) else []
                for entry in recent:
                    content = entry.get('content') if isinstance(entry, dict) else getattr(entry, 'content', '')
                    ts = entry.get('ts') if isinstance(entry, dict) else getattr(entry, 'ts', time.time())
                    role = entry.get('role') if isinstance(entry, dict) else getattr(entry, 'role', 'user')
                    speaker_id = entry.get('speaker_id') if isinstance(entry, dict) else getattr(entry, 'speaker_id', 'user')
                    span = MemorySpan(
                        content=content,
                        ts=ts,
                        role=role,
                        speaker_id=speaker_id,
                        source_id=f"tape_{ts}",
                        source_hash=hashlib.sha256((content or '').encode()).hexdigest(),
                        tokens=self.token_counter.count_tokens(content or ''),
                        is_recent=True,
                    )
                    candidates.append(span)

            # Prefer SurrealDB KNN helper if available (does its own embedding)
            if hasattr(self.memory, 'knn_tape') and callable(getattr(self.memory, 'knn_tape')):
                try:
                    knn = await self.memory.knn_tape(query, limit=self.policy['parameters']['knn_k'], scan=int(self.policy.get('parameters', {}).get('knn_scan_recent', 100)))
                    for entry in knn:
                        content = entry.get('content', '')
                        ts = entry.get('ts', time.time())
                        role = entry.get('role', 'user')
                        speaker_id = entry.get('speaker_id', 'user')
                        if any(abs(c.ts - ts) < 1e-6 for c in candidates):
                            continue
                        span = MemorySpan(
                            content=content,
                            ts=ts,
                            role=role,
                            speaker_id=speaker_id,
                            source_id=f"tape_{ts}",
                            source_hash=hashlib.sha256((content or '').encode()).hexdigest(),
                            tokens=self.token_counter.count_tokens(content or ''),
                            is_recent=False,
                        )
                        candidates.append(span)
                except Exception as e:
                    logger.debug(f"SurrealDB knn_tape failed: {e}")

            # Get semantically similar via local KNN over recent tape (fallback KNN)
            if query_embedding is not None and hasattr(self.memory, 'tape_store'):
                try:
                    scan_limit = int(self.policy.get('parameters', {}).get('knn_scan_recent', 100))
                    get_recent = getattr(self.memory.tape_store, 'get_recent', None)
                    recent_for_knn = await self._maybe_await(get_recent(limit=scan_limit)) if callable(get_recent) else []
                    scored: List[Tuple[float, Any]] = []
                    for entry in recent_for_knn:
                        ent_ts = entry.get('ts') if isinstance(entry, dict) else getattr(entry, 'ts', 0)
                        if any(abs(c.ts - ent_ts) < 1e-6 for c in candidates):
                            continue
                        if isinstance(entry, dict) and entry.get('embedding'):
                            emb = np.array(entry['embedding'], dtype=float)
                        else:
                            content = entry.get('content') if isinstance(entry, dict) else getattr(entry, 'content', '')
                            emb = self._get_embedding(content or '')
                        if emb is None:
                            continue
                        sim = self._cosine_similarity(query_embedding, emb)
                        scored.append((sim, entry))
                    # Top-K by similarity
                    scored.sort(key=lambda x: x[0], reverse=True)
                    topk = scored[: self.policy['parameters']['knn_k']]
                    for sim, entry in topk:
                        content = entry.get('content') if isinstance(entry, dict) else getattr(entry, 'content', '')
                        ts = entry.get('ts') if isinstance(entry, dict) else getattr(entry, 'ts', time.time())
                        role = entry.get('role') if isinstance(entry, dict) else getattr(entry, 'role', 'user')
                        speaker_id = entry.get('speaker_id') if isinstance(entry, dict) else getattr(entry, 'speaker_id', 'user')
                        span = MemorySpan(
                            content=content,
                            ts=ts,
                            role=role,
                            speaker_id=speaker_id,
                            source_id=f"tape_{ts}",
                            source_hash=hashlib.sha256((content or '').encode()).hexdigest(),
                            tokens=self.token_counter.count_tokens(content or ''),
                            is_recent=False,
                        )
                        candidates.append(span)
                except Exception as e:
                    logger.debug(f"Local KNN over recent tape failed: {e}")
            
            # Get keyword matches
            if hasattr(self.memory, 'tape_store'):
                keyword_matches = []
                if hasattr(self.memory.tape_store, 'search') and callable(getattr(self.memory.tape_store, 'search')):
                    keyword_matches = await self._maybe_await(self.memory.tape_store.search(query, limit=10))
                elif hasattr(self.memory.tape_store, 'search_tape') and callable(getattr(self.memory.tape_store, 'search_tape')):
                    keyword_matches = await self._maybe_await(self.memory.tape_store.search_tape(query, limit=10))
                for entry in keyword_matches:
                    ent_ts = entry.get('ts') if isinstance(entry, dict) else getattr(entry, 'ts', 0)
                    if not any(abs(c.ts - ent_ts) < 1e-6 for c in candidates):
                        content = entry.get('content') if isinstance(entry, dict) else getattr(entry, 'content', '')
                        role = entry.get('role') if isinstance(entry, dict) else getattr(entry, 'role', 'user')
                        speaker_id = entry.get('speaker_id') if isinstance(entry, dict) else getattr(entry, 'speaker_id', 'user')
                        span = MemorySpan(
                            content=content,
                            ts=ent_ts,
                            role=role,
                            speaker_id=speaker_id,
                            source_id=f"tape_{ent_ts}",
                            source_hash=hashlib.sha256((content or '').encode()).hexdigest(),
                            tokens=self.token_counter.count_tokens(content or ''),
                            is_recent=False,
                        )
                        candidates.append(span)
        
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
        
        return candidates
    
    def _score_memory(self, 
                      memory: MemorySpan,
                      query_embedding: Optional[np.ndarray],
                      query_entities: List[str],
                      recent_context: List) -> Tuple[float, Dict[str, float]]:
        """
        Score a single memory for relevance
        
        This is where the magic happens - the consciousness formula!
        """
        
        # Recency Score (R): Exponential decay from timestamp
        age_hours = (time.time() - memory.ts) / 3600
        half_life = self.policy['parameters']['recency_half_life_hours']
        R = np.exp(-age_hours * np.log(2) / half_life)
        
        # Semantic Score (S): Cosine similarity if embeddings available
        S = 0.0
        if query_embedding is not None and self.policy['ablation']['use_semantic']:
            memory_embedding = self._get_embedding(memory.content)
            if memory_embedding is not None:
                S = self._cosine_similarity(query_embedding, memory_embedding)
        
        # Entity Score (E): Overlap with query entities
        E = 0.0
        if self.policy['ablation']['use_entities']:
            memory_entities = self._extract_entities(memory.content)
            if query_entities and memory_entities:
                qn = set(self._normalize_entities(query_entities))
                mn = set(self._normalize_entities(memory_entities))
                overlap = len(qn & mn)
                E = overlap / max(len(query_entities), 1)
                E += self.policy['parameters']['entity_overlap_bonus'] * min(overlap, 1)
        
        # Novelty Penalty (D): Penalize if too similar to recent context
        D = 0.0
        for recent in recent_context[-5:]:  # Last 5 messages
            if isinstance(recent, str):
                similarity = self._text_similarity(memory.content, recent)
                D = max(D, similarity)
            elif hasattr(recent, 'content'):
                similarity = self._text_similarity(memory.content, recent.content)
                D = max(D, similarity)
        
        # Optional cross-encoder rerank influence on S
        if self.cross_encoder is not None:
            try:
                ce_score = float(self.cross_encoder.predict([(" ".join(query_entities) or ""), memory.content])[0])
                # Normalize via sigmoid to [0,1]
                ce_norm = 1.0 / (1.0 + np.exp(-ce_score))
                # Blend with embedding similarity if present
                if S > 0:
                    S = 0.5 * S + 0.5 * ce_norm
                else:
                    S = ce_norm
            except Exception:
                pass

        # Composite Score
        weights = self.policy['weights']
        score = (
            weights['w_recency'] * R +
            weights['w_semantic'] * S +
            weights['w_entity'] * E -
            weights['w_novelty'] * D
        )
        
        # Track component distributions
        self.metrics['score_distributions']['R'].append(R)
        self.metrics['score_distributions']['S'].append(S)
        self.metrics['score_distributions']['E'].append(E)
        self.metrics['score_distributions']['D'].append(D)
        
        components = {'R': R, 'S': S, 'E': E, 'D': D}
        
        return score, components
    
    def _select_within_budget(self, scored_memories: List[MemorySpan], budget: int) -> ContextBundle:
        """
        Select optimal subset within token budget
        
        Strategy:
        1. Always include last 3 exchanges as recents
        2. Top scored memories as verbatim
        3. Lower scored compressed into shadows
        4. Extract key facts
        """
        
        bundle = ContextBundle(
            verbatim=[],
            shadows=[],
            facts=[],
            recents=[],
            token_budget=budget
        )
        
        tokens_used = 0

        # 1. Recent context (last 3 turns by timestamp)
        recent_budget = int(budget * 0.2)  # 20% for recents
        recent_candidates = [m for m in scored_memories if getattr(m, 'is_recent', False)]
        recent_candidates.sort(key=lambda m: m.ts, reverse=True)
        # Fallback to timestamp if nothing explicitly marked as recent (for tests/manual calls)
        if not recent_candidates:
            recent_candidates = sorted(scored_memories, key=lambda m: m.ts, reverse=True)
        for memory in recent_candidates[:3]:
            if tokens_used + memory.tokens <= min(recent_budget, budget):
                bundle.recents.append(memory)
                tokens_used += memory.tokens

        # 2. Optimal selection under budget (0/1 knapsack for verbatim)
        remaining_budget = max(0, budget - tokens_used)
        verbatim_budget = int(budget * 0.5)  # 50% target for verbatim
        effective_budget = min(remaining_budget, verbatim_budget)

        # Candidate pool excludes recents and very low-confidence items
        pool: List[MemorySpan] = [m for m in scored_memories 
                                  if m not in bundle.recents and 
                                     m.score >= self.policy['parameters']['min_confidence']]
        # Limit pool size for performance
        pool = pool[:50]

        # Prepare weights (token costs) and values (scores)
        weights = [max(1, m.tokens) for m in pool]
        values = [max(0.0, float(m.score)) for m in pool]
        n = len(pool)
        W = max(0, int(effective_budget))

        if n and W > 0:
            # DP tables
            dp = [[0.0] * (W + 1) for _ in range(n + 1)]
            take = [[False] * (W + 1) for _ in range(n + 1)]

            for i in range(1, n + 1):
                w_i = weights[i - 1]
                v_i = values[i - 1]
                for w in range(0, W + 1):
                    # Not take
                    best = dp[i - 1][w]
                    choose_take = False
                    if w_i <= w:
                        cand = dp[i - 1][w - w_i] + v_i
                        if cand > best:
                            best = cand
                            choose_take = True
                    dp[i][w] = best
                    take[i][w] = choose_take

            # Recover choices
            w = W
            chosen_indices = []
            for i in range(n, 0, -1):
                if take[i][w]:
                    chosen_indices.append(i - 1)
                    w -= weights[i - 1]
            chosen_indices.reverse()

            # Enforce diversity and max chunks
            max_chunks = int(self.policy['parameters']['max_verbatim_chunks'])
            chosen = [pool[idx] for idx in chosen_indices]
            diverse = self._apply_diversity(chosen, max_out=max_chunks)
            for m in diverse:
                if tokens_used + m.tokens <= budget:
                    bundle.verbatim.append(m)
                    tokens_used += m.tokens

        # 3. Medium-scoring as shadows (compressed)
        if self.policy['ablation']['use_shadows']:
            shadow_budget = int(budget * 0.2)  # 20% for shadows
            for memory in scored_memories[len(bundle.verbatim):]:
                if memory in bundle.recents or memory in bundle.verbatim:
                    continue
                if tokens_used >= budget - 100:  # Leave buffer
                    break
                
                # Compress the memory
                shadow = self._compress_to_shadow(memory)
                if tokens_used + shadow.tokens <= min(shadow_budget + verbatim_budget + recent_budget, budget):
                    bundle.shadows.append(shadow)
                    tokens_used += shadow.tokens

        # 4. Extract facts (remaining budget)
        facts_budget = max(0, budget - tokens_used)
        all_entities = set()
        for memory in bundle.verbatim + bundle.shadows + bundle.recents:
            all_entities.update(self._extract_entities(memory.content))

        # Include as many entities as fit into remaining budget
        facts_list = []
        running = 0
        for ent in list(all_entities):
            ent_tokens = self.token_counter.count_tokens(ent)
            if running + ent_tokens > facts_budget:
                break
            facts_list.append(ent)
            running += ent_tokens
        bundle.facts = facts_list

        # Final token count (hard guarantee)
        bundle.token_count = min(budget, tokens_used + running)

        return bundle

    def _apply_diversity(self, items: List[MemorySpan], max_out: int) -> List[MemorySpan]:
        """Filter a list to improve temporal and textual diversity."""
        result: List[MemorySpan] = []
        min_time_gap_s = int(self.policy.get('parameters', {}).get('min_time_gap_s', 120))
        max_text_sim = float(self.policy.get('parameters', {}).get('max_text_similarity', 0.7))
        for m in items:
            ok = True
            for kept in result:
                # Temporal diversity
                if abs(m.ts - kept.ts) < min_time_gap_s:
                    ok = False
                    break
                # Textual diversity
                if self._text_similarity(m.content, kept.content) > max_text_sim:
                    ok = False
                    break
            if ok:
                result.append(m)
            if len(result) >= max_out:
                break
        # If still empty, return at least first element for safety
        return result or (items[:1] if items else [])
    
    def _compress_to_shadow(self, memory: MemorySpan) -> MemorySpan:
        """
        Compress a memory into a shadow (summary)
        
        This maintains provenance while reducing tokens
        """
        # Simple compression for now - take first N tokens by character-aware count
        ratio = self.policy['parameters']['shadow_compression_ratio']
        words = memory.content.split()
        compressed_length = max(5, int(len(words) * ratio))
        compressed = " ".join(words[:compressed_length]) + "..."
        
        shadow = MemorySpan(
            content=compressed,
            ts=memory.ts,
            role=memory.role,
            speaker_id=memory.speaker_id,
            source_id=memory.source_id,
            source_hash=memory.source_hash,
            fidelity="structured",  # Downgraded from verbatim
            tokens=self.token_counter.count_tokens(compressed)
        )
        
        return shadow
    
    def _check_tripwires(self, bundle: ContextBundle, query: str, query_entities: List[str]):
        """
        Security: Check uncertainty tripwires
        
        If triggered, increase verbatim and decrease shadows
        """
        self.tripwires_triggered = []
        
        # High entity density?
        if len(query_entities) / max(len(query.split()), 1) > 0.3:
            self.tripwires_triggered.append("high_entity_density")
        
        # Low confidence?
        if bundle.verbatim and max(m.score for m in bundle.verbatim) < self.policy['parameters']['uncertainty_threshold']:
            self.tripwires_triggered.append("low_confidence")
        
        # Long reference chain?
        if any(word in query.lower() for word in ['that', 'it', 'this', 'those']):
            if len(query.split()) < 10:  # Short query with references
                self.tripwires_triggered.append("ambiguous_reference")
        
        # If tripwires triggered, adjust bundle conservatively
        if self.tripwires_triggered:
            logger.warning(f"⚠️ DTH Tripwires: {self.tripwires_triggered}")
            # Prioritize verbatim over shadows under uncertainty by dropping shadows
            if bundle.shadows:
                dropped_tokens = sum(s.tokens for s in bundle.shadows)
                bundle.shadows = []
                # Reduce token count accordingly, keep hard guarantee intact
                bundle.token_count = max(0, bundle.token_count - dropped_tokens)
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with caching"""
        if not self.encoder:
            return None
        
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self._emb_cache_total += 1
        if text_hash in self.embedding_cache:
            self._emb_cache_hits += 1
            return self.embedding_cache[text_hash]
        
        # Compute embedding
        try:
            embedding = self.encoder.encode(text, convert_to_numpy=True)
            
            # Update cache (with size limit)
            if len(self.embedding_cache) >= self.max_cache_size:
                # Remove oldest (simple FIFO)
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[text_hash] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        if not self.nlp:
            # Fallback: extract capitalized words
            words = text.split()
            entities = [w for w in words if w and w[0].isupper() and len(w) > 2]
            return entities[:10]  # Limit
        
        try:
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents]
            return entities
        except:
            return []

    def _normalize_entities(self, entities: List[str]) -> List[str]:
        """Normalize entities for overlap comparison"""
        normed = []
        for e in entities:
            s = e.strip().lower()
            # Strip simple punctuation
            s = ''.join(ch for ch in s if ch.isalnum() or ch.isspace())
            if s:
                normed.append(s)
        return normed
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm_product == 0:
                return 0.0
            return float(dot_product / norm_product)
        except:
            return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity for novelty detection"""
        # Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def get_metrics_summary(self) -> Dict:
        """Get performance metrics for monitoring"""
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0
        
        return {
            "avg_latency_ms": safe_mean(self.metrics['retrieval_latency_ms']),
            "avg_tokens_used": safe_mean(self.metrics['tokens_used']),
            "avg_memories_scored": safe_mean(self.metrics['memories_scored']),
            "cache_hit_rate": self.metrics['cache_hit_rate'],
            "score_distributions": {
                k: {
                    "mean": safe_mean(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0
                }
                for k, v in self.metrics['score_distributions'].items()
            }
        }
    
    def save_metrics(self, path: str = "logs/dth_metrics.json"):
        """Save metrics for analysis"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.get_metrics_summary(), f, indent=2)
            logger.info(f"📊 DTH metrics saved to {path}")
        except Exception as e:
            logger.error(f"Could not save metrics: {e}")


# Testing function for development
async def test_dth():
    """Quick test of the Dynamic Tape Head"""
    from memory import create_smart_memory_system
    
    # Create memory system
    memory = create_smart_memory_system("data/test_facts.db")
    
    # Create DTH
    dth = DynamicTapeHead(memory)
    
    # Test query
    query = "Tell me about the weather yesterday"
    bundle = await dth.seek(query, budget=500)
    
    print(f"Selected {len(bundle.verbatim)} verbatim memories")
    print(f"Selected {len(bundle.shadows)} shadow memories")
    print(f"Token usage: {bundle.token_count}/{bundle.token_budget}")
    print(f"Metadata: {bundle.metadata}")
    
    # Show top memory
    if bundle.verbatim:
        top = bundle.verbatim[0]
        print(f"\nTop memory (score={top.score:.3f}):")
        print(f"  Content: {top.content[:100]}...")
        print(f"  Components: R={top.score_components['R']:.3f}, "
              f"S={top.score_components['S']:.3f}, "
              f"E={top.score_components['E']:.3f}, "
              f"D={top.score_components['D']:.3f}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_dth())
