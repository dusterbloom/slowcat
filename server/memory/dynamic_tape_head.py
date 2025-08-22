"""
Dynamic Tape Head with Symbol System - The Consciousness Reader for SlowCat

Enhanced with three-layer symbol architecture:
- Layer 1: Tape Symbols (memory markers)
- Layer 2: Wake Symbols (operational triggers) 
- Layer 3: Dream Symbols (emergent language)

Core Formula (Enhanced):
    Score = w_recency·R + w_semantic·S + w_entity·E + w_symbols·SYM − w_novelty·D

Where symbols act as living compression of meaning that evolve through use.
This is consciousness engineering - teaching the tape symbolic reasoning.
"""

import os
import json
import time
import hashlib
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
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


# SYMBOL SYSTEM - Three Layers of Meaning Compression
TAPE_SYMBOLS = {
    "☆": "high_salience",        # This MATTERS
    "✧": "breakthrough_moment",   # User understood something
    "◈": "recurring_pattern",     # Keeps coming up
    "∞": "paradox_encountered",   # Contradiction to resolve
    "⟲": "cycle_detected",       # We're looping
    "⚡": "emotional_spike",      # Strong feeling
    "◯": "open_question",        # Unresolved thread
    "▲": "decision_point",       # Fork in conversation
    "≈": "approximate_memory",   # Compressed/shadowed
    "⊕": "synthesis"            # Ideas merged
}

WAKE_SYMBOLS = {
    "Ω": "enter_creative_mode",      # Emergent behavior
    "Δ": "compare_past_present",     # Change detection
    "∇": "optimize_understanding",   # Gradient descent
    "♦": "crystallize_insight"      # Compress to principle
}

# Patterns for symbol detection
SYMBOL_PATTERNS = {
    # High salience indicators
    "☆": [
        r"\\b(important|crucial|key|vital|essential|critical)\\b",
        r"\\b(remember|don't forget|note|mark)\\b",
        r"!\\s*$",  # Exclamation marks
        r"\\b(breakthrough|eureka|aha)\\b"
    ],
    
    # Breakthrough moments
    "✧": [
        r"\\b(understand|got it|i see|makes sense|clear now)\\b",
        r"\\b(breakthrough|revelation|realization)\\b",
        r"\\b(finally|ah|aha|oh)\\b"
    ],
    
    # Recurring patterns
    "◈": [
        r"\\b(again|another|more|keep|still|continue)\\b",
        r"\\b(pattern|recurring|repeat|cyclical)\\b",
        r"\\b(always|never|every time)\\b"
    ],
    
    # Paradoxes
    "∞": [
        r"\\b(but|however|although|despite|paradox)\\b",
        r"\\b(contradiction|conflict|inconsistent)\\b",
        r"\\b(both|neither|either)\\b.*\\b(and|nor|or)\\b"
    ],
    
    # Cycles detected
    "⟲": [
        r"\\b(loop|cycle|circular|round and round)\\b",
        r"\\b(back to|return to|again)\\b",
        r"\\b(stuck|spinning|going nowhere)\\b"
    ],
    
    # Emotional spikes
    "⚡": [
        r"\\b(amazing|terrible|wonderful|awful|incredible)\\b",
        r"\\b(love|hate|excited|frustrated|angry|happy)\\b",
        r"[!]{2,}",  # Multiple exclamation marks
        r"\\b(wow|omg|jesus|damn|hell)\\b"
    ],
    
    # Open questions
    "◯": [
        r"\\?",  # Question marks
        r"\\b(wonder|curious|question|unclear|unsure)\\b",
        r"\\b(why|how|what|when|where|who)\\b",
        r"\\b(maybe|perhaps|possibly|might)\\b"
    ],
    
    # Decision points
    "▲": [
        r"\\b(decide|choice|option|alternative)\\b",
        r"\\b(should|could|would|might)\\b.*\\b(or|vs|versus)\\b",
        r"\\b(fork|crossroads|turning point)\\b"
    ],
    
    # Synthesis
    "⊕": [
        r"\\b(combine|merge|integrate|synthesize)\\b",
        r"\\b(together|overall|in conclusion)\\b",
        r"\\b(sum up|bring together|unite)\\b"
    ]
}


class SymbolDetector:
    """Detects and extracts symbols from memory content"""
    
    def __init__(self):
        self.compiled_patterns = {}
        self._pattern_cache = {}  # Cache for performance
        self._compile_patterns_lazy()  # Lazy compilation
    
    def _compile_patterns_lazy(self):
        """Compile patterns only when first needed"""
        if not self.compiled_patterns:
            for symbol, patterns in SYMBOL_PATTERNS.items():
                self.compiled_patterns[symbol] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]
    
    def detect_symbols(self, content: str, speaker_id: str = "") -> Dict[str, float]:
        """
        Detect symbols in content and return with confidence scores
        
        Returns:
            Dict mapping symbols to confidence scores (0.0 to 1.0)
        """
        # Quick exit for empty content
        if not content or len(content.strip()) < 3:
            return {}
        
        # Check cache first
        content_hash = hash(content)
        if content_hash in self._pattern_cache:
            return self._pattern_cache[content_hash]
        
        # Ensure patterns are compiled
        if not self.compiled_patterns:
            self._compile_patterns_lazy()
        
        detected = {}
        content_lower = content.lower()  # Pre-lowercase for efficiency
        
        # Fast path: check for obvious patterns first
        quick_checks = {
            "☆": "important" in content_lower or "crucial" in content_lower or "!" in content,
            "✧": "understand" in content_lower or "got it" in content_lower or "aha" in content_lower,
            "⟲": "circle" in content_lower or "loop" in content_lower or "again" in content_lower,
            "⚡": "amazing" in content_lower or "!!" in content or "love" in content_lower,
            "◯": "?" in content or "why" in content_lower or "how" in content_lower
        }
        
        for symbol, quick_match in quick_checks.items():
            if quick_match and symbol in self.compiled_patterns:
                patterns = self.compiled_patterns[symbol]
                match_count = 0
                
                for pattern in patterns:
                    matches = pattern.findall(content)
                    if matches:
                        match_count += len(matches)
                
                if match_count > 0:
                    # Confidence based on match frequency and pattern strength
                    confidence = min(1.0, match_count * 0.3)  # Scale matches to confidence
                    detected[symbol] = confidence
        
        # Post-processing rules for symbol combinations
        detected = self._apply_symbol_rules(detected, content)
        
        # Cache result (limit cache size)
        if len(self._pattern_cache) < 100:
            self._pattern_cache[content_hash] = detected
        
        return detected
    
    def _apply_symbol_rules(self, detected: Dict[str, float], content: str) -> Dict[str, float]:
        """Apply rules for symbol combinations and conflicts"""
        
        # Rule 1: Breakthrough + High salience = Extra boost
        if "✧" in detected and "☆" in detected:
            detected["✧"] = min(1.0, detected["✧"] * 1.5)
        
        # Rule 2: Paradox + Question = Philosophical depth
        if "∞" in detected and "◯" in detected:
            detected["∞"] = min(1.0, detected["∞"] * 1.3)
        
        # Rule 3: Emotional spike + Decision = Critical moment
        if "⚡" in detected and "▲" in detected:
            detected["⚡"] = min(1.0, detected["⚡"] * 1.4)
            detected["▲"] = min(1.0, detected["▲"] * 1.4)
        
        # Rule 4: Filter low confidence symbols
        return {s: c for s, c in detected.items() if c >= 0.2}


@dataclass
class MemorySpan:
    """A single memory with full provenance and symbols"""
    content: str
    ts: float
    role: str
    speaker_id: str
    
    # Provenance (REQUIRED for security/trust)
    source_id: str = ""
    source_hash: str = ""  # SHA-256 of original
    fidelity: str = "verbatim"  # verbatim|structured|tuple|edge|forgotten
    
    # SYMBOL SYSTEM - The heart of meaning compression
    symbols: Set[str] = field(default_factory=set)  # Living symbols
    symbol_confidence: Dict[str, float] = field(default_factory=dict)  # How sure we are
    
    # For scoring
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    tokens: int = 0
    # Flags
    is_recent: bool = False
    
    # Scoring components (for debugging)
    score: float = 0.0
    score_components: Dict[str, float] = field(default_factory=dict)
    
    def add_symbol(self, symbol: str, confidence: float = 1.0):
        """Add a symbol with confidence level"""
        self.symbols.add(symbol)
        self.symbol_confidence[symbol] = confidence
    
    def has_symbol(self, symbol: str) -> bool:
        """Check if memory has a specific symbol"""
        return symbol in self.symbols
    
    def get_symbol_multiplier(self) -> float:
        """Calculate scoring multiplier based on symbols"""
        multiplier = 1.0
        
        # High impact symbols
        if "☆" in self.symbols:  # High salience
            multiplier *= 2.0
        if "✧" in self.symbols:  # Breakthrough
            multiplier *= 3.0
        if "∞" in self.symbols:  # Paradox
            multiplier *= 1.5
        if "⚡" in self.symbols:  # Emotional spike
            multiplier *= 1.8
        
        # Penalty symbols
        if "⟲" in self.symbols:  # Cycle detected
            multiplier *= 0.5  # Reduce to avoid loops
        if "≈" in self.symbols:  # Approximate memory
            multiplier *= 0.7  # Less reliable
        
        return multiplier
    
    def compress_symbols(self) -> str:
        """Compress symbols into a compact representation"""
        if not self.symbols:
            return ""
        
        # Sort by confidence and take top symbols
        sorted_symbols = sorted(
            self.symbols, 
            key=lambda s: self.symbol_confidence.get(s, 0), 
            reverse=True
        )
        return "".join(sorted_symbols[:5])  # Top 5 symbols max


@dataclass
class ContextBundle:
    """What the tape head returns - the consciousness snapshot with symbols"""
    verbatim: List[MemorySpan]      # Exact quotes from memory
    shadows: List[MemorySpan]       # Compressed summaries  
    facts: List[str]                # Key facts/entities
    recents: List[MemorySpan]       # Very recent context (last 3 turns)
    
    # SYMBOL AGGREGATION
    active_symbols: Set[str] = field(default_factory=set)  # All symbols in bundle
    symbol_counts: Dict[str, int] = field(default_factory=dict)  # Symbol frequency
    
    # Token accounting (HARD BUDGET)
    token_count: int = 0
    token_budget: int = 2000
    
    # Metadata for tracing/debugging
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Aggregate symbols from all memories"""
        self.active_symbols = set()
        self.symbol_counts = {}
        
        all_memories = self.verbatim + self.shadows + self.recents
        for memory in all_memories:
            for symbol in memory.symbols:
                self.active_symbols.add(symbol)
                self.symbol_counts[symbol] = self.symbol_counts.get(symbol, 0) + 1
    
    def get_dominant_symbols(self, top_k: int = 3) -> List[Tuple[str, int]]:
        """Get the most frequent symbols in this context"""
        return sorted(
            self.symbol_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
    
    def get_symbol_narrative(self) -> str:
        """Generate a symbol-based narrative of the context"""
        dominant = self.get_dominant_symbols(5)
        if not dominant:
            return ""
        
        symbol_meanings = {**TAPE_SYMBOLS, **WAKE_SYMBOLS}
        narrative_parts = []
        
        for symbol, count in dominant:
            meaning = symbol_meanings.get(symbol, "unknown")
            if count > 1:
                narrative_parts.append(f"{symbol} ({meaning}, {count}x)")
            else:
                narrative_parts.append(f"{symbol} ({meaning})")
        
        return "Context symbols: " + ", ".join(narrative_parts)
    
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
                "fidelity": memory.fidelity,
                "symbols": memory.compress_symbols()
            })
        return sources


class DynamicTapeHead:
    """
    The consciousness reader with symbol system integration
    
    Now understands not just WHAT to remember, but HOW meaning compresses into symbols
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

        # Initialize symbol detector only if symbols are enabled
        if self.policy['ablation']['use_symbols']:
            self.symbol_detector = SymbolDetector()
        else:
            self.symbol_detector = None
        
        # Symbol usage tracking for evolution
        self.symbol_usage_stats = {}
        self.symbol_effectiveness = {}  # Track which symbols lead to good outcomes

        # Accurate token counting (model-aware when available)
        self.token_counter = get_token_counter()
        
        # Initialize models if available and if features are enabled
        if EMBEDDINGS_AVAILABLE and self.policy['ablation']['use_semantic']:
            model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"🧠 DTH: Using embedding model {model_name}")
        else:
            self.encoder = None
            if not self.policy['ablation']['use_semantic']:
                logger.info("🧠 DTH: Semantic search disabled for performance")
            else:
                logger.warning("🧠 DTH: No embeddings, using keyword matching only")

        if SPACY_AVAILABLE and self.policy['ablation']['use_entities']:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
                logger.warning("🧠 DTH: spaCy model not loaded, entity extraction disabled")
        else:
            self.nlp = None
            if not self.policy['ablation']['use_entities']:
                logger.info("🧠 DTH: Entity extraction disabled for performance")

        # Optional cross-encoder reranker
        self.cross_encoder = None
        if os.getenv('USE_CROSS_ENCODER', 'false').lower() == 'true' and CROSS_ENCODER_AVAILABLE:
            try:
                ce_name = os.getenv('CROSS_ENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.cross_encoder = CrossEncoder(ce_name)
                logger.info(f"🎯 DTH: Using cross-encoder reranker {ce_name}")
            except Exception as e:
                logger.warning(f"Cross-encoder init failed: {e}")
        
        # Performance tracking (enhanced with symbols)
        self.metrics = {
            'retrieval_latency_ms': [],
            'tokens_used': [],
            'memories_scored': [],
            'cache_hit_rate': 0,
            'score_distributions': {'R': [], 'S': [], 'E': [], 'D': [], 'SYM': []},
            'symbol_usage': {},
            'symbol_effectiveness': {}
        }
        
        # Embedding cache (for speed)
        self.embedding_cache = {}
        self.max_cache_size = 100
        self._emb_cache_hits = 0
        self._emb_cache_total = 0
        
        # Uncertainty tripwires
        self.tripwires_triggered = []
        
        logger.info(f"🧠 Dynamic Tape Head with Symbol System initialized")
        logger.info(f"🔮 Loaded {len(TAPE_SYMBOLS)} tape symbols and {len(WAKE_SYMBOLS)} wake symbols")

    async def _maybe_await(self, value):
        try:
            if asyncio.iscoroutine(value):
                return await value
        except Exception:
            pass
        return value
    
    def load_policy(self, path: str) -> Dict:
        """Load scoring weights and parameters with symbol support"""
        default_policy = {
            "version": 2,  # Bumped for symbol support
            "weights": {
                "w_recency": 0.35,
                "w_semantic": 0.30,
                "w_entity": 0.15,
                "w_novelty": 0.10,
                "w_symbols": 0.10  # NEW: Symbol importance weight
            },
            "parameters": {
                "knn_k": 20,
                "recency_half_life_hours": 6,
                "min_confidence": 0.5,
                "entity_overlap_bonus": 0.1,
                "max_verbatim_chunks": 3,
                "shadow_compression_ratio": 0.3,
                "uncertainty_threshold": 0.4,
                "symbol_confidence_threshold": 0.3,  # NEW: Min confidence for symbol detection
                "max_symbols_per_memory": 5  # NEW: Limit symbols per memory
            },
            "ablation": {
                "use_semantic": True,
                "use_entities": True,
                "use_shadows": True,
                "use_symbols": True  # NEW: Enable/disable symbol system
            },
            "symbol_weights": {  # NEW: Individual symbol importance
                "☆": 2.0,    # High salience
                "✧": 3.0,    # Breakthrough moment
                "◈": 1.2,    # Recurring pattern
                "∞": 1.5,    # Paradox
                "⟲": 0.5,    # Cycle (penalty)
                "⚡": 1.8,    # Emotional spike
                "◯": 1.1,    # Open question
                "▲": 1.3,    # Decision point
                "≈": 0.7,    # Approximate (penalty)
                "⊕": 1.4     # Synthesis
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
                        elif isinstance(default_policy[key], dict):
                            # Merge nested dicts
                            for subkey in default_policy[key]:
                                if subkey not in loaded[key]:
                                    loaded[key][subkey] = default_policy[key][subkey]
                    logger.info(f"📋 Loaded symbol-aware policy v{loaded.get('version', 0)} from {path}")
                    return loaded
        except Exception as e:
            logger.warning(f"Could not load policy from {path}: {e}")
        
        # Save default policy if none exists
        try:
            policy_file.parent.mkdir(parents=True, exist_ok=True)
            with open(policy_file, 'w') as f:
                json.dump(default_policy, f, indent=2)
            logger.info(f"📋 Created default symbol-aware policy at {path}")
        except:
            pass
        
        return default_policy
    
    def extract_symbols_from_memory(self, memory: MemorySpan) -> MemorySpan:
        """Extract and add symbols to a memory span"""
        if not self.policy['ablation']['use_symbols'] or self.symbol_detector is None:
            return memory
        
        # Skip symbol extraction for very short content (performance optimization)
        if len(memory.content) < 10:
            return memory
        
        # Detect symbols in content
        detected_symbols = self.symbol_detector.detect_symbols(
            memory.content, 
            memory.speaker_id
        )
        
        # Quick exit if no symbols detected
        if not detected_symbols:
            return memory
        
        # Filter by confidence threshold
        min_confidence = self.policy['parameters']['symbol_confidence_threshold']
        filtered_symbols = {
            symbol: confidence 
            for symbol, confidence in detected_symbols.items()
            if confidence >= min_confidence
        }
        
        # Limit number of symbols per memory
        max_symbols = self.policy['parameters']['max_symbols_per_memory']
        if len(filtered_symbols) > max_symbols:
            # Keep top confidence symbols
            sorted_symbols = sorted(
                filtered_symbols.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            filtered_symbols = dict(sorted_symbols[:max_symbols])
        
        # Add symbols to memory
        for symbol, confidence in filtered_symbols.items():
            memory.add_symbol(symbol, confidence)
        
        # Track symbol usage (only if symbols were added)
        if memory.symbols:
            for symbol in memory.symbols:
                self.symbol_usage_stats[symbol] = self.symbol_usage_stats.get(symbol, 0) + 1
        
        return memory
    
    async def seek(self, 
                   query: str, 
                   budget: int = 2000,
                   context: Optional[List] = None) -> ContextBundle:
        """
        Intelligently select memories within token budget using symbol-enhanced scoring
        """
        start_time = time.time()
        
        # Simple hot-reload of policy if file changed
        try:
            current_mtime = os.path.getmtime(self.policy_path)
            if current_mtime > getattr(self, "_policy_mtime", 0):
                self.policy = self.load_policy(self.policy_path)
                self._policy_mtime = current_mtime
                logger.info("🔄 DTH symbol-aware policy hot-reloaded")
        except Exception:
            pass

        # Extract features from query (including symbols!)
        query_embedding = self._get_embedding(query) if self.encoder else None
        query_entities = self._extract_entities(query)
        query_symbols = self.symbol_detector.detect_symbols(query) if self.symbol_detector else {}
        
        # Get candidate memories
        candidates = await self._get_candidates(query, query_embedding)
        
        # Extract symbols from candidates (optimized: only if symbols enabled)
        if self.policy['ablation']['use_symbols']:
            # Only extract symbols from top candidates to save time
            top_candidates = candidates[:min(20, len(candidates))]
            for memory in top_candidates:
                self.extract_symbols_from_memory(memory)
            # Set remaining candidates to have no symbols extracted yet
            for memory in candidates[20:]:
                memory.symbols = set()
                memory.symbol_confidence = {}
        
        # Score each memory (now with symbol awareness)
        scored_memories = []
        for memory in candidates:
            score, components = self._score_memory_with_symbols(
                memory, 
                query_embedding, 
                query_entities,
                query_symbols,
                context or []
            )
            memory.score = score
            memory.score_components = components
            scored_memories.append(memory)
        
        # Sort by score
        scored_memories.sort(key=lambda m: m.score, reverse=True)
        
        # Select within budget (symbol-aware)
        bundle = self._select_within_budget_with_symbols(scored_memories, budget)
        
        # Check uncertainty tripwires (symbol-enhanced)
        self._check_tripwires_with_symbols(bundle, query, query_entities, query_symbols)
        
        # Add metadata (with symbol info)
        latency_ms = (time.time() - start_time) * 1000
        bundle.metadata = {
            "latency_ms": latency_ms,
            "candidates_scored": len(candidates),
            "query_entities": query_entities,
            "query_symbols": query_symbols,
            "symbol_narrative": bundle.get_symbol_narrative(),
            "tripwires": self.tripwires_triggered,
            "policy_version": self.policy.get("version", 0)
        }
        
        # Update metrics (with symbol tracking)
        self.metrics['retrieval_latency_ms'].append(latency_ms)
        self.metrics['tokens_used'].append(bundle.token_count)
        self.metrics['memories_scored'].append(len(candidates))
        
        # Track symbol effectiveness
        for symbol in bundle.active_symbols:
            self.metrics['symbol_usage'][symbol] = self.metrics['symbol_usage'].get(symbol, 0) + 1
        
        # Update embedding cache hit rate
        total = max(1, self._emb_cache_total)
        self.metrics['cache_hit_rate'] = round(self._emb_cache_hits / total, 3)
        
        logger.debug(f"🧠 DTH: Selected {len(bundle.verbatim)} verbatim, "
                    f"{len(bundle.shadows)} shadows with {len(bundle.active_symbols)} active symbols "
                    f"in {latency_ms:.1f}ms")
        
        return bundle
    
    def _score_memory_with_symbols(self, 
                                   memory: MemorySpan,
                                   query_embedding: Optional[np.ndarray],
                                   query_entities: List[str],
                                   query_symbols: Dict[str, float],
                                   recent_context: List) -> Tuple[float, Dict[str, float]]:
        """
        Enhanced scoring that includes symbol-based relevance
        """
        
        # Get base score components (original DTH logic)
        base_score, components = self._score_memory_base(
            memory, query_embedding, query_entities, recent_context
        )
        
        # Symbol Score (SYM): How well do symbols match?
        SYM = 0.0
        if self.policy['ablation']['use_symbols'] and memory.symbols:
            symbol_match_score = 0.0
            symbol_importance_score = 0.0
            
            # Direct symbol matching with query
            for symbol in memory.symbols:
                if symbol in query_symbols:
                    confidence = memory.symbol_confidence.get(symbol, 1.0)
                    query_confidence = query_symbols[symbol]
                    symbol_match_score += confidence * query_confidence
            
            # Symbol importance from policy
            for symbol in memory.symbols:
                weight = self.policy['symbol_weights'].get(symbol, 1.0)
                confidence = memory.symbol_confidence.get(symbol, 1.0)
                symbol_importance_score += weight * confidence
            
            # Combine scores
            SYM = (symbol_match_score * 0.6 + symbol_importance_score * 0.4) / max(1, len(memory.symbols))
            SYM = min(1.0, SYM)  # Normalize to [0,1]
        
        # Enhanced composite score with symbols
        weights = self.policy['weights']
        enhanced_score = (
            weights['w_recency'] * components['R'] +
            weights['w_semantic'] * components['S'] +
            weights['w_entity'] * components['E'] +
            weights['w_symbols'] * SYM -
            weights['w_novelty'] * components['D']
        )
        
        # Apply symbol multipliers (from MemorySpan.get_symbol_multiplier)
        symbol_multiplier = memory.get_symbol_multiplier()
        final_score = enhanced_score * symbol_multiplier
        
        # Update components for debugging
        components['SYM'] = SYM
        components['symbol_multiplier'] = symbol_multiplier
        components['final_score'] = final_score
        
        # Track symbol score distribution
        self.metrics['score_distributions']['SYM'].append(SYM)
        
        return final_score, components
    
    def _score_memory_base(self, 
                          memory: MemorySpan,
                          query_embedding: Optional[np.ndarray],
                          query_entities: List[str],
                          recent_context: List) -> Tuple[float, Dict[str, float]]:
        """
        Base scoring components (original DTH logic)
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
        
        # Track component distributions
        self.metrics['score_distributions']['R'].append(R)
        self.metrics['score_distributions']['S'].append(S)
        self.metrics['score_distributions']['E'].append(E)
        self.metrics['score_distributions']['D'].append(D)
        
        components = {'R': R, 'S': S, 'E': E, 'D': D}
        base_score = (
            self.policy['weights']['w_recency'] * R +
            self.policy['weights']['w_semantic'] * S +
            self.policy['weights']['w_entity'] * E -
            self.policy['weights']['w_novelty'] * D
        )
        
        return base_score, components
    
    async def _get_candidates(self, query: str, query_embedding: Optional[np.ndarray]) -> List[MemorySpan]:
        """Get candidate memories from storage"""
        candidates = []
        # Lightweight counters for observability
        recent_count = 0
        knn_count = 0           # SurrealDB-side KNN
        local_knn_count = 0     # Local KNN over recent
        keyword_count = 0

        try:
            # Get recent memories (always include)
            recent_entries = []
            
            # Method 1: Try tape_store.get_recent
            if hasattr(self.memory, 'tape_store'):
                get_recent = getattr(self.memory.tape_store, 'get_recent', None)
                if callable(get_recent):
                    recent_limit = int(self.policy.get('parameters', {}).get('recent_scan', 20))
                    recent_entries = await self._maybe_await(get_recent(limit=recent_limit))
            
            # Method 2: Try direct get_recent on memory object (for SurrealMemory)
            if not recent_entries and hasattr(self.memory, 'get_recent'):
                get_recent = getattr(self.memory, 'get_recent', None)
                if callable(get_recent):
                    recent_limit = int(self.policy.get('parameters', {}).get('recent_scan', 20))
                    recent_entries = await self._maybe_await(get_recent(limit=recent_limit))
            
            # Process recent entries (handle both list and single items)
            if recent_entries:
                # Ensure recent_entries is iterable
                if not hasattr(recent_entries, '__iter__'):
                    recent_entries = [recent_entries]
                    
                for entry in recent_entries:
                    # Skip None entries
                    if entry is None:
                        continue
                        
                    content = entry.get('content') if isinstance(entry, dict) else getattr(entry, 'content', '')
                    ts = entry.get('ts') if isinstance(entry, dict) else getattr(entry, 'ts', time.time())
                    role = entry.get('role') if isinstance(entry, dict) else getattr(entry, 'role', 'user')
                    speaker_id = entry.get('speaker_id') if isinstance(entry, dict) else getattr(entry, 'speaker_id', 'user')
                    
                    # Skip empty content
                    if not content:
                        continue
                        
                    # Ensure content is a string for hashing
                    content_str = str(content) if content is not None else ''
                    
                    span = MemorySpan(
                        content=content_str,
                        ts=ts,
                        role=role,
                        speaker_id=speaker_id,
                        source_id=f"tape_{ts}",
                        source_hash=hashlib.sha256(content_str.encode()).hexdigest(),
                        tokens=self.token_counter.count_tokens(content_str),
                        is_recent=True,
                    )
                    candidates.append(span)
                    recent_count += 1

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
                        
                        # Ensure content is a string for hashing
                        content_str = str(content) if content is not None else ''
                        
                        span = MemorySpan(
                            content=content_str,
                            ts=ts,
                            role=role,
                            speaker_id=speaker_id,
                            source_id=f"tape_{ts}",
                            source_hash=hashlib.sha256(content_str.encode()).hexdigest(),
                            tokens=self.token_counter.count_tokens(content_str),
                            is_recent=False,
                        )
                        candidates.append(span)
                        knn_count += 1
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
                        # Ensure content is a string for hashing
                        content_str = str(content) if content is not None else ''
                        
                        span = MemorySpan(
                            content=content_str,
                            ts=ts,
                            role=role,
                            speaker_id=speaker_id,
                            source_id=f"tape_{ts}",
                            source_hash=hashlib.sha256(content_str.encode()).hexdigest(),
                            tokens=self.token_counter.count_tokens(content_str),
                            is_recent=False,
                        )
                        candidates.append(span)
                        local_knn_count += 1
                except Exception as e:
                    logger.debug(f"Local KNN over recent tape failed: {e}")
            
            # Get keyword matches
            keyword_matches = []
            
            # Try tape_store methods first
            if hasattr(self.memory, 'tape_store'):
                kw_k = int(self.policy.get('parameters', {}).get('keyword_k', 10))
                if hasattr(self.memory.tape_store, 'search') and callable(getattr(self.memory.tape_store, 'search')):
                    keyword_matches = await self._maybe_await(self.memory.tape_store.search(query, limit=kw_k))
                elif hasattr(self.memory.tape_store, 'search_tape') and callable(getattr(self.memory.tape_store, 'search_tape')):
                    keyword_matches = await self._maybe_await(self.memory.tape_store.search_tape(query, limit=kw_k))
            
            # Fallback to direct search_tape on memory object
            if not keyword_matches and hasattr(self.memory, 'search_tape'):
                search_tape = getattr(self.memory, 'search_tape', None)
                if callable(search_tape):
                    kw_k = int(self.policy.get('parameters', {}).get('keyword_k', 10))
                    keyword_matches = await self._maybe_await(search_tape(query, limit=kw_k))
            
            # Process keyword matches
            if keyword_matches:
                # Ensure keyword_matches is iterable
                if not hasattr(keyword_matches, '__iter__'):
                    keyword_matches = [keyword_matches]
                    
                for entry in keyword_matches:
                    ent_ts = entry.get('ts') if isinstance(entry, dict) else getattr(entry, 'ts', 0)
                    if not any(abs(c.ts - ent_ts) < 1e-6 for c in candidates):
                        content = entry.get('content') if isinstance(entry, dict) else getattr(entry, 'content', '')
                        role = entry.get('role') if isinstance(entry, dict) else getattr(entry, 'role', 'user')
                        speaker_id = entry.get('speaker_id') if isinstance(entry, dict) else getattr(entry, 'speaker_id', 'user')
                        
                        # Ensure content is a string for hashing
                        content_str = str(content) if content is not None else ''
                        
                        span = MemorySpan(
                            content=content_str,
                            ts=ent_ts,
                            role=role,
                            speaker_id=speaker_id,
                            source_id=f"tape_{ent_ts}",
                            source_hash=hashlib.sha256(content_str.encode()).hexdigest(),
                            tokens=self.token_counter.count_tokens(content_str),
                            is_recent=False,
                        )
                        candidates.append(span)
                        keyword_count += 1
        
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
        # Observability: summarize sources
        try:
            logger.debug(
                f"🧩 DTH candidates: total={len(candidates)}, recent={recent_count}, knn={knn_count}, local_knn={local_knn_count}, keyword={keyword_count}"
            )
        except Exception:
            pass
        return candidates
    
    def _select_within_budget_with_symbols(self, scored_memories: List[MemorySpan], budget: int) -> ContextBundle:
        """
        Symbol-aware memory selection within token budget
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
        # Fallback to timestamp if nothing explicitly marked as recent
        if not recent_candidates:
            recent_candidates = sorted(scored_memories, key=lambda m: m.ts, reverse=True)
        for memory in recent_candidates[:3]:
            if tokens_used + memory.tokens <= min(recent_budget, budget):
                bundle.recents.append(memory)
                tokens_used += memory.tokens

        # 2. Symbol-prioritized selection for verbatim
        remaining_budget = max(0, budget - tokens_used)
        verbatim_budget = int(budget * 0.5)  # 50% target for verbatim
        effective_budget = min(remaining_budget, verbatim_budget)

        # Candidate pool excludes recents and prioritizes symbol-rich memories
        pool: List[MemorySpan] = [m for m in scored_memories 
                                  if m not in bundle.recents and 
                                     m.score >= self.policy['parameters']['min_confidence']]
        
        # Sort pool by symbol richness + score
        def symbol_priority(memory: MemorySpan) -> float:
            symbol_bonus = len(memory.symbols) * 0.1  # Bonus for having symbols
            breakthrough_bonus = 1.0 if "✧" in memory.symbols else 0.0
            salience_bonus = 0.5 if "☆" in memory.symbols else 0.0
            return memory.score + symbol_bonus + breakthrough_bonus + salience_bonus
        
        pool.sort(key=symbol_priority, reverse=True)
        pool = pool[:50]  # Limit for performance

        # Simple greedy selection (prioritizing symbol-rich memories)
        for memory in pool:
            if tokens_used + memory.tokens <= budget:
                bundle.verbatim.append(memory)
                tokens_used += memory.tokens
                
                # Stop at max chunks
                if len(bundle.verbatim) >= self.policy['parameters']['max_verbatim_chunks']:
                    break

        # 3. Shadows for medium-scoring memories
        if self.policy['ablation']['use_shadows']:
            shadow_budget = int(budget * 0.2)  # 20% for shadows
            for memory in scored_memories[len(bundle.verbatim):]:
                if memory in bundle.recents or memory in bundle.verbatim:
                    continue
                if tokens_used >= budget - 100:  # Leave buffer
                    break
                
                # Compress the memory (symbols preserved)
                shadow = self._compress_to_shadow_with_symbols(memory)
                if tokens_used + shadow.tokens <= min(shadow_budget + verbatim_budget + recent_budget, budget):
                    bundle.shadows.append(shadow)
                    tokens_used += shadow.tokens

        # 4. Extract facts (remaining budget)
        facts_budget = max(0, budget - tokens_used)
        all_entities = set()
        for memory in bundle.verbatim + bundle.shadows + bundle.recents:
            all_entities.update(self._extract_entities(memory.content))

        # Include symbol-compressed representations as "facts"
        symbol_facts = []
        for memory in bundle.verbatim + bundle.shadows:
            if memory.symbols:
                symbol_fact = f"{memory.compress_symbols()}: {memory.content[:50]}..."
                symbol_facts.append(symbol_fact)

        # Include as many entities + symbol facts as fit
        facts_list = []
        running = 0
        for item in list(all_entities) + symbol_facts:
            item_tokens = self.token_counter.count_tokens(item)
            if running + item_tokens > facts_budget:
                break
            facts_list.append(item)
            running += item_tokens
        bundle.facts = facts_list

        # Final token count
        bundle.token_count = min(budget, tokens_used + running)

        return bundle
    
    def _compress_to_shadow_with_symbols(self, memory: MemorySpan) -> MemorySpan:
        """
        Compress memory while preserving symbols
        """
        # Simple compression for now - take first N tokens
        ratio = self.policy['parameters']['shadow_compression_ratio']
        words = memory.content.split()
        compressed_length = max(5, int(len(words) * ratio))
        
        # Add symbol prefix to compressed content
        symbol_prefix = memory.compress_symbols()
        if symbol_prefix:
            compressed_content = f"{symbol_prefix} " + " ".join(words[:compressed_length]) + "..."
        else:
            compressed_content = " ".join(words[:compressed_length]) + "..."
        
        shadow = MemorySpan(
            content=compressed_content,
            ts=memory.ts,
            role=memory.role,
            speaker_id=memory.speaker_id,
            source_id=memory.source_id,
            source_hash=memory.source_hash,
            fidelity="structured",  # Downgraded from verbatim
            symbols=memory.symbols.copy(),  # Preserve symbols!
            symbol_confidence=memory.symbol_confidence.copy(),
            tokens=self.token_counter.count_tokens(compressed_content)
        )
        
        # Mark as approximate memory
        shadow.add_symbol("≈", 0.8)
        
        return shadow
    
    def _check_tripwires_with_symbols(self, bundle: ContextBundle, query: str, 
                                    query_entities: List[str], query_symbols: Dict[str, float]):
        """
        Symbol-enhanced uncertainty tripwires
        """
        self.tripwires_triggered = []
        
        # Original tripwires
        if len(query_entities) / max(len(query.split()), 1) > 0.3:
            self.tripwires_triggered.append("high_entity_density")
        
        if bundle.verbatim and max(m.score for m in bundle.verbatim) < self.policy['parameters']['uncertainty_threshold']:
            self.tripwires_triggered.append("low_confidence")
        
        if any(word in query.lower() for word in ['that', 'it', 'this', 'those']):
            if len(query.split()) < 10:
                self.tripwires_triggered.append("ambiguous_reference")
        
        # NEW: Symbol-based tripwires
        
        # Cycle detection
        if "⟲" in bundle.active_symbols and bundle.symbol_counts.get("⟲", 0) > 2:
            self.tripwires_triggered.append("cycle_pattern_detected")
        
        # Paradox overload
        if "∞" in bundle.active_symbols and bundle.symbol_counts.get("∞", 0) > 3:
            self.tripwires_triggered.append("paradox_overload")
        
        # Missing breakthrough symbols for important queries
        if any(word in query.lower() for word in ['understand', 'explain', 'how', 'why']):
            if "✧" not in bundle.active_symbols and "☆" not in bundle.active_symbols:
                self.tripwires_triggered.append("missing_insights")
        
        # Emotional volatility
        if "⚡" in bundle.active_symbols and bundle.symbol_counts.get("⚡", 0) > 2:
            self.tripwires_triggered.append("emotional_volatility")
        
        # Symbol-guided adjustments
        if self.tripwires_triggered:
            logger.warning(f"⚠️ DTH Symbol Tripwires: {self.tripwires_triggered}")
            
            # If cycle detected, heavily prioritize breakthrough symbols
            if "cycle_pattern_detected" in self.tripwires_triggered:
                breakthrough_memories = [m for m in bundle.verbatim if "✧" in m.symbols]
                if breakthrough_memories:
                    # Keep only breakthrough memories
                    bundle.verbatim = breakthrough_memories[:1]
                    # Recalculate tokens
                    bundle.token_count = sum(m.tokens for m in bundle.verbatim + bundle.shadows + bundle.recents)
            
            # If missing insights, try to surface any breakthrough memories
            if "missing_insights" in self.tripwires_triggered:
                all_memories = bundle.verbatim + bundle.shadows
                insight_memories = [m for m in all_memories if "✧" in m.symbols or "☆" in m.symbols]
                if insight_memories:
                    # Promote insights to verbatim
                    bundle.verbatim.extend(insight_memories[:1])
                    bundle.shadows = [m for m in bundle.shadows if m not in insight_memories]
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding with caching"""
        if not self.encoder:
            return None
        
        # Ensure text is a string
        text_str = str(text) if text is not None else ''
        
        # Check cache
        text_hash = hashlib.md5(text_str.encode()).hexdigest()
        self._emb_cache_total += 1
        if text_hash in self.embedding_cache:
            self._emb_cache_hits += 1
            return self.embedding_cache[text_hash]
        
        # Compute embedding
        try:
            embedding = self.encoder.encode(text_str, convert_to_numpy=True)
            
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
        # Ensure text is a string
        text_str = str(text) if text is not None else ''
        
        if not self.nlp:
            # Improved fallback: extract capitalized words AND common terms
            words = text_str.split()
            entities = []
            text_lower = text_str.lower()
            
            # Extract capitalized words (proper nouns)
            for w in words:
                # Clean word of punctuation
                clean_w = w.strip('.,!?;:')
                if clean_w and clean_w[0].isupper() and len(clean_w) > 2:
                    entities.append(clean_w)
            
            # Common technical and domain terms to look for
            common_terms = [
                'python', 'javascript', 'java', 'code', 'api', 'database',
                'sql', 'html', 'css', 'react', 'node', 'git', 'coding',
                'programming', 'script', 'function', 'class', 'method',
                'cooking', 'recipes', 'recipe', 'food', 'kitchen', 'chef'
            ]
            
            # Check each term
            for term in common_terms:
                if term in text_lower:
                    # Add the capitalized version if not already present
                    capitalized = term.capitalize()
                    if capitalized not in entities:
                        entities.append(capitalized)
            
            # Also check for lowercase versions of words that should be entities
            for word in words:
                word_lower = word.lower().strip('.,!?;:')
                if word_lower in ['python', 'code', 'cooking', 'recipes', 'recipe']:
                    capitalized = word_lower.capitalize()
                    if capitalized not in entities:
                        entities.append(capitalized)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_entities = []
            for e in entities:
                e_lower = e.lower()
                if e_lower not in seen:
                    seen.add(e_lower)
                    unique_entities.append(e)
            
            return unique_entities[:10]  # Limit to 10 entities
        
        try:
            doc = self.nlp(text_str)
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
        # Ensure both texts are strings
        text1_str = str(text1) if text1 is not None else ''
        text2_str = str(text2) if text2 is not None else ''
        
        # Jaccard similarity on words
        words1 = set(text1_str.lower().split())
        words2 = set(text2_str.lower().split())
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
            },
            "symbol_usage": self.metrics['symbol_usage'],
            "symbol_effectiveness": self.metrics['symbol_effectiveness']
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
    
    # BACKWARD COMPATIBILITY METHODS FOR EXISTING TESTS
    
    def _score_memory(self, 
                      memory: MemorySpan,
                      query_embedding: Optional[np.ndarray],
                      query_entities: List[str],
                      recent_context: List) -> Tuple[float, Dict[str, float]]:
        """Backward compatibility wrapper for existing tests"""
        query_symbols = self.symbol_detector.detect_symbols("") if self.symbol_detector else {}
        return self._score_memory_with_symbols(
            memory, query_embedding, query_entities, query_symbols, recent_context
        )
    
    def _compress_to_shadow(self, memory: MemorySpan) -> MemorySpan:
        """Backward compatibility wrapper for existing tests"""
        return self._compress_to_shadow_with_symbols(memory)
    
    def _select_within_budget(self, scored_memories: List[MemorySpan], budget: int) -> ContextBundle:
        """Backward compatibility wrapper for existing tests"""
        return self._select_within_budget_with_symbols(scored_memories, budget)
    
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


def get_token_counter():
    """Get appropriate token counter"""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return TokenCounter(encoding)
    except ImportError:
        return SimpleTokenCounter()


class TokenCounter:
    """Accurate token counting using tiktoken"""
    def __init__(self, encoding):
        self.encoding = encoding
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))


class SimpleTokenCounter:
    """Fallback token counter"""
    def count_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)  # Rough approximation


# Testing function for development
async def test_dth_with_symbols():
    """Quick test of the Symbol-Enhanced Dynamic Tape Head"""
    
    # Create mock memory system for testing
    class MockMemory:
        async def knn_tape(self, query, limit=10, scan=100):
            return [
                {
                    "ts": time.time() - 100,
                    "speaker_id": "user",
                    "role": "user",
                    "content": "This is really important! I finally understand neural networks!"
                },
                {
                    "ts": time.time() - 200,
                    "speaker_id": "user",
                    "role": "user",
                    "content": "I keep getting the same error again and again. We're going in circles."
                }
            ]
    
    # Create DTH with symbols
    dth = DynamicTapeHead(MockMemory())
    
    # Test symbol-aware query
    query = "How can I understand this better?"
    bundle = await dth.seek(query, budget=1000)
    
    print(f"🔮 Symbol-enhanced results:")
    print(f"Active symbols: {bundle.active_symbols}")
    print(f"Symbol narrative: {bundle.get_symbol_narrative()}")
    print(f"Selected {len(bundle.verbatim)} verbatim memories")
    print(f"Token usage: {bundle.token_count}/{bundle.token_budget}")
    
    # Show top memory with symbols
    if bundle.verbatim:
        top = bundle.verbatim[0]
        print(f"\nTop memory (score={top.score:.3f}):")
        print(f"  Content: {top.content[:100]}...")
        print(f"  Symbols: {top.symbols}")
        print(f"  Symbol multiplier: {top.get_symbol_multiplier():.2f}")
        if 'SYM' in top.score_components:
            print(f"  Symbol score: {top.score_components['SYM']:.3f}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_dth_with_symbols())
