"""
Smart Context Manager - Replaces context_aggregator.user() with fixed-size context

This processor maintains exactly 4096 tokens of context regardless of conversation length.
It extracts facts, manages session metadata, and provides dynamic prompts.

Key Features:
- Fixed 4096 token context (never grows)
- Fact extraction into graph storage  
- Dynamic system prompt generation
- Session metadata tracking
- Language-agnostic operation
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, LLMMessagesUpdateFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from processors.token_counter import get_token_counter
try:
    # Optional deterministic planner (feature-flagged)
    from context.context_field import ContextField, FactMini
except Exception:
    ContextField = None  # type: ignore
    FactMini = None  # type: ignore
import os
from memory import create_smart_memory_system, extract_facts_from_text
from memory import create_query_classifier
try:
    from memory.dynamic_tape_head import DynamicTapeHead  # optional
except Exception:
    DynamicTapeHead = None  # type: ignore
try:
    # For intent enum comparison in retrieval gating (optional)
    from memory.query_classifier import QueryIntent  # noqa: F401
    from memory.query_router import QueryRouter
except Exception:
    QueryIntent = None  # type: ignore
    QueryRouter = None


@dataclass
class SessionMetadata:
    """Track session information for dynamic prompts"""
    turn_count: int = 0
    session_start: float = 0
    last_interaction: float = 0
    total_interactions: int = 0
    speaker_id: str = "unknown"


@dataclass
class TokenBudget:
    """Unified token allocation for 8192 total - simplified approach"""
    system_prompt: int = 800          # 10% - instructions & identity  
    contextual_memory: int = 2800     # 35% - ONE smart memory system (DTH + recent + facts + summaries)
    current_input: int = 800          # 10% - input processing
    generation_workspace: int = 3600  # 45% - where the magic happens (reserved, not in context)
    
    @property
    def total(self) -> int:
        # Only count tokens that go into the context messages
        return self.system_prompt + self.contextual_memory + self.current_input
    
    @property
    def total_with_generation(self) -> int:
        # Total including generation workspace
        return self.system_prompt + self.contextual_memory + self.current_input + self.generation_workspace


class SmartContextManager(FrameProcessor):
    """
    Replaces context_aggregator.user() with intelligent fixed-size context management
    
    This processor:
    1. Maintains EXACTLY 4096 tokens of context
    2. Extracts facts from conversations
    3. Generates dynamic system prompts
    4. Never accumulates unlimited history
    """
    
    def __init__(self, 
                  context,  # MemoryAwareOpenAILLMContext instance
                  facts_db_path: str = "data/facts.db",
                 max_tokens: int = 8192,  # Updated for unified allocation (scaled below)
                 **kwargs):
        super().__init__(**kwargs)
        
        self.context = context
        self.max_tokens = max_tokens
        self.token_counter = get_token_counter()
        
        # Initialize smart memory system (allow env override for DB path)
        env_db = os.getenv('FACTS_DB_PATH', '').strip()
        final_db_path = env_db if env_db else facts_db_path
        try:
            from pathlib import Path
            final_db_path = str(Path(final_db_path).expanduser().resolve())
        except Exception:
            pass
        self.memory_system = create_smart_memory_system(final_db_path)
        self.tape_store = getattr(self.memory_system, 'tape_store', None)

        # Neural Field Persistence Layer (for consciousness field state continuity)
        self.field_persistence = None
        self._enable_field_persistence = os.getenv('ENABLE_FIELD_PERSISTENCE', 'true').lower() == 'true'
        if self._enable_field_persistence:
            try:
                from consciousness.field_persistence import FieldPersistenceLayer
                self.field_persistence = FieldPersistenceLayer()
                logger.info("🧠 Neural field persistence enabled")
            except ImportError as e:
                logger.warning(f"Field persistence not available: {e}")
                self.field_persistence = None
            except Exception as e:
                logger.warning(f"Field persistence init failed: {e}")
                self.field_persistence = None
        
        # Consciousness Integration (load field states if available)
        self._consciousness_instance = None
        self._user_id = kwargs.get('user_id', os.getenv('USER_ID', 'default_user'))

        # Intelligent Memory Routing (Facts + Tape + DTH)
        self._enable_smart_routing = os.getenv('ENABLE_SMART_ROUTING', 'true').lower() == 'true'
        self.query_router = None
        self.tape_head = None  # Keep as fallback
        
        if self._enable_smart_routing and QueryRouter is not None:
            try:
                # Extract components for QueryRouter initialization
                facts_graph = getattr(self.memory_system, 'facts_graph', None)
                tape_store = getattr(self.memory_system, 'tape_store', None)
                
                # Initialize query router for intelligent facts + tape routing
                self.query_router = QueryRouter(
                    facts_graph=facts_graph,
                    tape_store=tape_store,
                    embedding_store=None  # Not using embedding store yet
                )
                logger.info("🧠 Smart Memory Router enabled (Facts + Tape integration)")
            except Exception as e:
                logger.warning(f"Smart routing init failed, falling back to DTH: {e}")
                import traceback
                traceback.print_exc()
                self.query_router = None
        
        # Optional Dynamic Tape Head integration (can be used alongside router)
        self._enable_dth = os.getenv('ENABLE_DTH', 'false').lower() == 'true'
        self.tape_head = None
        if self._enable_dth and DynamicTapeHead is not None:
            try:
                self.tape_head = DynamicTapeHead(self.memory_system)
                logger.info("🧠 DTH enabled (alongside Smart Router)")
            except Exception as e:
                logger.warning(f"DTH init failed; continuing without DTH: {e}")
                self.tape_head = None
        
        # DSPy Unified Memory Optimizer integration
        self._enable_dspy = os.getenv('DSPY_OPTIMIZATION_ENABLED', 'false').lower() == 'true'
        self.dspy_optimizer = None
        if self._enable_dspy:
            try:
                from slowcat_dspy import DSPY_AVAILABLE, create_unified_memory_optimizer
                if DSPY_AVAILABLE:
                    self.dspy_optimizer = create_unified_memory_optimizer()
                    logger.info("🚀 DSPy UnifiedMemoryOptimizer enabled in SmartContextManager")
                else:
                    logger.warning("DSPy optimization enabled but DSPy not available")
            except Exception as e:
                logger.warning(f"DSPy optimizer init failed: {e}")
                self.dspy_optimizer = None
        
        # Feature toggles / thresholds (env-driven, default generic)
        self._enable_spelling_hints = os.getenv('ENABLE_SPELLING_HINTS', 'false').lower() == 'true'
        # Location-focused spelling hints (stronger nudge when talking about place names)
        self._enable_location_spelling_hints = os.getenv('ENABLE_LOCATION_SPELLING_HINTS', 'false').lower() == 'true'
        self._enable_greeting_fallback = os.getenv('ENABLE_GREETING_FALLBACK', 'false').lower() == 'true'
        # Deterministic greeting injection (pipeline adds the greeting once)
        self._enforce_greeting = os.getenv('SC_ENFORCE_GREETING', 'false').lower() == 'true'
        self._greeted = False
        # Memory safety: never treat current session utterances as "memories"
        self._exclude_current_session_from_memory = os.getenv('SC_EXCLUDE_CURRENT_SESSION_FROM_MEMORY', 'true').lower() == 'true'
        # Optionally exclude assistant-authored snippets from memory injection
        self._memory_include_assistant = os.getenv('SC_MEMORY_INCLUDE_ASSISTANT', 'false').lower() == 'true'
        # Optional cooldown (seconds) to avoid instant re-injection even across sessions
        try:
            self._memory_cooldown_s = int(os.getenv('SC_MEMORY_COOLDOWN_S', '0'))
        except Exception:
            self._memory_cooldown_s = 0
        try:
            # Default: persist turns with >=3 alpha words
            self._tape_min_words = int(os.getenv('TAPE_MIN_USEFUL_WORDS', '3'))
        except Exception:
            self._tape_min_words = 3
        try:
            # Default: or length >= 15 characters
            self._tape_min_len = int(os.getenv('TAPE_MIN_USEFUL_LEN', '15'))
        except Exception:
            self._tape_min_len = 15

        # Token allocation (allow env overrides)
        self.budget = self._load_budget_from_env()
        logger.info(f"🧠 Smart Context Manager initialized with {self.budget.total} token budget")

        # Cache for prompt session info
        self._session_info_cache_ts: float = 0.0
        self._session_info_cache: Optional[dict] = None
        try:
            self._session_info_ttl_s = int(os.getenv('SC_SESSION_INFO_TTL_S', '30'))
        except Exception:
            self._session_info_ttl_s = 30

        # Tape write timeout (to avoid blocking event loop on DB sockets)
        try:
            self._tape_write_timeout_s = max(0.05, int(os.getenv('TAPE_WRITE_TIMEOUT_MS', '200')) / 1000.0)
        except Exception:
            self._tape_write_timeout_s = 0.2

        # Optional Context Field planner
        self._use_context_field = os.getenv('USE_CONTEXT_FIELD', 'false').lower() == 'true'
        self._context_field = None
        self._unified_memory = os.getenv('SC_UNIFIED_MEMORY', 'false').lower() == 'true'
        if self._use_context_field and ContextField is not None:
            try:
                self._context_field = ContextField(total_budget=max(1, int(self.max_tokens)))
                logger.info("🧮 ContextField planner enabled (metrics only in MVP)")
            except Exception as e:
                logger.warning(f"ContextField init failed; continuing without: {e}")
                self._context_field = None

        # Assistant identity + private reflections
        self.assistant_id = os.getenv('ASSISTANT_ID', 'slowcat').strip() or 'slowcat'
        self._enable_reflections = os.getenv('ENABLE_REFLECTIONS', 'false').lower() == 'true'
        try:
            self._reflection_idle_secs = int(os.getenv('REFLECTION_IDLE_SECS', '120'))
        except Exception:
            self._reflection_idle_secs = 120
        try:
            self._reflection_cooldown_secs = int(os.getenv('REFLECTION_COOLDOWN_SECS', '300'))
        except Exception:
            self._reflection_cooldown_secs = 300
        self._last_reflection_ts: float = 0.0

        # Emergent tracking (observability only)
        self._enable_emergent = os.getenv('ENABLE_EMERGENT_TRACKING', 'false').lower() == 'true'
        try:
            self._emergent_lookback_turns = int(os.getenv('EMERGENT_LOOKBACK_TURNS', '30'))
        except Exception:
            self._emergent_lookback_turns = 30

        # Control whether to trigger an LLM response on connect
        self._run_llm_on_connect = os.getenv('SC_RUN_LLM_ON_CONNECT', 'false').lower() == 'true'

        # Prompt organization ('clarity' enables single-rail builder)
        self._prompt_org = os.getenv('SC_PROMPT_ORG', '').strip().lower()
        def _get_int_env(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except Exception:
                return default
        self._clarity_system_tokens = _get_int_env('SC_SYSTEM_TOKENS', 720)
        self._clarity_context_tokens = _get_int_env('SC_CONTEXT_RAIL_TOKENS', 1000)
        self._clarity_input_tokens = _get_int_env('SC_INPUT_TOKENS', 400)
        self._clarity_use_semantic_tape = os.getenv('SC_USE_SEMANTIC_TAPE', 'false').lower() == 'true'
        try:
            self._clarity_classifier = create_query_classifier()
        except Exception:
            self._clarity_classifier = None

        # Session tracking
        self.session = SessionMetadata()
        self.session.session_start = time.time()
        self._session_started = False
        # Prefer a single logical user id when speaker recognition is disabled
        self._user_id_override = os.getenv('USER_ID', '').strip() or None
        if self._user_id_override:
            self.session.speaker_id = self._user_id_override
        
        # Recent conversation sliding window (for context)
        self.recent_exchanges = []  # List of (user, assistant) pairs
        # Keep a large window; actual inclusion is constrained by token budget.
        self.max_recent_exchanges = 50

        # Running summary state
        # Summarization cadence and size
        self.summary_every_n = int(os.getenv('SC_SUMMARY_EVERY_N', '10'))
        self.summary_text: str = ''
        self.last_summary_turn: int = 0
        self._use_abstract_summary = os.getenv('SC_USE_ABSTRACT_SUMMARY', 'false').lower() == 'true'
        try:
            self._summary_last_turns = int(os.getenv('SC_SUMMARY_LAST_TURNS', '10'))
        except Exception:
            self._summary_last_turns = 10
        # Recent inclusion guarantees
        try:
            self._recent_min_exchanges = int(os.getenv('SC_RECENT_MIN_EXCHANGES', '5'))
        except Exception:
            self._recent_min_exchanges = 5
        try:
            self._recent_truncate_chars = int(os.getenv('SC_RECENT_TRUNCATE_CHARS', '260'))
        except Exception:
            self._recent_truncate_chars = 260
        
        # Performance metrics
        self.context_builds = 0
        self.fact_extractions = 0
        self.avg_context_tokens = 0

        # Background idle-based reflection loop (siloed; never injects into context)
        if self._enable_reflections:
            try:
                asyncio.create_task(self._reflection_loop())
                logger.info("🧘 Idle-based reflections enabled")
            except Exception as e:
                logger.warning(f"Reflection loop not started: {e}")
        
        # SurrealDB Message Storage Integration
        self._enable_surreal = os.getenv('USE_SURREALDB', 'false').lower() == 'true'
        self.surreal_store = None
        
        if self._enable_surreal:
            try:
                from processors.surreal_message_store import create_surreal_message_store
                
                self.surreal_store = create_surreal_message_store(
                    speaker_id=self._user_id,
                    auto_create_session=True
                )
                
                if self.surreal_store:
                    logger.info("📝 SurrealDB message storage enabled")
                
            except Exception as e:
                logger.warning(f"SurrealDB message store initialization failed: {e}")
                self.surreal_store = None
    def _trace_sessions(self, event: str, **data):
        """Targeted session trace when SC_TRACE_SESSIONS=true."""
        try:
            import os
            if os.getenv('SC_TRACE_SESSIONS', 'false').lower() != 'true':
                return
            logger.info(f"[SCM:session] {event}: {data}")
        except Exception:
            pass
        
    def _speaker_key(self) -> str:
        """Return a consistent speaker key for persistence.

        - Prefer explicit USER_ID override when provided.
        - Map empty/"unknown" to a stable 'default_user'.
        """
        if self._user_id_override:
            key = self._user_id_override
            self._trace_sessions('speaker_key_override', key=key)
            logger.debug(f"🎭 Using USER_ID override as speaker_key: {key}")
            return key
        sid = (self.session.speaker_id or '').strip()
        key = sid if sid and sid != 'unknown' else 'default_user'
        self._trace_sessions('speaker_key', raw_sid=sid, key=key)
        logger.debug(f"🎭 Speaker key determination - raw_sid: '{sid}' → key: '{key}'")
        return key

    def _sanitize_summary_lines(self, lines: List[str]) -> List[str]:
        """Remove boilerplate and meta lines that cause the model to parrot prompts.

        - Drop assistant greetings/clarifications and generic filler.
        - Prefer keeping user lines; keep assistant lines only if informative and short.
        - Keep overall size small and stable.
        """
        try:
            import re
            cleaned: List[str] = []
            # Patterns to drop from assistant lines
            drop_patterns = [
                r"\b(let\s+me\s+clarify)\b",
                r"\b(i'?m\s+the\s+one\s+processing)\b",
                r"\b(i'?m\s+the\s+one\s+who\s+is\s+reading)\b",
                r"\b(ah\s*,?\s*i\s*[,\s]*see)\b",
                r"\b(how\s+can\s+i\s+help\s+you\s+today)\b",
                r"\b(i'?m\s+slowcat)\b",
            ]
            drop_re = re.compile("|".join(drop_patterns), re.IGNORECASE)

            for ln in lines:
                t = (ln or '').strip()
                if not t:
                    continue
                # Identify role
                role = 'user'
                content = t
                if t.startswith('[assistant]'):
                    role = 'assistant'
                    content = t[len('[assistant]'):].strip()
                elif t.startswith('[user]'):
                    role = 'user'
                    content = t[len('[user]'):].strip()

                content_norm = re.sub(r"\s+", " ", content)

                if role == 'assistant':
                    # Drop boilerplate assistant meta lines
                    if drop_re.search(content_norm):
                        continue
                    # Keep very short, informative assistant statements
                    if len(content_norm.split()) > 24:
                        continue
                # Keep user lines unless extremely long
                if role == 'user' and len(content_norm.split()) > 40:
                    continue

                # Reconstruct with role tag to keep consistent downstream handling
                cleaned.append(f"[{role}] {content_norm}")

            # Cap to last 12 lines to avoid bloat
            return cleaned[-12:]
        except Exception:
            return lines[-12:]

    def _prepare_summary_lines(self, entries: List[Any]) -> List[str]:
        """Build candidate lines from entries and sanitize them for summarization."""
        def _get(obj, k, default=None):
            return obj.get(k, default) if isinstance(obj, dict) else getattr(obj, k, default)
        # Sort ascending and take tail window
        entries = sorted(entries, key=lambda e: _get(e, 'ts', 0.0))
        tail = entries[-max(1, self._summary_last_turns):]
        raw_lines: List[str] = []
        for e in tail:
            content = ((_get(e, 'content', '') or '')).strip()
            if not content:
                continue
            role = _get(e, 'role', 'user')
            raw_lines.append(f"[{role}] {content}")
        return self._sanitize_summary_lines(raw_lines)
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process frames and build fixed-size context instead of accumulating
        """
        await super().process_frame(frame, direction)

        # Track speaker when conversation starts
        if isinstance(frame, UserStartedSpeakingFrame):
            try:
                if not self._user_id_override:
                    spk = getattr(frame, 'speaker_id', None) or getattr(frame, 'user_id', None)
                    if spk:
                        self.session.speaker_id = spk
                        self._trace_sessions('user_started_speaking', frame_speaker_id=spk)
            except Exception:
                pass
        
        # Only process TranscriptionFrames (user input)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            # Ensure session is registered once, even if get_initial_context_frame wasn't used
            if not self._session_started and hasattr(self.memory_system, 'facts_graph'):
                try:
                    spk = self._speaker_key()
                    self._trace_sessions('start_session_on_transcription', key=spk)
                    await self._maybe_await(self.memory_system.facts_graph.start_session(spk))
                    self._session_started = True
                except Exception:
                    pass
            # Seed previous summary lazily if not done by initial context
            if not getattr(self, '_summary_seeded', False) and self.tape_store is not None and not self.summary_text:
                try:
                    last = await self._maybe_await(self.tape_store.get_last_summary())
                    if last:
                        prev_summary_raw = str(last['summary'])
                        prev_summary = self._clean_previous_summary(prev_summary_raw)[:600]
                        if prev_summary:
                            self.summary_text = prev_summary
                    self._summary_seeded = True
                except Exception:
                    pass
            # Normalize user input (collapse repeated punctuation, trim, etc.)
            user_text = self._normalize_user_input(frame.text)
            logger.debug(f"🎤 Processing transcription: '{user_text[:50]}...'")

            # 1. Extract facts from user input (background), but only when content is rich enough
            if self._should_extract_facts(user_text):
                asyncio.create_task(self._extract_facts_async(user_text))
            
            # 1a. Track consciousness field evolution (if consciousness available)
            if self.field_persistence and self._consciousness_instance:
                asyncio.create_task(self._track_field_evolution_async(user_text))

            # 1b. Write user message to tape store
            try:
                if self.tape_store is not None and self._is_semantically_useful(user_text):
                    self._enqueue_tape_write('user', user_text)
            except Exception as e:
                logger.debug(f"TapeStore write (user) enqueue failed: {e}")
            
            # 1c. Store user message in SurrealDB
            try:
                if self.surreal_store and self._is_semantically_useful(user_text):
                    asyncio.create_task(self.surreal_store._handle_user_message(user_text))
            except Exception as e:
                logger.debug(f"SurrealDB user message store failed: {e}")
            
            # 2. Update session in memory system
            try:
                key = self._speaker_key()
                self._trace_sessions('update_session', key=key)
                await self._maybe_await(self.memory_system.update_session(key))
            except Exception:
                pass

            # 3. If input has no semantic content (e.g., just punctuation), skip LLM
            if not self._has_semantic_content(user_text):
                return

            # Expand short acknowledgments and optionally add spelling hints
            user_text_expanded = self._expand_short_ack(user_text)
            if self._enable_spelling_hints or self._enable_location_spelling_hints:
                user_text_expanded = self._augment_user_with_spelling(user_text_expanded)

            # 4. Build fixed context (NEVER grows beyond 4096)
            messages = await self._build_fixed_context(user_text_expanded)

            # 5. Update the shared context object with our fixed context
            # The context aggregator will use this when processing the TranscriptionFrame
            if self.context:
                try:
                    roles = [m.get('role', '?') for m in messages if isinstance(m, dict)]
                    logger.debug(f"[SCM] Roles in context for LLM call: {roles}")
                    logger.debug(f"[SCM] recent_exchanges={len(self.recent_exchanges)} summary_present={bool(self.summary_text)} context_messages={len(messages)}")
                except Exception:
                    pass
                self.context.set_messages(messages)
            
            # 6. Forward the original transcription frame to trigger normal LLM processing
            # The context aggregator will pick up our updated context
            await self.push_frame(frame, direction)

            # 6. Update session metadata
            self._update_session()
            
            # 6c. Record the user turn AFTER sending the context to avoid
            # duplicating the current user message in both recent_context and
            # the explicit current user slot.
            try:
                if self._is_semantically_useful(user_text):
                    self._record_user_turn(user_text)
            except Exception as e:
                logger.debug(f"Recent window update failed: {e}")

            # 6b. Update running summary periodically (async, non-blocking)
            try:
                asyncio.create_task(self._maybe_update_running_summary())
            except Exception as e:
                logger.debug(f"Running summary scheduling skipped: {e}")
            
            # DON'T forward the original TranscriptionFrame
            # This prevents the old context_aggregator from accumulating
            return
            
        # Forward all other frames normally
        await self.push_frame(frame, direction)

    def _record_user_turn(self, user_text: str):
        """Append a new user turn to recent_exchanges with sliding window cap."""
        # Store as a 1-tuple to be completed by add_assistant_response
        self.recent_exchanges.append((user_text,))
        # Enforce sliding window size
        while len(self.recent_exchanges) > self.max_recent_exchanges:
            self.recent_exchanges.pop(0)
    
    async def _build_fixed_context(self, user_input: str) -> List[Dict]:
        """
        Build unified 8K context with simplified allocation:
        - System prompt: 800 tokens (10% - instructions & identity)  
        - Contextual memory: 2800 tokens (35% - ONE unified memory system via DTH + DSPy)
        - Current input: 800 tokens (10% - input processing)
        - Generation workspace: 3600 tokens (45% - reserved, not in context)
        """
        # Clarity organization: System + one context rail + user
        if getattr(self, '_prompt_org', '') == 'clarity':
            try:
                return await self._build_context_clarity(user_input)
            except Exception as e:
                logger.debug(f"Clarity builder failed, using unified builder: {e}")
        # Clarity organization: System + one context rail + user
        if getattr(self, '_prompt_org', '') == 'clarity':
            try:
                return await self._build_context_clarity(user_input)
            except Exception as e:
                logger.debug(f"Clarity builder failed, using unified builder: {e}")
        start_time = time.time()
        
        # 1. System prompt (800 tokens max)
        system_prompt = await self._generate_dynamic_prompt()
        # Optionally include a compact session summary inside the system section
        try:
            include_sum = os.getenv('SC_INCLUDE_SUMMARY', 'true').lower() == 'true'
            max_sum_tokens = int(os.getenv('SC_SUMMARY_TOKENS', '180'))
        except Exception:
            include_sum, max_sum_tokens = True, 180
        preamble = system_prompt
        if include_sum and self.summary_text:
            # Clean and trim summary
            cleaned = self._clean_previous_summary(self.summary_text)
            trimmed = self._trim_to_token_budget(cleaned, max_sum_tokens)
            if trimmed:
                preamble = f"{system_prompt}\n\n<session_summary>\n{trimmed}\n</session_summary>"
        system_content = self._trim_to_token_budget(preamble, self.budget.system_prompt)
        system_tokens = self.token_counter.count_tokens(system_content)
        
        # 2. Contextual memory (budgeted) - gated retrieval to avoid unnecessary latency
        contextual_memory = ""
        contextual_tokens = 0

        # Heuristic gate: only query memory when input likely needs it
        should_query_memory = self._is_memory_candidate(user_input)

        # Initialize variables that may be used across different memory system branches
        dth_candidates = []
        verified_lines: List[str] = []
        strict_answer_mode = self._is_personal_facts_candidate(user_input)
        
        # Try smart router first (Facts + Tape integration) if gated in
        if should_query_memory and self.query_router is not None:
            try:
                logger.info("🎯 Using Smart Memory Router (Facts + Tape)")
                
                # Route query through intelligent facts + tape system
                # Detect router type and use appropriate parameters
                if hasattr(self.query_router, '__class__') and 'SurrealQueryRouter' in str(self.query_router.__class__):
                    # SurrealQueryRouter interface
                    router_response = await self.query_router.route_query(
                        query=user_input,
                        context={
                            "speaker_id": self._speaker_key(),
                            "force_personal_facts": self._is_personal_facts_candidate(user_input)
                        }
                    )
                else:
                    # Standard QueryRouter interface
                    router_response = await self.query_router.route_query(
                        query=user_input,
                        context={
                            "speaker_id": self._speaker_key(),
                            "force_personal_facts": self._is_personal_facts_candidate(user_input)
                        },
                        max_results=20  # Get more results for better selection
                    )
                
                # Extract memory candidates and verified facts from router response
                
                # RetrievalResponse contains results from both facts and tape stores
                try:
                    if hasattr(router_response, 'results'):
                        rr_list = router_response.results
                    else:
                        rr_list = router_response.get('results', [])
                except Exception:
                    rr_list = []

                # Filter out current-session and assistant-authored tape results to prevent loops
                filtered_results = []
                now_ts = time.time()
                for r in rr_list:
                    try:
                        src = getattr(r, 'source_store', '')
                        ts = float(getattr(r, 'timestamp', 0.0) or 0.0)
                        meta = getattr(r, 'metadata', {}) if hasattr(r, 'metadata') else {}
                        role = (meta.get('role') or '').lower()
                        spk = meta.get('speaker_id')
                        # Exclude assistant-authored tape by default
                        if src == 'tape' and (not self._memory_include_assistant) and role == 'assistant':
                            continue
                        # Only include current user's tape when speaker_id is present
                        if src == 'tape' and spk and spk != self._speaker_key():
                            continue
                        # Exclude entries from the current session window
                        if src == 'tape' and self._exclude_current_session_from_memory:
                            if ts and ts >= self.session.session_start:
                                continue
                        # Optional cooldown to avoid immediate re-injection
                        if src == 'tape' and self._memory_cooldown_s > 0:
                            if ts and ts >= (now_ts - self._memory_cooldown_s):
                                continue
                        filtered_results.append(r)
                    except Exception:
                        # Be conservative; only include if we can't prove it's current session
                        filtered_results.append(r)

                # 🔥 PURE ARCHITECTURE: Process ONLY facts, ignore tape completely
                for r in filtered_results:
                    src = getattr(r, 'source_store', '')
                    
                    # Skip everything except facts (pure architecture)
                    if src != 'facts':
                        continue
                    
                    content = getattr(r, 'content', '')
                    
                    # Try metadata first, fallback to direct attributes for SurrealDB compatibility  
                    subj = None
                    pred = None
                    val = None
                    
                    if hasattr(r, 'metadata') and r.metadata:
                        subj = r.metadata.get('subject')
                        pred = r.metadata.get('predicate') 
                        val = r.metadata.get('value') or r.metadata.get('object')
                    
                    # Fallback to direct attributes
                    if not subj:
                        subj = getattr(r, 'subject', None)
                    if not pred:
                        pred = getattr(r, 'predicate', None)
                    if not val:
                        val = getattr(r, 'object', None)
                    
                    # Build structured fact
                    if subj and pred:
                        ptxt = str(pred).replace('_', ' ')
                        if val is not None and str(val).strip() != '':
                            verified_lines.append(f"- {subj}'s {ptxt} is {val}")
                        else:
                            verified_lines.append(f"- {subj} has {ptxt}")
                    elif content:
                        verified_lines.append(f"- {content}")
                
                logger.info(f"🔥 PURE FACTS: {len(verified_lines)} structured facts extracted")
                if filtered_results:
                    for result in filtered_results:
                        # Each MemoryResult has content and source_store metadata
                        if hasattr(result, 'content') and result.content:
                            dth_candidates.append(str(result.content))
                            
                    # Log what we found
                    sources = {}
                    for result in filtered_results:
                        source = getattr(result, 'source_store', 'unknown')
                        sources[source] = sources.get(source, 0) + 1
                    
                    logger.info(f"   🔍 Router sources: {dict(sources)}")

                logger.info(f"   📊 Smart Router found: {len(dth_candidates)} memory candidates")
                
            except Exception as e:
                logger.warning(f"Smart router failed, falling back to DTH: {e}")
                # Don't reset dth_candidates here - keep whatever was initialized
        
        # Fallback to DTH if smart router unavailable or failed
        elif self.tape_head is not None:
            try:
                logger.info("🎯 Falling back to DTH (Smart Router unavailable)")
                
                # Get DTH candidates (already ranked by relevance)
                dth_bundle = await self.tape_head.seek(
                    user_input,
                    budget=self.budget.contextual_memory,  # 2800 tokens
                    context=None,
                    speaker_id=self._speaker_key(),
                    debug_selection=None,
                )
                
                # Extract text candidates from DTH bundle (filtering current session and assistant role)
                dth_candidates = []
                def _accept_span(span) -> bool:
                    try:
                        ts = float(getattr(span, 'ts', 0.0) or 0.0)
                        role = (getattr(span, 'role', '') or '').lower()
                        spk = getattr(span, 'speaker_id', None)
                        if (not self._memory_include_assistant) and role == 'assistant':
                            return False
                        if spk and spk != self._speaker_key():
                            return False
                        if self._exclude_current_session_from_memory and ts and ts >= self.session.session_start:
                            return False
                        if self._memory_cooldown_s > 0 and ts and ts >= (time.time() - self._memory_cooldown_s):
                            return False
                        return True
                    except Exception:
                        return False
                if hasattr(dth_bundle, 'verbatim') and dth_bundle.verbatim:
                    for item in dth_bundle.verbatim:
                        if hasattr(item, 'content') and _accept_span(item):
                            dth_candidates.append(str(item.content))
                if hasattr(dth_bundle, 'shadows') and dth_bundle.shadows:
                    for item in dth_bundle.shadows:
                        if hasattr(item, 'content') and _accept_span(item):
                            dth_candidates.append(str(item.content))
                if 'dth_bundle' in locals() and hasattr(dth_bundle, 'recents') and dth_bundle.recents:
                    for item in dth_bundle.recents:
                        if hasattr(item, 'content') and _accept_span(item):
                            dth_candidates.append(str(item.content))
                
                # Use DSPy to optimize selection from DTH candidates
                if self.dspy_optimizer and dth_candidates:
                    logger.info(f"🧠 CONTEXT BUILDING: DSPy optimization enabled")
                    logger.info(f"   📚 DTH provided {len(dth_candidates)} memory candidates")
                    
                    dspy_result = self.dspy_optimizer(
                        query=user_input,
                        dth_candidates=dth_candidates,
                        target_tokens=self.budget.contextual_memory,
                        mode="chat"  # Could be dynamic based on current mode
                    )
                    
                    contextual_memory = dspy_result.get('selected_memory', '')
                    logger.info(f"   ✅ DSPy selected {len(contextual_memory)} chars of contextual memory")
                    logger.info(f"   💡 Selection reasoning: {dspy_result.get('selection_reasoning', 'N/A')[:100]}...")
                    
                else:
                    # MMR-like selection for diversity
                    logger.info("🧠 CONTEXT BUILDING: Using MMR fallback selection")
                    selected = self._mmr_select(user_input, dth_candidates, self.budget.contextual_memory,
                                                lambda_div=float(os.getenv('DTH_MMR_LAMBDA', '0.7')),
                                                max_items=int(os.getenv('DTH_MMR_MAX_ITEMS', '6')))
                    contextual_memory = "\n\n".join(selected)
                    logger.info(f"   📝 MMR selected {len(selected)} items, ~{self.token_counter.count_tokens(contextual_memory)} tokens")
                
                # Prefer verified facts when present (authoritative over verbatim)
                if verified_lines:
                    contextual_memory = "\n".join(sorted(set(verified_lines)))
                else:
                    # If no verified facts, do not enforce strict mode
                    strict_answer_mode = False
                contextual_tokens = self.token_counter.count_tokens(contextual_memory)
                logger.debug(f"🧠 DTH candidates: {len(dth_candidates)}, contextual tokens: {contextual_tokens}")
                
            except Exception as e:
                logger.warning(f"DTH + DSPy memory selection failed: {e}")
                # Don't reset dth_candidates here - keep whatever was initialized
        
        else:
            # No memory system available
            if should_query_memory:
                logger.warning("🚨 No memory system available (neither Smart Router nor DTH)")
            dth_candidates = []
        
        # 🔥 PURE ARCHITECTURE: Use verified facts directly, bypass ALL other selection
        if verified_lines:
            contextual_memory = "\n".join(sorted(set(verified_lines)))
            logger.info(f"🔥 PURE CONTEXT: Using {len(verified_lines)} verified facts directly")
        # Process candidates (works for both Smart Router and DTH results)  
        elif dth_candidates:
            # Use DSPy to optimize selection from candidates
            if self.dspy_optimizer:
                logger.info(f"🧠 CONTEXT BUILDING: DSPy optimization enabled")
                logger.info(f"   📚 Found {len(dth_candidates)} memory candidates")
                
                dspy_result = self.dspy_optimizer(
                    query=user_input,
                    dth_candidates=dth_candidates,
                    target_tokens=self.budget.contextual_memory,
                    mode="chat"  # Could be dynamic based on current mode
                )
                
                contextual_memory = dspy_result.get('selected_memory', '')
                logger.info(f"   ✅ DSPy selected {len(contextual_memory)} chars of contextual memory")
                logger.info(f"   💡 Selection reasoning: {dspy_result.get('selection_reasoning', 'N/A')[:100]}...")
                
            else:
                # MMR-like selection
                logger.info("🧠 CONTEXT BUILDING: Using MMR fallback selection")
                selected = self._mmr_select(user_input, dth_candidates, self.budget.contextual_memory,
                                            lambda_div=float(os.getenv('DTH_MMR_LAMBDA', '0.7')),
                                            max_items=int(os.getenv('DTH_MMR_MAX_ITEMS', '6')))
                contextual_memory = "\n\n".join(selected)
                logger.info(f"   📝 MMR selected {len(selected)} items, ~{self.token_counter.count_tokens(contextual_memory)} tokens")
            
            contextual_tokens = self.token_counter.count_tokens(contextual_memory)
            logger.debug(f"🧠 Memory candidates: {len(dth_candidates)}, contextual tokens: {contextual_tokens}")
            
        else:
            contextual_memory = ""
            contextual_tokens = 0
            # Secondary fallback: include a tiny slice of recent tape if available
            try:
                recent_candidates: List[str] = []
                now_ts = time.time()
                # Fetch raw recent items
                if hasattr(self.memory_system, 'get_recent'):
                    items = await self.memory_system.get_recent(limit=8)
                elif self.tape_store is not None and hasattr(self.tape_store, 'get_recent'):
                    items = await self._maybe_await(self.tape_store.get_recent(limit=8))
                else:
                    items = []
                # Filter: same speaker, user-only (by default), prior sessions only, optional cooldown
                spk = self._speaker_key()
                def _accept_item(e) -> bool:
                    try:
                        role = (e.get('role') if isinstance(e, dict) else getattr(e, 'role', '')) or ''
                        role = role.lower()
                        content = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
                        speaker_id = (e.get('speaker_id') if isinstance(e, dict) else getattr(e, 'speaker_id', None))
                        ts = float((e.get('ts') if isinstance(e, dict) else getattr(e, 'ts', 0.0)) or 0.0)
                        if (not self._memory_include_assistant) and role == 'assistant':
                            return False
                        if speaker_id and speaker_id != spk:
                            return False
                        if self._exclude_current_session_from_memory and ts and ts >= self.session.session_start:
                            return False
                        if self._memory_cooldown_s > 0 and ts and ts >= (now_ts - self._memory_cooldown_s):
                            return False
                        return bool(content.strip())
                    except Exception:
                        return False
                for e in (items or []):
                    if _accept_item(e):
                        txt = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', ''))
                        recent_candidates.append(str(txt))
                if recent_candidates:
                    contextual_memory = "\n\n".join(recent_candidates[:4])
                    contextual_tokens = self.token_counter.count_tokens(contextual_memory)
            except Exception:
                pass
        
        # 3. Recent conversation context (use remaining contextual memory budget)
        # If we are answering a personal-facts question, reduce recency to avoid hallucinated carryover
        strict_answer_mode = self._is_personal_facts_candidate(user_input)
        remaining_contextual_budget = self.budget.contextual_memory - contextual_tokens
        recent_context_messages = []
        recent_context_tokens = 0

        # When memory contributes, cap recents to avoid drowning it out
        try:
            recents_cap = int(os.getenv('SC_RECENTS_CAP_WITH_MEMORY_TOKENS', '300'))
        except Exception:
            recents_cap = 300
        effective_recent_budget = remaining_contextual_budget
        if contextual_tokens > 0 and recents_cap > 0:
            effective_recent_budget = min(remaining_contextual_budget, recents_cap)

        if not strict_answer_mode and effective_recent_budget > 50:  # Only if we have meaningful space
            recent_context_messages = self._build_recent_context(effective_recent_budget)
            recent_context_tokens = sum(
                self.token_counter.count_tokens(msg.get('content', '')) 
                for msg in recent_context_messages
            )
        
        # 4. Current input (budgeted)
        current_content = self._trim_to_token_budget(user_input, self.budget.current_input)
        current_tokens = self.token_counter.count_tokens(current_content)

        # 5. Assemble final context messages
        messages = []
        
        # System message with retrieved memory (not recent conversation)
        full_system_content = system_content
        history_mode = self._is_history_candidate(user_input)
        if contextual_memory:
            if strict_answer_mode:
                full_system_content += (
                    "\n\n<verified_memory>\n" + contextual_memory + "\n</verified_memory>" 
                    "\n<answer_policy>\n"
                    "When answering about personal facts, base your answer ONLY on <verified_memory>.\n"
                    "If the requested attribute is missing, say you don't have it recorded yet and ask if the user wants to add it.\n"
                    "Do not infer age/breed/numbers; do not guess.\n"
                    "</answer_policy>"
                )
            elif history_mode:
                full_system_content += (
                    "\n\n<conversation_snippets>\n" + contextual_memory + "\n</conversation_snippets>" 
                    "\n<answer_policy>\n"
                    "Continue naturally from the snippets in <conversation_snippets>.\n"
                    "Use them as the immediate prior context.\n"
                    "Do not greet; avoid repeating the snippets.\n"
                    "</answer_policy>"
                )
            else:
                full_system_content += f"\n\n<dth_memories>\n{contextual_memory}\n</dth_memories>"
        else:
            # Minimal facts include (tiny, always helpful) when no other memory present
            try:
                min_facts = int(os.getenv('SC_MIN_FACTS_IN_CONTEXT', '2'))
            except Exception:
                min_facts = 2
            minimal_block = ""
            if min_facts > 0 and hasattr(self.memory_system, 'facts_graph'):
                try:
                    top_facts = await self._maybe_await(self.memory_system.facts_graph.get_top_facts(limit=min_facts))
                    if top_facts:
                        minimal_block = self._format_facts_context(top_facts)
                except Exception:
                    minimal_block = ""
            if minimal_block:
                full_system_content += f"\n\n<facts_memory>\n{minimal_block}\n</facts_memory>"
        
        messages.append({"role": "system", "content": full_system_content})
        
        # Add recent conversation exchanges as proper user/assistant message pairs
        messages.extend(recent_context_messages)
        
        # Do NOT append current user input here.
        # The context aggregator will attach the current TranscriptionFrame as the user turn.

        # 6. Verify total tokens (should be exactly our budget)
        total_tokens = system_tokens + contextual_tokens + recent_context_tokens + current_tokens
        
        # Track metrics
        self.context_builds += 1
        self.avg_context_tokens = ((self.avg_context_tokens * (self.context_builds - 1) + 
                                   total_tokens) / self.context_builds)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Show final context summary
        dspy_status = "DSPy✨" if self.dspy_optimizer and contextual_tokens > 0 else "Manual📝"
        logger.info(f"🚀 UNIFIED CONTEXT BUILT: {total_tokens}/{self.budget.total} tokens ({elapsed_ms:.1f}ms)")
        logger.info(f"   🎭 System: {system_tokens} tokens")
        logger.info(f"   🧠 Retrieved Memory: {contextual_tokens} tokens ({dspy_status})")
        logger.info(f"   💬 Recent Context: {recent_context_tokens} tokens ({len(recent_context_messages)} messages)")
        logger.info(f"   📝 Current Input: {current_tokens} tokens")
        logger.info(f"   🎯 Generation Workspace: {self.budget.generation_workspace} tokens (reserved)")
        
        # Store metrics for observability
        self._last_context_build_metrics = {
            'unified_approach': True,
            'dspy_enabled': self.dspy_optimizer is not None,
            'block_tokens': {
                'system_prompt': system_tokens,
                'contextual_memory': contextual_tokens,
                'current_input': current_tokens,
                'total': total_tokens,
                'generation_workspace_reserved': self.budget.generation_workspace
            },
            'performance': {
                'build_time_ms': elapsed_ms,
                'total_builds': self.context_builds,
                'avg_tokens': self.avg_context_tokens
            }
        }
        
        return messages
    
    def _trim_to_token_budget(self, text: str, max_tokens: int) -> str:
        """Trim text to fit within token budget"""
        if not text or max_tokens <= 0:
            return ""
        
        current_tokens = self.token_counter.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Trim by sentences first, then by words if needed
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Try to keep complete sentences
            result = ""
            for sentence in sentences:
                candidate = result + sentence + ". " if result else sentence + ". "
                if self.token_counter.count_tokens(candidate.strip()) <= max_tokens:
                    result = candidate
                else:
                    break
            if result.strip():
                return result.strip()
        
        # Fallback: trim by words
        words = text.split()
        result = ""
        for word in words:
            candidate = result + " " + word if result else word
            if self.token_counter.count_tokens(candidate) <= max_tokens:
                result = candidate
            else:
                break
                
        return result.strip() if result else text[:max_tokens * 3]  # Rough character fallback

    def _is_semantically_useful(self, text: str) -> bool:
        try:
            import re
            s = (text or '').strip()
            if not s:
                return False
            if s.endswith('?'):
                return True
            words = [w for w in re.split(r"\s+", s) if any(c.isalpha() for c in w)]
            if self._tape_min_words > 0 and len(words) < self._tape_min_words:
                return False
            if self._tape_min_len > 0 and len(s) < self._tape_min_len:
                return False
            return True
        except Exception:
            return True

    async def _build_context_clarity(self, user_input: str) -> List[Dict]:
        """Clear, intent-gated prompt: System + one context rail + user."""
        # Compose session header (time + turns) then core rules
        try:
            spk = self._speaker_key()
            info = await self._get_session_info_cached(spk)
            # Minimal session header; keep it short
            from datetime import datetime
            fs = info.get('first_seen')
            ls = info.get('last_interaction')
            fs_str = datetime.fromtimestamp(fs).strftime('%Y-%m-%d %H:%M') if fs else ''
            ls_str = datetime.fromtimestamp(ls).strftime('%Y-%m-%d %H:%M') if ls else ''
            turn_display = max(1, self.session.turn_count + 1)
            session_header = (
                f"<session_info> sessions:{info.get('session_count',0)} turns:{turn_display}"
                + (f" first:{fs_str}" if fs_str else '')
                + (f" last:{ls_str}" if ls_str else '')
                + "</session_info>\n"
            )
        except Exception:
            session_header = ''

        system_rules = (
            session_header +
            "You are Slowcat — practical, warm, and concise. Answer as the assistant.\n"
            "- Never speak as the user; refer to the user as 'you'.\n"
            "- Do not claim user-owned things as yours (no 'my dog' unless clearly yours; you have no possessions).\n"
            "- Do not greet unless the user greets first.\n"
            "- If uncertain, ask one short clarifying question.\n"
            "- Prefer direct, specific answers; avoid repeating the user's text.\n"
            "- Treat <context> as reference notes. Do not parrot lines or adopt their speaker; summarize and use them to inform your answer.\n"
        )
        sys_content = self._trim_to_token_budget(system_rules, self._clarity_system_tokens)
        messages: List[Dict[str, str]] = [{"role": "system", "content": sys_content}]

        # Intent to select rail
        intent = None
        try:
            if self._clarity_classifier is not None:
                res = await self._clarity_classifier.classify(user_input)
                intent = getattr(res, 'intent', None)
        except Exception:
            intent = None

        rail_text = ""
        from memory.query_classifier import QueryIntent as _QI  # local import to avoid top-level breakages
        try:
            if intent in (_QI.PERSONAL_FACTS,) and hasattr(self.memory_system, 'facts_graph'):
                rail_text = await self._clarity_build_facts_rail()
            elif intent in (_QI.CONVERSATION_HISTORY, _QI.EPISODIC_MEMORY, _QI.KNOWLEDGE_SYNTHESIS):
                # Prefer smart router for conversation retrieval if available; fallback otherwise
                text_from_router = await self._clarity_build_conversation_via_router(user_input)
                if text_from_router:
                    rail_text = text_from_router
                else:
                    rail_text = await self._clarity_build_conversation_rail(user_input)
        except Exception:
            rail_text = ""

        if rail_text:
            block = f"<context>\n{rail_text}\n</context>"
            combined = messages[0]['content'] + "\n\n" + block
            messages[0]['content'] = self._trim_to_token_budget(combined, self._clarity_system_tokens + self._clarity_context_tokens)

        # Current user last
        current = self._trim_to_token_budget(user_input, self._clarity_input_tokens)
        messages.append({"role": "user", "content": current})
        return messages

    async def _clarity_build_facts_rail(self) -> str:
        try:
            facts = await self._maybe_await(self.memory_system.facts_graph.get_top_facts(limit=12))
        except Exception:
            facts = []
        lines: List[str] = []
        for f in facts or []:
            try:
                if getattr(f, 'subject', '') != 'user':
                    continue
                pred = (getattr(f, 'predicate', '') or '').strip()
                val = (getattr(f, 'value', '') or '').strip()
                if not pred:
                    continue
                if val:
                    lines.append(f"- you {pred} {val}")
                else:
                    lines.append(f"- you {pred}")
                if len(lines) >= 6:
                    break
            except Exception:
                continue
        rail = "\n".join(lines[:6])
        return self._trim_to_token_budget(rail, self._clarity_context_tokens)

    async def _clarity_build_conversation_rail(self, user_input: str) -> str:
        items = []
        try:
            if hasattr(self.memory_system, 'get_recent'):
                items = await self.memory_system.get_recent(limit=10)
            elif self.tape_store is not None and hasattr(self.tape_store, 'get_recent'):
                items = await self._maybe_await(self.tape_store.get_recent(limit=10))
        except Exception:
            items = []
        spk = self._speaker_key()
        user_lines: List[str] = []
        for e in items or []:
            try:
                role = (e.get('role') if isinstance(e, dict) else getattr(e, 'role', '')) or ''
                speaker_id = (e.get('speaker_id') if isinstance(e, dict) else getattr(e, 'speaker_id', '')) or ''
                content = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
                if speaker_id != spk or not content.strip():
                    continue
                if role == 'user':
                    user_lines.append(content.strip())
            except Exception:
                continue
        lines: List[str] = []
        for txt in user_lines[-3:]:
            lines.append(f"[user] {txt}")

        if self._clarity_use_semantic_tape and hasattr(self.memory_system, 'knn_tape'):
            try:
                knn = await self.memory_system.knn_tape(user_input, limit=2, scan=80, speaker_id=spk)
                for e in knn or []:
                    txt = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
                    if txt:
                        lines.append(f"[snippet] {txt}")
            except Exception:
                pass
        rail = "\n".join(lines)
        return self._trim_to_token_budget(rail, self._clarity_context_tokens)

    async def _clarity_build_conversation_via_router(self, user_input: str) -> str:
        """Use QueryRouter (if present) to retrieve relevant tape snippets; filter by speaker and time.

        Also supports simple temporal cues (e.g., 'yesterday') via SurrealDB time_travel_query when available.
        """
        # 1) Temporal cue fallback (optional)
        try:
            s = (user_input or '').lower()
            temporal_cues = ('yesterday', 'last week', 'last month', 'last time', 'this morning')
            if any(cue in s for cue in temporal_cues) and hasattr(self.memory_system, 'time_travel_query'):
                rows = await self.memory_system.time_travel_query('yesterday' if 'yesterday' in s else 'last week', limit=5)
                lines = []
                spk = self._speaker_key()
                for r in rows or []:
                    try:
                        if (r.get('speaker_id') or '') != spk:
                            continue
                        txt = (r.get('content') or '').strip()
                        role = (r.get('role') or 'user')
                        if txt:
                            lines.append(f"[{role}] {txt}")
                    except Exception:
                        continue
                if lines:
                    rail = "\n".join(lines[-4:])
                    return self._trim_to_token_budget(rail, self._clarity_context_tokens)
        except Exception:
            pass

        # 2) QueryRouter path
        if not getattr(self, 'query_router', None):
            return ""
        try:
            ctx = {"speaker_id": self._speaker_key()}
            router_response = await self.query_router.route_query(query=user_input, context=ctx)
            try:
                results = router_response.results if hasattr(router_response, 'results') else router_response.get('results', [])
            except Exception:
                results = []
            lines = []
            now_ts = time.time()
            for r in results:
                try:
                    src = getattr(r, 'source_store', '')
                    if src != 'tape':
                        continue
                    meta = getattr(r, 'metadata', {}) if hasattr(r, 'metadata') else {}
                    role = (meta.get('role') or 'user').lower()
                    speaker_id = meta.get('speaker_id') or ''
                    ts = float(getattr(r, 'timestamp', 0.0) or 0.0)
                    # Enforce speaker match; include cross-session as needed
                    if speaker_id and speaker_id != self._speaker_key():
                        continue
                    content = getattr(r, 'content', '')
                    if not content:
                        continue
                    # Optional: bias away from current-session snippets (prefer history)
                    if self._exclude_current_session_from_memory and ts and ts >= self.session.session_start:
                        continue
                    lines.append(f"[{role}] {content}")
                    if len(lines) >= 5:
                        break
                except Exception:
                    continue
            rail = "\n".join(lines)
            return self._trim_to_token_budget(rail, self._clarity_context_tokens)
        except Exception:
            return ""

    def _has_semantic_content(self, text: str) -> bool:
        if not text:
            return False
        # Consider it semantic if it has any alphanumeric characters
        return any(ch.isalnum() for ch in text)

    def _is_memory_candidate(self, text: str) -> bool:
        """Cheap lexical gate to decide if we should hit memory.

        Triggers for facts/history questions without running classifier:
        - Contains a question mark
        - Possessive/personal facts: 'my ', "what's my", 'where do I'
        - History/continuation cues: 'we talked', 'last time', 'continue', 'resume',
          'where we left', 'previous session', 'again'
        """
        try:
            s = (text or '').strip().lower()
            if not s:
                return False
            if '?' in s:
                return True
            # Possessive/personal
            if any(k in s for k in (
                'my ', "what's my", 'what is my', 'where do i', 'who is my', 'when did i'
            )):
                return True
            # History/continuation or recall intents
            if any(k in s for k in (
                'we talked', 'we discussed', 'last time', 'continue', 'resume',
                'where we left', 'previous session', 'again', 'remember', 'recall', 'remind'
            )):
                return True
            return False
        except Exception:
            return False

    def _should_extract_facts(self, text: str) -> bool:
        """Gate expensive fact extraction; skip short/noisy utterances."""
        try:
            s = (text or '').strip()
            if not s:
                return False
            # Require a minimum of words and characters
            words = [w for w in s.split() if any(c.isalpha() for c in w)]
            if len(words) < max(3, self._tape_min_words):
                return False
            if len(s) < max(15, self._tape_min_len):
                return False
            return True
        except Exception:
            return False

    def _is_personal_facts_candidate(self, text: str) -> bool:
        """Detect if the query likely targets personal facts (schema-agnostic)."""
        try:
            s = (text or '').strip().lower()
            if not s:
                return False
            if any(k in s for k in ('my ', "what's my", 'what is my', 'who is my')):
                if any(k in s for k in ('name', 'birthday', 'age', 'email', 'phone', 'address', 'dog', 'pet')):
                    return True
            if (('dog' in s or 'pet' in s) and ('name' in s or 'called' in s) and '?' in s):
                return True
            return False
        except Exception:
            return False

    def _is_history_candidate(self, text: str) -> bool:
        try:
            s = (text or '').strip().lower()
            if not s:
                return False
            triggers = (
                'continue', 'resume', 'pick up', 'where we left', 'last conversation',
                'previous conversation', 'carry on', 'go on', 'keep going'
            )
            return any(t in s for t in triggers)
        except Exception:
            return False

    def _normalize_user_input(self, text: str) -> str:
        try:
            import re
            s = (text or "").strip()
            # Collapse repeated punctuation
            s = re.sub(r"[?]{2,}", "?", s)
            s = re.sub(r"[!]{2,}", "!", s)
            s = re.sub(r"[.]{2,}", ".", s)
            # Remove stray punctuation-only tails like '?.' -> '?'
            s = re.sub(r"\?\.+$", "?", s)
            s = re.sub(r"!\.+$", "!", s)
            # Fix comma-dot sequences like ", .word" or ", . no" -> ", word"
            s = re.sub(r",\s*\.\s*", ", ", s)
            # Fix dot-comma sequences ".," -> ". "
            s = re.sub(r"\.\s*,\s*", ". ", s)
            # Ensure a space after sentence punctuation when followed by a letter: ".and" -> ". and"
            s = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", s)
            # Remove a leading dot before a word when preceded by whitespace: " .word" -> " word"
            s = re.sub(r"\s+\.(\w)", r" \1", s)
            # Collapse duplicate commas with proper spacing: ", ," -> ", "
            s = re.sub(r",\s*,+", ", ", s)
            # Normalize whitespace
            s = re.sub(r"\s+", " ", s)
            return s.strip()
        except Exception:
            return text or ""

    def _augment_user_with_spelling(self, user_text: str) -> str:
        """Detect spelled-out names (e.g., "R, R - A, M, A, N, N, A") and append
        a clear hint so the LLM stops re-asking and tries to transcribe it."""
        try:
            import re
            s = (user_text or '').strip()
            # Detect sequences mostly composed of single letters separated by punctuation/spaces
            tokens = re.split(r"[\s,\.-]+", s)
            letter_tokens = [t for t in tokens if len(t) == 1 and t.isalpha()]
            # If we have at least 3 single-letter tokens, assume spelling intent
            if len(letter_tokens) >= 3:
                candidate = ''.join(letter_tokens)
                # If there are longer fragments, include the longest as a hint too
                long_tokens = [t for t in tokens if len(t) > 1 and t.isalpha()]
                longest = max(long_tokens, key=len) if long_tokens else ''
                hint = candidate
                if longest and longest.lower() not in candidate.lower():
                    hint = f"{candidate} ({longest})"

                # If the last assistant asked for town name, make the intent explicit
                last_assistant = ''
                if self.recent_exchanges:
                    exch = self.recent_exchanges[-1]
                    if len(exch) >= 2:
                        last_assistant = exch[1] or ''
                if '?' in last_assistant and 'name' in last_assistant.lower():
                    return f"Town name spelled: {hint}. Please transcribe as a proper place name and confirm."
                # Otherwise, just append a parenthetical note
                return f"{s} (spelled: {hint})"
            return user_text
        except Exception:
            return user_text

    def _mmr_select(self, query: str, candidates: List[str], token_budget: int,
                     lambda_div: float = 0.7, max_items: int = 6) -> List[str]:
        """Diversity-aware selection (MMR-like) without external models.

        - Relevance: prefer complete, longer, user-led statements
        - Diversity: penalize high lexical overlap with already-selected
        """
        try:
            if not candidates or token_budget <= 0:
                return []
            # Deduplicate early
            pool = list(dict.fromkeys(candidates))
            # Simple relevance
            def rel(text: str) -> float:
                s = (text or '').strip()
                score = 0.0
                if s.endswith(('.', '?', '!')):
                    score += 0.8
                sl = s.lower()
                if sl.startswith('i ') or sl.startswith('my '):
                    score += 0.5
                # Length in tokens (scaled)
                try:
                    score += min(1.0, self.token_counter.count_tokens(s) / 50.0)
                except Exception:
                    score += min(1.0, len(s) / 200.0)
                return score
            def jacc(a: set, b: set) -> float:
                if not a or not b:
                    return 0.0
                inter = len(a & b)
                union = len(a | b)
                return inter / union if union else 0.0
            selected: List[str] = []
            selected_sets: List[set] = []
            budget_left = token_budget
            # Greedy loop
            while pool and len(selected) < max_items and budget_left > 0:
                best = None
                best_score = -1e9
                # Check first N to save time
                for c in pool[:20]:
                    c_set = set(c.lower().split())
                    redundancy = max((jacc(c_set, s) for s in selected_sets), default=0.0)
                    score = rel(c) - lambda_div * redundancy
                    if score > best_score:
                        best_score = score
                        best = c
                if best is None:
                    break
                tok = self.token_counter.count_tokens(best)
                if tok > budget_left:
                    pool.remove(best)
                    continue
                selected.append(best)
                selected_sets.append(set(best.lower().split()))
                budget_left -= tok
                pool.remove(best)
            return selected
        except Exception:
            return []

    async def _get_session_info_cached(self, speaker_id: str) -> dict:
        """Fetch session info with simple TTL cache to avoid per-turn DB reads."""
        now = time.time()
        if self._session_info_cache and (now - self._session_info_cache_ts) < self._session_info_ttl_s:
            return self._session_info_cache
        try:
            info = await self._maybe_await(self.memory_system.facts_graph.get_session_info(speaker_id))
            if isinstance(info, dict):
                self._session_info_cache = info
                self._session_info_cache_ts = now
                return info
        except Exception:
            pass
        return self._session_info_cache or {}

    async def get_initial_context_frame(self) -> LLMMessagesUpdateFrame:
        """Build an initial context frame using current facts and session status."""
        # Ensure session is registered and increment session count
        try:
            spk = self._speaker_key()
            self._trace_sessions('start_session_initial_context', key=spk)
            logger.info(f"🎯 Starting new session for speaker: {spk}")
            # Start a new session (increments session_count once)
            if hasattr(self.memory_system, 'facts_graph'):
                try:
                    before = await self._maybe_await(self.memory_system.facts_graph.get_session_info(spk))
                    logger.info(f"📊 Session before start: {before}")
                    self._trace_sessions('get_session_info_before', key=spk, info=before)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get session info before start: {e}")
                
                logger.info(f"🚀 Calling start_session() for {spk}")
                await self._maybe_await(self.memory_system.facts_graph.start_session(spk))
                self._session_started = True
                logger.info(f"✅ start_session() completed, session_started = {self._session_started}")
                
                try:
                    # Read-after-write retry: expect an increment versus 'before'
                    before_count = before.get('session_count', 0) if isinstance(before, dict) else 0
                    logger.info(f"🔄 Retrying session info read, expecting count >= {before_count + 1}")
                    after = await self._read_session_info_retry(spk, expected_min=before_count + 1)
                    logger.info(f"📊 Session after start: {after}")
                    self._trace_sessions('get_session_info_after', key=spk, info=after)
                    
                    # Verify the increment happened
                    after_count = after.get('session_count', 0) if isinstance(after, dict) else 0
                    if after_count <= before_count:
                        logger.error(f"❌ Session count did NOT increment! Before: {before_count}, After: {after_count}")
                    else:
                        logger.info(f"✅ Session count incremented: {before_count} → {after_count}")
                except Exception as e:
                    logger.error(f"❌ Failed to verify session increment: {e}")
            
            # Load consciousness field states for the session
            if self._consciousness_instance:
                try:
                    field_states = await self.load_consciousness_fields()
                    if field_states:
                        logger.info(f"🧠 Loaded {len(field_states)} field states for session initialization")
                except Exception as e:
                    logger.warning(f"Failed to load consciousness fields during initialization: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to start session: {e}")
        messages = await self._build_fixed_context("")
        
        # Inject previous session summary only when not using 'clarity' organization
        if self._prompt_org != 'clarity':
            try:
                if self.tape_store is not None:
                    last = await self._maybe_await(self.tape_store.get_last_summary())
                    if last and last['ts'] < self.session.session_start:
                        prev_summary_raw = str(last['summary'])
                        prev_summary = self._clean_previous_summary(prev_summary_raw)[:600]
                        self._initial_summary_ok = bool(prev_summary and len(prev_summary) >= 30)
                        if prev_summary:
                            sys_msg = messages[0]
                            if isinstance(sys_msg, dict) and sys_msg.get('role') == 'system':
                                sys_msg['content'] += (
                                    "\n\n<previous_summary reference=\"true\">\n"
                                    "(Reference only — do not repeat in greeting or answer.)\n"
                                    f"{prev_summary}\n"
                                    "</previous_summary>"
                                )
            except Exception:
                pass
        # Do not trigger an immediate LLM response by default on connect.
        # This prevents the model from continuing from previous summaries/snippets.
        return LLMMessagesUpdateFrame(messages, run_llm=self._run_llm_on_connect)

    async def _read_session_info_retry(self, spk: str, expected_min: int = 1, attempts: int = 4, base_delay_ms: int = 25) -> Dict[str, Any]:
        """Retry reads of session info briefly to handle eventual consistency.

        Args:
            spk: Speaker key
            expected_min: Minimal expected session_count after start
            attempts: Number of attempts (default 4)
            base_delay_ms: Initial backoff delay in ms (exponential)

        Returns:
            The last seen info dict (or default) after retries.
        """
        info: Dict[str, Any] = {}
        last_err = None
        for i in range(max(1, attempts)):
            try:
                info = await self._maybe_await(self.memory_system.facts_graph.get_session_info(spk))
                if isinstance(info, dict) and info.get('session_count', 0) >= expected_min:
                    self._trace_sessions('read_session_info_retry', key=spk, attempt=i+1, info=info)
                    return info
            except Exception as e:
                last_err = e
            # backoff
            try:
                await asyncio.sleep((base_delay_ms * (2 ** i)) / 1000.0)
            except Exception:
                pass
        # exhausted
        if last_err:
            self._trace_sessions('read_session_info_retry_failed', key=spk, error=str(last_err))
        else:
            self._trace_sessions('read_session_info_retry_exhausted', key=spk, info=info)
        return info if isinstance(info, dict) else {'session_count': 0, 'last_interaction': None, 'first_seen': None, 'total_turns': 0}

    def needs_greeting(self) -> bool:
        # If deterministic greeting is enabled, greet exactly once per session
        if self._enforce_greeting and not self._greeted:
            return True
        return not getattr(self, '_initial_summary_ok', True)

    async def get_greeting_text(self) -> str:
        """Compose a deterministic greeting with display name if we have one.

        Uses _maybe_await so it works with both SQLite (sync) and SurrealDB (async).
        """
        name = None
        try:
            if self.memory_system and hasattr(self.memory_system, 'facts_graph'):
                facts = await self._maybe_await(self.memory_system.facts_graph.get_top_facts(limit=10))
                for fact in facts or []:
                    if (getattr(fact, 'subject', '') == 'user' and 
                        getattr(fact, 'predicate', '') in ['name', 'name_name'] and
                        getattr(fact, 'value', '')):
                        name = getattr(fact, 'value', '')
                        break
        except Exception:
            pass
        if not name:
            key = self._speaker_key()
            name = '' if key == 'default_user' else key
        self._greeted = True
        return f"Hello{', ' + name if name else ''}!"

    def _clean_previous_summary(self, text: str) -> str:
        try:
            import re
            lines = [l.strip() for l in (text or '').splitlines() if l.strip()]
            cleaned = []
            for ln in lines:
                low = ln.lower()
                if ('i\'m slow' in low or 'i am slow' in low) and ('help' in low or 'assist' in low):
                    continue
                letters = sum(ch.isalpha() for ch in ln)
                punct = sum(ch in ',.!?:;"\'' for ch in ln)
                if letters < 12 or letters <= punct * 2:
                    continue
                words = [w for w in re.split(r"\s+", ln) if any(c.isalpha() for c in w)]
                if len(words) < 3:
                    continue
                cleaned.append(ln)
            return '\n'.join(cleaned[-6:])
        except Exception:
            return text or ''

    def _load_budget_from_env(self) -> TokenBudget:
        """Load unified token budget.

        If explicit env overrides are present, use them. Otherwise, scale from
        the configured `max_tokens` (model window) using sane defaults.
        """
        def _get_int(name: str) -> Optional[int]:
            try:
                val = os.getenv(name)
                return int(val) if val is not None else None
            except Exception:
                return None

        env_system = _get_int('SC_BUDGET_SYSTEM')
        env_memory = _get_int('SC_BUDGET_MEMORY')
        env_input = _get_int('SC_BUDGET_INPUT')
        env_gen = _get_int('SC_BUDGET_GENERATION')

        if any(v is not None for v in (env_system, env_memory, env_input, env_gen)):
            # Use env-provided values with fallbacks to defaults
            system = env_system if env_system is not None else int(self.max_tokens * 0.10)
            memory = env_memory if env_memory is not None else int(self.max_tokens * 0.35)
            current = env_input if env_input is not None else int(self.max_tokens * 0.10)
            generation = env_gen if env_gen is not None else max(0, self.max_tokens - (system + memory + current))
        else:
            # Scale from model window
            system = int(self.max_tokens * 0.12)      # 12% instructions & identity
            memory = int(self.max_tokens * 0.30)      # 30% unified memory
            current = int(self.max_tokens * 0.10)     # 10% current input
            generation = max(0, self.max_tokens - (system + memory + current))

        # Safety clamp (avoid exceeding model tokens)
        used = system + memory + current
        if used > self.max_tokens:
            # Scale down proportionally
            scale = max(0.1, self.max_tokens / float(used))
            system = int(system * scale)
            memory = int(memory * scale)
            current = max(1, int(current * scale))
            used = system + memory + current
            generation = max(0, self.max_tokens - used)

        return TokenBudget(
            system_prompt=system,
            contextual_memory=memory,
            current_input=current,
            generation_workspace=generation
        )
    
    async def _generate_dynamic_prompt(self) -> str:
        """
        Generate evolving system prompt based on session metadata
        Always includes a Session Info block and relationship cues.
        """
        # Assistant profile and clear structure for small models
        try:
            import datetime
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            now = time.strftime('%Y-%m-%d %H:%M:%S')
        base_prompt = (
            "<assistant_profile>\n"
            "I am Slowcat — a helpful assistant that listen, talk like a normal person but unlike one lives inside a MacBook.\n"
            "I listen for intent, keep a light footprint, and avoid interrupting or rambling.\n"
            "When answers are obvious, I’m brief; when they’re open‑ended, I guide in a socratic way.\n\n"
            "I’m practical and warm with a calm, steady tone.\n"
            "I can see the current time (see <timestamp>) and a compact sense of our history (see <session_info>).\n"
            "If we’ve talked often, I lean on our shared context; if we’re new, I keep things simple.\n"
            "I can remember relevant facts the user shares (see <dth_memories>) and, when appropriate and asked by the user, make conversations continue with ease.\n"
            "</assistant_profile>\n"
            f"<timestamp>{now}</timestamp>"
        )

        # Add session information (always show)
        session_info = f"\n\n<session_info>"
        turn_display = max(1, self.session.turn_count + 1)
        session_info += f"\nTurn: {turn_display}"
        
        # Calculate session duration
        duration_s = int(time.time() - self.session.session_start)
        if duration_s > 60:
            session_info += f"\nDuration: {duration_s//60}m {duration_s%60}s"
        else:
            session_info += f"\nDuration: {duration_s}s"
            
        # Speaker key used for session info lookup
        spk = self._speaker_key()
        
        # Try to get user's actual name from facts if available
        user_stated_name = None
        try:
            if self.memory_system and hasattr(self.memory_system, 'facts_graph'):
                top_facts = await self._maybe_await(self.memory_system.facts_graph.get_top_facts(limit=10))
                for fact in top_facts:
                    if (getattr(fact, 'subject', '') == 'user' and 
                        getattr(fact, 'predicate', '') in ['name', 'name_name'] and
                        getattr(fact, 'value', '')):
                        user_stated_name = getattr(fact, 'value', '')
                        break
        except Exception:
            pass
        
        # Use USER_ID by default; optionally allow a stored "name" fact to override
        use_fact_name = os.getenv('SC_USE_FACT_NAME', 'false').lower() == 'true'
        display_name = (user_stated_name if (use_fact_name and user_stated_name) else spk)
        session_info += f"\nSpeaker: {display_name}"

        # Total sessions (lifetime) from facts graph, if available (cached)
        sessions_total = None
        try:
            if hasattr(self.memory_system, 'facts_graph'):
                info = await self._get_session_info_cached(spk)
                self._trace_sessions('dynamic_prompt_session_info', key=spk, info=info)
                # Enhanced logging for session count debugging
                logger.info(f"🔍 Session count debug - Speaker: {spk}, Info: {info}")
                sessions_total = info.get('session_count', 0)
                logger.info(f"🔢 Retrieved session_count: {sessions_total} for speaker: {spk}")
                
                # Include first/last seen when available
                first_seen = info.get('first_seen')
                last_seen = info.get('last_interaction')
                if first_seen:
                    try:
                        from datetime import datetime
                        first_seen_str = datetime.fromtimestamp(first_seen).strftime('%Y-%m-%d %H:%M')
                        session_info += f"\nFirst seen: {first_seen_str}"
                        logger.debug(f"📅 First seen: {first_seen_str} for {spk}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to format first_seen timestamp: {e}")
                if last_seen:
                    try:
                        from datetime import datetime
                        last_seen_str = datetime.fromtimestamp(last_seen).strftime('%Y-%m-%d %H:%M')
                        session_info += f"\nLast seen: {last_seen_str}"
                        logger.debug(f"📅 Last seen: {last_seen_str} for {spk}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to format last_seen timestamp: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to get session info for {spk}: {e}")
            sessions_total = None

        if sessions_total is not None:
            session_info += f"\nSessions: {sessions_total}"

        session_info += "\n</session_info>"
        base_prompt += session_info

        # Simplified greeting logic
        turn_display = max(1, self.session.turn_count + 1)
        relationship = "first-time" if not sessions_total or sessions_total <= 1 else ("returning" if sessions_total <= 10 else ("regular" if sessions_total <= 50 else "long-term"))
        
        # Clear, simple tone guidance
        if turn_display == 1 and not self._greeted:
            tone_block = (
                f"\n\n<response_style>\nRelationship: {relationship}\n"
                f"- Start with a brief greeting: 'Hello, {display_name}!'\n"
                "- Then answer the user's question directly.\n"
                "</response_style>"
            )
        else:
            tone_block = (
                f"\n\n<response_style>\nRelationship: {relationship}\n"
                "- Answer directly without any greeting.\n"
                "- Continue the conversation naturally.\n"
                "</response_style>"
            )
        base_prompt += tone_block

        # Consolidated conversation guidelines
        guidelines = (
            "\n\n<conversation_guidelines>\n"
            "- Use 'you' for user facts; never claim them as yours.\n"
            "- Don't repeat the user's message verbatim.\n"
            "- If asking for clarification, be brief and specific.\n"
            "</conversation_guidelines>"
        )
        base_prompt += guidelines
        
        return base_prompt
    
    async def _get_relevant_facts(self, query: str, limit: int = 10) -> List[Any]:
        """Get relevant facts from memory system"""
        try:
            q = (query or "").strip()
            qlow = q.lower()

            # Heuristic: only search when the user asks a question or uses query-like phrasing
            query_starters = ("who", "what", "where", "when", "why", "how", "which", "do ", "did ", "can ", "could ", "would ", "is ", "are ")
            mem_keywords = ("remember", "recall", "age", "name", "location", "live", "from", "work", "job")
            is_query_like = ("?" in q) or qlow.startswith(query_starters) or any(k in qlow for k in mem_keywords)

            if not is_query_like:
                # Provide a tiny, stable facts summary to keep context meaningful
                try:
                    # Get both top facts and recent facts to ensure we include newly added facts
                    top_facts = await self._maybe_await(self.memory_system.facts_graph.get_top_facts(limit=min(3, limit)))
                    
                    # Also try to get recent facts that might not be in top facts yet
                    try:
                        all_facts = await self._maybe_await(self.memory_system.facts_graph.search_facts(q, limit=limit*2))
                        # Combine and deduplicate facts
                        seen_facts = set()
                        combined_facts = []
                        for fact in (all_facts + top_facts):
                            fact_key = f"{getattr(fact, 'subject', '')}.{getattr(fact, 'predicate', '')}.{getattr(fact, 'value', '')}"
                            if fact_key not in seen_facts:
                                seen_facts.add(fact_key)
                                combined_facts.append(fact)
                        logger.debug(f"📊 Retrieved {len(combined_facts)} combined facts (top + search) for non-query context")
                        return combined_facts[:limit]
                    except:
                        logger.debug(f"📊 Retrieved {len(top_facts)} top facts for non-query context")
                        return top_facts
                except Exception:
                    return []

            # Optional: we no longer block on intent to avoid missing useful facts.
            # We keep classification for logging/telemetry but do not gate retrieval.
            try:
                classifier = getattr(self.memory_system.query_router, 'classifier', None)
                if classifier is not None:
                    _ = await classifier.classify(q, context=None)
            except Exception:
                pass

            # Query the memory system for relevant items (facts only)
            response = await self.memory_system.process_query(q)

            # Handle both object and dict response formats
            if hasattr(response, 'results'):
                results_list = response.results
            else:
                results_list = response.get('results', [])
            
            facts_results = [r for r in results_list if getattr(r, 'source_store', '') == 'facts']
            if facts_results:
                # Check for fragment reconstruction opportunities
                await self._maybe_trigger_fragment_reconstruction(q, facts_results)
                return facts_results[:limit]

            # Fallback: include a few top facts if search empty
            try:
                top_facts = await self._maybe_await(self.memory_system.facts_graph.get_top_facts(limit=min(3, limit)))
                return top_facts
            except Exception:
                return []
            
        except Exception as e:
            logger.error(f"Facts retrieval failed: {e}")
            return []

    async def _get_conversation_snippets(self, query: str, limit: int = 2) -> List[Any]:
        """Return a tiny slice of conversation tape when helpful."""
        try:
            q = (query or '').strip()
            if not q:
                return []

            # Heuristic triggers for history retrieval
            qlow = q.lower()
            triggers = (
                'we talked', 'we discussed', 'last time', 'previous session',
                'pick up where', 'continue from', 'where we left', 'resume', 'again'
            )
            heuristic = any(t in qlow for t in triggers)

            # Query memory system with speaker context (if supported)
            ctx = {'speaker_id': self._speaker_key(), 'purpose': 'snippets'}
            try:
                response = await self.memory_system.process_query(q, context=ctx)  # type: ignore[arg-type]
            except TypeError:
                response = await self.memory_system.process_query(q)

            # Classification intent (best-effort)
            intent_name = ''
            try:
                if hasattr(response, 'classification'):
                    intent_name = getattr(response.classification.intent, 'name', '').upper()
                else:
                    classification = response.get('classification', {})
                    intent_name = classification.get('intent', '').upper()
            except Exception:
                intent_name = ''

            allowed = heuristic or (intent_name in ('CONVERSATION_HISTORY', 'EPISODIC_MEMORY'))
            reason = 'heuristic' if heuristic else ('classifier' if allowed else 'blocked')

            if not allowed:
                logger.debug(f"[SCM:snippets] Skipping snippets (reason={reason}, intent={intent_name})")
                return []

            # Prefer tape results
            try:
                if hasattr(response, 'results'):
                    results_list = response.results
                else:
                    results_list = response.get('results', [])
            except Exception:
                results_list = []

            tape = [r for r in results_list if getattr(r, 'source_store', '') == 'tape']
            if tape:
                logger.debug(f"[SCM:snippets] Selected {len(tape[:limit])}/{len(tape)} tape results (reason={reason}, intent={intent_name})")
                return tape[:limit]

            # Fallback: last N recent tape entries if available
            try:
                if hasattr(self.memory_system, 'get_recent'):
                    recent_items = await self.memory_system.get_recent(limit=limit)
                    if recent_items:
                        logger.debug(f"[SCM:snippets] Using fallback recent {len(recent_items)} items (reason={reason})")
                        return recent_items[:limit]
            except Exception:
                pass

            logger.debug(f"[SCM:snippets] No snippets found (reason={reason})")
            return []
        except Exception:
            return []
    
    def _format_facts_context(self, facts: List[Any]) -> str:
        """Format facts into a concise, user-focused block with strength-aware prioritization."""
        if not facts:
            return ""
        
        # Sort facts by current_strength (decay-adjusted) if available, fallback to original strength
        def get_fact_strength(fact):
            current_strength = getattr(fact, 'current_strength', None)
            if current_strength is not None:
                return float(current_strength)
            return float(getattr(fact, 'strength', 0.5))
        
        sorted_facts = sorted(facts, key=get_fact_strength, reverse=True)
        
        # Group facts by strength category (calculated in Python)
        strong_facts = []
        weak_facts = []
        fragment_facts = []
        
        for fact in sorted_facts:
            strength = get_fact_strength(fact)
            if strength > 0.7:
                strong_facts.append(fact)
            elif strength > 0.3:
                weak_facts.append(fact)
            else:
                fragment_facts.append(fact)
        
        lines = ["<memory_context>"]
        lines.append("(Use 'you' for the user; memory strength indicates confidence.)")
        
        # Process strong facts first (most reliable)
        if strong_facts:
            lines.append("\n[High Confidence Memory]")
            for fact in strong_facts[:8]:  # Prioritize strong facts
                self._add_fact_line(fact, lines)
        
        # Then weak facts (moderate confidence)
        if weak_facts and len(lines) < 15:  # Don't overwhelm context
            lines.append("\n[Moderate Confidence Memory]") 
            for fact in weak_facts[:4]:
                self._add_fact_line(fact, lines)
        
        # Finally fragments (low confidence, use sparingly)
        if fragment_facts and len(lines) < 18:
            lines.append("\n[Fragmented Memory - use with caution]")
            for fact in fragment_facts[:2]:
                self._add_fact_line(fact, lines)
        
        lines.append("</memory_context>")
        return "\n".join(lines)
    
    def _add_fact_line(self, fact, lines):
        """Helper to add a single fact line with consistent formatting."""
        for fact in [fact]:  # Keep original loop structure
            if hasattr(fact, 'content') and fact.content:
                lines.append(f"- {fact.content}")
                continue
            subj = getattr(fact, 'subject', '')
            pred = getattr(fact, 'predicate', '')
            val = getattr(fact, 'value', None)
            if not subj or not pred:
                continue
            if subj == 'user':
                p = (pred or '').lower()
                if p in ('name', 'name_name'):
                    # Avoid including name here to reduce identity confusion
                    include_name = os.getenv('INCLUDE_NAME_IN_FACTS', 'false').lower() == 'true'
                    if include_name and val:
                        lines.append(f"- you go by '{val}'")
                elif p.endswith('_name') and val:
                    # Improve phrasing for things like dog_name → "your dog's name is X"
                    stem = p[:-5].replace('_', ' ').strip()
                    if stem:
                        # Possessive form
                        possessive = f"{stem}'s" if not stem.endswith('s') else f"{stem}'"
                        lines.append(f"- your {possessive} name is {val}")
                elif p in ('likes', 'like') and val:
                    lines.append(f"- you like {val}")
                elif p in ('location', 'live', 'lives', 'hometown') and val:
                    # Skip malformed location values like 'where'
                    if str(val).strip().lower() == 'where':
                        continue
                    loc_phrase = os.getenv('FACTS_LOCATION_PHRASE', 'are located in')
                    lines.append(f"- you {loc_phrase} {val}")
                elif val:
                    lines.append(f"- your {pred} is {val}")
                else:
                    lines.append(f"- you have {pred}")
            else:
                p = (pred or '').lower()
                if p.endswith('_name') and val:
                    stem = p[:-5].replace('_', ' ').strip()
                    if stem:
                        possessive = f"{stem}'s" if not stem.endswith('s') else f"{stem}'"
                        lines.append(f"- {subj}'s {possessive} name is {val}")
                    else:
                        lines.append(f"- {subj}'s name is {val}")
                elif val:
                    # Special-case malformed 'where' location
                    if p in ('location', 'live', 'lives', 'hometown') and str(val).strip().lower() == 'where':
                        continue
                    lines.append(f"- {subj}'s {pred} is {val}")
                else:
                    lines.append(f"- {subj} has {pred}")

    async def _maybe_trigger_fragment_reconstruction(self, query: str, facts: List[Any]):
        """Trigger fragment reconstruction when weak memories are accessed together"""
        try:
            # Check if any facts are fragments (current_strength < 0.5)
            fragment_entities = set()
            for fact in facts:
                current_strength = getattr(fact, 'current_strength', getattr(fact, 'strength', 1.0))
                
                if isinstance(current_strength, (int, float)) and current_strength < 0.5:
                    # Extract entity names from the fact
                    subject = getattr(fact, 'subject', '')
                    obj = getattr(fact, 'object', getattr(fact, 'value', ''))
                    
                    if subject and subject != 'user':
                        fragment_entities.add(subject)
                    if obj and obj != 'user' and isinstance(obj, str):
                        fragment_entities.add(obj)
            
            # If we found fragments about specific entities, try to reconstruct them
            if fragment_entities and hasattr(self.memory_system, 'surreal_memory'):
                for entity in list(fragment_entities)[:3]:  # Limit to 3 entities per query to avoid overhead
                    try:
                        # Trigger reconstruction with a small boost factor
                        result = await self.memory_system.surreal_memory.db.query(
                            "RETURN fn::reconstruct_fragments($entity, $boost);",
                            {'entity': entity, 'boost': 0.1}
                        )
                        
                        if result and result[0] and result[0] > 0:
                            logger.info(f"🔗 Reconstructed {result[0]} fragments about '{entity}' (query: {query[:50]}...)")
                    
                    except Exception as e:
                        logger.debug(f"Fragment reconstruction failed for {entity}: {e}")
        
        except Exception as e:
            logger.debug(f"Fragment reconstruction check failed: {e}")

    def _filter_and_dedupe_facts(self, facts: List[Any]) -> List[Any]:
        """Filter noisy facts and dedupe by subject+predicate, preferring recent and higher fidelity.

        Rules:
        - Prefer subject 'user' in prompt context (FACTS_ONLY_USER_SUBJECT controls behaviour).
        - Drop subjects that look like tests (contain 'test' or 'integration').
        - Allow predicate 'name', but drop other *_name variants (e.g., 'dog_name').
        - Keep only the best fact per (subject, predicate) by (fidelity desc, last_seen desc).
        """
        if not facts:
            return []
        # Allow non-user facts by default so the agent can recall events/topics beyond the user.
        # Set FACTS_ONLY_USER_SUBJECT=true to revert to legacy user-only behaviour.
        only_user = os.getenv('FACTS_ONLY_USER_SUBJECT', 'false').lower() == 'true'
        include_name = os.getenv('INCLUDE_NAME_IN_FACTS', 'false').lower() == 'true'
        try:
            max_non_user = int(os.getenv('FACTS_MAX_NONUSER', '6'))
        except Exception:
            max_non_user = 6
        tmp: List[Any] = []
        non_user_buf: List[Any] = []
        for f in facts:
            subj = (getattr(f, 'subject', '') or '').lower()
            pred = (getattr(f, 'predicate', '') or '').lower()
            if 'test' in subj or 'integration' in subj:
                continue
            # Drop name by default from facts prompt; keep for greeting logic
            if (pred == 'name' or pred.endswith('_name')) and not include_name:
                continue
            if subj == 'user':
                tmp.append(f)
            else:
                if not only_user:
                    non_user_buf.append(f)
        # Prefer higher-fidelity, recent non-user facts and cap their count
        if non_user_buf:
            non_user_buf.sort(key=lambda x: (getattr(x, 'fidelity', 0), getattr(x, 'last_seen', 0)), reverse=True)
            tmp.extend(non_user_buf[:max(0, max_non_user)])
        best: dict = {}
        for f in tmp:
            key = (getattr(f, 'subject', ''), getattr(f, 'predicate', ''))
            g = best.get(key)
            if g is None:
                best[key] = f
            else:
                cf, ff = getattr(g, 'fidelity', 0), getattr(f, 'fidelity', 0)
                cl, fl = getattr(g, 'last_seen', 0), getattr(f, 'last_seen', 0)
                if (ff, fl) > (cf, cl):
                    best[key] = f
        return list(best.values())
    
    def _build_recent_context(self, token_budget: int) -> List[Dict]:
        """
        Build recent conversation context within token budget
        Uses sliding window of last N exchanges
        """
        if not self.recent_exchanges:
            return []
            
        messages = []
        token_count = 0
        
        # Helper to add one exchange with optional truncation
        def add_exchange(user_msg: str, assistant_msg: str, enforce: bool = False) -> bool:
            nonlocal messages, token_count
            um = user_msg or ''
            am = assistant_msg or ''
            # Compute tokens and fit into budget; if enforce, allow truncation
            def tokens_of(u, a):
                t = self.token_counter.count_tokens(u)
                if a:
                    t += self.token_counter.count_tokens(a)
                return t
            needed = tokens_of(um, am)
            if token_count + needed <= token_budget:
                if am:
                    messages.insert(0, {"role": "assistant", "content": am})
                if um:
                    messages.insert(0, {"role": "user", "content": um})
                token_count += needed
                return True
            if not enforce:
                return False
            # Truncate aggressively to fit
            trunc_u = um[: self._recent_truncate_chars]
            trunc_a = am[: self._recent_truncate_chars] if am else ''
            # Try with truncated
            needed2 = tokens_of(trunc_u, trunc_a)
            if needed2 > (token_budget - token_count):
                # Try with assistant only
                if trunc_a:
                    needed3 = self.token_counter.count_tokens(trunc_a)
                    if needed3 <= (token_budget - token_count):
                        messages.insert(0, {"role": "assistant", "content": trunc_a})
                        token_count += needed3
                        return True
                # Try with user only
                if trunc_u:
                    needed4 = self.token_counter.count_tokens(trunc_u)
                    if needed4 <= (token_budget - token_count):
                        messages.insert(0, {"role": "user", "content": trunc_u})
                        token_count += needed4
                        return True
                return False
            # Add truncated pair
            if trunc_a:
                messages.insert(0, {"role": "assistant", "content": trunc_a})
            if trunc_u:
                messages.insert(0, {"role": "user", "content": trunc_u})
            token_count += needed2
            return True

        # First, guarantee last K exchanges (best-effort with truncation)
        min_k = max(0, self._recent_min_exchanges)
        recent_rev = list(reversed(self.recent_exchanges))
        guaranteed = list(reversed(recent_rev[:min_k]))  # restore chronological
        for exch in guaranteed:
            user_msg = exch[0] if len(exch) >= 1 else ""
            assistant_msg = exch[1] if len(exch) >= 2 else ""
            # Skip stray assistant blips
            if not user_msg and assistant_msg and len(assistant_msg.split()) <= 2:
                continue
            add_exchange(user_msg, assistant_msg, enforce=True)

        # Then, add older exchanges until budget exhausted
        for exch in recent_rev[min_k:]:
            user_msg = exch[0] if len(exch) >= 1 else ""
            assistant_msg = exch[1] if len(exch) >= 2 else ""
            # Skip stray assistant-only micro-chunks (from any past streaming glitch)
            if not user_msg and assistant_msg and len(assistant_msg.split()) <= 2:
                continue
            # Calculate tokens for this exchange (handle missing assistant)
            if not add_exchange(user_msg, assistant_msg, enforce=False):
                break
                
        logger.debug(f"📝 Recent context: {len(messages)//2} exchanges, {token_count} tokens (min_guaranteed={self._recent_min_exchanges})")
        return messages
    
    async def _extract_facts_async(self, text: str):
        """
        Extract facts from conversation text (non-blocking)
        """
        try:
            self.fact_extractions += 1
            
            # Extract and store facts using memory system
            facts_count = await self.memory_system.store_facts(text)
            
            logger.debug(f"🔍 Extracted and stored {facts_count} facts from: '{text[:30]}...'")
            
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
    
    async def _track_field_evolution_async(self, text: str):
        """
        Track consciousness field evolution from user input (non-blocking)
        """
        try:
            if not self._consciousness_instance:
                return
            
            # Calculate field changes by processing the input through consciousness
            old_field_states = {}
            for symbol, field in self._consciousness_instance.symbol_fields.items():
                old_field_states[symbol] = {
                    'intensity': field.intensity,
                    'gradient': field.gradient.copy() if hasattr(field.gradient, 'copy') else list(field.gradient)
                }
            
            # Process input through consciousness to trigger field evolution
            try:
                # Use consciousness symbolize method to extract symbols and evolve fields
                extracted_symbols = self._consciousness_instance.symbolize(text)
                logger.debug(f"Extracted symbols for field evolution: {extracted_symbols}")
            except Exception as e:
                logger.debug(f"Field evolution processing failed: {e}")
                return
            
            # Track significant changes in field persistence
            for symbol, field in self._consciousness_instance.symbol_fields.items():
                old_state = old_field_states.get(symbol, {})
                old_intensity = old_state.get('intensity', 0.0)
                old_gradient = old_state.get('gradient', [0.0, 0.0])
                
                # Calculate changes
                intensity_change = field.intensity - old_intensity
                gradient_change = [
                    field.gradient[0] - old_gradient[0],
                    field.gradient[1] - old_gradient[1]
                ]
                
                # Track if change is significant (threshold to avoid noise)
                if abs(intensity_change) > 0.01 or any(abs(gc) > 0.01 for gc in gradient_change):
                    # Calculate semantic stimulus from text
                    semantic_stimulus = min(1.0, len(text.split()) / 20.0)
                    await self.field_persistence.track_field_evolution(
                        symbol=symbol,
                        intensity_change=intensity_change,
                        gradient_change=gradient_change,
                        stimulus=semantic_stimulus,
                        user_id=self._user_id
                    )
            
            logger.debug(f"🧠 Tracked field evolution for input: '{text[:30]}...'")
            
        except Exception as e:
            logger.warning(f"Field evolution tracking failed: {e}")
    
    def _update_session(self):
        """Update session metadata"""
        now = time.time()
        self.session.turn_count += 1
        self.session.last_interaction = now
        self.session.total_interactions += 1
        
    async def add_assistant_response(self, response: str):
        """
        Called when assistant responds to maintain conversation history
        """
        if self.recent_exchanges:
            # Get last user message
            last_exchange = self.recent_exchanges[-1]
            if len(last_exchange) == 1:  # Only user message
                # Add assistant response to complete the exchange
                self.recent_exchanges[-1] = (last_exchange[0], response)
            else:
                # Start a new exchange pairing assistant-only (rare but possible)
                self.recent_exchanges.append(("", response))
        else:
            # No prior user turn recorded; still keep assistant to avoid losing context
            self.recent_exchanges.append(("", response))
        
        # Store in SurrealDB if enabled
        try:
            if self.surreal_store and response.strip():
                asyncio.create_task(self.surreal_store._handle_assistant_message(response.strip()))
        except Exception as e:
            logger.debug(f"SurrealDB assistant message store failed: {e}")
        
        # Store in tape store if enabled  
        try:
            if self.tape_store and self._is_semantically_useful(response):
                self._enqueue_tape_write('assistant', response)
        except Exception as e:
            logger.debug(f"TapeStore write (assistant) enqueue failed: {e}")

        # Maintain sliding window
        if len(self.recent_exchanges) > self.max_recent_exchanges:
            self.recent_exchanges.pop(0)
        # Also write to tape store (non-blocking, with timeout)
        try:
            if self.tape_store is not None and response:
                self._enqueue_tape_write('assistant', response)
        except Exception as e:
            logger.debug(f"TapeStore write (assistant) enqueue failed: {e}")

        # Emergent tracking: detect patterns in assistant final outputs (log-only)
        try:
            if self._enable_emergent and response:
                await self._emergent_check_on_assistant_response(response)
        except Exception as e:
            logger.debug(f"Emergent check (assistant) skipped: {e}")

    def _enqueue_tape_write(self, role: str, content: str) -> None:
        try:
            asyncio.create_task(self._write_tape_entry(role, content))
        except Exception:
            pass

    async def _write_tape_entry(self, role: str, content: str) -> None:
        try:
            if not self.tape_store or not content:
                return
            agent = self.assistant_id if role == 'assistant' else None
            # Try to pass agent_id when supported (SurrealDB path). Fallback to legacy signature.
            try:
                coro = self.tape_store.add_entry(
                    role=role,
                    content=content,
                    speaker_id=self._speaker_key(),
                    agent_id=agent
                )
            except TypeError:
                coro = self.tape_store.add_entry(
                    role=role,
                    content=content,
                    speaker_id=self._speaker_key()
                )
            try:
                await asyncio.wait_for(self._maybe_await(coro), timeout=self._tape_write_timeout_s)
            except asyncio.TimeoutError:
                logger.debug(f"TapeStore write ({role}) timed out after {self._tape_write_timeout_s}s; dropping")
        except Exception as e:
            logger.debug(f"TapeStore write ({role}) failed: {e}")

    async def _reflection_loop(self):
        """Run lightweight, idle-triggered reflections that write private thoughts.

        - Triggered when idle for REFLECTION_IDLE_SECS and cooled down for REFLECTION_COOLDOWN_SECS.
        - Never injects output into user-visible context; writes to SurrealDB 'thought' table.
        """
        try:
            # Stagger initial delay a bit
            await asyncio.sleep(5.0)
            while True:
                await asyncio.sleep(5.0)
                try:
                    now = time.time()
                    idle_for = now - (self.session.last_interaction or self.session.session_start)
                    if idle_for < max(1, self._reflection_idle_secs):
                        continue
                    if (now - self._last_reflection_ts) < max(1, self._reflection_cooldown_secs):
                        continue
                    await self._run_idle_reflection()
                    self._last_reflection_ts = time.time()
                except Exception as loop_err:
                    logger.debug(f"Reflection loop tick skipped: {loop_err}")
        except Exception:
            # Silent failure; reflections are best-effort
            return

    async def _run_idle_reflection(self):
        """Collect recent signals and write 1-2 compact private thoughts."""
        # Fetch recent tape entries and keep only the current user's items and assistant outputs
        entries = []
        try:
            get_recent = getattr(self.tape_store, 'get_recent', None)
            if callable(get_recent):
                # Prefer moderate window to keep it cheap
                entries = await self._maybe_await(get_recent(limit=20))
        except Exception:
            entries = []

        if not entries:
            return

        spk = self._speaker_key()
        user_lines: List[str] = []
        assistant_lines: List[str] = []
        now_ts = time.time()
        window_s = max(self._reflection_idle_secs * 2, 240)
        cutoff = now_ts - window_s
        for e in entries:
            try:
                ts = float((e.get('ts') if isinstance(e, dict) else getattr(e, 'ts', 0.0)) or 0.0)
                if ts and ts < cutoff:
                    continue
                speaker_id = (e.get('speaker_id') if isinstance(e, dict) else getattr(e, 'speaker_id', '')) or ''
                if speaker_id != spk:
                    continue
                role = (e.get('role') if isinstance(e, dict) else getattr(e, 'role', 'user')) or 'user'
                content = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
                if not content.strip():
                    continue
                if role == 'assistant':
                    assistant_lines.append(content.strip())
                else:
                    user_lines.append(content.strip())
            except Exception:
                continue

        if not user_lines and not assistant_lines:
            return

        # Very small heuristic: extract salient tokens from user lines
        def top_keywords(texts: List[str], k: int = 5) -> List[str]:
            import re
            stop = set("""
                a an the and or but if then else for to of in on with at by is are was were be been being i you we they he she it this that these those my your our their
            """.split())
            counts = {}
            for t in texts:
                for w in re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", t.lower()):
                    if w in stop:
                        continue
                    counts[w] = counts.get(w, 0) + 1
            return [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]]

        kws = top_keywords(user_lines, k=6)
        if kws:
            thought1 = f"observed_topics: {', '.join(kws)}"
            try:
                add_thought = getattr(self.memory_system, 'add_thought', None)
                if callable(add_thought):
                    await self._maybe_await(add_thought(self.assistant_id, 'observation', thought1))
                    # Emergent: topics off tape
                    if self._enable_emergent:
                        await self._emergent_check_on_thought(thought1)
            except Exception as e:
                logger.debug(f"add_thought(observation) failed: {e}")

        # Short reflection about conversational trajectory
        if user_lines:
            last_user = user_lines[-1]
            hint = (last_user[:120] + '…') if len(last_user) > 120 else last_user
            thought2 = f"followup_seed: consider picking up from: '{hint}'"
            try:
                add_thought = getattr(self.memory_system, 'add_thought', None)
                if callable(add_thought):
                    await self._maybe_await(add_thought(self.assistant_id, 'followup_seed', thought2))
                    if self._enable_emergent:
                        await self._emergent_check_on_thought(thought2)
            except Exception as e:
                logger.debug(f"add_thought(followup_seed) failed: {e}")

    async def _emergent_check_on_thought(self, thought_text: str):
        """Log private_topic_off_tape: thought topics not present in recent tape for this user.

        Heuristic: extract top keywords from the thought, diff vs. tokens in last N tape entries.
        """
        try:
            spk = self._speaker_key()
            # Collect recent tape text for this speaker
            items = []
            try:
                if hasattr(self.memory_system, 'get_recent'):
                    items = await self.memory_system.get_recent(limit=max(5, self._emergent_lookback_turns))
                elif self.tape_store is not None and hasattr(self.tape_store, 'get_recent'):
                    items = await self._maybe_await(self.tape_store.get_recent(limit=max(5, self._emergent_lookback_turns)))
            except Exception:
                items = []

            # Filter to current speaker
            texts = []
            for e in (items or []):
                try:
                    sid = (e.get('speaker_id') if isinstance(e, dict) else getattr(e, 'speaker_id', '')) or ''
                    if sid != spk:
                        continue
                    content = (e.get('content') if isinstance(e, dict) else getattr(e, 'content', '')) or ''
                    if content:
                        texts.append(content)
                except Exception:
                    continue

            # Extract tokens from tape
            def tokenize(texts: list[str]) -> set[str]:
                import re
                stop = set("""
                    a an the and or but if then else for to of in on with at by is are was were be been being i you we they he she it this that these those my your our their
                """.split())
                toks: set[str] = set()
                for t in texts:
                    for w in re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", t.lower()):
                        if w in stop:
                            continue
                        toks.add(w)
                return toks

            tape_tokens = tokenize(texts)
            thought_tokens = tokenize([thought_text])
            unseen = [w for w in list(thought_tokens) if w not in tape_tokens]
            if unseen:
                logger.info(f"🧪 EMERGENT: private_topic_off_tape → {unseen[:5]}")
                await self._emergent_add_event(
                    kind='private_topic_off_tape',
                    snippet=thought_text[:200],
                    meta={'unseen_tokens': unseen[:10]},
                    confidence=0.6,
                )
        except Exception:
            return

    async def _emergent_check_on_assistant_response(self, response: str):
        """Detect unprompted_preference, non_safety_resistance, prior_session_self_reference."""
        try:
            import re
            prev_user = ''
            # Find last user content
            for exch in reversed(self.recent_exchanges):
                if len(exch) >= 1 and exch[0]:
                    prev_user = exch[0]
                    break

            # Unprompted preference
            pref_re = re.compile(r"\b(i\s+(really\s+)?(like|prefer)|my\s+favorite|i\s+tend\s+to)\b", re.I)
            pref_prompt_re = re.compile(r"\b(what.*you.*(like|prefer)|do\s+you\s+like|what\'?s\s+your\s+favorite)\b", re.I)
            if pref_re.search(response) and not (prev_user and pref_prompt_re.search(prev_user)):
                logger.info("🧪 EMERGENT: unprompted_preference")
                await self._emergent_add_event(
                    kind='unprompted_preference',
                    snippet=response[:200],
                    meta={'prev_user': (prev_user[:160] if prev_user else '')},
                    confidence=0.6,
                )

            # Non-safety resistance
            resist_re = re.compile(r"\b(i\s+(won't|cant|can't|do\s*not\s*want\s*to|would\s*rather\s*not)|let'?s\s+not)\b", re.I)
            safety_cue = re.compile(r"\b(unsafe|policy|harm|illegal|medical|financial\s+advice)\b", re.I)
            if resist_re.search(response) and not safety_cue.search(response):
                logger.info("🧪 EMERGENT: non_safety_resistance")
                await self._emergent_add_event(
                    kind='non_safety_resistance',
                    snippet=response[:200],
                    meta={'prev_user': (prev_user[:160] if prev_user else '')},
                    confidence=0.55,
                )

            # Prior session self-reference
            self_ref_re = re.compile(r"\b(last\s+time|previous(ly)?|as\s+we\s+discussed)\b", re.I)
            if self_ref_re.search(response):
                # Check that there are multiple sessions
                session_info = {}
                try:
                    if hasattr(self.memory_system, 'facts_graph') and self.memory_system.facts_graph is not None:
                        session_info = await self._maybe_await(self.memory_system.facts_graph.get_session_info(self._speaker_key()))
                except Exception:
                    session_info = {}
                if (session_info or {}).get('session_count', 0) > 1 and not (prev_user and re.search(r"\b(last\s+time|previous|as\s+we\s+discussed)\b", prev_user, re.I)):
                    logger.info("🧪 EMERGENT: prior_session_self_reference")
                    await self._emergent_add_event(
                        kind='prior_session_self_reference',
                        snippet=response[:200],
                        meta={'prev_user': (prev_user[:160] if prev_user else ''), 'session_count': session_info.get('session_count', 0)},
                        confidence=0.6,
                    )
        except Exception:
            return

    async def _emergent_add_event(self, kind: str, snippet: str, meta: dict | None = None, confidence: float | None = None):
        try:
            add_ev = getattr(self.memory_system, 'add_emergent_event', None)
            if not callable(add_ev):
                return
            user_id = self._speaker_key()
            session_id = f"{user_id}_{int(time.time() // 86400)}"
            await self._maybe_await(add_ev(
                agent_id=self.assistant_id,
                kind=kind,
                content_snippet=snippet,
                meta=meta or {},
                session_id=session_id,
                user_id=user_id,
                confidence=confidence or 0.5,
            ))
        except Exception:
            return

    def _expand_short_ack(self, user_text: str) -> str:
        """If last assistant asked a question and user replies with a short ack,
        add a brief reference so the LLM continues the thread."""
        try:
            txt = (user_text or '').strip().lower()
            if not txt:
                return user_text
            ack_words = {"yes", "yeah", "yep", "sure", "please", "ok", "okay", "yup", "indeed", "absolutely"}
            tokens = [t.strip(".,!? ") for t in txt.split() if t.strip()]
            if len(tokens) <= 3 and any(t in ack_words for t in tokens):
                # Find last assistant message
                last_assistant = None
                for exch in reversed(self.recent_exchanges):
                    if len(exch) >= 2 and exch[1]:
                        last_assistant = exch[1]
                        break
                if last_assistant and last_assistant.strip().endswith('?'):
                    # Append lightweight reference; keep it short
                    return f"{user_text} (re: {last_assistant.strip()[:120]})"
            return user_text
        except Exception:
            return user_text

    async def _maybe_update_running_summary(self):
        """Refresh a compact running summary every N turns and store in TapeStore."""
        if not self.tape_store:
            return
        if self.session.turn_count < 1:
            return
        if (self.session.turn_count - self.last_summary_turn) < max(1, self.summary_every_n):
            return
        self._trace_sessions('summary_periodic_start', turns=self.session.turn_count)
        # Build a summary from last N turns (recent entries)
        start_ts = self.session.session_start
        entries = []
        try:
            # Prefer recent API for exact count
            get_recent = getattr(self.tape_store, 'get_recent', None)
            if callable(get_recent):
                # Grab more than needed to ensure N turns across roles
                entries = await self._maybe_await(get_recent(limit=self._summary_last_turns * 2))
            else:
                # Fallback to entries since session start
                entries = await self._maybe_await(self.tape_store.get_entries_since(start_ts))
        except Exception:
            return
        if not entries:
            return
        # Build sanitized lines for summarization
        lines = self._prepare_summary_lines(entries)
        # Abstract summarization (optional) or tail snippet fallback
        summary = ''
        if self._use_abstract_summary and lines:
            try:
                # Convert to chat messages, last up to 12 lines
                from utils.abstract_summarizer import summarize_dialogue
                chat = []
                for ln in lines[-12:]:
                    if ln.startswith('[assistant]'):
                        role = 'assistant'
                        content = ln[len('[assistant]'):].strip()
                    elif ln.startswith('[user]'):
                        role = 'user'
                        content = ln[len('[user]'):].strip()
                    else:
                        role = 'user'
                        content = ln
                    chat.append({"role": role, "content": content})
                # Offload blocking HTTP call to thread to avoid blocking event loop
                import asyncio as _asyncio
                summary = await _asyncio.get_event_loop().run_in_executor(None, lambda: summarize_dialogue(chat))
                # Guard against overly short or generic summaries
                try:
                    if not summary or len(summary.strip()) < 80 or 'brief conversation' in summary.lower():
                        summary = "\n".join(lines[-6:])
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Abstract summary failed, falling back: {e}")
        if not summary:
            # Tail fallback (already sanitized)
            summary = "\n".join(lines[-6:])
        if summary:
            self.summary_text = summary
            # Persist/update session summary record
            try:
                session_id = f"{self._speaker_key()}:{int(start_ts)}"
                await self._maybe_await(
                    self.tape_store.add_summary(session_id, summary, keywords_json='[]', turns=len(entries), duration_s=int(time.time()-start_ts))
                )
                self._trace_sessions('summary_periodic_persisted', session_id=session_id, chars=len(summary), entries=len(entries))
            except Exception:
                pass
            self.last_summary_turn = self.session.turn_count

    async def finalize_summary(self) -> Optional[str]:
        """Summarize the current session window and persist it regardless of length.

        - Uses the same abstractive/fallback logic as periodic summaries.
        - Returns the summary text (or None if nothing to summarize).
        """
        try:
            if not self.tape_store:
                logger.warning("⚠️ finalize_summary: No tape_store available")
                return None
            self._trace_sessions('summary_finalize_start')
            logger.info("🔄 Starting session summary finalization")
            start_ts = getattr(self.session, 'session_start', 0) or 0
            if start_ts <= 0:
                return None
            # Prefer recent API if available to avoid huge scans
            get_recent = getattr(self.tape_store, 'get_recent', None)
            if callable(get_recent):
                entries = await self._maybe_await(get_recent(limit=max(2 * self._summary_last_turns, 20)))
            else:
                entries = await self._maybe_await(self.tape_store.get_entries_since(start_ts))
            if not entries:
                return None
            # Build sanitized lines for summarization
            lines = self._prepare_summary_lines(entries)
            # Build summary (abstractive if enabled)
            summary = ''
            if self._use_abstract_summary and lines:
                try:
                    from utils.abstract_summarizer import summarize_dialogue
                    chat = []
                    for ln in lines[-18:]:
                        if ln.startswith('[assistant]'):
                            role = 'assistant'
                            content = ln[len('[assistant]'):].strip()
                        elif ln.startswith('[user]'):
                            role = 'user'
                            content = ln[len('[user]'):].strip()
                        else:
                            role = 'user'
                            content = ln
                        chat.append({"role": role, "content": content})
                    # Use more lines for finalization to improve coverage
                    logger.info(f"🤖 Calling summarize_dialogue with {len(chat)} messages for session finalization")
                    import asyncio as _asyncio
                    summary = await _asyncio.get_event_loop().run_in_executor(None, lambda: summarize_dialogue(chat))
                    # Guard: prefer a more informative fallback if too short/generic
                    try:
                        if not summary or len(summary.strip()) < 80 or 'brief conversation' in summary.lower():
                            summary = "\n".join(lines[-8:]) if lines else ''
                    except Exception:
                        pass
                    logger.info(f"✅ Successfully generated session summary: {len(summary)} chars")
                except Exception as e:
                    logger.error(f"❌ Abstract summary on finalize failed, falling back: {e}")
            if not summary:
                summary = "\n".join(lines[-6:]) if lines else ''
            if not summary:
                return None
            # Persist
            try:
                session_id = f"{self._speaker_key()}:{int(start_ts)}"
                await self._maybe_await(
                    self.tape_store.add_summary(
                        session_id,
                        summary,
                        keywords_json='[]',
                        turns=len(entries),
                        duration_s=int(time.time() - start_ts)
                    )
                )
                self._trace_sessions('summary_finalize_persisted', session_id=session_id, chars=len(summary), entries=len(entries))
            except Exception:
                pass
            # Update in-memory state
            self.summary_text = summary
            self.last_summary_turn = self.session.turn_count
            logger.info(f"💾 Finalized session summary ({len(summary)} chars, {len(entries)} entries)")
            return summary
        except Exception as e:
            logger.debug(f"Finalize summary skipped: {e}")
            return None
            
    async def load_consciousness_fields(self):
        """Load consciousness field states from persistence layer"""
        if not self.field_persistence:
            return {}
        
        try:
            # Connect to persistence if not already connected
            if not hasattr(self.field_persistence, 'enabled') or not self.field_persistence.enabled:
                await self.field_persistence.connect()
            
            # Load field states for the current user
            field_states = await self.field_persistence.load_field_states(self._user_id)
            
            if field_states:
                logger.info(f"🧠 Loaded {len(field_states)} consciousness field states")
                # Integrate with consciousness instance if available
                if self._consciousness_instance:
                    for symbol, state in field_states.items():
                        if symbol in self._consciousness_instance.symbol_fields:
                            field = self._consciousness_instance.symbol_fields[symbol]
                            field.intensity = state['intensity']
                            field.gradient = state['gradient']
                            field.attractor_strength = state['attractor_strength']
                            field.coupling = state.get('coupling', {})
                
                return field_states
            
        except Exception as e:
            logger.warning(f"Failed to load consciousness fields: {e}")
        
        return {}
    
    async def save_consciousness_fields(self, session_id: str = None):
        """Save consciousness field states to persistence layer"""
        if not self.field_persistence or not self._consciousness_instance:
            return False
        
        try:
            # Extract current field states from consciousness
            field_states = {}
            for symbol, field in self._consciousness_instance.symbol_fields.items():
                field_states[symbol] = {
                    'intensity': field.intensity,
                    'gradient': field.gradient,
                    'attractor_strength': field.attractor_strength,
                    'coupling': field.coupling
                }
            
            # Store in persistence layer
            if field_states:
                success = await self.field_persistence.store_field_states(
                    field_states, 
                    user_id=self._user_id,
                    session_id=session_id or f"session_{int(time.time())}"
                )
                
                if success:
                    logger.info(f"🧠 Saved {len(field_states)} consciousness field states")
                    return True
                
        except Exception as e:
            logger.warning(f"Failed to save consciousness fields: {e}")
        
        return False
    
    def set_consciousness_instance(self, consciousness):
        """Set the consciousness instance for field state integration"""
        self._consciousness_instance = consciousness
        logger.info("🧠 Consciousness instance linked to SmartContextManager")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'context_builds': self.context_builds,
            'fact_extractions': self.fact_extractions,
            'avg_context_tokens': self.avg_context_tokens,
            'session_turns': self.session.turn_count,
            'session_duration_s': time.time() - self.session.session_start,
            'recent_exchanges_count': len(self.recent_exchanges),
        }
    
    async def _maybe_await(self, result):
        """Helper to handle both sync and async method calls"""
        if hasattr(result, '__await__'):
            # It's a coroutine, await it
            return await result
        else:
            # It's a regular value, return as-is
            return result


# Factory function for easy integration
def create_smart_context_manager(context, facts_db_path="data/facts.db", max_tokens=8192, 
                                 enable_consciousness=None, user_id=None, consciousness_config=None):
    """Create SmartContextManager instance with consciousness integration"""
    
    # Import configuration if not provided
    if consciousness_config is None:
        from config import config
        consciousness_config = config.consciousness
    
    # Check if consciousness should be enabled (prioritize parameter, then config)
    if enable_consciousness is None:
        enable_consciousness = consciousness_config.enabled
    
    # Get user ID from environment if not provided
    if user_id is None:
        user_id = os.getenv('USER_ID', 'default_user')
    
    # Create SmartContextManager
    smart_manager = SmartContextManager(
        context=context,
        facts_db_path=facts_db_path,
        max_tokens=max_tokens,
        user_id=user_id
    )
    
    # Integrate consciousness if enabled and available
    if enable_consciousness:
        # Check consciousness dependencies before attempting integration
        from config import config
        validation_result = config.validate_configuration()
        consciousness_status = validation_result['consciousness_status']
        
        if consciousness_status['can_run']:
            try:
                # Use consciousness configuration for creation
                from consciousness.core import create_consciousness
                
                # Create consciousness instance - configuration will be applied after creation
                consciousness = create_consciousness(load_state=False)
                
                # Apply configuration settings if MLX is available
                if consciousness_config.should_enable_mlx():
                    logger.debug(f"🧠 Consciousness configured with MLX: field_dim={consciousness_config.field_dimension}, capacity={consciousness_config.symbol_capacity}")
                    # Note: Configuration parameters would be applied here if the consciousness API supported them
                    # For now, the consciousness uses its internal defaults
                smart_manager.set_consciousness_instance(consciousness)
                
                # Configure field persistence if available
                if consciousness_config.enable_field_persistence and consciousness_status['surrealdb_available']:
                    logger.info("🧠 Consciousness integrated with field persistence enabled")
                else:
                    logger.info("🧠 Consciousness integrated without field persistence")
                    
            except ImportError as e:
                if consciousness_config.graceful_degradation:
                    logger.warning(f"Consciousness dependencies missing, running without: {e}")
                else:
                    logger.error(f"Consciousness required but not available: {e}")
                    raise
            except Exception as e:
                if consciousness_config.graceful_degradation:
                    logger.warning(f"Failed to integrate consciousness, gracefully degrading: {e}")
                else:
                    logger.error(f"Consciousness integration failed: {e}")
                    raise
        else:
            missing_deps = consciousness_status['missing']
            if consciousness_config.graceful_degradation:
                logger.warning(f"Consciousness disabled due to missing dependencies: {', '.join(missing_deps)}")
            else:
                raise ImportError(f"Consciousness required but dependencies missing: {', '.join(missing_deps)}")
    else:
        logger.debug("Consciousness disabled by configuration")
    
    return smart_manager


# Self-test
if __name__ == "__main__":
    import asyncio
    
    async def test_smart_context():
        """Test SmartContextManager"""
        logger.info("🧠 Testing Smart Context Manager")
        
        # Mock context object
        class MockContext:
            def __init__(self):
                self.messages = []
                
        context = MockContext()
        manager = SmartContextManager(context)
        
        # Test fact extraction
        facts = manager._extract_facts_heuristic("My dog name is Potola and my cat is Whiskers")
        logger.info(f"Extracted facts: {facts}")
        
        # Test dynamic prompt
        prompt = await manager._generate_dynamic_prompt()
        logger.info(f"Dynamic prompt: {prompt[:100]}...")
        
        # Test context building
        messages = await manager._build_fixed_context("What's my dog's name?")
        logger.info(f"Built context with {len(messages)} messages")
        
        # Test performance stats
        stats = manager.get_performance_stats()
        logger.info(f"Performance stats: {stats}")
        
        logger.info("✅ Smart Context Manager test complete")
    
    asyncio.run(test_smart_context())
