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
    from memory.session_manager import SessionManager
except ImportError:
    SessionManager = None
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
                # Debug: Check what memory_system actually contains
                logger.info(f"🔍 DEBUG: memory_system type: {type(self.memory_system)}")
                logger.info(f"🔍 DEBUG: memory_system attrs: {dir(self.memory_system) if self.memory_system else 'None'}")
                
                # Use existing query router from memory system (unified SurrealDB approach)
                self.query_router = getattr(self.memory_system, 'query_router', None)
                logger.info(f"🔍 DEBUG: Retrieved query_router: {self.query_router}")
                
                if self.query_router:
                    logger.info("🧠 Smart Memory Router enabled (using unified SurrealDB router)")
                else:
                    # Fallback: create router if memory system doesn't have one
                    facts_graph = getattr(self.memory_system, 'facts_graph', None)
                    tape_store = getattr(self.memory_system, 'tape_store', None)
                    
                    self.query_router = QueryRouter(
                        facts_graph=facts_graph,
                        tape_store=tape_store,
                        embedding_store=None
                    )
                    logger.info("🧠 Smart Memory Router enabled (fallback router created)")
                    
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
        
        # SurrealDB Message Storage Integration with global session management
        self._enable_surreal = os.getenv('USE_SURREALDB', 'false').lower() == 'true'
        self.surreal_store = None
        
        if self._enable_surreal:
            try:
                from processors.surreal_message_store import create_surreal_message_store
                
                # Ensure global session is available before creating store
                if SessionManager:
                    existing_session = SessionManager.get_current_session()
                    if not existing_session:
                        # Create global session that all processors will use
                        session_id = SessionManager.create_new_session(
                            speaker_id=self._user_id,
                            metadata={'created_by': 'SmartContextManager', 'global_session': True}
                        )
                        logger.info(f"🎬 SmartContextManager created global session: {session_id}")
                
                self.surreal_store = create_surreal_message_store(
                    speaker_id=self._user_id,
                    session_id=SessionManager.get_current_session() if SessionManager else None,
                    auto_create_session=True
                )
                
                if self.surreal_store:
                    logger.info("📝 SurrealDB message storage enabled with global session management")
                
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
        
        # PURE ARCHITECTURE: Smart Router → Facts ONLY (no fallbacks, no tape)
        if should_query_memory and self.query_router is not None:
            try:
                logger.info("🔥 Using PURE Smart Router (Facts Only)")
                
                # Route query through Smart Router
                if hasattr(self.query_router, '__class__') and 'SurrealQueryRouter' in str(self.query_router.__class__):
                    router_response = await self.query_router.route_query(
                        query=user_input,
                        context={
                            "speaker_id": self._speaker_key(),
                            "force_personal_facts": self._is_personal_facts_candidate(user_input)
                        }
                    )
                else:
                    router_response = await self.query_router.route_query(
                        query=user_input,
                        context={
                            "speaker_id": self._speaker_key(),
                            "force_personal_facts": self._is_personal_facts_candidate(user_input)
                        },
                        max_results=20
                    )
                
                # Extract results
                try:
                    if hasattr(router_response, 'results'):
                        rr_list = router_response.results
                    else:
                        rr_list = router_response.get('results', [])
                except Exception:
                    rr_list = []

                # Process ONLY facts - completely ignore tape
                facts_count = 0
                for r in rr_list:
                    src = getattr(r, 'source_store', '')
                    
                    # Skip everything that's not a fact
                    if src != 'facts':
                        continue
                    
                    facts_count += 1
                    content = getattr(r, 'content', '')
                    
                    # Extract structured fact metadata
                    subj = None
                    pred = None
                    val = None
                    
                    if hasattr(r, 'metadata') and r.metadata:
                        subj = r.metadata.get('subject')
                        pred = r.metadata.get('predicate') 
                        val = r.metadata.get('value') or r.metadata.get('object')
                    
                    if not subj:
                        subj = getattr(r, 'subject', None)
                    if not pred:
                        pred = getattr(r, 'predicate', None)
                    if not val:
                        val = getattr(r, 'object', None)
                    
                    # Build clean fact representation
                    if subj and pred:
                        ptxt = str(pred).replace('_', ' ')
                        if val is not None and str(val).strip() != '':
                            verified_lines.append(f"- {subj}'s {ptxt} is {val}")
                        else:
                            verified_lines.append(f"- {subj} has {ptxt}")
                    elif content:
                        verified_lines.append(f"- {content}")
                
                logger.info(f"✅ Pure facts only: {facts_count} facts processed → {len(verified_lines)} structured lines")
                
            except Exception as e:
                logger.warning(f"Smart router failed: {e}")
                # No fallback - if Smart Router fails, we have no context
        
        # PURE CONTEXT BUILDING: Use verified facts directly, no fallbacks
        if verified_lines:
            contextual_memory = "\n".join(sorted(set(verified_lines)))
            logger.info(f"🔥 PURE CONTEXT: {len(verified_lines)} facts → LLM context")
        else:
            contextual_memory = ""
            logger.info("📭 PURE CONTEXT: No facts found → Empty context")
        
        contextual_tokens = self.token_counter.count_tokens(contextual_memory)
        
        # 3. Recent conversation context (minimal budget)
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

        strict_answer_mode = self._is_personal_facts_candidate(user_input)
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
        
        # Build system message first with dynamic prompt
        system_prompt = await self._generate_dynamic_prompt()
        
        # Add contextual memory (facts) to system prompt if available
        full_system_content = system_prompt
        if contextual_memory.strip():
            full_system_content += f"\n\n<verified_memory>\n{contextual_memory}\n</verified_memory>"
            if strict_answer_mode:
                full_system_content += (
                    "\n<answer_policy>\n"
                    "When answering about personal facts, base your answer ONLY on <verified_memory>.\n"
                    "If the requested attribute is missing, say you don't have it recorded yet and ask if the user wants to add it.\n"
                    "Do not infer age/breed/numbers; do not guess.\n"
                    "</answer_policy>"
                )
        
        system_tokens = self.token_counter.count_tokens(full_system_content)
        
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
        
        logger.info(f"🚀 UNIFIED CONTEXT BUILT: {total_tokens}/{self.budget.total_context} tokens ({elapsed_ms:.1f}ms)")
        logger.info(f"   🎭 System: {system_tokens} tokens")
        logger.info(f"   🧠 Retrieved Memory: {contextual_tokens} tokens ({dspy_status})")
        logger.info(f"   💬 Recent Context: {recent_context_tokens} tokens ({len(recent_context_messages)} messages)")
        logger.info(f"   📝 Current Input: {current_tokens} tokens")
        logger.info(f"   🎯 Generation Workspace: {self.budget.total_context - total_tokens} tokens (reserved)")
        
        return {
            "messages": messages,
            "content": full_system_content,
            "tokens": {
                "total": total_tokens,
                "system": system_tokens,
                "contextual": contextual_tokens,
                "recent": recent_context_tokens,
                "current": current_tokens
            },
            "elapsed_ms": elapsed_ms
        }
