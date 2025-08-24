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
import os
from memory import create_smart_memory_system, extract_facts_from_text
try:
    from memory.dynamic_tape_head import DynamicTapeHead  # optional
except Exception:
    DynamicTapeHead = None  # type: ignore
try:
    # For intent enum comparison in retrieval gating (optional)
    from memory.query_classifier import QueryIntent  # noqa: F401
except Exception:
    QueryIntent = None  # type: ignore


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
    """Fixed token allocation for 4096 total"""
    system_prompt: int = 500
    facts_context: int = 800
    recent_conversation: int = 2000
    current_input: int = 696
    buffer: int = 100
    
    @property
    def total(self) -> int:
        return (self.system_prompt + self.facts_context + 
                self.recent_conversation + self.current_input + self.buffer)


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
                 max_tokens: int = 4096,
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

        # Optional Dynamic Tape Head integration (feature flag)
        self._enable_dth = os.getenv('ENABLE_DTH', 'false').lower() == 'true'
        self.tape_head = None
        if self._enable_dth and DynamicTapeHead is not None:
            try:
                self.tape_head = DynamicTapeHead(self.memory_system)
                logger.info("🧠 DTH enabled in SmartContextManager")
            except Exception as e:
                logger.warning(f"DTH init failed; continuing without: {e}")
                self.tape_head = None
        
        # Feature toggles / thresholds (env-driven, default generic)
        self._enable_spelling_hints = os.getenv('ENABLE_SPELLING_HINTS', 'false').lower() == 'true'
        # Location-focused spelling hints (stronger nudge when talking about place names)
        self._enable_location_spelling_hints = os.getenv('ENABLE_LOCATION_SPELLING_HINTS', 'false').lower() == 'true'
        self._enable_greeting_fallback = os.getenv('ENABLE_GREETING_FALLBACK', 'false').lower() == 'true'
        # Deterministic greeting injection (pipeline adds the greeting once)
        self._enforce_greeting = os.getenv('SC_ENFORCE_GREETING', 'false').lower() == 'true'
        self._greeted = False
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

            # 1. Extract facts from user input (background, non-blocking)
            if self._has_semantic_content(user_text):
                asyncio.create_task(self._extract_facts_async(user_text))

            # 1b. Write user message to tape store
            try:
                if self.tape_store is not None and self._is_semantically_useful(user_text):
                    await self._maybe_await(
                        self.tape_store.add_entry(
                            role='user',
                            content=user_text,
                            speaker_id=self._speaker_key()
                        )
                    )
            except Exception as e:
                logger.debug(f"TapeStore write (user) failed: {e}")
            
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
        Build exactly 4096 tokens of context (never more, never less)
        
        Token allocation:
        - System prompt: 500 tokens (dynamic, includes session info)
        - Facts context: 800 tokens (structured knowledge)
        - Recent conversation: 2000 tokens (sliding window)
        - Current input: 696 tokens (user's current message)
        - Buffer: 100 tokens (safety margin)
        """
        start_time = time.time()
        
        # 1. Generate dynamic system prompt
        system_prompt = await self._generate_dynamic_prompt()
        system_tokens = self.token_counter.count_tokens(system_prompt)

        # 1b. Running conversation summary (small)
        summary_block = ''
        if self.summary_text:
            summary_block = f"\n\n[Conversation Summary]\n{self.summary_text[:600]}"  # keep tight
        summary_tokens = self.token_counter.count_tokens(summary_block) if summary_block else 0

        # 2. Get facts context if available
        facts_context = ""
        if self.memory_system:
            facts = await self._get_relevant_facts(user_input)
            logger.debug(f"🧠 Retrieved {len(facts)} facts for context")
            facts = self._filter_and_dedupe_facts(facts)
            facts_context = self._format_facts_context(facts)
            logger.debug(f"📝 Facts context: '{facts_context[:100]}{'...' if len(facts_context) > 100 else ''}'")
        
        facts_tokens = self.token_counter.count_tokens(facts_context)

        # 2b. Conversation snippets (separate block) when relevant
        snippets_context = ""
        if self.memory_system:
            snippets = await self._get_conversation_snippets(user_input)
            if snippets:
                lines = ["<conversation_snippets>"]
                for r in snippets[:3]:
                    text = getattr(r, 'content', '')
                    if text:
                        lines.append(f"- {text}")
                lines.append("</conversation_snippets>")
                snippets_context = "\n".join(lines)
        snippets_tokens = self.token_counter.count_tokens(snippets_context)

        # 2c. Optional DTH memory block (uses its own budget from facts_context share)
        dth_context = ""
        if self.tape_head is not None and user_input:
            try:
                # Reserve up to half of facts_context budget for DTH verbatim
                dth_budget = max(100, self.budget.facts_context // 2)
                bundle = await self.tape_head.seek(user_input, budget=dth_budget)
                try:
                    # Observability: brief summary of what DTH returned
                    logger.debug(
                        f"[SCM:retrieval] DTH verbatim={len(bundle.verbatim)}, shadows={len(bundle.shadows)}, recents={len(bundle.recents)}, tokens={bundle.token_count}/{dth_budget}"
                    )
                except Exception:
                    pass
                if bundle.verbatim:
                    lines = ["<dth_memories>"]
                    for m in bundle.verbatim[:3]:
                        lines.append(f"- {m.content}")
                    lines.append("</dth_memories>")
                    dth_context = "\n".join(lines)
            except Exception as e:
                logger.debug(f"DTH block skipped: {e}")
        dth_tokens = self.token_counter.count_tokens(dth_context)

        # 3. Build recent conversation context with dynamic budget
        # Use remaining space after system + summary + facts + current input + buffer
        current_tokens_est = self.token_counter.count_tokens(user_input)
        dynamic_recent_budget = max(
            0,
            self.max_tokens - (system_tokens + summary_tokens + facts_tokens + snippets_tokens + dth_tokens + current_tokens_est + self.budget.buffer)
        )
        recent_tokens_budget = dynamic_recent_budget
        recent_context = self._build_recent_context(recent_tokens_budget)
        recent_tokens = self.token_counter.count_tokens(str(recent_context))

        # 4. Current user input
        # IMPORTANT: Do not include the current user message here, since the
        # context aggregator will append it automatically when we forward the
        # original TranscriptionFrame. Including it here would duplicate it.
        current_user = None
        current_tokens = self.token_counter.count_tokens(user_input)

        # 5. Assemble final context
        messages = []
        
        # System message with facts
        full_system = system_prompt
        if summary_block:
            full_system += summary_block
        if facts_context:
            full_system += f"\n\n{facts_context}"
        if snippets_context:
            full_system += f"\n\n{snippets_context}"
        if dth_context:
            full_system += f"\n\n{dth_context}"
        
        messages.append({"role": "system", "content": full_system})
        
        # Recent conversation
        messages.extend(recent_context)
        
        # Do NOT append current user input here to avoid duplication.
        
        # 6. Verify token count
        total_tokens = (system_tokens + summary_tokens + facts_tokens + snippets_tokens + 
                       recent_tokens + current_tokens)
        
        # Track metrics
        self.context_builds += 1
        self.avg_context_tokens = ((self.avg_context_tokens * (self.context_builds - 1) + 
                                   total_tokens) / self.context_builds)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"🧠 Built fixed context: {total_tokens}/{self.max_tokens} tokens "
                   f"({elapsed_ms:.1f}ms)")
        logger.debug(f"   System: {system_tokens}, Summary: {summary_tokens}, Facts: {facts_tokens}, Snippets: {snippets_tokens}, "
                    f"Recent: {recent_tokens} (budget {recent_tokens_budget}), Current: {current_tokens}")
        
        # Warn if over budget
        if total_tokens > self.max_tokens:
            logger.warning(f"⚠️  Context over budget: {total_tokens}/{self.max_tokens}")
            
        return messages

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

    def _has_semantic_content(self, text: str) -> bool:
        if not text:
            return False
        # Consider it semantic if it has any alphanumeric characters
        return any(ch.isalnum() for ch in text)

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
        except Exception as e:
            logger.error(f"❌ Failed to start session: {e}")
        messages = await self._build_fixed_context("")
        
        # Inject previous session summary, if available
        try:
            if self.tape_store is not None:
                last = await self._maybe_await(self.tape_store.get_last_summary())
                # Only include if it predates this session start
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
        return LLMMessagesUpdateFrame(messages, run_llm=True)

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
        """Load token budget from environment variables if set."""
        def _get(name: str, default: int) -> int:
            try:
                val = os.getenv(name)
                return int(val) if val is not None else default
            except Exception:
                return default
        return TokenBudget(
            system_prompt=_get('SC_BUDGET_SYSTEM', 500),
            facts_context=_get('SC_BUDGET_FACTS', 800),
            recent_conversation=_get('SC_BUDGET_RECENT', 2000),
            current_input=_get('SC_BUDGET_INPUT', 696),
            buffer=_get('SC_BUDGET_BUFFER', 100),
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

        # Total sessions (lifetime) from facts graph, if available
        sessions_total = None
        try:
            if hasattr(self.memory_system, 'facts_graph'):
                info = await self._maybe_await(self.memory_system.facts_graph.get_session_info(spk))
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
            response = await self.memory_system.process_query(q)
            
            # Handle both object and dict response formats for classification
            if hasattr(response, 'classification'):
                intent_name = getattr(response.classification.intent, 'name', '').upper()
            else:
                classification = response.get('classification', {})
                intent_name = classification.get('intent', '').upper()
            
            if intent_name not in ('CONVERSATION_HISTORY', 'EPISODIC_MEMORY'):
                return []
            
            # Handle both object and dict response formats for results
            if hasattr(response, 'results'):
                results_list = response.results
            else:
                results_list = response.get('results', [])
                
            return [r for r in results_list if getattr(r, 'source_store', '') == 'tape'][:limit]
        except Exception:
            return []
    
    def _format_facts_context(self, facts: List[Any]) -> str:
        """Format facts into a concise, user-focused block with clear tags and natural phrasing."""
        if not facts:
            return ""
        lines = ["<relevant_facts>", "(Use 'you' for the user; do not claim as yours.)"]
        for fact in facts[:12]:
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
        lines.append("</relevant_facts>")
        return "\n".join(lines)

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

        # Maintain sliding window
        if len(self.recent_exchanges) > self.max_recent_exchanges:
            self.recent_exchanges.pop(0)
        # Also write to tape store
        try:
            if self.tape_store is not None and response:
                await self._maybe_await(
                    self.tape_store.add_entry(
                        role='assistant',
                        content=response,
                        speaker_id=self._speaker_key()
                    )
                )
        except Exception as e:
            logger.debug(f"TapeStore write (assistant) failed: {e}")

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
                summary = summarize_dialogue(chat)
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
                    logger.info(f"🤖 Calling summarize_dialogue with {len(chat)} messages for session finalization")
                    summary = summarize_dialogue(chat)
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
def create_smart_context_manager(context, facts_db_path="data/facts.db", max_tokens=4096):
    """Create SmartContextManager instance"""
    return SmartContextManager(
        context=context,
        facts_db_path=facts_db_path,
        max_tokens=max_tokens
    )


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
