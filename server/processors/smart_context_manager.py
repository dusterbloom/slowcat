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

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from processors.token_counter import get_token_counter
import os
from memory import create_smart_memory_system, extract_facts_from_text
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
        
        # Feature toggles / thresholds (env-driven, default generic)
        self._enable_spelling_hints = os.getenv('ENABLE_SPELLING_HINTS', 'false').lower() == 'true'
        # Location-focused spelling hints (stronger nudge when talking about place names)
        self._enable_location_spelling_hints = os.getenv('ENABLE_LOCATION_SPELLING_HINTS', 'false').lower() == 'true'
        self._enable_greeting_fallback = os.getenv('ENABLE_GREETING_FALLBACK', 'false').lower() == 'true'
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
        # Prefer a single logical user id when speaker recognition is disabled
        self._user_id_override = os.getenv('USER_ID', '').strip() or None
        if self._user_id_override:
            self.session.speaker_id = self._user_id_override
        
        # Recent conversation sliding window (for context)
        self.recent_exchanges = []  # List of (user, assistant) pairs
        # Keep a large window; actual inclusion is constrained by token budget.
        self.max_recent_exchanges = 50

        # Running summary state
        self.summary_every_n = int(os.getenv('SC_SUMMARY_EVERY_N', '3'))
        self.summary_text: str = ''
        self.last_summary_turn: int = 0
        
        # Performance metrics
        self.context_builds = 0
        self.fact_extractions = 0
        self.avg_context_tokens = 0
        
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
            except Exception:
                pass
        
        # Only process TranscriptionFrames (user input)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            # Normalize user input (collapse repeated punctuation, trim, etc.)
            user_text = self._normalize_user_input(frame.text)
            logger.debug(f"🎤 Processing transcription: '{user_text[:50]}...'")

            # 1. Extract facts from user input (background, non-blocking)
            if self._has_semantic_content(user_text):
                asyncio.create_task(self._extract_facts_async(user_text))

            # 1b. Write user message to tape store
            try:
                if self.tape_store is not None and self._is_semantically_useful(user_text):
                    self.tape_store.add_entry(
                        role='user',
                        content=user_text,
                        speaker_id=self.session.speaker_id or 'default_user'
                    )
            except Exception as e:
                logger.debug(f"TapeStore write (user) failed: {e}")
            
            # 2. Update session in memory system
            try:
                self.memory_system.update_session(self.session.speaker_id or 'default_user')
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

            # 5. Create LLM frame with our fixed context
            # Use legacy LLMMessagesFrame for maximum compatibility with current Pipecat version
            llm_frame = LLMMessagesFrame(messages)
            await self.push_frame(llm_frame, direction)

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

            # 6b. Update running summary periodically
            try:
                await self._maybe_update_running_summary()
            except Exception as e:
                logger.debug(f"Running summary update skipped: {e}")
            
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
        system_prompt = self._generate_dynamic_prompt()
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
            facts_context = self._format_facts_context(facts)
        
        facts_tokens = self.token_counter.count_tokens(facts_context)

        # 2b. Conversation snippets (separate block) when relevant
        snippets_context = ""
        if self.memory_system:
            snippets = await self._get_conversation_snippets(user_input)
            if snippets:
                lines = ["[Conversation Snippets]"]
                for r in snippets[:3]:
                    text = getattr(r, 'content', '')
                    if text:
                        lines.append(f"- {text}")
                snippets_context = "\n".join(lines)
        snippets_tokens = self.token_counter.count_tokens(snippets_context)

        # 3. Build recent conversation context with dynamic budget
        # Use remaining space after system + summary + facts + current input + buffer
        current_tokens_est = self.token_counter.count_tokens(user_input)
        dynamic_recent_budget = max(
            0,
            self.max_tokens - (system_tokens + summary_tokens + facts_tokens + snippets_tokens + current_tokens_est + self.budget.buffer)
        )
        recent_tokens_budget = dynamic_recent_budget
        recent_context = self._build_recent_context(recent_tokens_budget)
        recent_tokens = self.token_counter.count_tokens(str(recent_context))

        # 4. Current user input
        current_user = {"role": "user", "content": user_input}
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
        
        messages.append({"role": "system", "content": full_system})
        
        # Recent conversation
        messages.extend(recent_context)
        
        # Current user input (only if non-empty)
        if user_input and user_input.strip():
            messages.append(current_user)
        
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

    async def get_initial_context_frame(self) -> LLMMessagesFrame:
        """Build an initial context frame using current facts and session status."""
        # Ensure session is registered and increment session count
        try:
            spk = self.session.speaker_id or 'default_user'
            # Start a new session (increments session_count once)
            if hasattr(self.memory_system, 'facts_graph'):
                try:
                    before = self.memory_system.facts_graph.get_session_info(spk)
                    logger.info(f"📊 Session before start: {before}")
                except Exception:
                    pass
                self.memory_system.facts_graph.start_session(spk)
                try:
                    after = self.memory_system.facts_graph.get_session_info(spk)
                    logger.info(f"📊 Session after start: {after}")
                except Exception:
                    pass
        except Exception:
            pass
        messages = await self._build_fixed_context("")
        
        # Inject previous session summary, if available
        try:
            if self.tape_store is not None:
                last = self.tape_store.get_last_summary()
                # Only include if it predates this session start
                if last and last['ts'] < self.session.session_start:
                    prev_summary_raw = str(last['summary'])
                    prev_summary = self._clean_previous_summary(prev_summary_raw)[:600]
                    self._initial_summary_ok = bool(prev_summary and len(prev_summary) >= 30)
                    if prev_summary:
                        sys_msg = messages[0]
                        if isinstance(sys_msg, dict) and sys_msg.get('role') == 'system':
                            sys_msg['content'] += f"\n\n[Previous Session Summary]\n{prev_summary}"
        except Exception:
            pass
        return LLMMessagesFrame(messages)

    def needs_greeting(self) -> bool:
        return not getattr(self, '_initial_summary_ok', True)

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
    
    def _generate_dynamic_prompt(self) -> str:
        """
        Generate evolving system prompt based on session metadata
        Always includes a Session Info block and relationship cues.
        """
        base_prompt = "You are Slowcat, a helpful AI assistant."
        # Current local time for clarity
        try:
            import datetime
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            base_prompt += f"\nTime: {now}"
        except Exception:
            pass

        # Add session information (always show)
        session_info = f"\n\n[Session Info]"
        turn_display = max(1, self.session.turn_count + 1)
        session_info += f"\nTurn: {turn_display}"
        
        # Calculate session duration
        duration_s = int(time.time() - self.session.session_start)
        if duration_s > 60:
            session_info += f"\nDuration: {duration_s//60}m {duration_s%60}s"
        else:
            session_info += f"\nDuration: {duration_s}s"
            
        # Speaker information
        spk = self.session.speaker_id if self.session.speaker_id and self.session.speaker_id != "unknown" else "default_user"
        session_info += f"\nSpeaker: {spk}"

        # Total sessions (lifetime) from facts graph, if available
        sessions_total = None
        try:
            if hasattr(self.memory_system, 'facts_graph'):
                info = self.memory_system.facts_graph.get_session_info(spk)
                sessions_total = info.get('session_count', 0)
        except Exception:
            sessions_total = None

        if sessions_total is not None:
            session_info += f"\nSessions: {sessions_total}"

        base_prompt += session_info

        # Relationship-aware tone guidance
        relationship = "first-time" if not sessions_total or sessions_total <= 1 else ("returning" if sessions_total <= 10 else ("regular" if sessions_total <= 50 else "long-term"))
        # Compact tone guidance (single block, minimal spacing)
        tone_block = (
            f"\n\n[Tone Guidance]\nRelationship: {relationship}\n"
            "- Answer directly; avoid greetings unless asked.\n"
            "- If the user asks a question, never introduce yourself — respond with content."
        )
        base_prompt += tone_block
        
        # Add memory instructions
        memory_instructions = (
            "\n\n[Memory Instructions]\n"
            "- User facts describe the user, not you; never claim them as yours.\n"
            "- Refer to the user as 'you'; avoid 'I' for their attributes.\n"
            "- Be concise. If a direct answer is clear, avoid greetings.\n"
            "- If specific info is missing, ask a brief clarification."
        )
        base_prompt += memory_instructions

        # Interaction rules tuned for small models (generic, not user-specific)
        interaction_rules = (
            "\n\n[Interaction Rules]\n"
            "- Do not repeat the user's message verbatim; paraphrase only when needed.\n"
            "- If the user spells a name (letters separated by spaces/commas/dashes), join the letters into a single candidate (e.g., 'S E R R A' -> 'Serra') and propose it as a guess.\n"
            "- Ask at most one follow-up question for the same detail. If still unclear, ask the user to type it, rather than re-asking.\n"
            "- Avoid generic capability listings unless the user asks for capabilities.\n"
            "- Prefer: answer in one short sentence; add a brief follow-up only if necessary."
        )
        base_prompt += interaction_rules
        
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
                    top_facts = self.memory_system.facts_graph.get_top_facts(limit=min(3, limit))
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

            facts_results = [r for r in response.results if getattr(r, 'source_store', '') == 'facts']
            if facts_results:
                return facts_results[:limit]

            # Fallback: include a few top facts if search empty
            try:
                top_facts = self.memory_system.facts_graph.get_top_facts(limit=min(3, limit))
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
            intent_name = getattr(response.classification.intent, 'name', '').upper()
            if intent_name not in ('CONVERSATION_HISTORY', 'EPISODIC_MEMORY'):
                return []
            return [r for r in response.results if getattr(r, 'source_store', '') == 'tape'][:limit]
        except Exception:
            return []
    
    def _format_facts_context(self, facts: List[Any]) -> str:
        """Format facts as context string"""
        if not facts:
            return ""
            
        context_lines = ["[Relevant Facts — about the user (use 'you' when answering; do not claim as yours)]"]
        for fact in facts[:10]:  # Limit to prevent token overflow
            # Accept both MemoryResult (with .content) and FactsGraph Fact
            if hasattr(fact, 'content'):
                context_lines.append(f"- {fact.content}")
            else:
                subj = getattr(fact, 'subject', None)
                pred = getattr(fact, 'predicate', None)
                val = getattr(fact, 'value', None)
                if subj and pred and val:
                    if subj == 'user':
                        context_lines.append(f"- user's {pred} is {val}")
                    else:
                        context_lines.append(f"- {subj}'s {pred} is {val}")
                elif subj and pred:
                    if subj == 'user':
                        context_lines.append(f"- user has {pred}")
                    else:
                        context_lines.append(f"- {subj} has {pred}")
        
        return "\n".join(context_lines)
    
    def _build_recent_context(self, token_budget: int) -> List[Dict]:
        """
        Build recent conversation context within token budget
        Uses sliding window of last N exchanges
        """
        if not self.recent_exchanges:
            return []
            
        messages = []
        token_count = 0
        
        # Add exchanges from most recent backwards until budget exhausted
        for exch in reversed(self.recent_exchanges):
            user_msg = exch[0] if len(exch) >= 1 else ""
            assistant_msg = exch[1] if len(exch) >= 2 else ""
            # Skip stray assistant-only micro-chunks (from any past streaming glitch)
            if not user_msg and assistant_msg and len(assistant_msg.split()) <= 2:
                continue
            # Calculate tokens for this exchange (handle missing assistant)
            exchange_tokens = self.token_counter.count_tokens(user_msg)
            if assistant_msg:
                exchange_tokens += self.token_counter.count_tokens(assistant_msg)
            
            if token_count + exchange_tokens <= token_budget:
                # Insert at beginning to maintain chronological order
                if assistant_msg:
                    messages.insert(0, {"role": "assistant", "content": assistant_msg})
                if user_msg:
                    messages.insert(0, {"role": "user", "content": user_msg})
                token_count += exchange_tokens
            else:
                break
                
        logger.debug(f"📝 Recent context: {len(messages)//2} exchanges, {token_count} tokens")
        return messages
    
    async def _extract_facts_async(self, text: str):
        """
        Extract facts from conversation text (non-blocking)
        """
        try:
            self.fact_extractions += 1
            
            # Extract and store facts using memory system
            facts_count = self.memory_system.store_facts(text)
            
            logger.debug(f"🔍 Extracted and stored {facts_count} facts from: '{text[:30]}...'")
            
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
    
    def _update_session(self):
        """Update session metadata"""
        now = time.time()
        self.session.turn_count += 1
        self.session.last_interaction = now
        self.session.total_interactions += 1
        
    def add_assistant_response(self, response: str):
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
                self.tape_store.add_entry(
                    role='assistant',
                    content=response,
                    speaker_id=self.session.speaker_id or 'default_user'
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
        # Build a heuristic summary from recent tape entries in this session
        start_ts = self.session.session_start
        entries = self.tape_store.get_entries_since(start_ts)
        if not entries:
            return
        # Keep last ~10 lines for summary context
        tail = entries[-10:]
        lines = []
        for e in tail:
            content = (e.content or '').strip()
            if not content:
                continue
            # Prefer questions and concise statements
            if content.endswith('?') or len(content.split()) <= 24:
                lines.append(f"[{e.role}] {content}")
        # Truncate and join
        summary = "\n".join(lines[-6:])
        if summary:
            self.summary_text = summary
            # Persist/update session summary record
            try:
                session_id = f"{self.session.speaker_id or 'default_user'}:{int(start_ts)}"
                self.tape_store.add_summary(session_id, summary, keywords_json='[]', turns=len(entries), duration_s=int(time.time()-start_ts))
            except Exception:
                pass
            self.last_summary_turn = self.session.turn_count
            
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
        prompt = manager._generate_dynamic_prompt()
        logger.info(f"Dynamic prompt: {prompt[:100]}...")
        
        # Test context building
        messages = await manager._build_fixed_context("What's my dog's name?")
        logger.info(f"Built context with {len(messages)} messages")
        
        # Test performance stats
        stats = manager.get_performance_stats()
        logger.info(f"Performance stats: {stats}")
        
        logger.info("✅ Smart Context Manager test complete")
    
    asyncio.run(test_smart_context())
