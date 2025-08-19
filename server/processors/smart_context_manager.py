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
        
        # Initialize smart memory system
        self.memory_system = create_smart_memory_system(facts_db_path)
        self.tape_store = getattr(self.memory_system, 'tape_store', None)
        
        # Token allocation (allow env overrides)
        self.budget = self._load_budget_from_env()
        logger.info(f"🧠 Smart Context Manager initialized with {self.budget.total} token budget")
        
        # Session tracking
        self.session = SessionMetadata()
        self.session.session_start = time.time()
        
        # Recent conversation sliding window (for context)
        self.recent_exchanges = []  # List of (user, assistant) pairs
        self.max_recent_exchanges = 5  # Keep last 5 exchanges
        
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
                if self.tape_store is not None and self._has_semantic_content(user_text):
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

            # 4. Build fixed context (NEVER grows beyond 4096)
            messages = await self._build_fixed_context(user_text)

            # 5. Create LLM frame with our fixed context
            # Use legacy LLMMessagesFrame for maximum compatibility with current Pipecat version
            llm_frame = LLMMessagesFrame(messages)
            await self.push_frame(llm_frame, direction)

            # 6. Update session metadata
            self._update_session()
            
            # DON'T forward the original TranscriptionFrame
            # This prevents the old context_aggregator from accumulating
            return
            
        # Forward all other frames normally
        await self.push_frame(frame, direction)
    
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
        
        # 2. Get facts context if available
        facts_context = ""
        if self.memory_system:
            facts = await self._get_relevant_facts(user_input)
            facts_context = self._format_facts_context(facts)
        
        facts_tokens = self.token_counter.count_tokens(facts_context)
        
        # 3. Build recent conversation context
        recent_tokens_budget = self.budget.recent_conversation
        recent_context = self._build_recent_context(recent_tokens_budget)
        recent_tokens = self.token_counter.count_tokens(str(recent_context))
        
        # 4. Current user input
        current_user = {"role": "user", "content": user_input}
        current_tokens = self.token_counter.count_tokens(user_input)
        
        # 5. Assemble final context
        messages = []
        
        # System message with facts
        full_system = system_prompt
        if facts_context:
            full_system += f"\n\n{facts_context}"
        
        messages.append({"role": "system", "content": full_system})
        
        # Recent conversation
        messages.extend(recent_context)
        
        # Current user input
        messages.append(current_user)
        
        # 6. Verify token count
        total_tokens = (system_tokens + facts_tokens + 
                       recent_tokens + current_tokens)
        
        # Track metrics
        self.context_builds += 1
        self.avg_context_tokens = ((self.avg_context_tokens * (self.context_builds - 1) + 
                                   total_tokens) / self.context_builds)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"🧠 Built fixed context: {total_tokens}/{self.max_tokens} tokens "
                   f"({elapsed_ms:.1f}ms)")
        logger.debug(f"   System: {system_tokens}, Facts: {facts_tokens}, "
                    f"Recent: {recent_tokens}, Current: {current_tokens}")
        
        # Warn if over budget
        if total_tokens > self.max_tokens:
            logger.warning(f"⚠️  Context over budget: {total_tokens}/{self.max_tokens}")
            
        return messages

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

    async def get_initial_context_frame(self) -> LLMMessagesFrame:
        """Build an initial context frame using current facts and session status."""
        messages = await self._build_fixed_context("")
        return LLMMessagesFrame(messages)

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
        """
        base_prompt = "You are Slowcat, a helpful AI assistant."
        # Current local time for clarity
        try:
            import datetime
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            base_prompt += f"\nTime: {now}"
        except Exception:
            pass
        
        # Add session information
        if self.session.turn_count > 0:
            session_info = f"\n\n[Session Info]"
            session_info += f"\nTurn: {self.session.turn_count}"
            
            # Calculate session duration
            duration_s = int(time.time() - self.session.session_start)
            if duration_s > 60:
                session_info += f"\nDuration: {duration_s//60}m {duration_s%60}s"
            else:
                session_info += f"\nDuration: {duration_s}s"
                
            # Speaker information
            if self.session.speaker_id != "unknown":
                session_info += f"\nSpeaker: {self.session.speaker_id}"
                
            base_prompt += session_info
        
        # Add memory instructions
        memory_instructions = """

[Memory Instructions]
- The following facts are about the user (the person you are talking to), NOT about you.
- Never present user facts as your own attributes; do not say "I'm ..." for the user's info.
- When answering about the user, use second person ("you") or neutral phrasing.
- Be concise. If a direct answer is clear, avoid greetings and extra fluff.
- If you need to recall something specific that is not provided, ask a short clarification question."""
        
        base_prompt += memory_instructions
        
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

            # Query the memory system for relevant facts
            response = await self.memory_system.process_query(q)

            # Filter for facts from the facts store
            facts_results = [r for r in response.results if r.source_store == 'facts']
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
        for user_msg, assistant_msg in reversed(self.recent_exchanges):
            # Calculate tokens for this exchange
            exchange_tokens = (self.token_counter.count_tokens(user_msg) +
                             self.token_counter.count_tokens(assistant_msg))
            
            if token_count + exchange_tokens <= token_budget:
                # Insert at beginning to maintain chronological order
                messages.insert(0, {"role": "assistant", "content": assistant_msg})
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
                # Start new exchange with user message (shouldn't happen)
                logger.warning("Unexpected assistant response without user message")
        
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
