"""
M3 Integrated Context Manager - Production-ready M3 integration with SmartContextManager

Replaces standard SmartContextManager with M3 context retrieval system:
- Uses M3 SimilaritySearch, EquivalenceResolver, and ContextRetriever
- Maintains backward compatibility with standard memory when M3 disabled
- Graceful degradation if SurrealDB unavailable
- Same interface as SmartContextManager for seamless pipeline integration
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, LLMMessagesUpdateFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from processors.token_counter import get_token_counter
from services.embedding_service import EmbeddingService
from processors.surreal_message_store import create_surreal_message_store
try:
    from memory.session_manager import SessionManager
except Exception:
    SessionManager = None

# Import M3 components with graceful fallback
try:
    from memory.m3_similarity_search import M3SimilaritySearch
    from memory.m3_equivalence_resolver import M3EquivalenceResolver
    from memory.m3_conversation_buffer import M3ConversationBuffer  
    from memory.m3_context_retriever import M3ContextRetriever
    from memory.surreal_connection import SurrealConnectionManager
    from memory.m3_surreal_integration import M3SurrealIntegration
    M3_AVAILABLE = True
except ImportError as e:
    logger.warning(f"M3 components not available: {e}")
    M3_AVAILABLE = False

# Import standard memory components for fallback
try:
    from memory import create_smart_memory_system, extract_facts_from_text
    from memory import create_query_classifier
    from memory.query_router import QueryRouter
    from memory.graph_integration import get_graph_integration
    STANDARD_MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Standard memory components not available: {e}")
    STANDARD_MEMORY_AVAILABLE = False


@dataclass
class M3SessionMetadata:
    """Session metadata compatible with both M3 and standard memory"""
    turn_count: int = 0
    session_start: float = 0
    last_interaction: float = 0
    total_interactions: int = 0
    speaker_id: str = "unknown"
    user_id: str = "default_user"
    session_id: Optional[str] = None


@dataclass
class TokenBudget:
    """Token allocation for M3 and fallback modes"""
    system_prompt: int = 800
    contextual_memory: int = 2800
    current_input: int = 800
    generation_workspace: int = 3600
    
    @property
    def total(self) -> int:
        return self.system_prompt + self.contextual_memory + self.current_input


class M3IntegratedContextManager(FrameProcessor):
    """
    Production M3 integration that replaces SmartContextManager with backward compatibility.
    
    Features:
    - M3 context retrieval when enabled and available
    - Graceful fallback to standard memory system
    - Same interface as SmartContextManager
    - Automatic SurrealDB connection management
    - Performance monitoring and error recovery
    """
    
    def __init__(self, 
                 context,  # LLMContext instance
                 config=None,  # M3Config instance
                 facts_db_path: str = "data/facts.db",
                 max_tokens: int = 4096,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.context = context
        self.config = config
        self.max_tokens = max_tokens
        self.token_counter = get_token_counter()
        
        # Session tracking
        self.session_metadata = M3SessionMetadata()
        self.session_metadata.session_start = time.time()
        # Default speaker/user ids (match storage)
        import os as _os
        self.session_metadata.speaker_id = _os.getenv('USER_ID', 'default_user') or 'default_user'
        self.session_metadata.user_id = self.session_metadata.speaker_id
        
        # M3 conversation buffer for batch processing
        self.conversation_buffer = None
        if M3_AVAILABLE:
            try:
                buffer_turns = int(_os.getenv('M3_BUFFER_TURNS', '5'))
                buffer_seconds = float(_os.getenv('M3_BUFFER_SECONDS', '30.0'))
                self.conversation_buffer = M3ConversationBuffer(
                    max_turns=buffer_turns,
                    max_seconds=buffer_seconds
                )
                logger.info(f"🔄 M3 conversation buffer enabled: {buffer_turns} turns OR {buffer_seconds}s")
            except Exception as e:
                logger.warning(f"Failed to initialize M3 conversation buffer: {e}")
        
        # Session info cache
        self._session_count: int = 0
        # Try to attach existing global session id
        try:
            if SessionManager:
                sid = SessionManager.get_current_session()
                if sid:
                    self.session_metadata.session_id = sid
        except Exception:
            pass
        
        # M3 components (will be initialized if available)
        self.m3_similarity_search = None
        self.m3_equivalence_resolver = None
        self.m3_context_retriever = None
        self.surreal_connection = None
        self.m3_integration = None
        self.m3_enabled = False
        self.embedding_service: Optional[EmbeddingService] = None
        
        # Optional SurrealDB message storage (existing message/session storage)
        import os
        # Auto-enable Surreal message storage when SurrealDB is configured or M3 is enabled
        env_use = os.getenv('USE_SURREALDB', '').lower()
        has_sdb_url = bool(os.getenv('SURREALDB_URL', '').strip())
        m3_cfg_on = bool(self.config and getattr(self.config, 'enabled', False))
        self._enable_surreal_store = (env_use == 'true') or has_sdb_url or m3_cfg_on
        self.surreal_store = None
        if self._enable_surreal_store:
            try:
                self.surreal_store = create_surreal_message_store(
                    speaker_id=self.session_metadata.user_id,
                    auto_create_session=True
                )
                if self.surreal_store:
                    logger.info("📝 SurrealDB message storage enabled (M3IntegratedContextManager)")
            except Exception as e:
                logger.warning(f"SurrealDB message store init failed: {e}")
        
        # Standard memory fallback components
        self.standard_memory_system = None
        self.query_classifier = None
        self.query_router = None
        
        # Performance tracking
        self.stats = {
            'm3_queries': 0,
            'standard_queries': 0,
            'fallback_events': 0,
            'retrieval_time_ms': [],
            'context_generation_time_ms': []
        }
        
        # Conversation window management (stable context size like SmartContextManager)
        self.recent_exchanges: List[tuple] = []  # [(user, assistant)]
        self.max_recent_pairs: int = 10
        # Strict context mode: build bounded context each turn and run the LLM directly
        import os
        self.strict_context_mode: bool = os.getenv('M3_STRICT_CONTEXT', 'true').lower() == 'true'
        
        # Initialize memory systems
        asyncio.create_task(self._initialize_memory_systems())

    async def _refresh_session_info(self):
        """Refresh session_count from SurrealDB for header display."""
        try:
            if self.surreal_connection:
                speaker = self.session_metadata.speaker_id or 'default_user'
                info = await self.surreal_connection.get_session_info(speaker)
                if isinstance(info, dict):
                    self._session_count = int(info.get('session_count', 0))
        except Exception:
            pass
    
    async def _initialize_memory_systems(self):
        """Initialize M3 and/or standard memory systems based on availability"""
        # Try to initialize M3 system first
        if self.config and self.config.enabled and M3_AVAILABLE:
            await self._initialize_m3_system()
        
        # Initialize standard memory system as fallback
        if not self.m3_enabled and STANDARD_MEMORY_AVAILABLE:
            await self._initialize_standard_memory_system()
        
        if not self.m3_enabled and not self.standard_memory_system:
            logger.error("❌ No memory system available - neither M3 nor standard memory could be initialized")
    
    async def _initialize_m3_system(self):
        """Initialize M3 memory system with connection retry logic"""
        try:
            logger.info("🔄 Initializing M3 memory system...")
            
            # Connect to SurrealDB with retry logic
            url = f"ws://{self.config.surrealdb_host}:{self.config.surrealdb_port}/rpc"
            self.surreal_connection = SurrealConnectionManager(
                url=url,
                namespace=self.config.surrealdb_namespace,
                database=self.config.surrealdb_database
            )
            
            for attempt in range(self.config.startup_retry_attempts):
                try:
                    await asyncio.wait_for(
                        self.surreal_connection.connect(),
                        timeout=self.config.connection_timeout_seconds
                    )
                    logger.info("✅ SurrealDB connection established")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ SurrealDB connection attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.startup_retry_attempts - 1:
                        await asyncio.sleep(self.config.startup_retry_delay_seconds)
            else:
                raise Exception("Failed to connect to SurrealDB after all retries")
            
            # Initialize M3 integration
            self.m3_integration = M3SurrealIntegration(self.surreal_connection)
            await self.m3_integration.initialize()
            
            # Create M3 components (order matters - similarity_search needed for equivalence_resolver)
            self.m3_similarity_search = M3SimilaritySearch(self.m3_integration)
            self.m3_equivalence_resolver = M3EquivalenceResolver(self.m3_integration, self.m3_similarity_search)
            # Initialize embedding service for query embeddings
            try:
                self.embedding_service = EmbeddingService()
                embedding_test = await self.embedding_service.test_embedding_generation()
                if embedding_test:
                    logger.info("✅ Embedding service initialized and tested successfully")
                else:
                    logger.warning("⚠️ Embedding service test failed, using fallback methods")
            except Exception as e:
                logger.warning(f"EmbeddingService init failed, retrieval may be limited: {e}")
                self.embedding_service = None
            self.m3_context_retriever = M3ContextRetriever(
                self.m3_integration,
                self.m3_similarity_search,
                self.m3_equivalence_resolver,
                embedding_service=self.embedding_service
            )
            
            self.m3_enabled = True
            logger.info("✅ M3 memory system initialized successfully")
            # Refresh session count for header
            await self._refresh_session_info()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize M3 system: {e}")
            if self.config.fallback_to_standard_memory:
                logger.info("🔄 Falling back to standard memory system")
                self.stats['fallback_events'] += 1
            else:
                raise
    
    async def _initialize_standard_memory_system(self):
        """Initialize standard memory system as fallback"""
        try:
            logger.info("🔄 Initializing standard memory system...")
            
            # Get facts DB path from the constructor parameter or config
            facts_path = getattr(self, 'facts_db_path', None)
            if not facts_path:
                # Try to get from main config
                from config import config as main_config
                facts_path = main_config.memory.facts_db_path
            
            # Create standard memory components
            self.standard_memory_system = create_smart_memory_system(
                facts_db_path=facts_path
            )
            self.query_classifier = create_query_classifier()
            self.query_router = QueryRouter()
            
            logger.info("✅ Standard memory system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize standard memory system: {e}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with M3 or standard memory context injection"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, UserStartedSpeakingFrame):
            self.session_metadata.turn_count += 1
            self.session_metadata.last_interaction = time.time()
        
        if isinstance(frame, TranscriptionFrame):
            # Extract user input text
            user_text = frame.text.strip()
            if not user_text:
                await self.push_frame(frame, direction)
                return
            
            # M3 conversation buffering for batch processing
            should_process_chunk = False
            if self.conversation_buffer and self.m3_enabled:
                should_process_chunk = self.conversation_buffer.add_turn(
                    text=user_text,
                    speaker_id=self.session_metadata.speaker_id
                )
                
                if should_process_chunk:
                    # Process conversation chunk asynchronously
                    asyncio.create_task(self._process_conversation_chunk())
            
            # Persist user message if enabled
            try:
                if self.surreal_store:
                    await self.surreal_store._handle_user_message(user_text)
            except Exception as e:
                logger.debug(f"Surreal user store failed: {e}")
            # Update session info for header
            await self._refresh_session_info()
            
            # Update recent exchanges window
            self._track_user_turn(user_text)
            
            start_time = time.time()
            try:
                if self.strict_context_mode:
                    # Build bounded context (system + relevant memories + last N pairs + current user)
                    if self.m3_enabled:
                        logger.info(f"🔍 M3 ENABLED: Creating bounded context for query: '{user_text[:50]}...'")
                        context_frame = await self._create_m3_bounded_context_frame(user_text)
                        self.stats['m3_queries'] += 1
                    elif self.standard_memory_system:
                        logger.info(f"🔍 STANDARD MEMORY: Using standard memory for query: '{user_text[:50]}...'")
                        context_frame = await self._create_standard_bounded_context_frame(user_text)
                        self.stats['standard_queries'] += 1
                    else:
                        logger.info(f"🔍 BASIC CONTEXT: No memory system available for query: '{user_text[:50]}...'")
                        context_frame = await self._create_basic_context_frame(user_text)
                    retrieval_time = (time.time() - start_time) * 1000
                    self.stats['retrieval_time_ms'].append(retrieval_time)
                    # Push only the bounded context; do NOT forward raw transcription (prevents saturation)
                    if context_frame:
                        await self.push_frame(context_frame, direction)
                    else:
                        await self.push_frame(frame, direction)
                else:
                    # System-only update + forward original frame (aggregators maintain history)
                    update_frame = None
                    if self.m3_enabled:
                        update_frame = await self._build_m3_system_update(user_text)
                        self.stats['m3_queries'] += 1
                    elif self.standard_memory_system:
                        update_frame = await self._build_standard_system_update()
                        self.stats['standard_queries'] += 1
                    else:
                        update_frame = await self._build_basic_system_update()
                    retrieval_time = (time.time() - start_time) * 1000
                    self.stats['retrieval_time_ms'].append(retrieval_time)
                    if update_frame:
                        await self.push_frame(update_frame, direction)
                    await self.push_frame(frame, direction)
                # Refresh session_id from SessionManager after first message
                try:
                    if SessionManager and not self.session_metadata.session_id:
                        sid = SessionManager.get_current_session()
                        if sid:
                            self.session_metadata.session_id = sid
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"❌ Error creating/updating context: {e}")
                await self.push_frame(frame, direction)
        else:
            # Forward non-transcription frames
            await self.push_frame(frame, direction)
    
    async def _create_m3_context_frame(self, user_text: str) -> Optional[LLMMessagesUpdateFrame]:
        """Create context frame using M3 retrieval system"""
        try:
            # Retrieve context using M3 system
            logger.info(f"🔍 M3 RETRIEVAL: Starting retrieval for query: '{user_text[:50]}...' speaker_id={self.session_metadata.speaker_id}")
            retrieved_context = await self.m3_context_retriever.retrieve_context(
                query=user_text,
                max_items=self.config.max_retrieval_items,
                speaker_id=self.session_metadata.speaker_id
            )
            logger.info(f"🔍 M3 RETRIEVAL RESULT: {len(retrieved_context.items) if retrieved_context and retrieved_context.items else 0} items retrieved")
            items = self._filter_context_items(retrieved_context.items, user_text)
            
            # Build context messages
            messages = []
            
            # System prompt
            system_prompt = self._build_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
            
            # Add M3 context if available
            if items:
                context_content = self._format_m3_context(items)
                messages.append({"role": "system", "content": f"Relevant memories:\n{context_content}"})
            
            # Add bounded recent exchanges (last N pairs)
            messages.extend(self._build_recent_exchange_messages(self.max_recent_pairs))
            
            # Add current user input
            messages.append({"role": "user", "content": user_text})
            
            # Use update frame so aggregators can manage history consistently
            return LLMMessagesUpdateFrame(messages=messages, run_llm=True)
            
        except Exception as e:
            logger.error(f"❌ M3 context retrieval failed: {e}")
            return None

    async def _build_m3_system_update(self, user_text: str) -> Optional[LLMMessagesUpdateFrame]:
        """Build a system-only update frame with M3 context (do not run LLM)."""
        try:
            retrieved_context = await self.m3_context_retriever.retrieve_context(
                query=user_text,
                max_items=self.config.max_retrieval_items,
                speaker_id=self.session_metadata.speaker_id
            )
            items = self._filter_context_items(retrieved_context.items, user_text)
            messages = []
            system_prompt = self._build_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
            if items:
                context_content = self._format_m3_context(items)
                messages.append({"role": "system", "content": f"Relevant memories:\n{context_content}"})
            return LLMMessagesUpdateFrame(messages=messages, run_llm=False)
        except Exception as e:
            logger.error(f"❌ M3 system update failed: {e}")
            return None

    async def _build_standard_system_update(self) -> Optional[LLMMessagesUpdateFrame]:
        try:
            messages = [{"role": "system", "content": self._build_system_prompt()}]
            return LLMMessagesUpdateFrame(messages=messages, run_llm=False)
        except Exception:
            return None

    async def _build_basic_system_update(self) -> Optional[LLMMessagesUpdateFrame]:
        try:
            return LLMMessagesUpdateFrame(messages=[{"role": "system", "content": "You are Slowcat, a helpful voice assistant."}], run_llm=False)
        except Exception:
            return None

    async def _create_m3_bounded_context_frame(self, user_text: str) -> Optional[LLMMessagesUpdateFrame]:
        """Build full, bounded context like SmartContextManager (token-safe sliding window)."""
        try:
            frame = await self._create_m3_context_frame(user_text)
            if not frame:
                return None
            # Optionally trim to token budget
            if self.token_counter and hasattr(frame, 'messages'):
                frame.messages = self._truncate_messages_to_budget(frame.messages, self.max_tokens)
            return frame
        except Exception as e:
            logger.error(f"❌ M3 bounded context build failed: {e}")
            return None

    async def _create_standard_bounded_context_frame(self, user_text: str) -> Optional[LLMMessagesUpdateFrame]:
        try:
            # Reuse standard prompt + recent exchanges; no memory injection
            messages = [{"role": "system", "content": self._build_system_prompt()}]
            messages.extend(self._build_recent_exchange_messages(self.max_recent_pairs))
            messages.append({"role": "user", "content": user_text})
            frame = LLMMessagesUpdateFrame(messages=messages, run_llm=True)
            if self.token_counter:
                frame.messages = self._truncate_messages_to_budget(frame.messages, self.max_tokens)
            return frame
        except Exception:
            return None

    def _build_recent_exchange_messages(self, max_pairs: int) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        pairs = self.recent_exchanges[-max_pairs:]
        # Skip unfinished last pair (to avoid duplicating the current user input)
        if pairs and pairs[-1] and not pairs[-1][1]:
            pairs = pairs[:-1]
        for user_text, assistant_text in pairs:
            if user_text:
                msgs.append({"role": "user", "content": user_text})
            if assistant_text:
                msgs.append({"role": "assistant", "content": assistant_text})
        return msgs

    def _track_user_turn(self, user_text: str):
        # If last pair has empty assistant, start a new pair to avoid merging unfinished ones
        if self.recent_exchanges and not self.recent_exchanges[-1][1]:
            # Replace the last unfinished user message with the latest (keep only most recent text)
            self.recent_exchanges[-1] = (user_text, "")
        else:
            self.recent_exchanges.append((user_text, ""))
        if len(self.recent_exchanges) > self.max_recent_pairs * 2:
            self.recent_exchanges = self.recent_exchanges[-self.max_recent_pairs:]

    def _truncate_messages_to_budget(self, messages: List[Dict[str, str]], budget_tokens: int) -> List[Dict[str, str]]:
        try:
            total = sum(self.token_counter.count_tokens(m.get('content', '')) for m in messages)
            if total <= budget_tokens:
                return messages
            # Drop oldest conversational lines (keep system first)
            sys_msgs = [m for m in messages if m.get('role') == 'system']
            convo_msgs = [m for m in messages if m.get('role') != 'system']
            while sys_msgs + convo_msgs and total > budget_tokens and convo_msgs:
                dropped = convo_msgs.pop(0)
                total -= self.token_counter.count_tokens(dropped.get('content', ''))
            return sys_msgs + convo_msgs
        except Exception:
            return messages
    
    async def _process_conversation_chunk(self):
        """Process conversation chunk with M3 batch fact extraction"""
        if not self.conversation_buffer:
            return
            
        try:
            # Get conversation chunk
            chunk = self.conversation_buffer.get_chunk()
            if not chunk:
                return
                
            logger.info(f"🔄 Processing M3 conversation chunk: {chunk.total_turns} turns, {chunk.duration_seconds:.1f}s")
            
            # Get recent context for better extraction
            recent_context = ""
            try:
                if self.m3_context_retriever:
                    # Get some recent context for better understanding
                    recent_clips = await self.m3_context_retriever.retrieve_context(
                        query=chunk.combined_text[:100],  # Use beginning of chunk
                        max_items=3,
                        speaker_id=self.session_metadata.speaker_id
                    )
                    if recent_clips.items:
                        recent_context = " ".join([item.content[:100] for item in recent_clips.items[:2]])
            except Exception as e:
                logger.debug(f"Failed to get recent context: {e}")
            
            # Extract facts from conversation chunk
            try:
                from memory.dspy_integration import extract_facts_from_chunk_dspy
                
                facts = extract_facts_from_chunk_dspy(
                    chunk_text=chunk.combined_text,
                    previous_context=recent_context
                )
                
                if facts and self.surreal_store and self.surreal_store.surreal:
                    logger.info(f"🧠 Storing {len(facts)} facts from conversation chunk")
                    
                    # Store facts extracted from chunk
                    for fact in facts:
                        try:
                            await self.surreal_store.surreal.store_knowledge_relation(
                                subject_name=fact.get('subject', 'user'),
                                predicate=fact.get('predicate', 'related_to'),
                                object_name=fact.get('value', ''),
                                confidence=fact.get('confidence', 0.7)
                            )
                        except Exception as e:
                            logger.debug(f"Failed to store chunk fact: {e}")
                
                # Clear processed buffer
                self.conversation_buffer.clear_buffer()
                
                logger.info(f"✅ M3 chunk processing complete: {len(facts)} facts extracted")
                
            except Exception as e:
                logger.error(f"❌ M3 chunk fact extraction failed: {e}")
                # Still clear buffer to avoid blocking
                self.conversation_buffer.clear_buffer()
                
        except Exception as e:
            logger.error(f"❌ M3 conversation chunk processing failed: {e}")
    
    async def _create_standard_context_frame(self, user_text: str) -> Optional[LLMMessagesUpdateFrame]:
        """Create context frame using standard memory system"""
        try:
            # Use standard memory retrieval logic
            # This would need to be implemented based on your existing SmartContextManager
            # For now, return a basic frame
            messages = [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_text}
            ]
            
            return LLMMessagesUpdateFrame(messages=messages, run_llm=True)
            
        except Exception as e:
            logger.error(f"❌ Standard memory context creation failed: {e}")
            return None
    
    async def _create_basic_context_frame(self, user_text: str) -> LLMMessagesUpdateFrame:
        """Create basic context frame with no memory"""
        messages = [
            {"role": "system", "content": "You are Slowcat, a helpful voice assistant."},
            {"role": "user", "content": user_text}
        ]
        return LLMMessagesUpdateFrame(messages=messages, run_llm=True)
    
    def _build_system_prompt(self) -> str:
        """Build dynamic system prompt based on session state (match SCM style)."""
        base_prompt = (
            "You are Slowcat, a helpful voice assistant.\n"
            "- Never speak as the user; always refer to them as 'you'.\n"
            "- Treat 'Relevant memories' as reference notes. Do not quote or mimic their tone; summarize facts.\n"
            "- Do not repeat the user's words back verbatim unless explicitly asked.\n"
            "- Prefer concise, direct answers using relevant facts."
        )
        from datetime import datetime
        import time as time_module
        
        def _fmt(ts):
            try:
                return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else ''
            except Exception:
                return ''
        
        # Current time information
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Session timing information
        session_start_str = _fmt(self.session_metadata.session_start)
        session_duration = time_module.time() - self.session_metadata.session_start
        duration_mins = int(session_duration / 60)
        
        first_str = _fmt(getattr(self, '_first_seen_ts', None))
        last_str = _fmt(getattr(self, '_last_interaction_ts', None))
        
        # Build comprehensive timing context
        info = f"\nCurrent time: {current_time_str}"
        info += f"\nSession info: Sessions {self._session_count}, Turn {self.session_metadata.turn_count}"
        info += f", Started: {session_start_str}, Duration: {duration_mins}min"
        
        if first_str:
            info += f", First seen: {first_str}"
        if last_str:
            info += f", Last interaction: {last_str}"
        if self.session_metadata.speaker_id and self.session_metadata.speaker_id != "unknown":
            info += f", Speaker: {self.session_metadata.speaker_id}"
        if self.session_metadata.session_id:
            info += f", Session: {self.session_metadata.session_id}"
        
        return base_prompt + info
    
    def _format_m3_context(self, context_items) -> str:
        """Format M3 context items for inclusion in prompt with simple de-duplication"""
        formatted_items = []
        seen = set()
        for item in context_items:
            content = (getattr(item, 'content', '') or '').strip()
            key = content.lower()
            if not content or key in seen:
                continue
            seen.add(key)
            formatted_items.append(f"- {content} (relevance: {item.relevance_score:.2f})")
        return "\n".join(formatted_items)
    
    async def get_initial_context_frame(self) -> LLMMessagesUpdateFrame:
        """Get initial context frame for session start"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": "Hello"}
        ]
        return LLMMessagesUpdateFrame(messages=messages, run_llm=True)
    
    def needs_greeting(self) -> bool:
        """Check if greeting is needed"""
        return self.session_metadata.turn_count == 0
    
    async def get_greeting_text(self) -> str:
        """Get greeting text"""
        return "Hello, I'm Slowcat!"
    
    async def finalize_summary(self):
        """Finalize session and store summary"""
        try:
            if self.m3_enabled and self.m3_integration:
                # Store session summary in M3 system
                session_duration = time.time() - self.session_metadata.session_start
                logger.info(f"📊 Session completed: {self.session_metadata.turn_count} turns, {session_duration:.1f}s")
                # Could implement session summarization here
            
            # Finalize SurrealDB session if any
            if self.surreal_store:
                try:
                    await self.surreal_store.finalize_session()
                except Exception:
                    pass

            # Close connections (after finalizing session)
            if self.surreal_connection:
                try:
                    await self.surreal_connection.close()
                except Exception:
                    # Be resilient during shutdown
                    pass
                
        except Exception as e:
            logger.error(f"❌ Error finalizing session: {e}")

    async def add_assistant_response(self, response: str):
        """Called by ResponseTap to persist assistant messages."""
        try:
            cleaned = self._dedup_text(response or '')
            if self.surreal_store and cleaned.strip():
                await self.surreal_store._handle_assistant_message(cleaned.strip())
            # Track assistant side of the last exchange
            if self.recent_exchanges:
                user_text, _ = self.recent_exchanges[-1]
                self.recent_exchanges[-1] = (user_text, cleaned)
            else:
                self.recent_exchanges.append(("", cleaned))
        except Exception as e:
            logger.debug(f"Surreal assistant store failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_retrieval_time = sum(self.stats['retrieval_time_ms']) / len(self.stats['retrieval_time_ms']) if self.stats['retrieval_time_ms'] else 0
        
        return {
            'm3_enabled': self.m3_enabled,
            'total_queries': self.stats['m3_queries'] + self.stats['standard_queries'],
            'm3_queries': self.stats['m3_queries'],
            'standard_queries': self.stats['standard_queries'],
            'fallback_events': self.stats['fallback_events'],
            'avg_retrieval_time_ms': round(avg_retrieval_time, 2),
            'session_turns': self.session_metadata.turn_count
        }

    def _filter_context_items(self, items, user_text: str):
        """Remove voice items and entries too similar to the current user text."""
        def _sim(a: str, b: str) -> float:
            try:
                sa, sb = set((a or '').lower().split()), set((b or '').lower().split())
                if not sa or not sb:
                    return 0.0
                inter = len(sa & sb)
                union = len(sa | sb)
                return inter / union if union else 0.0
            except Exception:
                return 0.0
        filtered = []
        for it in items or []:
            try:
                # ContextItem has attributes: content (str), context_type (enum)
                ctype = getattr(it, 'context_type', None)
                if str(ctype).lower().endswith('voice'):
                    continue
                content = getattr(it, 'content', '') or ''
                if content and _sim(content, user_text) >= 0.7:
                    continue
                filtered.append(it)
            except Exception:
                continue
        return filtered

    def _dedup_text(self, text: str) -> str:
        """Basic deduplication to clean repeated segments in stored assistant text."""
        try:
            import re
            if not text:
                return text
            # Simple repeated word pattern
            pattern = r'(\b\w+[\'.,!?]*)\1+'
            cleaned = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
            complex_pattern = r'(\b\w+\'?\w*)\1+([^\w]|\s|$)'
            cleaned = re.sub(complex_pattern, r'\1\2', cleaned)
            return cleaned
        except Exception:
            return text


def create_m3_integrated_context_manager(context, config=None, **kwargs):
    """Factory function to create M3 integrated context manager"""
    return M3IntegratedContextManager(context=context, config=config, **kwargs)
