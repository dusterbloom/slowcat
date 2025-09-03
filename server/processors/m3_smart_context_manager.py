"""
M3SmartContextManager - M3-inspired context management with SurrealDB backend

Replaces the token budgeting approach with M3's entity-centric memory retrieval:
- SurrealDB-based M3 node storage and retrieval
- Optional voice node creation from transcriptions
- Episodic and semantic memory generation from batches
- M3-style similarity search for context selection

Based on M3-Agent's control architecture with voice-first optimizations.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, LLMMessagesUpdateFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# M3 SurrealDB components  
from memory.m3_surreal_integration import M3SurrealIntegration
from memory.m3_similarity_search import M3SimilaritySearch, ModalityType
from memory.surreal_connection import SurrealConnectionManager
from services.embedding_service import EmbeddingService

# Existing components for integration
try:
    from processors.token_counter import get_token_counter
except ImportError:
    get_token_counter = None


@dataclass
class M3SessionMetadata:
    """M3-style session tracking with clip-based organization."""
    session_id: int = 0
    clip_id: int = 0  # M3's temporal segmentation
    turn_count: int = 0
    session_start: float = 0
    last_interaction: float = 0
    speaker_id: Optional[str] = "unknown"
    voice_node_id: Optional[int] = None


class M3SmartContextManager(FrameProcessor):
    """
    M3-inspired context manager replacing token budgeting with entity-centric memory.
    
    Key M3 patterns:
    - AudioGraph for unified memory storage
    - Voice node processing for speaker recognition
    - Episodic/semantic memory generation  
    - Similarity-based context retrieval
    - Progressive speaker annotation
    """
    
    def __init__(self, 
                 context,  # LLMContext instance
                 config=None,  # M3Config instance 
                 max_context_tokens: int = 4096,  # Total context limit
                 memory_tokens: int = 2000,       # Tokens for memory context
                 **kwargs):
        super().__init__(**kwargs)
        
        self.context = context
        self.config = config
        self.max_context_tokens = max_context_tokens
        self.memory_tokens = memory_tokens
        
        # M3 SurrealDB components (will be initialized async)
        self.surreal_connection = None
        self.m3_integration = None
        self.similarity_search = None
        self.m3_initialized = False
        self.embedding_service: Optional[EmbeddingService] = None
        
        # Session tracking (M3's clip-based organization)
        self.session_metadata = M3SessionMetadata()
        self.session_metadata.session_start = time.time()
        
        # M3's conversation tracking
        self.conversation_buffer = []  # Recent conversation for memory generation
        self.last_memory_update = time.time()
        self.memory_update_interval = 30  # Update memory every 30 seconds
        self.memory_turn_threshold = 5    # Or every 5 turns
        
        # Token counting (if available)
        self.token_counter = get_token_counter() if get_token_counter else None
        
        # M3 processing statistics
        self.stats = {
            'frames_processed': 0,
            'voice_nodes_created': 0,
            'memory_nodes_created': 0,
            'context_retrievals': 0,
            'speaker_updates': 0
        }
        
        # Initialize M3 connection asynchronously
        asyncio.create_task(self._initialize_m3())
        
        logger.info(f"🧠 M3SmartContextManager initialized")
        logger.info(f"   Max context tokens: {max_context_tokens}")
        logger.info(f"   Memory context tokens: {memory_tokens}")
        logger.info(f"   M3 will connect to production database")
    
    async def _initialize_m3(self):
        """Initialize M3 SurrealDB integration"""
        try:
            if not self.config:
                logger.warning("No M3 config provided, using defaults")
                # Create default config
                from config import M3Config
                self.config = M3Config()
            
            # Create SurrealDB connection to production database
            self.surreal_connection = SurrealConnectionManager(
                url=f"ws://{self.config.surrealdb_host}:{self.config.surrealdb_port}/rpc",
                namespace=self.config.surrealdb_namespace,
                database=self.config.surrealdb_database
            )
            
            await self.surreal_connection.connect()
            logger.info(f"✅ M3 connected to {self.config.surrealdb_namespace}/{self.config.surrealdb_database}")
            
            # Initialize M3 components
            self.m3_integration = M3SurrealIntegration(self.surreal_connection)
            await self.m3_integration.initialize()
            
            self.similarity_search = M3SimilaritySearch(self.m3_integration)
            self.embedding_service = EmbeddingService()
            await self.embedding_service.test_embedding_generation()
            
            self.m3_initialized = True
            logger.info("✅ M3 integration fully initialized")
            
        except Exception as e:
            logger.error(f"❌ M3 initialization failed: {e}")
            self.m3_initialized = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """M3-style frame processing with AudioGraph integration."""
        await super().process_frame(frame, direction)
        
        try:
            if isinstance(frame, TranscriptionFrame):
                await self._handle_transcription_frame(frame, direction)
            elif isinstance(frame, UserStartedSpeakingFrame):
                await self._handle_user_started_speaking(frame, direction)
            else:
                # Forward all other frames
                await self.push_frame(frame, direction)
                
            self.stats['frames_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error in M3SmartContextManager frame processing: {e}")
            # Forward frame even on error to prevent pipeline blocking
            await self.push_frame(frame, direction)
    
    async def _handle_transcription_frame(self, frame: TranscriptionFrame, direction: FrameDirection):
        """Handle transcription with M3's voice and memory processing."""
        text = frame.text.strip()
        if not text:
            await self.push_frame(frame, direction)
            return
        
        # Update session tracking
        self.session_metadata.turn_count += 1
        self.session_metadata.last_interaction = time.time()
        
        # Add to conversation buffer for memory processing
        self.conversation_buffer.append({
            'text': text,
            'timestamp': time.time(),
            'turn': self.session_metadata.turn_count,
            'frame_type': 'user'
        })
        
        # M3's voice processing (if audio data available)
        voice_node_id = None
        if hasattr(frame, 'audio_data') and frame.audio_data:
            voice_node_id = await self._process_voice_data(
                frame.audio_data, text, self.session_metadata.clip_id
            )
            
            if voice_node_id:
                self.session_metadata.voice_node_id = voice_node_id
                self.stats['voice_nodes_created'] += 1
        
        # M3's memory processing (periodic updates)
        await self._maybe_update_memories()
        
        # M3's context retrieval and injection
        enriched_context = await self._build_m3_context(text, voice_node_id)
        
        # Create enriched LLM frame with M3 context
        enriched_frame = await self._create_context_frame(text, enriched_context, direction)
        
        if enriched_frame:
            await self.push_frame(enriched_frame, direction)
        else:
            await self.push_frame(frame, direction)
    
    async def _handle_user_started_speaking(self, frame: UserStartedSpeakingFrame, direction: FrameDirection):
        """Handle start of user speech with M3's session management."""
        # M3's clip advancement every N interactions
        if self.session_metadata.turn_count > 0 and self.session_metadata.turn_count % 10 == 0:
            try:
                if self.m3_initialized and self.m3_integration:
                    # Ensure we have a session id
                    if not self.session_metadata.session_id:
                        self.session_metadata.session_id = f"session_{int(time.time())}"
                    new_clip_id = await self.m3_integration.close_current_clip(
                        create_new=True,
                        session_id=self.session_metadata.session_id
                    )
                    if new_clip_id:
                        self.session_metadata.clip_id = new_clip_id
                        logger.debug(f"🎬 Advanced to new clip {self.session_metadata.clip_id}")
            except Exception as e:
                logger.warning(f"Clip rotation failed: {e}")
        
        # Forward frame
        await self.push_frame(frame, direction)
    
    async def _process_voice_data(self, audio_data: bytes, transcript: str, clip_id: int) -> Optional[int]:
        """Create a voice node in M3 based on transcription (no audio embeddings yet)."""
        try:
            if not self.m3_initialized:
                return None
            # Generate text embedding for transcript if available
            embeddings = []
            if self.embedding_service and transcript:
                try:
                    emb = await self.embedding_service.get_embedding(transcript)
                    embeddings = [emb] if emb else []
                except Exception as ee:
                    logger.warning(f"Embedding for voice transcript failed: {ee}")

            node_id = await self.m3_integration.store_m3_node(
                node_type='voice',
                contents=[transcript],
                embeddings=embeddings,
                clip_id=clip_id or None,
                speaker_id=self.session_metadata.speaker_id or 'unknown',
                extraction_method='transcription',
                confidence=0.85
            )
            return node_id
        except Exception as e:
            logger.error(f"Voice processing failed: {e}")
            return None
    
    async def _maybe_update_memories(self):
        """M3's periodic memory updates with batching."""
        current_time = time.time()
        
        # Check if it's time for memory update
        time_since_update = current_time - self.last_memory_update
        turns_since_update = len(self.conversation_buffer)
        
        if (time_since_update > self.memory_update_interval or 
            turns_since_update >= self.memory_turn_threshold):
            
            await self._update_memories()
            self.last_memory_update = current_time
    
    async def _update_memories(self):
        """M3's memory generation and storage."""
        if not self.conversation_buffer or not self.m3_initialized:
            return
        
        try:
            # Prepare conversation context
            conversation_text = " ".join([
                entry['text'] for entry in self.conversation_buffer
            ])
            
            # Create or get current clip
            if self.session_metadata.session_id:
                session_id = self.session_metadata.session_id
            else:
                session_id = f"session_{int(time.time())}"
                self.session_metadata.session_id = session_id
            
            if not self.session_metadata.clip_id:
                self.session_metadata.clip_id = await self.m3_integration.create_new_clip(session_id)
                logger.debug(f"🎬 Created new clip {self.session_metadata.clip_id} for session {session_id}")
            
            # Generate episodic memory from conversation batch
            if len(self.conversation_buffer) > 0:
                episodic_content = f"Conversation: {conversation_text}"
                episodic_node_id = await self.m3_integration.store_m3_node(
                    node_type='episodic',
                    contents=[episodic_content],
                    embeddings=[[0.5] * 384],  # Simple embedding for now
                    clip_id=self.session_metadata.clip_id,
                    speaker_id=self.session_metadata.speaker_id or 'unknown'
                )
                
                if episodic_node_id:
                    self.stats['memory_nodes_created'] += 1
                    logger.debug(f"📝 Created episodic memory node {episodic_node_id}")
            
            # Generate semantic memory if conversation contains facts/concepts
            if len(conversation_text) > 50:  # Only for substantial conversations
                semantic_content = f"Key concepts: {conversation_text[:100]}..."
                semantic_node_id = await self.m3_integration.store_m3_node(
                    node_type='semantic',
                    contents=[semantic_content],
                    embeddings=[[0.7] * 384],  # Simple embedding for now
                    clip_id=self.session_metadata.clip_id,
                    speaker_id='system'
                )
                
                if semantic_node_id:
                    self.stats['memory_nodes_created'] += 1
                    logger.debug(f"🧠 Created semantic memory node {semantic_node_id}")
            
            # Clear processed conversation buffer
            self.conversation_buffer = []
            logger.info(f"🎉 M3 memory update completed: {self.stats['memory_nodes_created']} total nodes")
            
        except Exception as e:
            logger.error(f"M3 memory update failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _build_m3_context(self, query_text: str, voice_node_id: Optional[int]) -> str:
        """Build context using M3's similarity-based retrieval (SurrealDB backend)."""
        try:
            if not self.m3_initialized or not self.embedding_service:
                return ""

            # Generate embedding for the query
            query_embedding = await self.embedding_service.get_embedding(query_text)
            if not query_embedding:
                return ""

            # Search semantic and episodic nodes
            semantic_results = await self.similarity_search.search_nodes(
                query_embedding=query_embedding,
                modality=ModalityType.SEMANTIC,
                max_results=8,
                node_type_filter='semantic'
            )
            episodic_results = await self.similarity_search.search_nodes(
                query_embedding=query_embedding,
                modality=ModalityType.EPISODIC,
                max_results=6,
                node_type_filter='episodic'
            )

            # Optionally include voice-related context if we had a recent voice node
            voice_results = []
            if voice_node_id:
                # Use the same query embedding but mark modality as VOICE for thresholding
                voice_results = await self.similarity_search.search_nodes(
                    query_embedding=query_embedding,
                    modality=ModalityType.VOICE,
                    max_results=4,
                    node_type_filter='voice'
                )

            # Merge and format context
            combined = semantic_results + episodic_results + voice_results
            context_parts = self._format_memory_context_from_results(combined[:10])
            self.stats['context_retrievals'] += 1

            # Prepend speaker if known
            if self.session_metadata.speaker_id and self.session_metadata.speaker_id != "unknown":
                context_parts.insert(0, f"Speaker: {self.session_metadata.speaker_id}")

            full_context = "\n".join(context_parts) if context_parts else ""

            # Token limiting
            if self.token_counter and full_context:
                context_tokens = self.token_counter.count_tokens(full_context)
                if context_tokens > self.memory_tokens:
                    truncated_context = self._truncate_context(full_context, self.memory_tokens)
                    logger.debug(f"🔄 Context truncated: {context_tokens} -> {self.memory_tokens} tokens")
                    return truncated_context

            return full_context

        except Exception as e:
            logger.error(f"Context building failed: {e}")
            return ""
    
    def _format_memory_context_from_results(self, results) -> List[str]:
        """Format M3SimilaritySearch results into context strings."""
        context_parts: List[str] = []
        try:
            for res in results:
                # res may be SearchResult or dict-like
                node_type = getattr(res, 'node_type', None) or res.get('node_type', 'unknown')
                similarity = getattr(res, 'similarity_score', None) or res.get('similarity', 0.0)
                contents = getattr(res, 'content', None) or res.get('contents', [])
                if not contents:
                    continue
                content = contents[0]
                indicator = "●" if similarity >= 0.8 else ("◐" if similarity >= 0.6 else "○")
                formatted = f"{indicator} {node_type.capitalize()}: {content}"
                context_parts.append(formatted)
        except Exception:
            pass
        return context_parts
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit token limit."""
        if not self.token_counter:
            # Simple character-based truncation fallback
            max_chars = max_tokens * 4  # Rough estimate
            return context[:max_chars] + "..." if len(context) > max_chars else context
        
        lines = context.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.token_counter.count_tokens(line)
            if current_tokens + line_tokens > max_tokens:
                break
            truncated_lines.append(line)
            current_tokens += line_tokens
        
        result = '\n'.join(truncated_lines)
        if len(truncated_lines) < len(lines):
            result += "\n..."
        
        return result
    
    async def _create_context_frame(self, user_text: str, context: str, 
                                   direction: FrameDirection) -> Optional[Frame]:
        """Create LLM frame with M3 context integration."""
        try:
            # Build messages with M3 context
            messages = []
            
            # System message with M3 context
            if context:
                system_content = f"""You are Slowcat, a voice agent with access to conversation memory.

Recent relevant context:
{context}

Respond naturally based on the context and current conversation."""
                messages.append({"role": "system", "content": system_content})
            
            # User message
            messages.append({"role": "user", "content": user_text})
            
            # Create LLM frame
            if hasattr(LLMMessagesFrame, '__init__'):
                return LLMMessagesFrame(messages=messages)
            else:
                # Fallback for different LLMMessagesFrame interface
                return LLMMessagesFrame(messages)
                
        except Exception as e:
            logger.error(f"Failed to create context frame: {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics (M3-focused)."""
        return {
            **self.stats,
            'session_metadata': {
                'session_id': self.session_metadata.session_id,
                'clip_id': self.session_metadata.clip_id,
                'turn_count': self.session_metadata.turn_count,
                'speaker_id': self.session_metadata.speaker_id,
                'voice_node_id': self.session_metadata.voice_node_id
            },
            'm3_initialized': self.m3_initialized,
        }
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get M3 memory summary from SurrealDB statistics."""
        try:
            if not self.m3_initialized:
                return {
                    'total_nodes': 0,
                    'voice_nodes': 0,
                    'episodic_nodes': 0,
                    'semantic_nodes': 0,
                    'total_edges': 0,
                    'equivalence_sets': 0,
                    'clips_processed': self.session_metadata.clip_id or 0,
                    'speakers_identified': 0,
                }
            stats = await self.m3_integration.get_statistics()
            # Derive by type if available
            nodes_by_type = {row['node_type']: row['count'] for row in stats.get('nodes_by_type', [])} if stats else {}
            return {
                'total_nodes': stats.get('total_nodes', 0) if stats else 0,
                'voice_nodes': nodes_by_type.get('voice', 0),
                'episodic_nodes': nodes_by_type.get('episodic', 0),
                'semantic_nodes': nodes_by_type.get('semantic', 0),
                'total_edges': stats.get('total_edges', 0) if stats else 0,
                'equivalence_sets': stats.get('total_equivalences', 0) if stats else 0,
                'clips_processed': stats.get('total_clips', 0) if stats else 0,
                'speakers_identified': 0  # Not tracked here yet
            }
        except Exception as e:
            logger.error(f"Failed to get memory summary: {e}")
            return {}
