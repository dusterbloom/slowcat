"""M3 Memory Processor for Frame-to-Memory Integration

This processor handles the conversion of Pipecat frames into M3 memory nodes,
including voice transcription, semantic extraction, and graph relationships.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
import json
import re
from datetime import datetime

from pipecat.frames.frames import (
    Frame, StartFrame, EndFrame, CancelFrame, 
    AudioRawFrame, TextFrame, TranscriptionFrame,
    LLMFullResponseStartFrame, LLMFullResponseEndFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from memory.m3_surreal_integration import M3SurrealIntegration
from memory.m3_llm_generator import M3LLMGenerator, EpisodicMemory, SemanticMemory
from services.embedding_service import EmbeddingService
# Optional imports for enhanced functionality
try:
    from memory.spacy_fact_extractor import SpacyFactExtractor
except ImportError:
    SpacyFactExtractor = None

logger = logging.getLogger(__name__)

class M3MemoryProcessor(FrameProcessor):
    """Process frames into M3 memory nodes with graph relationships"""
    
    def __init__(self, 
                 m3_integration: M3SurrealIntegration,
                 embedding_service: Optional[EmbeddingService] = None,
                 llm_generator: Optional[M3LLMGenerator] = None,
                 fact_extractor: Optional[SpacyFactExtractor] = None,
                 similarity_threshold: float = 0.7,
                 auto_create_edges: bool = True,
                 clip_duration_seconds: int = 30,
                 enable_llm_generation: bool = True,
                 **kwargs):
        """Initialize M3 Memory Processor
        
        Args:
            m3_integration: M3 SurrealDB integration instance
            embedding_service: Service for generating embeddings
            llm_generator: LLM-powered memory generator
            fact_extractor: Service for extracting facts from text
            similarity_threshold: Minimum similarity for auto-edge creation
            auto_create_edges: Whether to automatically infer edges
            clip_duration_seconds: Duration before creating new clips
            enable_llm_generation: Whether to use LLM for enhanced memory generation
        """
        super().__init__(**kwargs)
        self.m3_integration = m3_integration
        self.embedding_service = embedding_service
        self.llm_generator = llm_generator
        self.fact_extractor = fact_extractor
        self.similarity_threshold = similarity_threshold
        self.auto_create_edges = auto_create_edges
        self.clip_duration_seconds = clip_duration_seconds
        self.enable_llm_generation = enable_llm_generation
        
        # State tracking
        self._current_session_id: str = "default"
        self._current_speaker_id: str = "unknown"
        self._is_assistant_response: bool = False
        self._assistant_response_buffer: List[str] = []
        self._last_clip_time: Optional[datetime] = None
        
        # Audio processing state
        self._audio_buffer: List[AudioRawFrame] = []
        self._transcription_buffer: List[str] = []
        
        # LLM-enhanced memory generation state
        self._conversation_window: List[Dict[str, Any]] = []  # For episodic memory
        self._speaker_utterances: Dict[str, List[str]] = {}  # Track per-speaker utterances
        self._last_episodic_generation: Optional[datetime] = None
        
        logger.info(f"M3 Memory Processor initialized (LLM generation: {enable_llm_generation})")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process different frame types into M3 nodes"""
        await super().process_frame(frame, direction)
        
        try:
            if isinstance(frame, StartFrame):
                await self.push_frame(frame, direction)
                await self._initialize_processor()
                return
            
            elif isinstance(frame, EndFrame):
                await self._finalize_session()
                await self.push_frame(frame, direction)
                return
            
            elif isinstance(frame, TranscriptionFrame):
                await self._process_transcription_frame(frame)
            
            elif isinstance(frame, LLMFullResponseStartFrame):
                self._is_assistant_response = True
                self._assistant_response_buffer = []
            
            elif isinstance(frame, LLMFullResponseEndFrame):
                if self._is_assistant_response and self._assistant_response_buffer:
                    await self._process_assistant_response()
                self._is_assistant_response = False
                self._assistant_response_buffer = []
            
            elif isinstance(frame, TextFrame):
                if self._is_assistant_response:
                    self._assistant_response_buffer.append(frame.text)
                else:
                    # User text input (not transcription)
                    await self._process_user_text(frame.text)
            
            elif isinstance(frame, AudioRawFrame):
                await self._buffer_audio_frame(frame)
            
            # Handle data frames if needed
            # elif isinstance(frame, DataFrame):
            #     await self._process_data_frame(frame)
            
            # Always forward the frame
            await self.push_frame(frame, direction)
            
        except Exception as e:
            logger.error(f"Error processing frame in M3MemoryProcessor: {e}")
            await self.push_frame(frame, direction)
    
    async def _initialize_processor(self):
        """Initialize the M3 processor for a new session"""
        try:
            # Initialize M3 integration
            success = await self.m3_integration.initialize()
            if not success:
                logger.warning("M3 integration initialization failed")
                return
            
            # Set session ID based on timestamp
            self._current_session_id = f"session_{int(datetime.now().timestamp())}"
            
            # Create initial clip
            clip_id = await self.m3_integration.create_new_clip(self._current_session_id)
            if clip_id:
                self._last_clip_time = datetime.now()
                logger.info(f"Created initial clip {clip_id} for session {self._current_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize M3 processor: {e}")
    
    async def _process_transcription_frame(self, frame: TranscriptionFrame):
        """Process transcription frame into voice M3 node"""
        try:
            if not frame.text or not frame.text.strip():
                return
            
            # Check if we need a new clip based on time
            await self._check_clip_rotation()
            
            # Extract speaker information if available
            speaker_id = getattr(frame, 'speaker_id', self._current_speaker_id)
            confidence = getattr(frame, 'confidence', 0.8)
            
            # Generate embeddings for the transcription
            embeddings = []
            if self.embedding_service:
                try:
                    embedding = await self.embedding_service.get_embedding(frame.text)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for transcription: {e}")
            
            # Create voice node
            node_id = await self.m3_integration.store_m3_node(
                node_type="voice",
                contents=[frame.text],
                embeddings=embeddings,
                speaker_id=speaker_id,
                extraction_method="transcription",
                confidence=confidence
            )
            
            if node_id and self.auto_create_edges:
                # Automatically infer edges to similar nodes
                await self.m3_integration.infer_edges_for_node(
                    node_id, 
                    similarity_threshold=self.similarity_threshold
                )
            
            # Store in transcription buffer for potential episodic node creation
            self._transcription_buffer.append(frame.text)
            
            logger.info(f"Created voice node {node_id} for transcription: {frame.text[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to process transcription frame: {e}")
    
    async def _process_assistant_response(self):
        """Process accumulated assistant response into semantic nodes"""
        try:
            if not self._assistant_response_buffer:
                return
            
            full_response = " ".join(self._assistant_response_buffer)
            if not full_response.strip():
                return
            
            await self._check_clip_rotation()
            
            # Generate embeddings for the response
            embeddings = []
            if self.embedding_service:
                try:
                    embedding = await self.embedding_service.get_embedding(full_response)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for assistant response: {e}")
            
            # Enhanced semantic extraction using LLM if available
            facts = []
            semantic_memory = None
            
            if self.enable_llm_generation and self.llm_generator:
                try:
                    semantic_memory = await self.llm_generator.generate_semantic_memory(full_response)
                    if semantic_memory:
                        facts.extend(semantic_memory.facts)
                        logger.info(f"LLM extracted {len(semantic_memory.facts)} facts from assistant response")
                except Exception as e:
                    logger.warning(f"Failed to extract semantic memory via LLM: {e}")
            
            # Fallback to traditional fact extractor
            if not facts and self.fact_extractor:
                try:
                    extracted_facts = await self.fact_extractor.extract_facts(
                        full_response, self._current_speaker_id
                    )
                    facts = [fact['fact'] for fact in extracted_facts if fact.get('fact')]
                except Exception as e:
                    logger.warning(f"Failed to extract facts from response: {e}")
            
            # Create semantic node with response content and enhanced metadata
            contents = [full_response]
            if facts:
                contents.extend(facts)
            
            # Add concepts and relationships if available from LLM
            if semantic_memory:
                if semantic_memory.concepts:
                    contents.append(f"CONCEPTS: {', '.join(semantic_memory.concepts)}")
                if semantic_memory.relationships:
                    for rel in semantic_memory.relationships:
                        contents.append(f"RELATION: {rel.get('subject')} {rel.get('predicate')} {rel.get('object')}")
            
            node_id = await self.m3_integration.store_m3_node(
                node_type="semantic",
                contents=contents,
                embeddings=embeddings,
                speaker_id="assistant",
                extraction_method="assistant_response",
                confidence=0.9
            )
            
            if node_id and self.auto_create_edges:
                # Infer edges to related nodes
                await self.m3_integration.infer_edges_for_node(
                    node_id,
                    similarity_threshold=self.similarity_threshold
                )
            
            logger.info(f"Created semantic node {node_id} for assistant response")
            
            # Check if we should create episodic node from recent interactions
            await self._maybe_create_episodic_node()
            
        except Exception as e:
            logger.error(f"Failed to process assistant response: {e}")
    
    async def _process_user_text(self, text: str):
        """Process user text input (non-transcription) into semantic node"""
        try:
            if not text.strip():
                return
            
            await self._check_clip_rotation()
            
            # Generate embeddings
            embeddings = []
            if self.embedding_service:
                try:
                    embedding = await self.embedding_service.get_embedding(text)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for user text: {e}")
            
            # Extract facts
            facts = []
            if self.fact_extractor:
                try:
                    extracted_facts = await self.fact_extractor.extract_facts(
                        text, self._current_speaker_id
                    )
                    facts = [fact['fact'] for fact in extracted_facts if fact.get('fact')]
                except Exception as e:
                    logger.warning(f"Failed to extract facts from user text: {e}")
            
            # Create semantic node
            contents = [text]
            if facts:
                contents.extend(facts)
            
            node_id = await self.m3_integration.store_m3_node(
                node_type="semantic",
                contents=contents,
                embeddings=embeddings,
                speaker_id=self._current_speaker_id,
                extraction_method="user_input",
                confidence=0.8
            )
            
            if node_id and self.auto_create_edges:
                await self.m3_integration.infer_edges_for_node(
                    node_id,
                    similarity_threshold=self.similarity_threshold
                )
            
            logger.info(f"Created semantic node {node_id} for user text")
            
        except Exception as e:
            logger.error(f"Failed to process user text: {e}")
    
    async def _maybe_create_episodic_node(self):
        """Create episodic node from recent interaction if appropriate"""
        try:
            # Only create episodic nodes if we have sufficient interaction
            if len(self._transcription_buffer) < 2:
                return
            
            # Get recent transcriptions
            recent_transcriptions = self._transcription_buffer[-5:]  # Last 5 utterances
            combined_text = " ".join(recent_transcriptions)
            
            # Generate embedding for the episode
            embeddings = []
            if self.embedding_service and combined_text.strip():
                try:
                    embedding = await self.embedding_service.get_embedding(combined_text)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for episode: {e}")
            
            # Create episodic node
            node_id = await self.m3_integration.store_m3_node(
                node_type="episodic",
                contents=recent_transcriptions,
                embeddings=embeddings,
                speaker_id=self._current_speaker_id,
                extraction_method="episodic_aggregation",
                confidence=0.7
            )
            
            if node_id and self.auto_create_edges:
                await self.m3_integration.infer_edges_for_node(
                    node_id,
                    similarity_threshold=self.similarity_threshold * 0.8  # Lower threshold for episodes
                )
            
            logger.info(f"Created episodic node {node_id} from recent interactions")
            
            # Clear part of the buffer to avoid over-creation
            self._transcription_buffer = self._transcription_buffer[-2:]
            
        except Exception as e:
            logger.error(f"Failed to create episodic node: {e}")
    
    async def _buffer_audio_frame(self, frame: AudioRawFrame):
        """Buffer audio frames for potential audio processing"""
        try:
            # Keep audio buffer limited
            self._audio_buffer.append(frame)
            
            # Limit buffer size
            if len(self._audio_buffer) > 100:  # Arbitrary limit
                self._audio_buffer = self._audio_buffer[-50:]
            
            # TODO: Could process audio directly into voice nodes with audio embeddings
            
        except Exception as e:
            logger.error(f"Failed to buffer audio frame: {e}")
    
    async def _process_data_frame(self, frame):
        """Process data frames that might contain memory-relevant information"""
        try:
            # Look for speaker identification or other metadata
            if hasattr(frame, 'data') and isinstance(frame.data, dict):
                data = frame.data
                
                # Update speaker ID if provided
                if 'speaker_id' in data:
                    self._current_speaker_id = str(data['speaker_id'])
                    logger.info(f"Updated speaker ID to {self._current_speaker_id}")
                
                # Handle memory-related metadata
                if 'memory_context' in data:
                    await self._process_memory_context(data['memory_context'])
            
        except Exception as e:
            logger.error(f"Failed to process data frame: {e}")
    
    async def _process_memory_context(self, context: Dict[str, Any]):
        """Process memory context from data frames"""
        try:
            # Handle explicit memory instructions
            if context.get('action') == 'store_fact':
                fact = context.get('fact')
                if fact:
                    await self._store_explicit_fact(fact)
            
            elif context.get('action') == 'update_speaker':
                speaker_id = context.get('speaker_id')
                if speaker_id:
                    self._current_speaker_id = str(speaker_id)
            
            elif context.get('action') == 'new_clip':
                session_id = context.get('session_id', self._current_session_id)
                await self.m3_integration.create_new_clip(session_id)
                self._last_clip_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to process memory context: {e}")
    
    async def _store_explicit_fact(self, fact: str):
        """Store an explicitly provided fact as a semantic node"""
        try:
            if not fact.strip():
                return
            
            # Generate embedding
            embeddings = []
            if self.embedding_service:
                try:
                    embedding = await self.embedding_service.get_embedding(fact)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for fact: {e}")
            
            # Create semantic node for the fact
            node_id = await self.m3_integration.store_m3_node(
                node_type="semantic",
                contents=[fact],
                embeddings=embeddings,
                speaker_id=self._current_speaker_id,
                extraction_method="explicit_fact",
                confidence=0.95
            )
            
            if node_id and self.auto_create_edges:
                await self.m3_integration.infer_edges_for_node(
                    node_id,
                    similarity_threshold=self.similarity_threshold
                )
            
            logger.info(f"Stored explicit fact as node {node_id}: {fact}")
            
        except Exception as e:
            logger.error(f"Failed to store explicit fact: {e}")
    
    async def _check_clip_rotation(self):
        """Check if we need to rotate to a new clip based on time"""
        try:
            if not self._last_clip_time:
                return
            
            time_since_clip = datetime.now() - self._last_clip_time
            if time_since_clip.total_seconds() > self.clip_duration_seconds:
                # Create new clip
                clip_id = await self.m3_integration.close_current_clip(
                    create_new=True,
                    session_id=self._current_session_id
                )
                
                if clip_id:
                    self._last_clip_time = datetime.now()
                    logger.info(f"Rotated to new clip {clip_id}")
            
        except Exception as e:
            logger.error(f"Failed to check clip rotation: {e}")
    
    async def _finalize_session(self):
        """Finalize the current session and clean up"""
        try:
            # Close current clip
            if self.m3_integration.current_clip_id:
                await self.m3_integration.close_current_clip(create_new=False)
            
            # Create final episodic node if we have interactions
            if self._transcription_buffer:
                await self._maybe_create_episodic_node()
            
            # Clear buffers
            self._transcription_buffer = []
            self._audio_buffer = []
            self._assistant_response_buffer = []
            self._conversation_window = []
            self._speaker_utterances = {}
            
            logger.info(f"Finalized M3 memory session {self._current_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to finalize session: {e}")
    
    async def get_memory_context(self, 
                                query: str, 
                                max_nodes: int = 10) -> Dict[str, Any]:
        """Get memory context for a query (for LLM integration)
        
        Args:
            query: Query to find relevant memory for
            max_nodes: Maximum nodes to return
            
        Returns:
            Memory context with relevant nodes and relationships
        """
        try:
            if not self.embedding_service:
                return {"nodes": [], "context": ""}
            
            # Generate embedding for query
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Search for similar nodes
            similar_nodes = await self.m3_integration.search_similar_nodes(
                query_embedding,
                limit=max_nodes,
                min_similarity=0.3  # Lower threshold for context retrieval
            )
            
            # Format context for LLM
            context_parts = []
            for node in similar_nodes:
                node_type = node.get('node_type', 'unknown')
                contents = node.get('contents', [])
                similarity = node.get('similarity', 0)
                
                for content in contents:
                    context_parts.append(f"[{node_type.upper()}] {content} (similarity: {similarity:.2f})")
            
            context = "\n".join(context_parts)
            
            return {
                "nodes": similar_nodes,
                "context": context,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return {"nodes": [], "context": "", "query": query}
    
    async def _track_conversation_turn(self, turn_data: Dict[str, Any]):
        """Track conversation turns for episodic memory generation"""
        try:
            # Add to conversation window
            self._conversation_window.append(turn_data)
            
            # Keep window size manageable (last 10 turns)
            if len(self._conversation_window) > 10:
                self._conversation_window = self._conversation_window[-10:]
            
            # Track per-speaker utterances
            speaker_id = turn_data.get('speaker_id', 'unknown')
            if speaker_id not in self._speaker_utterances:
                self._speaker_utterances[speaker_id] = []
            
            self._speaker_utterances[speaker_id].append(turn_data['text'])
            
            # Keep per-speaker history reasonable
            if len(self._speaker_utterances[speaker_id]) > 20:
                self._speaker_utterances[speaker_id] = self._speaker_utterances[speaker_id][-20:]
            
            logger.debug(f"Tracked conversation turn for {speaker_id}: {turn_data['text'][:30]}...")
            
        except Exception as e:
            logger.error(f"Failed to track conversation turn: {e}")
    
    async def _extract_speaker_facts_llm(self, text: str, speaker_id: str):
        """Extract speaker-specific facts using LLM"""
        try:
            if not self.llm_generator:
                return
            
            facts = await self.llm_generator.extract_speaker_facts(text, speaker_id)
            
            if facts:
                # Store each fact as a semantic node
                for fact_data in facts:
                    # Generate embedding for the fact
                    fact_text = f"{fact_data['subject']} {fact_data['predicate']} {fact_data['object']}"
                    
                    embeddings = []
                    if self.embedding_service:
                        try:
                            embedding = await self.embedding_service.get_embedding(fact_text)
                            embeddings = [embedding]
                        except Exception as e:
                            logger.warning(f"Failed to generate embedding for fact: {e}")
                    
                    # Create semantic node for the fact
                    node_id = await self.m3_integration.store_m3_node(
                        node_type="semantic",
                        contents=[fact_text, f"FACT: {fact_data['subject']} -> {fact_data['object']}"],
                        embeddings=embeddings,
                        speaker_id=speaker_id,
                        extraction_method="llm_fact_extraction",
                        confidence=fact_data.get('confidence', 0.8)
                    )
                    
                    if node_id and self.auto_create_edges:
                        await self.m3_integration.infer_edges_for_node(
                            node_id,
                            similarity_threshold=self.similarity_threshold * 0.9  # Higher threshold for facts
                        )
                
                logger.info(f"Extracted and stored {len(facts)} speaker facts via LLM")
        
        except Exception as e:
            logger.error(f"Failed to extract speaker facts via LLM: {e}")
    
    async def _maybe_create_episodic_node(self):
        """Create episodic node from recent interaction using LLM if available"""
        try:
            # Only create episodic nodes if we have sufficient interaction
            if len(self._conversation_window) < 3:
                return
            
            # Check time since last episodic generation
            if self._last_episodic_generation:
                time_since_last = datetime.now() - self._last_episodic_generation
                if time_since_last.total_seconds() < 60:  # Don't generate too frequently
                    return
            
            if self.enable_llm_generation and self.llm_generator:
                # Use LLM for enhanced episodic memory generation
                await self._create_episodic_memory_llm()
            else:
                # Fall back to simple episodic aggregation
                await self._create_simple_episodic_memory()
                
        except Exception as e:
            logger.error(f"Failed to create episodic node: {e}")
    
    async def _create_episodic_memory_llm(self):
        """Create episodic memory using LLM-powered generation"""
        try:
            # Get recent conversation sequence
            conversation_texts = [turn['text'] for turn in self._conversation_window[-5:]]
            speaker_ids = [turn['speaker_id'] for turn in self._conversation_window[-5:]]
            
            if len(conversation_texts) < 2:
                return
            
            # Generate episodic memory using LLM
            episodic_memory = await self.llm_generator.generate_episodic_memory(
                conversation_texts, speaker_ids
            )
            
            if not episodic_memory:
                logger.warning("LLM failed to generate episodic memory")
                return
            
            # Generate embedding for the episodic summary
            embeddings = []
            if self.embedding_service:
                try:
                    embedding = await self.embedding_service.get_embedding(episodic_memory.summary)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for episodic memory: {e}")
            
            # Create episodic node with rich content
            contents = [
                episodic_memory.summary,
                f"SEQUENCE: {' | '.join(conversation_texts[:3])}",
            ]
            
            if episodic_memory.key_events:
                contents.append(f"KEY_EVENTS: {'; '.join(episodic_memory.key_events)}")
            
            if episodic_memory.participants:
                contents.append(f"PARTICIPANTS: {', '.join(episodic_memory.participants)}")
            
            if episodic_memory.temporal_markers:
                contents.append(f"TEMPORAL: {', '.join(episodic_memory.temporal_markers)}")
            
            node_id = await self.m3_integration.store_m3_node(
                node_type="episodic",
                contents=contents,
                embeddings=embeddings,
                speaker_id="conversation",
                extraction_method="llm_episodic_generation",
                confidence=episodic_memory.confidence
            )
            
            if node_id and self.auto_create_edges:
                await self.m3_integration.infer_edges_for_node(
                    node_id,
                    similarity_threshold=self.similarity_threshold * 0.8  # Lower threshold for episodes
                )
            
            self._last_episodic_generation = datetime.now()
            
            logger.info(f"Created LLM-enhanced episodic node {node_id}: {episodic_memory.summary[:50]}...")
            
            # Clear part of the conversation window to avoid over-creation
            self._conversation_window = self._conversation_window[-3:]
            
        except Exception as e:
            logger.error(f"Failed to create LLM episodic memory: {e}")
    
    async def _create_simple_episodic_memory(self):
        """Create simple episodic memory without LLM (fallback)"""
        try:
            # Get recent transcriptions
            recent_transcriptions = self._transcription_buffer[-5:]  # Last 5 utterances
            combined_text = " ".join(recent_transcriptions)
            
            if len(recent_transcriptions) < 2:
                return
            
            # Generate embedding for the episode
            embeddings = []
            if self.embedding_service and combined_text.strip():
                try:
                    embedding = await self.embedding_service.get_embedding(combined_text)
                    embeddings = [embedding]
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for episode: {e}")
            
            # Create episodic node
            node_id = await self.m3_integration.store_m3_node(
                node_type="episodic",
                contents=recent_transcriptions,
                embeddings=embeddings,
                speaker_id=self._current_speaker_id,
                extraction_method="episodic_aggregation",
                confidence=0.7
            )
            
            if node_id and self.auto_create_edges:
                await self.m3_integration.infer_edges_for_node(
                    node_id,
                    similarity_threshold=self.similarity_threshold * 0.8  # Lower threshold for episodes
                )
            
            logger.info(f"Created simple episodic node {node_id} from recent interactions")
            
            # Clear part of the buffer to avoid over-creation
            self._transcription_buffer = self._transcription_buffer[-2:]
            
        except Exception as e:
            logger.error(f"Failed to create simple episodic memory: {e}")
    
    async def get_enhanced_memory_context(self, 
                                        query: str, 
                                        max_nodes: int = 10,
                                        include_episodic: bool = True,
                                        include_semantic: bool = True,
                                        include_voice: bool = False) -> Dict[str, Any]:
        """Get enhanced memory context with LLM summarization
        
        Args:
            query: Query to find relevant memory for
            max_nodes: Maximum nodes to return
            include_episodic: Include episodic memories
            include_semantic: Include semantic memories 
            include_voice: Include voice transcriptions
            
        Returns:
            Enhanced memory context with LLM-generated summary
        """
        try:
            if not self.embedding_service:
                return {"nodes": [], "context": "", "summary": ""}
            
            # Generate embedding for query
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Search for similar nodes with type filtering
            all_memories = []
            
            if include_episodic:
                episodic_nodes = await self.m3_integration.search_similar_nodes(
                    query_embedding,
                    node_type="episodic",
                    limit=max_nodes // 3,
                    min_similarity=0.3
                )
                all_memories.extend(episodic_nodes)
            
            if include_semantic:
                semantic_nodes = await self.m3_integration.search_similar_nodes(
                    query_embedding,
                    node_type="semantic",
                    limit=max_nodes // 2,
                    min_similarity=0.3
                )
                all_memories.extend(semantic_nodes)
            
            if include_voice:
                voice_nodes = await self.m3_integration.search_similar_nodes(
                    query_embedding,
                    node_type="voice",
                    limit=max_nodes // 3,
                    min_similarity=0.4
                )
                all_memories.extend(voice_nodes)
            
            # Sort by similarity and limit
            all_memories.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            top_memories = all_memories[:max_nodes]
            
            # Format basic context
            context_parts = []
            for node in top_memories:
                node_type = node.get('node_type', 'unknown')
                contents = node.get('contents', [])
                similarity = node.get('similarity', 0)
                
                for content in contents:
                    context_parts.append(f"[{node_type.upper()}] {content} (similarity: {similarity:.2f})")
            
            basic_context = "\n".join(context_parts)
            
            # Generate LLM summary if available
            summary = ""
            if self.enable_llm_generation and self.llm_generator and top_memories:
                try:
                    summary = await self.llm_generator.generate_context_summary(top_memories, query)
                except Exception as e:
                    logger.warning(f"Failed to generate context summary: {e}")
            
            return {
                "nodes": top_memories,
                "context": basic_context,
                "summary": summary,
                "query": query,
                "node_types": {
                    "episodic": len([n for n in top_memories if n.get('node_type') == 'episodic']),
                    "semantic": len([n for n in top_memories if n.get('node_type') == 'semantic']),
                    "voice": len([n for n in top_memories if n.get('node_type') == 'voice'])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get enhanced memory context: {e}")
            return {"nodes": [], "context": "", "summary": "", "query": query, "node_types": {}}