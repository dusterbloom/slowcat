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
    AudioFrame, TextFrame, TranscriptionFrame,
    LLMFullResponseStartFrame, LLMFullResponseEndFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from ..memory.m3_surreal_integration import M3SurrealIntegration
# Optional imports for enhanced functionality
try:
    from ..memory.spacy_fact_extractor import SpacyFactExtractor
except ImportError:
    SpacyFactExtractor = None

try:
    from ..services.embedding_service import EmbeddingService  
except ImportError:
    EmbeddingService = None

logger = logging.getLogger(__name__)

class M3MemoryProcessor(FrameProcessor):
    """Process frames into M3 memory nodes with graph relationships"""
    
    def __init__(self, 
                 m3_integration: M3SurrealIntegration,
                 embedding_service: Optional[EmbeddingService] = None,
                 fact_extractor: Optional[SpacyFactExtractor] = None,
                 similarity_threshold: float = 0.7,
                 auto_create_edges: bool = True,
                 clip_duration_seconds: int = 30,
                 **kwargs):
        """Initialize M3 Memory Processor
        
        Args:
            m3_integration: M3 SurrealDB integration instance
            embedding_service: Service for generating embeddings
            fact_extractor: Service for extracting facts from text
            similarity_threshold: Minimum similarity for auto-edge creation
            auto_create_edges: Whether to automatically infer edges
            clip_duration_seconds: Duration before creating new clips
        """
        super().__init__(**kwargs)
        self.m3_integration = m3_integration
        self.embedding_service = embedding_service
        self.fact_extractor = fact_extractor
        self.similarity_threshold = similarity_threshold
        self.auto_create_edges = auto_create_edges
        self.clip_duration_seconds = clip_duration_seconds
        
        # State tracking
        self._current_session_id: str = "default"
        self._current_speaker_id: str = "unknown"
        self._is_assistant_response: bool = False
        self._assistant_response_buffer: List[str] = []
        self._last_clip_time: Optional[datetime] = None
        
        # Audio processing state
        self._audio_buffer: List[AudioFrame] = []
        self._transcription_buffer: List[str] = []
        
        logger.info("M3 Memory Processor initialized")
    
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
            
            elif isinstance(frame, AudioFrame):
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
            
            # Extract facts if fact extractor is available
            facts = []
            if self.fact_extractor:
                try:
                    extracted_facts = await self.fact_extractor.extract_facts(
                        full_response, self._current_speaker_id
                    )
                    facts = [fact['fact'] for fact in extracted_facts if fact.get('fact')]
                except Exception as e:
                    logger.warning(f"Failed to extract facts from response: {e}")
            
            # Create semantic node with response content
            contents = [full_response]
            if facts:
                contents.extend(facts)
            
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
    
    async def _buffer_audio_frame(self, frame: AudioFrame):
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