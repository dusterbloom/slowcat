"""
M3SmartContextManager - M3-inspired context management with AudioGraph integration

Replaces the token budgeting approach with M3's entity-centric memory retrieval:
- AudioGraph-based memory storage and retrieval
- Voice node processing with speaker recognition
- Episodic and semantic memory generation
- M3's search patterns for context selection

Based on M3-Agent's control architecture with voice-first optimizations.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame, LLMMessagesUpdateFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# M3 AudioGraph components
from memory.audio_graph import AudioGraph
from memory.voice_processing import VoiceProcessor, process_voices
from memory.memory_processing import MemoryProcessor, process_memories, generate_memories

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
                 audio_graph: Optional[AudioGraph] = None,
                 max_context_tokens: int = 4096,  # Total context limit
                 memory_tokens: int = 2000,       # Tokens for memory context
                 **kwargs):
        super().__init__(**kwargs)
        
        self.context = context
        self.max_context_tokens = max_context_tokens
        self.memory_tokens = memory_tokens
        
        # M3's AudioGraph initialization
        if audio_graph is None:
            self.audio_graph = AudioGraph(
                max_voice_embeddings=20,    # M3 default
                max_text_embeddings=10,     # M3 default  
                voice_matching_threshold=0.6,  # M3's audio threshold
                text_matching_threshold=0.3    # M3's text threshold
            )
        else:
            self.audio_graph = audio_graph
        
        # M3 processors
        self.voice_processor = VoiceProcessor(self.audio_graph)
        self.memory_processor = MemoryProcessor(self.audio_graph)
        
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
        
        logger.info(f"🧠 M3SmartContextManager initialized with AudioGraph")
        logger.info(f"   Max context tokens: {max_context_tokens}")
        logger.info(f"   Memory context tokens: {memory_tokens}")
        logger.info(f"   AudioGraph stats: {self.audio_graph.get_stats()}")
    
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
            self.session_metadata.clip_id += 1
            logger.debug(f"🎬 Advanced to clip {self.session_metadata.clip_id}")
        
        # Forward frame
        await self.push_frame(frame, direction)
    
    async def _process_voice_data(self, audio_data: bytes, transcript: str, clip_id: int) -> Optional[int]:
        """Process voice data using M3's voice processing pipeline."""
        try:
            voice_node_id = self.voice_processor.process_voices(
                audio_data=audio_data,
                transcript=transcript,
                session_id=clip_id,
                speaker_id=self.session_metadata.speaker_id
            )
            
            if voice_node_id:
                # M3's speaker identity resolution
                speaker_identity = self.voice_processor.get_speaker_identity(voice_node_id)
                if speaker_identity != self.session_metadata.speaker_id:
                    self.session_metadata.speaker_id = speaker_identity
                    self.stats['speaker_updates'] += 1
                    logger.debug(f"🎯 Speaker identity updated: {speaker_identity}")
                
                # M3's equivalence resolution
                self.audio_graph.refresh_equivalences()
            
            return voice_node_id
            
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
        if not self.conversation_buffer:
            return
        
        try:
            # Prepare conversation context
            conversation_text = " ".join([
                entry['text'] for entry in self.conversation_buffer
            ])
            
            # M3's speaker voice mapping
            speaker_voices = {}
            if self.session_metadata.voice_node_id:
                speaker_voices[self.session_metadata.voice_node_id] = conversation_text
            
            # Generate memories using M3's patterns
            episodic_memories, semantic_memories = await generate_memories(
                conversation_context=conversation_text,
                speaker_voices=speaker_voices,
                session_id=self.session_metadata.clip_id,
                llm_service=None  # Use local processing
            )
            
            # Store memories in AudioGraph
            if episodic_memories:
                episodic_node_ids = process_memories(
                    self.audio_graph, episodic_memories, 
                    self.session_metadata.clip_id, 'episodic'
                )
                self.stats['memory_nodes_created'] += len(episodic_node_ids)
                logger.debug(f"📝 Created {len(episodic_node_ids)} episodic memory nodes")
            
            if semantic_memories:
                semantic_node_ids = process_memories(
                    self.audio_graph, semantic_memories,
                    self.session_metadata.clip_id, 'semantic'
                )
                self.stats['memory_nodes_created'] += len(semantic_node_ids)
                logger.debug(f"🧠 Created {len(semantic_node_ids)} semantic memory nodes")
            
            # M3's collision resolution and deduplication
            for node_id in semantic_node_ids:
                self.audio_graph.fix_collisions(node_id, mode='eq_only')
            
            # Clear processed conversation buffer
            self.conversation_buffer = []
            
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
    
    async def _build_m3_context(self, query_text: str, voice_node_id: Optional[int]) -> str:
        """Build context using M3's similarity-based retrieval."""
        try:
            context_parts = []
            
            # M3's multi-modal search
            if self.memory_processor.embedding_model:
                # Generate query embeddings
                query_embeddings = self.memory_processor._get_memory_embeddings([query_text])
                
                if query_embeddings:
                    # M3's text node search with different modes
                    relevant_memories = self.audio_graph.search_text_nodes(
                        query_embeddings=query_embeddings,
                        mode="max"  # M3's max pooling
                    )
                    
                    # M3's voice node search (if voice context available)
                    if voice_node_id and voice_node_id in self.audio_graph.nodes:
                        voice_embeddings = self.audio_graph.nodes[voice_node_id].embeddings
                        if voice_embeddings:
                            voice_memories = self.audio_graph.search_voice_nodes({
                                'embeddings': voice_embeddings[:1]  # Use first embedding
                            })
                            
                            # Add connected memories from voice nodes
                            for vid, _ in voice_memories[:3]:  # Top 3 voice matches
                                connected_texts = self.audio_graph.get_connected_nodes(
                                    vid, type=['episodic', 'semantic']
                                )
                                for text_node_id in connected_texts[:2]:  # Top 2 per voice
                                    relevant_memories.append((text_node_id, 0.8))  # High relevance
                    
                    # M3's context building from relevant memories
                    context_parts = self._format_memory_context(relevant_memories[:10])  # Top 10
                    self.stats['context_retrievals'] += 1
            
            # M3's speaker identity context
            if self.session_metadata.speaker_id and self.session_metadata.speaker_id != "unknown":
                context_parts.insert(0, f"Speaker: {self.session_metadata.speaker_id}")
            
            # Join context with M3's formatting
            full_context = "\n".join(context_parts) if context_parts else ""
            
            # M3's token limiting
            if self.token_counter and full_context:
                context_tokens = self.token_counter.count_tokens(full_context)
                if context_tokens > self.memory_tokens:
                    # Truncate to fit memory token budget
                    truncated_context = self._truncate_context(full_context, self.memory_tokens)
                    logger.debug(f"🔄 Context truncated: {context_tokens} -> {self.memory_tokens} tokens")
                    return truncated_context
            
            return full_context
            
        except Exception as e:
            logger.error(f"Context building failed: {e}")
            return ""
    
    def _format_memory_context(self, relevant_memories: List[tuple]) -> List[str]:
        """Format memories into context strings (M3's pattern)."""
        context_parts = []
        
        for node_id, similarity in relevant_memories:
            if node_id not in self.audio_graph.nodes:
                continue
                
            node = self.audio_graph.nodes[node_id]
            
            if node.metadata.get('contents'):
                content = node.metadata['contents'][0]
                
                # M3's content formatting with type and confidence
                memory_type = node.type.capitalize()
                confidence_indicator = "●" if similarity > 0.8 else "○"
                
                formatted = f"{confidence_indicator} {memory_type}: {content}"
                context_parts.append(formatted)
        
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
        """Get comprehensive processing statistics."""
        return {
            **self.stats,
            'session_metadata': {
                'session_id': self.session_metadata.session_id,
                'clip_id': self.session_metadata.clip_id,
                'turn_count': self.session_metadata.turn_count,
                'speaker_id': self.session_metadata.speaker_id,
                'voice_node_id': self.session_metadata.voice_node_id
            },
            'audio_graph_stats': self.audio_graph.get_stats(),
            'voice_processor_stats': self.voice_processor.get_processing_stats(),
            'memory_processor_stats': getattr(self.memory_processor, 'stats', {})
        }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get M3-style memory summary."""
        return {
            'total_nodes': len(self.audio_graph.nodes),
            'voice_nodes': len([n for n in self.audio_graph.nodes.values() if n.type == 'voice']),
            'episodic_nodes': len([n for n in self.audio_graph.nodes.values() if n.type == 'episodic']),
            'semantic_nodes': len([n for n in self.audio_graph.nodes.values() if n.type == 'semantic']),
            'total_edges': len(self.audio_graph.edges) // 2,  # Bidirectional
            'equivalence_sets': len(self.audio_graph.equivalences),
            'clips_processed': self.session_metadata.clip_id + 1,
            'speakers_identified': len(set(
                node.metadata.get('speaker_id') for node in self.audio_graph.nodes.values()
                if node.type == 'voice' and node.metadata.get('speaker_id')
            ))
        }