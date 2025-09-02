"""
Voice Processing Pipeline - M3-inspired voice node management

Adapts M3's face/voice processing for audio-first voice agents:
- Voice embedding extraction and storage
- Speaker identity resolution with equivalence detection
- Progressive speaker annotation and enrollment
- Voice node clustering and deduplication

Based on M3-Agent's voice_processing.py with audio-first optimizations.
"""

import os
import io
import json
import logging
import time
import base64
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Import voice recognition components
try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    VoiceEncoder = None
    logger.warning("Resemblyzer not available - voice node processing will be limited")

# Audio processing
try:
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    AudioSegment = None
    logger.warning("Pydub not available - audio processing will be limited")


class VoiceProcessor:
    """
    M3-inspired voice processor for AudioGraph integration.
    
    Handles:
    - Voice embedding extraction using Resemblyzer
    - Speaker identity resolution
    - Voice node creation and updates
    - Progressive speaker annotation
    """
    
    def __init__(self, audio_graph, config: Optional[Dict] = None):
        """
        Initialize voice processor.
        
        Args:
            audio_graph: AudioGraph instance for node management
            config: Optional configuration dictionary
        """
        self.audio_graph = audio_graph
        self.config = config or {}
        
        # M3's processing parameters
        self.voice_sample_rate = 16000  # Standard for Resemblyzer
        self.min_voice_duration = 1.0   # Minimum seconds for voice processing
        self.confidence_threshold = 0.7  # M3's voice matching threshold
        
        # Initialize voice encoder if available
        self.voice_encoder = None
        if RESEMBLYZER_AVAILABLE:
            try:
                # Use CPU to avoid GPU overhead on macOS
                self.voice_encoder = VoiceEncoder("cpu")
                logger.info("🎤 Resemblyzer VoiceEncoder initialized for voice processing")
            except Exception as e:
                logger.error(f"Failed to initialize VoiceEncoder: {e}")
                self.voice_encoder = None
        
        # Voice processing statistics
        self.stats = {
            'voices_processed': 0,
            'speakers_enrolled': 0,
            'equivalences_detected': 0,
            'nodes_created': 0,
            'nodes_updated': 0
        }
        
        logger.info(f"🔧 VoiceProcessor initialized with AudioGraph")
    
    def process_voices(self, audio_data: bytes, transcript: str, 
                      session_id: int, speaker_id: Optional[str] = None) -> Optional[int]:
        """
        M3's voice processing adapted for continuous audio streams.
        
        Args:
            audio_data: Raw audio bytes
            transcript: Corresponding transcript text
            session_id: Session/clip identifier (M3's clip_id)
            speaker_id: Optional known speaker identifier
            
        Returns:
            Voice node ID if processing successful, None otherwise
        """
        if not self.voice_encoder:
            logger.warning("VoiceEncoder not available - skipping voice processing")
            return None
        
        try:
            # M3's audio preprocessing
            voice_embeddings = self._extract_voice_embeddings(audio_data)
            if not voice_embeddings:
                logger.debug("No voice embeddings extracted from audio")
                return None
            
            # M3's voice node search for existing speakers
            voice_info = {
                'embeddings': voice_embeddings,
                'transcript': transcript,
                'session_id': session_id,
                'speaker_id': speaker_id
            }
            
            existing_voices = self.audio_graph.search_voice_nodes(voice_info)
            
            if existing_voices and existing_voices[0][1] > self.confidence_threshold:
                # M3's node update pattern for existing speaker
                node_id = existing_voices[0][0]
                similarity_score = existing_voices[0][1]
                
                # Update existing voice node
                update_success = self._update_voice_node(
                    node_id, voice_embeddings, transcript, session_id, similarity_score
                )
                
                if update_success:
                    self.stats['nodes_updated'] += 1
                    logger.debug(f"🔄 Updated voice node {node_id} (similarity: {similarity_score:.3f})")
                    return node_id
            
            else:
                # M3's new voice node creation
                node_id = self._create_voice_node(
                    voice_embeddings, transcript, session_id, speaker_id
                )
                
                if node_id is not None:
                    self.stats['nodes_created'] += 1
                    self.stats['speakers_enrolled'] += 1
                    logger.debug(f"🆕 Created new voice node {node_id}")
                    
                    # M3's progressive annotation - check for speaker equivalences
                    self._detect_speaker_equivalences(node_id)
                    
                    return node_id
            
            self.stats['voices_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error in voice processing: {e}")
            return None
        
        return None
    
    def _extract_voice_embeddings(self, audio_data: bytes) -> List[List[float]]:
        """
        Extract voice embeddings using Resemblyzer (M3's pattern).
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            List of voice embedding vectors
        """
        if not self.voice_encoder or not audio_data:
            return []
        
        try:
            # Convert audio bytes to format suitable for Resemblyzer
            if AUDIO_PROCESSING_AVAILABLE:
                # Use pydub for audio conversion
                audio_io = io.BytesIO(audio_data)
                audio_segment = AudioSegment.from_raw(
                    audio_io, 
                    sample_width=2,  # 16-bit
                    frame_rate=self.voice_sample_rate,
                    channels=1
                )
                
                # Convert to numpy array for Resemblyzer
                audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                # Normalize to [-1, 1] range
                audio_array = audio_array / 32768.0
                
            else:
                # Basic conversion without pydub
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Check minimum duration (M3's pattern)
            duration = len(audio_array) / self.voice_sample_rate
            if duration < self.min_voice_duration:
                logger.debug(f"Audio too short for processing: {duration:.2f}s < {self.min_voice_duration}s")
                return []
            
            # M3's embedding extraction
            try:
                embedding = self.voice_encoder.embed_utterance(audio_array)
                if embedding is not None and len(embedding) > 0:
                    return [embedding.tolist()]
                    
            except Exception as e:
                logger.error(f"Resemblyzer embedding extraction failed: {e}")
                return []
        
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return []
        
        return []
    
    def _create_voice_node(self, voice_embeddings: List[List[float]], 
                          transcript: str, session_id: int, 
                          speaker_id: Optional[str]) -> Optional[int]:
        """
        Create new voice node using M3's pattern.
        
        Args:
            voice_embeddings: Voice embedding vectors
            transcript: Corresponding transcript
            session_id: Session/clip identifier
            speaker_id: Optional speaker identifier
            
        Returns:
            New voice node ID if successful, None otherwise
        """
        try:
            # M3's voice data structure
            voice_data = {
                'embeddings': voice_embeddings,
                'contents': [transcript] if transcript else [],
                'speaker_id': speaker_id,
                'confidence': 1.0,  # New node starts with full confidence
                'session_id': session_id,
                'first_seen': time.time()
            }
            
            # Create voice node in AudioGraph
            node_id = self.audio_graph.add_voice_node(voice_data)
            
            # Update node metadata with session info
            if node_id is not None:
                node = self.audio_graph.nodes[node_id]
                node.metadata['timestamp'] = session_id
                node.metadata['session_id'] = session_id
                
                logger.debug(f"🎤 Voice node {node_id} created with {len(voice_embeddings)} embeddings")
                
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to create voice node: {e}")
            return None
    
    def _update_voice_node(self, node_id: int, voice_embeddings: List[List[float]], 
                          transcript: str, session_id: int, similarity_score: float) -> bool:
        """
        Update existing voice node using M3's pattern.
        
        Args:
            node_id: Existing voice node ID
            voice_embeddings: New voice embeddings
            transcript: New transcript
            session_id: Session identifier
            similarity_score: Similarity confidence
            
        Returns:
            Success indicator
        """
        try:
            # M3's update information structure
            update_info = {
                'embeddings': voice_embeddings,
                'contents': [transcript] if transcript else []
            }
            
            # Update node using AudioGraph method
            success = self.audio_graph.update_node(node_id, update_info)
            
            if success:
                # M3's confidence reinforcement based on similarity
                if similarity_score > 0.8:  # High confidence match
                    self.audio_graph.reinforce_node(node_id, delta_weight=0.5)
                    logger.debug(f"💪 Reinforced voice node {node_id} (high similarity)")
                
                # Update session tracking
                node = self.audio_graph.nodes[node_id]
                if 'sessions_seen' not in node.metadata:
                    node.metadata['sessions_seen'] = set()
                node.metadata['sessions_seen'].add(session_id)
                
                return True
            
        except Exception as e:
            logger.error(f"Failed to update voice node {node_id}: {e}")
            
        return False
    
    def _detect_speaker_equivalences(self, voice_node_id: int):
        """
        M3's progressive annotation for speaker identity resolution.
        
        Args:
            voice_node_id: New voice node to check for equivalences
        """
        try:
            # Look for other voice nodes with similar characteristics
            voice_node = self.audio_graph.nodes[voice_node_id]
            
            # Search for similar voice nodes
            similar_voices = self.audio_graph.search_voice_nodes({
                'embeddings': voice_node.embeddings[:1]  # Use first embedding
            })
            
            # M3's equivalence detection logic
            potential_equivalences = []
            
            for similar_node_id, similarity in similar_voices:
                if (similar_node_id != voice_node_id and 
                    similarity > 0.75):  # High similarity threshold
                    
                    # Check temporal proximity (M3's pattern)
                    similar_node = self.audio_graph.nodes[similar_node_id]
                    time_diff = abs(
                        voice_node.metadata.get('timestamp', 0) - 
                        similar_node.metadata.get('timestamp', 0)
                    )
                    
                    # M3's temporal equivalence window
                    if time_diff < 10:  # Within 10 sessions/clips
                        potential_equivalences.append((similar_node_id, similarity))
            
            # Create equivalence relationships
            if potential_equivalences:
                self._create_equivalence_relationships(voice_node_id, potential_equivalences)
                self.stats['equivalences_detected'] += len(potential_equivalences)
                
        except Exception as e:
            logger.error(f"Error in speaker equivalence detection: {e}")
    
    def _create_equivalence_relationships(self, voice_node_id: int, 
                                        equivalences: List[Tuple[int, float]]):
        """
        Create semantic equivalence nodes (M3's pattern).
        
        Args:
            voice_node_id: Primary voice node ID
            equivalences: List of (node_id, similarity) tuples
        """
        try:
            for equiv_node_id, similarity in equivalences:
                # M3's equivalence content format
                equiv_content = f"Equivalence: <voice_{voice_node_id}>, <voice_{equiv_node_id}> (similarity: {similarity:.3f})"
                
                # Create semantic node for equivalence
                equivalence_data = {
                    'embeddings': [],  # Equivalence nodes don't need embeddings
                    'contents': [equiv_content]
                }
                
                # Use current session as clip_id
                current_session = self.audio_graph.nodes[voice_node_id].metadata.get('timestamp', 0)
                equiv_node_id = self.audio_graph.add_text_node(
                    equivalence_data, 
                    current_session, 
                    'semantic'
                )
                
                # M3's edge creation for equivalence
                if equiv_node_id is not None:
                    self.audio_graph.add_edge(voice_node_id, equiv_node_id, weight=similarity)
                    
                    # Connect to equivalent voice node too
                    for equiv_voice_id, _ in equivalences:
                        self.audio_graph.add_edge(equiv_voice_id, equiv_node_id, weight=similarity)
                    
                    logger.debug(f"🔗 Created equivalence relationship: voice_{voice_node_id} <-> voice_{equiv_node_id}")
        
        except Exception as e:
            logger.error(f"Error creating equivalence relationships: {e}")
    
    def get_speaker_identity(self, voice_node_id: int) -> Optional[str]:
        """
        Get canonical speaker identity for a voice node.
        
        Args:
            voice_node_id: Voice node ID
            
        Returns:
            Canonical speaker identifier or None
        """
        if voice_node_id not in self.audio_graph.nodes:
            return None
        
        voice_node = self.audio_graph.nodes[voice_node_id]
        
        # Check for explicit speaker ID
        if voice_node.metadata.get('speaker_id'):
            return voice_node.metadata['speaker_id']
        
        # Use equivalence resolution
        entity = f"voice_{voice_node_id}"
        canonical = self.audio_graph.get_canonical_identity(entity)
        
        if canonical != entity:
            return canonical
        
        # Default to node-based identity
        return f"Speaker_{voice_node_id}"
    
    def enroll_speaker(self, speaker_name: str, voice_embeddings: List[List[float]], 
                      session_id: int) -> Optional[int]:
        """
        Explicitly enroll a named speaker (M3's pattern).
        
        Args:
            speaker_name: Human-readable speaker name
            voice_embeddings: Speaker's voice embeddings
            session_id: Session identifier
            
        Returns:
            Voice node ID for enrolled speaker
        """
        try:
            # Create voice node with explicit speaker identity
            voice_data = {
                'embeddings': voice_embeddings,
                'contents': [f"Speaker enrolled as: {speaker_name}"],
                'speaker_id': speaker_name,
                'confidence': 1.0
            }
            
            node_id = self.audio_graph.add_voice_node(voice_data)
            
            if node_id is not None:
                # Update metadata
                node = self.audio_graph.nodes[node_id]
                node.metadata['timestamp'] = session_id
                node.metadata['enrolled_name'] = speaker_name
                node.metadata['enrollment_time'] = time.time()
                
                self.stats['speakers_enrolled'] += 1
                logger.info(f"🎯 Speaker '{speaker_name}' enrolled as voice node {node_id}")
                
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to enroll speaker '{speaker_name}': {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        return {
            **self.stats,
            'encoder_available': self.voice_encoder is not None,
            'audio_processing_available': AUDIO_PROCESSING_AVAILABLE,
            'graph_voice_nodes': len(self.audio_graph.voice_nodes),
            'total_nodes': len(self.audio_graph.nodes),
            'equivalence_sets': len(self.audio_graph.equivalences)
        }


def process_voices(audio_graph, audio_data: bytes, transcript: str, 
                  session_id: int, speaker_id: Optional[str] = None,
                  save_path: Optional[str] = None, 
                  preprocessing: List[str] = []) -> Optional[int]:
    """
    M3-style voice processing function for integration.
    
    Args:
        audio_graph: AudioGraph instance
        audio_data: Raw audio bytes
        transcript: Corresponding transcript
        session_id: Session/clip identifier (M3's clip_id)
        speaker_id: Optional speaker identifier
        save_path: Optional path to save processing results
        preprocessing: Processing flags (for M3 compatibility)
        
    Returns:
        Voice node ID if successful, None otherwise
    """
    try:
        # Initialize voice processor
        processor = VoiceProcessor(audio_graph)
        
        # Process voice using M3 patterns
        voice_node_id = processor.process_voices(
            audio_data, transcript, session_id, speaker_id
        )
        
        # M3's save pattern (optional)
        if save_path and voice_node_id is not None:
            result_data = {
                'voice_node_id': voice_node_id,
                'session_id': session_id,
                'speaker_id': speaker_id,
                'transcript': transcript,
                'processing_stats': processor.get_processing_stats()
            }
            
            try:
                with open(save_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                logger.debug(f"💾 Voice processing results saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save processing results: {e}")
        
        return voice_node_id
        
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        return None