"""
M3 Voice Recognition Integration Module

Bridges LightweightVoiceRecognition with M3-AudioGraph system:
- Voice node creation and management
- Speaker identity resolution using AudioGraph equivalences  
- Migration utilities for existing profiles
- Progressive speaker annotation

This module provides the integration layer between existing voice recognition
and the new M3-inspired AudioGraph memory system.
"""

import logging
import pickle
import os
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# M3 components
try:
    import sys
    import os
    
    # Add server directory to Python path for imports
    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    
    from memory.audio_graph import AudioGraph
    from memory.voice_processing import VoiceProcessor
    from memory.m3_integration import M3AudioGraphFactory
    M3_AVAILABLE = True
except ImportError as e:
    logger.warning(f"M3 components not available: {e}")
    AudioGraph = None
    VoiceProcessor = None
    M3AudioGraphFactory = None
    M3_AVAILABLE = False


class M3VoiceIntegration:
    """
    Integration layer between LightweightVoiceRecognition and M3-AudioGraph.
    
    Provides:
    - Voice node creation from Resemblyzer embeddings
    - Speaker identity resolution using AudioGraph equivalences
    - Migration utilities for existing pickle profiles
    - Session-based voice node management
    """
    
    def __init__(self, audio_graph: AudioGraph = None, config: Optional[Dict] = None):
        """
        Initialize M3 voice integration.
        
        Args:
            audio_graph: Optional AudioGraph instance (will create if None)
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize AudioGraph
        if audio_graph:
            self.audio_graph = audio_graph
        else:
            self.audio_graph = M3AudioGraphFactory.create_audio_graph(config)
        
        # Initialize VoiceProcessor for embedding management
        self.voice_processor = M3AudioGraphFactory.create_voice_processor(
            self.audio_graph, config
        )
        
        # Session management
        self.current_session_id = int(time.time())
        self.voice_node_cache = {}  # speaker_name -> node_id mapping
        
        # Migration tracking
        self.migrated_profiles = set()
        
        logger.info("🔗 M3VoiceIntegration initialized with AudioGraph")
    
    def create_voice_node_from_embeddings(self, speaker_name: str, 
                                         embeddings: List[np.ndarray],
                                         transcript: str = "") -> Optional[int]:
        """
        Create voice node from Resemblyzer embeddings.
        
        Args:
            speaker_name: Speaker identifier
            embeddings: List of voice embeddings from Resemblyzer
            transcript: Optional transcript content
            
        Returns:
            Voice node ID if successful, None otherwise
        """
        try:
            # Convert numpy arrays to lists for AudioGraph
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Create voice node using VoiceProcessor
            node_id = self.voice_processor.enroll_speaker(
                speaker_name, 
                embedding_lists, 
                self.current_session_id
            )
            
            if node_id is not None:
                # Cache the mapping
                self.voice_node_cache[speaker_name] = node_id
                logger.debug(f"🎤 Voice node {node_id} created for speaker '{speaker_name}'")
                
                # Add transcript if provided
                if transcript:
                    self.audio_graph.update_node(node_id, {
                        'contents': [transcript]
                    })
            
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to create voice node for '{speaker_name}': {e}")
            return None
    
    def find_speaker_voice_node(self, embeddings: List[np.ndarray], 
                               confidence_threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        Find speaker using AudioGraph voice node matching.
        
        Args:
            embeddings: Voice embeddings to match
            confidence_threshold: Minimum confidence for match
            
        Returns:
            Tuple of (speaker_name, confidence) or (None, 0.0)
        """
        try:
            # Convert embeddings to lists
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Search for matching voice nodes
            voice_matches = self.audio_graph.search_voice_nodes({
                'embeddings': embedding_lists
            })
            
            logger.debug(f"Voice search found {len(voice_matches)} matches: {voice_matches}")
            
            if voice_matches:
                # Take the best match
                node_id, similarity = voice_matches[0]
                logger.debug(f"Best voice match: node {node_id}, similarity {similarity}")
                
                # Check if similarity meets threshold
                if similarity >= confidence_threshold:
                    # Get speaker identity from voice node
                    speaker_name = self.voice_processor.get_speaker_identity(node_id)
                    logger.debug(f"Retrieved speaker identity: {speaker_name}")
                    
                    if speaker_name:
                        # Check for canonical identity using equivalences
                        canonical_identity = self.audio_graph.get_canonical_identity(
                            f"speaker_{speaker_name}"
                        )
                        
                        # Extract speaker name from canonical identity
                        if canonical_identity.startswith("speaker_"):
                            canonical_name = canonical_identity[8:]  # Remove "speaker_" prefix
                            logger.debug(f"Canonical identity: {speaker_name} -> {canonical_name}")
                            return canonical_name, similarity
                        else:
                            logger.debug(f"Using original speaker name: {speaker_name}")
                            return speaker_name, similarity
                else:
                    logger.debug(f"Similarity {similarity} below threshold {confidence_threshold}")
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error in speaker voice node matching: {e}")
            return None, 0.0
    
    def update_voice_node(self, speaker_name: str, embeddings: List[np.ndarray],
                         transcript: str = "") -> bool:
        """
        Update existing voice node with new embeddings and content.
        
        Args:
            speaker_name: Speaker identifier
            embeddings: New voice embeddings
            transcript: Optional new transcript
            
        Returns:
            Success indicator
        """
        try:
            # Find existing voice node
            node_id = self.voice_node_cache.get(speaker_name)
            
            if node_id is None:
                # Try to find node by searching
                matches = self.find_speaker_voice_node(embeddings)
                if matches[0] == speaker_name:
                    # Find the actual node ID for this speaker
                    for cached_name, cached_id in self.voice_node_cache.items():
                        if cached_name == speaker_name:
                            node_id = cached_id
                            break
            
            if node_id is None:
                logger.warning(f"No voice node found for speaker '{speaker_name}'")
                return False
            
            # Convert embeddings to lists
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Update node using AudioGraph
            update_data = {'embeddings': embedding_lists}
            if transcript:
                update_data['contents'] = [transcript]
            
            success = self.audio_graph.update_node(node_id, update_data)
            
            if success:
                logger.debug(f"🔄 Updated voice node {node_id} for speaker '{speaker_name}'")
                
                # Reinforce node connections for frequent speakers
                self.audio_graph.reinforce_node(node_id, delta_weight=0.1)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update voice node for '{speaker_name}': {e}")
            return False
    
    def migrate_pickle_profile(self, speaker_name: str, profile_path: str) -> bool:
        """
        Migrate existing pickle-based speaker profile to AudioGraph.
        
        Args:
            speaker_name: Speaker identifier
            profile_path: Path to pickle profile file
            
        Returns:
            Success indicator
        """
        try:
            # Skip if already migrated
            if speaker_name in self.migrated_profiles:
                return True
            
            # Load pickle profile
            with open(profile_path, 'rb') as f:
                fingerprints = pickle.load(f)
            
            if not fingerprints:
                logger.warning(f"Empty profile for speaker '{speaker_name}'")
                return False
            
            # Create voice node from stored fingerprints
            node_id = self.create_voice_node_from_embeddings(
                speaker_name,
                fingerprints,
                f"Migrated profile for {speaker_name}"
            )
            
            if node_id is not None:
                # Mark as migrated
                self.migrated_profiles.add(speaker_name)
                
                # Set migration metadata
                node = self.audio_graph.nodes[node_id]
                node.metadata['migrated_from_pickle'] = True
                node.metadata['original_profile_path'] = profile_path
                node.metadata['migration_time'] = time.time()
                
                logger.info(f"✅ Migrated pickle profile for '{speaker_name}' to voice node {node_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to migrate profile for '{speaker_name}': {e}")
            return False
    
    def migrate_all_profiles(self, profile_dir: str, profile_extension: str = ".pkl") -> int:
        """
        Migrate all pickle profiles in directory to AudioGraph nodes.
        
        Args:
            profile_dir: Directory containing pickle profiles
            profile_extension: File extension for profiles
            
        Returns:
            Number of profiles successfully migrated
        """
        if not os.path.exists(profile_dir):
            logger.warning(f"Profile directory not found: {profile_dir}")
            return 0
        
        migrated_count = 0
        
        for filename in os.listdir(profile_dir):
            if filename.endswith(profile_extension):
                speaker_name = filename[:-len(profile_extension)]
                profile_path = os.path.join(profile_dir, filename)
                
                if self.migrate_pickle_profile(speaker_name, profile_path):
                    migrated_count += 1
        
        logger.info(f"📦 Migrated {migrated_count} speaker profiles to AudioGraph")
        return migrated_count
    
    def resolve_speaker_identity(self, speaker_name: str) -> str:
        """
        Resolve canonical speaker identity using AudioGraph equivalences.
        
        Args:
            speaker_name: Input speaker identifier
            
        Returns:
            Canonical speaker identifier
        """
        try:
            # Use AudioGraph equivalence resolution
            entity = f"speaker_{speaker_name}"
            canonical = self.audio_graph.get_canonical_identity(entity)
            
            # Extract speaker name from canonical identity
            if canonical.startswith("speaker_"):
                return canonical[8:]  # Remove "speaker_" prefix
            else:
                return speaker_name  # Fallback to original
                
        except Exception as e:
            logger.error(f"Error resolving identity for '{speaker_name}': {e}")
            return speaker_name
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """Get comprehensive speaker statistics from AudioGraph."""
        try:
            graph_stats = self.audio_graph.get_stats()
            processing_stats = self.voice_processor.get_processing_stats()
            
            return {
                'voice_nodes': graph_stats['voice_nodes'],
                'total_speakers': len(self.voice_node_cache),
                'migrated_profiles': len(self.migrated_profiles),
                'equivalence_sets': graph_stats['equivalence_sets'],
                'processing_stats': processing_stats,
                'current_session_id': self.current_session_id
            }
            
        except Exception as e:
            logger.error(f"Error getting speaker statistics: {e}")
            return {}
    
    def cleanup_weak_connections(self, threshold: float = 0.1) -> int:
        """
        Clean up weak voice node connections.
        
        Args:
            threshold: Minimum edge weight to keep
            
        Returns:
            Number of edges removed
        """
        try:
            removed = self.audio_graph.prune_weak_edges(threshold)
            logger.info(f"🧹 Cleaned up {removed} weak voice connections")
            return removed
            
        except Exception as e:
            logger.error(f"Error cleaning up connections: {e}")
            return 0
    
    def refresh_equivalences(self):
        """Refresh speaker identity equivalences in AudioGraph."""
        try:
            self.audio_graph.refresh_equivalences()
            logger.debug("🔄 Refreshed speaker identity equivalences")
            
        except Exception as e:
            logger.error(f"Error refreshing equivalences: {e}")
    
    def advance_session(self):
        """Advance to new session for temporal organization."""
        self.current_session_id = int(time.time())
        logger.debug(f"📅 Advanced to session {self.current_session_id}")


def create_m3_voice_integration(audio_graph: AudioGraph = None, 
                               config: Optional[Dict] = None) -> Optional[M3VoiceIntegration]:
    """
    Factory function for creating M3VoiceIntegration.
    
    Args:
        audio_graph: Optional AudioGraph instance
        config: Optional configuration
        
    Returns:
        M3VoiceIntegration instance or None if M3 not available
    """
    if not M3_AVAILABLE:
        logger.error("M3 components not available - cannot create voice integration")
        return None
    
    try:
        integration = M3VoiceIntegration(audio_graph, config)
        return integration
        
    except Exception as e:
        logger.error(f"Failed to create M3VoiceIntegration: {e}")
        return None