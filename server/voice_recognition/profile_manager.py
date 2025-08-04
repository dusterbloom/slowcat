"""Enhanced speaker profile management with quality control"""
import numpy as np
import logging
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SpeakerEmbedding:
    """Represents a single speaker embedding with metadata"""
    fingerprint: np.ndarray
    timestamp: datetime
    quality_score: float
    confidence: float
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'fingerprint': self.fingerprint.tolist(),
            'timestamp': self.timestamp.isoformat(),
            'quality_score': self.quality_score,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(
            fingerprint=np.array(data['fingerprint']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            quality_score=data['quality_score'],
            confidence=data['confidence']
        )


class ProfileManager:
    """Manages speaker profiles with quality control and pruning"""
    
    def __init__(self, config):
        self.config = config
        self.profiles: Dict[str, List[SpeakerEmbedding]] = {}
        
    def add_embedding(self, speaker_id: str, fingerprint: np.ndarray, 
                     quality_score: float, confidence: float):
        """Add a new embedding to a speaker profile"""
        embedding = SpeakerEmbedding(
            fingerprint=fingerprint,
            timestamp=datetime.now(),
            quality_score=quality_score,
            confidence=confidence
        )
        
        if speaker_id not in self.profiles:
            self.profiles[speaker_id] = []
            
        self.profiles[speaker_id].append(embedding)
        
        # Prune old or low quality embeddings
        self._prune_profile(speaker_id)
        
        logger.debug(f"Added embedding for {speaker_id}. Total embeddings: {len(self.profiles[speaker_id])}")
        
    def _prune_profile(self, speaker_id: str):
        """Remove old or low quality embeddings"""
        if speaker_id not in self.profiles:
            return
            
        embeddings = self.profiles[speaker_id]
        current_time = datetime.now()
        expiry_delta = timedelta(days=self.config.embedding_expiry_days)
        
        # Filter by age and quality
        valid_embeddings = [
            emb for emb in embeddings
            if (current_time - emb.timestamp) < expiry_delta
            and emb.quality_score >= self.config.profile_quality_threshold
        ]
        
        # Sort by quality and confidence
        valid_embeddings.sort(key=lambda x: (x.quality_score * x.confidence), reverse=True)
        
        # Keep only the best embeddings up to max limit
        self.profiles[speaker_id] = valid_embeddings[:self.config.max_embeddings_per_speaker]
        
        if len(embeddings) != len(self.profiles[speaker_id]):
            logger.debug(f"Pruned {speaker_id} profile: {len(embeddings)} -> {len(self.profiles[speaker_id])}")
    
    def get_embeddings(self, speaker_id: str) -> List[np.ndarray]:
        """Get all fingerprints for a speaker"""
        if speaker_id not in self.profiles:
            return []
        return [emb.fingerprint for emb in self.profiles[speaker_id]]
    
    def get_best_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        """Get the highest quality embedding for a speaker"""
        if speaker_id not in self.profiles or not self.profiles[speaker_id]:
            return None
            
        best_emb = max(self.profiles[speaker_id], 
                      key=lambda x: x.quality_score * x.confidence)
        return best_emb.fingerprint
    
    def get_centroid(self, speaker_id: str) -> Optional[np.ndarray]:
        """Get the centroid of all embeddings for a speaker"""
        embeddings = self.get_embeddings(speaker_id)
        if not embeddings:
            return None
            
        # Weight by quality scores
        weights = [emb.quality_score for emb in self.profiles[speaker_id]]
        weighted_sum = np.average(embeddings, axis=0, weights=weights)
        
        # Normalize
        centroid = weighted_sum / np.linalg.norm(weighted_sum)
        return centroid
    
    def save_profile(self, speaker_id: str, filepath: str):
        """Save a speaker profile to disk"""
        if speaker_id not in self.profiles:
            logger.warning(f"No profile found for {speaker_id}")
            return
            
        data = {
            'speaker_id': speaker_id,
            'embeddings': [emb.to_dict() for emb in self.profiles[speaker_id]],
            'saved_at': datetime.now().isoformat()
        }
        
        # Determine format based on extension
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Use pickle for binary format
            with open(filepath, 'wb') as f:
                pickle.dump(self.profiles[speaker_id], f)
    
    def load_profile(self, speaker_id: str, filepath: str):
        """Load a speaker profile from disk"""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            embeddings = [SpeakerEmbedding.from_dict(emb) for emb in data['embeddings']]
        else:
            # Load from pickle
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
                # Convert old format if necessary
                if embeddings and isinstance(embeddings[0], np.ndarray):
                    # Old format - convert to new
                    embeddings = [
                        SpeakerEmbedding(
                            fingerprint=fp,
                            timestamp=datetime.now(),
                            quality_score=0.8,  # Default for old profiles
                            confidence=0.8
                        ) for fp in embeddings
                    ]
        
        self.profiles[speaker_id] = embeddings
        self._prune_profile(speaker_id)  # Prune on load
        
    def get_profile_stats(self, speaker_id: str) -> dict:
        """Get statistics about a speaker profile"""
        if speaker_id not in self.profiles:
            return {'exists': False}
            
        embeddings = self.profiles[speaker_id]
        if not embeddings:
            return {'exists': True, 'num_embeddings': 0}
            
        quality_scores = [emb.quality_score for emb in embeddings]
        confidence_scores = [emb.confidence for emb in embeddings]
        ages = [(datetime.now() - emb.timestamp).days for emb in embeddings]
        
        return {
            'exists': True,
            'num_embeddings': len(embeddings),
            'avg_quality': np.mean(quality_scores),
            'avg_confidence': np.mean(confidence_scores),
            'oldest_days': max(ages),
            'newest_days': min(ages)
        }