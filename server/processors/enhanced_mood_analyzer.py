"""
Enhanced Mood Analyzer with Neural Field Integration

Extends the original MoodAnalyzer with:
1. Neural field consciousness integration
2. Prosody-to-field mapping for emotional activation
3. Real-time field state updates from voice features
4. Integration with NeuralFieldVoiceProcessor

This analyzer processes voice prosody and feeds emotional field activation
data to the consciousness system for richer field evolution.
"""

import time
import math
import numpy as np
from typing import Optional, Dict, Any, Callable, List
from loguru import logger

# Import original MoodAnalyzer as base
from processors.mood_analyzer import MoodAnalyzer as BaseMoodAnalyzer
from processors.neural_field_voice_processor import VoiceProsodyFeatures


class EnhancedMoodAnalyzer(BaseMoodAnalyzer):
    """Enhanced mood analyzer with neural field consciousness integration"""
    
    def __init__(self,
                 tape_store,
                 sample_rate: int = 16000,
                 max_buffer_seconds: float = 30.0,
                 neural_field_processor=None,
                 consciousness_instance=None,
                 emotional_field_mapping: bool = True):
        
        # Initialize base analyzer
        super().__init__(tape_store, sample_rate, max_buffer_seconds)
        
        # Neural field integration
        self.neural_field_processor = neural_field_processor
        self.consciousness_instance = consciousness_instance
        self.emotional_field_mapping = emotional_field_mapping
        
        # Enhanced prosody tracking
        self.prosody_history: List[VoiceProsodyFeatures] = []
        self.max_history_size = 10
        
        # Field influence parameters
        self.field_influence_config = {
            "arousal_threshold": 0.6,  # Minimum arousal for field activation
            "pitch_sensitivity": 0.8,  # How much pitch variation affects fields
            "energy_sensitivity": 0.7,  # How much energy affects fields
            "mood_field_mapping": {
                "excited": ["enthusiasm", "energy", "passion"],
                "engaged": ["focus", "attention", "interest"], 
                "calm": ["peace", "tranquility", "balance"],
                "stressed": ["tension", "pressure", "urgency"],
                "neutral": ["stability", "centeredness"]
            }
        }
        
        logger.info("🎵 Enhanced mood analyzer initialized with neural field integration")
    
    async def on_user_stopped(self):
        """Enhanced user stopped handler with field integration"""
        self._stop_ts = time.time()
        
        try:
            # Perform original analysis
            meta = self._analyze_buffer()
            
            if meta:
                # Create prosody features object
                prosody_features = self._create_prosody_features(meta)
                
                # Store in prosody history
                self._update_prosody_history(prosody_features)
                
                # Trigger neural field updates
                await self._trigger_field_updates(prosody_features)
                
                # Original tape store attachment
                try:
                    recent = self.tape_store.get_recent(limit=1)
                    if recent:
                        ts = getattr(recent[0], 'ts', None) if not isinstance(recent[0], dict) else recent[0].get('ts')
                        if ts:
                            # Enhanced metadata with field information
                            enhanced_meta = self._enhance_metadata_with_fields(meta, prosody_features)
                            self.tape_store.add_entry_meta(ts, enhanced_meta)
                            logger.info(f"🧩🧠 Enhanced mood analyzer stored meta: {meta.get('mood','?')} with field data")
                except Exception as e:
                    logger.debug(f"Enhanced mood analyzer could not attach meta: {e}")
                    
        except Exception as e:
            logger.debug(f"Enhanced mood analyzer analysis failed: {e}")
    
    def _create_prosody_features(self, meta: Dict[str, Any]) -> VoiceProsodyFeatures:
        """Create VoiceProsodyFeatures from analysis metadata"""
        return VoiceProsodyFeatures(
            mood=meta.get('mood', 'neutral'),
            arousal=meta.get('arousal', 0.5),
            pitch_mean_hz=meta.get('pitch_mean_hz', 0.0),
            pitch_std_hz=meta.get('pitch_std_hz', 0.0),
            energy_rms=meta.get('energy_rms', 0.0),
            duration_s=meta.get('duration_s', 0.0)
        )
    
    def _update_prosody_history(self, features: VoiceProsodyFeatures):
        """Update prosody history with size limit"""
        self.prosody_history.append(features)
        if len(self.prosody_history) > self.max_history_size:
            self.prosody_history.pop(0)
    
    async def _trigger_field_updates(self, prosody_features: VoiceProsodyFeatures):
        """Trigger neural field updates based on prosody features"""
        try:
            # Update neural field processor with prosody
            if self.neural_field_processor:
                self.neural_field_processor.update_prosody_features(prosody_features)
            
            # Direct consciousness field activation if enabled
            if self.emotional_field_mapping and self.consciousness_instance:
                await self._activate_emotional_fields(prosody_features)
                
        except Exception as e:
            logger.debug(f"Field update trigger failed: {e}")
    
    async def _activate_emotional_fields(self, prosody: VoiceProsodyFeatures):
        """Activate emotional fields based on prosody analysis"""
        try:
            # Get relevant field names for the detected mood
            field_names = self.field_influence_config["mood_field_mapping"].get(prosody.mood, [])
            
            if not field_names or not hasattr(self.consciousness_instance, 'symbol_fields'):
                return
            
            # Calculate activation strength based on prosody features
            activation_strength = self._calculate_activation_strength(prosody)
            
            # Activate relevant fields
            activated_fields = []
            for field_name in field_names:
                if field_name in self.consciousness_instance.symbol_fields:
                    field = self.consciousness_instance.symbol_fields[field_name]
                    
                    if hasattr(field, 'activation'):
                        # Increase field activation based on prosody
                        current_activation = getattr(field, 'activation', 0.5)
                        new_activation = min(1.0, current_activation + activation_strength)
                        field.activation = new_activation
                        activated_fields.append(field_name)
            
            if activated_fields:
                logger.debug(f"🎵🧠 Activated emotional fields: {activated_fields} (strength: {activation_strength:.3f})")
                
        except Exception as e:
            logger.debug(f"Emotional field activation failed: {e}")
    
    def _calculate_activation_strength(self, prosody: VoiceProsodyFeatures) -> float:
        """Calculate field activation strength from prosody features"""
        # Base activation from arousal
        base_strength = prosody.arousal * 0.3
        
        # Additional strength from pitch variation
        pitch_strength = min(0.2, prosody.pitch_std_hz / 200.0)
        
        # Additional strength from energy
        energy_strength = min(0.2, prosody.energy_rms * 2.0)
        
        # Mood-specific multipliers
        mood_multiplier = {
            "excited": 1.2,
            "engaged": 1.0,
            "stressed": 1.1, 
            "calm": 0.8,
            "neutral": 0.9
        }.get(prosody.mood, 1.0)
        
        total_strength = (base_strength + pitch_strength + energy_strength) * mood_multiplier
        return min(0.5, total_strength)  # Cap at 0.5 for safety
    
    def _enhance_metadata_with_fields(self, meta: Dict[str, Any], prosody: VoiceProsodyFeatures) -> Dict[str, Any]:
        """Enhance metadata with neural field information"""
        enhanced_meta = meta.copy()
        
        # Add field-related metadata
        enhanced_meta.update({
            'neural_field_data': {
                'prosody_features': {
                    'mood': prosody.mood,
                    'arousal': prosody.arousal,
                    'pitch_variation': prosody.pitch_std_hz,
                    'energy_level': prosody.energy_rms
                },
                'field_activation_strength': self._calculate_activation_strength(prosody),
                'activated_field_domains': self.field_influence_config["mood_field_mapping"].get(prosody.mood, [])
            }
        })
        
        # Add prosody trend analysis if we have history
        if len(self.prosody_history) > 1:
            enhanced_meta['neural_field_data']['prosody_trend'] = self._analyze_prosody_trend()
        
        return enhanced_meta
    
    def _analyze_prosody_trend(self) -> Dict[str, Any]:
        """Analyze prosody trends from recent history"""
        if len(self.prosody_history) < 2:
            return {}
        
        # Calculate trends over recent history
        recent = self.prosody_history[-3:] if len(self.prosody_history) >= 3 else self.prosody_history
        
        arousal_values = [p.arousal for p in recent]
        pitch_values = [p.pitch_std_hz for p in recent]
        energy_values = [p.energy_rms for p in recent]
        
        return {
            'arousal_trend': 'increasing' if arousal_values[-1] > arousal_values[0] else 'decreasing',
            'pitch_trend': 'increasing' if pitch_values[-1] > pitch_values[0] else 'decreasing', 
            'energy_trend': 'increasing' if energy_values[-1] > energy_values[0] else 'decreasing',
            'trend_strength': abs(arousal_values[-1] - arousal_values[0])
        }
    
    def set_neural_field_processor(self, processor):
        """Set neural field processor reference"""
        self.neural_field_processor = processor
        logger.info("🎵🧠 Enhanced mood analyzer linked to neural field processor")
    
    def set_consciousness_instance(self, consciousness):
        """Set consciousness instance for direct field activation"""
        self.consciousness_instance = consciousness
        logger.info("🎵🧠 Enhanced mood analyzer linked to consciousness instance")
    
    def get_recent_prosody_summary(self) -> Dict[str, Any]:
        """Get summary of recent prosody trends"""
        if not self.prosody_history:
            return {"status": "no_data"}
        
        recent = self.prosody_history[-5:] if len(self.prosody_history) >= 5 else self.prosody_history
        
        return {
            "recent_moods": [p.mood for p in recent],
            "avg_arousal": sum(p.arousal for p in recent) / len(recent),
            "avg_pitch_variation": sum(p.pitch_std_hz for p in recent) / len(recent),
            "avg_energy": sum(p.energy_rms for p in recent) / len(recent),
            "sample_count": len(recent),
            "trend_analysis": self._analyze_prosody_trend() if len(recent) > 1 else {}
        }


# Helper to wire enhanced analyzer with pipeline components
def attach_enhanced_mood_analyzer(tee, vad_bridge, tape_store, 
                                neural_field_processor=None,
                                consciousness_instance=None,
                                sample_rate: int = 16000) -> EnhancedMoodAnalyzer:
    """Attach EnhancedMoodAnalyzer to AudioTee and VADEventBridge with field integration.
    
    Args:
        tee: AudioTeeProcessor
        vad_bridge: VADEventBridge  
        tape_store: TapeStore for metadata storage
        neural_field_processor: Optional NeuralFieldVoiceProcessor instance
        consciousness_instance: Optional consciousness instance for direct field access
        sample_rate: Audio sample rate
    """
    analyzer = EnhancedMoodAnalyzer(
        tape_store=tape_store,
        sample_rate=sample_rate,
        neural_field_processor=neural_field_processor,
        consciousness_instance=consciousness_instance
    )
    
    try:
        # Register as audio consumer
        tee.register_audio_consumer(analyzer)
        logger.info("🎵 Enhanced mood analyzer registered with audio tee")
    except Exception as e:
        logger.warning(f"Could not register enhanced mood analyzer with audio tee: {e}")
    
    try:
        # Attach VAD callbacks
        vad_bridge.set_callbacks(on_started=analyzer.on_user_started, on_stopped=analyzer.on_user_stopped)
        logger.info("🎵 Enhanced mood analyzer attached to VAD bridge")
    except Exception as e:
        logger.warning(f"Could not attach enhanced mood analyzer to VAD: {e}")
    
    return analyzer