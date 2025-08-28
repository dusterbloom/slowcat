"""
Neural Field Voice Processor - Integrates consciousness field states with voice pipeline

Connects neural field consciousness to voice interactions by:
1. Processing STT transcripts through consciousness.experience()
2. Using prosody analysis to influence emotional field activation
3. Updating field states in real-time during voice processing
4. Providing field state context for response generation

Integration with existing pipeline:
- Receives TranscriptionFrames and processes through consciousness
- Integrates with MoodAnalyzer prosody features
- Updates hierarchical memory field states
- Maintains <200ms voice-to-voice latency
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame, LLMMessagesFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


@dataclass
class VoiceProsodyFeatures:
    """Prosody features from voice analysis"""
    mood: str = "neutral"
    arousal: float = 0.5
    pitch_mean_hz: float = 0.0
    pitch_std_hz: float = 0.0
    energy_rms: float = 0.0
    duration_s: float = 0.0


@dataclass
class FieldEvolutionContext:
    """Context for field evolution from voice interactions"""
    transcription: str
    prosody: VoiceProsodyFeatures
    timestamp: float
    speaker_id: str = "unknown"
    interaction_type: str = "voice_input"


class NeuralFieldVoiceProcessor(FrameProcessor):
    """
    Processes voice interactions through consciousness field evolution
    
    This processor:
    1. Captures voice transcriptions and prosody features
    2. Triggers field evolution through consciousness.experience()
    3. Updates field states in real-time during conversations
    4. Provides field context for downstream processing
    """
    
    def __init__(self, 
                 consciousness_instance=None,
                 hierarchical_memory=None,
                 field_influence_strength: float = 0.7,
                 enable_prosody_mapping: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.consciousness_instance = consciousness_instance
        self.hierarchical_memory = hierarchical_memory
        self.field_influence_strength = field_influence_strength
        self.enable_prosody_mapping = enable_prosody_mapping
        
        # Field state tracking
        self.current_field_states: Dict[str, Any] = {}
        self.field_evolution_history: List[Dict] = []
        
        # Prosody integration
        self.latest_prosody: Optional[VoiceProsodyFeatures] = None
        self.prosody_callbacks: List[Callable] = []
        
        # Performance tracking
        self.processing_stats = {
            "total_voice_interactions": 0,
            "field_evolutions": 0,
            "avg_processing_time_ms": 0.0,
            "last_field_update": 0.0
        }
        
        logger.info(f"🧠 Neural field voice processor initialized (influence: {field_influence_strength})")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and trigger field evolution for voice interactions"""
        await super().process_frame(frame, direction)
        
        # Process transcription frames for field evolution
        if isinstance(frame, TranscriptionFrame) and frame.text:
            await self._process_voice_transcription(frame)
        
        # Forward frame to continue pipeline
        await self.push_frame(frame, direction)
    
    async def _process_voice_transcription(self, frame: TranscriptionFrame):
        """Process voice transcription through consciousness field evolution"""
        start_time = time.time()
        
        try:
            # Extract transcription text
            transcription = frame.text.strip()
            if not transcription:
                return
            
            # Create field evolution context
            context = FieldEvolutionContext(
                transcription=transcription,
                prosody=self.latest_prosody or VoiceProsodyFeatures(),
                timestamp=frame.timestamp if hasattr(frame, 'timestamp') else time.time(),
                speaker_id=getattr(frame, 'speaker_id', 'unknown'),
                interaction_type="voice_input"
            )
            
            # Trigger field evolution through consciousness
            await self._evolve_consciousness_fields(context)
            
            # Update hierarchical memory field states
            await self._update_memory_field_states(context)
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_stats(processing_time)
            
            logger.debug(f"🧠 Field evolution completed: {transcription[:50]}... ({processing_time:.1f}ms)")
            
        except Exception as e:
            logger.error(f"Neural field voice processing failed: {e}")
    
    async def _evolve_consciousness_fields(self, context: FieldEvolutionContext):
        """Evolve consciousness fields based on voice interaction"""
        if not self.consciousness_instance:
            return
        
        try:
            # Use consciousness.experience() for field evolution
            if hasattr(self.consciousness_instance, 'experience'):
                # Process voice input as an experience
                await asyncio.to_thread(
                    self.consciousness_instance.experience,
                    context.transcription
                )
            elif hasattr(self.consciousness_instance, 'symbolize_async'):
                # Fallback to symbolize_async for field updates
                symbols = await self.consciousness_instance.symbolize_async(context.transcription)
                logger.debug(f"🧠 Extracted {len(symbols)} symbols from voice input")
            
            # Apply prosody influence to field states
            if self.enable_prosody_mapping and context.prosody:
                await self._apply_prosody_influence(context.prosody)
            
            # Update current field states
            self.current_field_states = self._extract_current_field_states()
            self.processing_stats["field_evolutions"] += 1
            
        except Exception as e:
            logger.warning(f"Consciousness field evolution failed: {e}")
    
    async def _apply_prosody_influence(self, prosody: VoiceProsodyFeatures):
        """Apply prosody features to influence emotional field activation"""
        if not self.consciousness_instance:
            return
        
        try:
            # Map prosody features to field influence
            field_modifiers = self._calculate_prosody_field_modifiers(prosody)
            
            # Apply modifiers to consciousness fields
            if hasattr(self.consciousness_instance, 'symbol_fields'):
                for symbol, field in self.consciousness_instance.symbol_fields.items():
                    if hasattr(field, 'activation'):
                        # Emotional field activation based on arousal and mood
                        mood_multiplier = self._get_mood_multiplier(prosody.mood)
                        arousal_influence = prosody.arousal * self.field_influence_strength
                        
                        # Adjust field activation (clamped to reasonable bounds)
                        new_activation = max(0.1, min(1.0, 
                            field.activation + (arousal_influence * mood_multiplier * 0.1)
                        ))
                        
                        if abs(new_activation - field.activation) > 0.05:
                            field.activation = new_activation
                            logger.debug(f"🎵 Prosody influenced field '{symbol}': {field.activation:.3f}")
            
        except Exception as e:
            logger.debug(f"Prosody influence application failed: {e}")
    
    def _calculate_prosody_field_modifiers(self, prosody: VoiceProsodyFeatures) -> Dict[str, float]:
        """Calculate field modifiers based on prosody features"""
        modifiers = {
            "arousal_influence": prosody.arousal,
            "pitch_variation": min(1.0, prosody.pitch_std_hz / 100.0),
            "energy_level": min(1.0, prosody.energy_rms * 10.0),
            "mood_factor": self._get_mood_multiplier(prosody.mood)
        }
        return modifiers
    
    def _get_mood_multiplier(self, mood: str) -> float:
        """Get multiplier based on detected mood"""
        mood_multipliers = {
            "excited": 1.2,
            "engaged": 1.1, 
            "neutral": 1.0,
            "calm": 0.9,
            "stressed": 1.3,
        }
        return mood_multipliers.get(mood, 1.0)
    
    async def _update_memory_field_states(self, context: FieldEvolutionContext):
        """Update hierarchical memory field states"""
        if not self.hierarchical_memory:
            return
        
        try:
            # Create field state update for hierarchical memory
            field_update = {
                "timestamp": context.timestamp,
                "interaction_type": context.interaction_type,
                "field_states": self.current_field_states.copy(),
                "prosody_features": {
                    "mood": context.prosody.mood,
                    "arousal": context.prosody.arousal,
                    "pitch_variation": context.prosody.pitch_std_hz
                }
            }
            
            # Update hierarchical memory if it supports field states
            if hasattr(self.hierarchical_memory, 'update_field_states'):
                await self.hierarchical_memory.update_field_states(field_update)
            
        except Exception as e:
            logger.debug(f"Memory field state update failed: {e}")
    
    def _extract_current_field_states(self) -> Dict[str, Any]:
        """Extract current field states from consciousness instance"""
        if not self.consciousness_instance:
            return {}
        
        field_states = {}
        try:
            if hasattr(self.consciousness_instance, 'symbol_fields'):
                for symbol, field in self.consciousness_instance.symbol_fields.items():
                    field_states[symbol] = {
                        "activation": getattr(field, 'activation', 0.5),
                        "coherence": getattr(field, 'coherence', 0.5),
                        "resonance": getattr(field, 'resonance', 0.5),
                        "last_update": time.time()
                    }
            
            # Get general field states if available
            if hasattr(self.consciousness_instance, 'get_field_states'):
                general_states = self.consciousness_instance.get_field_states()
                if general_states:
                    field_states.update(general_states)
            
        except Exception as e:
            logger.debug(f"Field state extraction failed: {e}")
        
        return field_states
    
    def register_prosody_callback(self, callback: Callable):
        """Register callback for prosody updates"""
        self.prosody_callbacks.append(callback)
    
    def update_prosody_features(self, prosody: VoiceProsodyFeatures):
        """Update latest prosody features from MoodAnalyzer"""
        self.latest_prosody = prosody
        
        # Notify callbacks
        for callback in self.prosody_callbacks:
            try:
                asyncio.create_task(callback(prosody))
            except Exception as e:
                logger.debug(f"Prosody callback failed: {e}")
    
    def get_field_states_for_generation(self) -> Dict[str, Any]:
        """Get current field states for response generation context"""
        return {
            "neural_field_states": self.current_field_states.copy(),
            "latest_prosody": {
                "mood": self.latest_prosody.mood if self.latest_prosody else "neutral",
                "arousal": self.latest_prosody.arousal if self.latest_prosody else 0.5,
            },
            "field_influence_strength": self.field_influence_strength,
            "last_field_update": self.processing_stats["last_field_update"]
        }
    
    def _update_processing_stats(self, processing_time_ms: float):
        """Update processing performance statistics"""
        self.processing_stats["total_voice_interactions"] += 1
        self.processing_stats["last_field_update"] = time.time()
        
        # Update rolling average processing time
        total = self.processing_stats["total_voice_interactions"]
        current_avg = self.processing_stats["avg_processing_time_ms"]
        self.processing_stats["avg_processing_time_ms"] = (
            (current_avg * (total - 1)) + processing_time_ms
        ) / total
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        return {
            **self.processing_stats,
            "field_evolution_rate": (
                self.processing_stats["field_evolutions"] / 
                max(1, self.processing_stats["total_voice_interactions"])
            ),
            "performance_target_met": self.processing_stats["avg_processing_time_ms"] <= 50.0  # <50ms target
        }
    
    def set_consciousness_instance(self, consciousness):
        """Set or update consciousness instance"""
        self.consciousness_instance = consciousness
        logger.info("🧠 Neural field voice processor consciousness updated")
    
    def set_hierarchical_memory(self, memory):
        """Set or update hierarchical memory instance"""
        self.hierarchical_memory = memory
        logger.info("🧠 Neural field voice processor memory updated")


def create_neural_field_voice_processor(consciousness_instance=None,
                                       hierarchical_memory=None,
                                       field_influence_strength: float = 0.7,
                                       enable_prosody_mapping: bool = True) -> NeuralFieldVoiceProcessor:
    """Create neural field voice processor with consciousness integration"""
    
    processor = NeuralFieldVoiceProcessor(
        consciousness_instance=consciousness_instance,
        hierarchical_memory=hierarchical_memory,
        field_influence_strength=field_influence_strength,
        enable_prosody_mapping=enable_prosody_mapping
    )
    
    logger.info(f"🧠 Created neural field voice processor (influence: {field_influence_strength})")
    return processor