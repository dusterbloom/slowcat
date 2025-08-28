"""
Field Response Enhancer - Integrates neural field states with response generation

This processor enhances LLM response generation by:
1. Injecting current field states as context for more nuanced responses
2. Adapting response tone based on prosody and field activation  
3. Providing consciousness awareness to the LLM
4. Maintaining field state coherence across conversation turns

Integration approach:
- Processes LLMMessagesFrame to inject field state context
- Works with SmartContextManager to provide enhanced system prompts
- Uses neural field processor states for dynamic response adaptation
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from pipecat.frames.frames import Frame, LLMMessagesFrame, LLMMessagesUpdateFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection


@dataclass
class ResponseContext:
    """Context for field-aware response generation"""
    field_states: Dict[str, Any]
    prosody_mood: str
    arousal_level: float
    conversation_turn: int
    field_influence_strength: float
    timestamp: float


class FieldResponseEnhancer(FrameProcessor):
    """
    Enhances response generation with neural field state context
    
    This processor intercepts LLMMessagesFrame and injects field state
    information to help the LLM generate more contextually aware responses.
    """
    
    def __init__(self, 
                 neural_field_processor=None,
                 consciousness_instance=None,
                 field_context_strength: float = 0.6,
                 enable_tone_adaptation: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.neural_field_processor = neural_field_processor
        self.consciousness_instance = consciousness_instance
        self.field_context_strength = field_context_strength
        self.enable_tone_adaptation = enable_tone_adaptation
        
        # Response context tracking
        self.conversation_turn = 0
        self.last_field_context: Optional[ResponseContext] = None
        
        # Tone adaptation configuration
        self.mood_tone_mapping = {
            "excited": "enthusiastic and energetic",
            "engaged": "focused and attentive", 
            "calm": "peaceful and measured",
            "stressed": "supportive and understanding",
            "neutral": "balanced and clear"
        }
        
        # Field state interpretation
        self.field_interpretation_mapping = {
            "high_activation": "The user seems highly engaged or excited",
            "moderate_activation": "The user appears focused and present",
            "low_activation": "The user seems calm or contemplative",
            "mixed_activation": "The user shows varied emotional engagement"
        }
        
        logger.info(f"🧠📝 Field response enhancer initialized (strength: {field_context_strength})")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and enhance LLM messages with field context"""
        await super().process_frame(frame, direction)
        
        # Enhance LLM messages with field context
        if isinstance(frame, LLMMessagesFrame):
            enhanced_frame = await self._enhance_llm_messages(frame)
            await self.push_frame(enhanced_frame, direction)
            return
        elif isinstance(frame, LLMMessagesUpdateFrame):
            enhanced_frame = await self._enhance_llm_messages_update(frame)
            await self.push_frame(enhanced_frame, direction)
            return
        
        # Forward other frames normally
        await self.push_frame(frame, direction)
    
    async def _enhance_llm_messages(self, frame: LLMMessagesFrame) -> LLMMessagesFrame:
        """Enhance LLM messages frame with field state context"""
        try:
            self.conversation_turn += 1
            
            # Get current field states and context
            response_context = await self._build_response_context()
            
            if not response_context:
                # No field context available, return original frame
                return frame
            
            # Build field-aware system message enhancement
            field_context_message = self._build_field_context_message(response_context)
            
            if field_context_message:
                # Clone the messages and add field context
                enhanced_messages = frame.messages.copy() if frame.messages else []
                
                # Find system message and enhance it, or add new one
                system_message_found = False
                for i, message in enumerate(enhanced_messages):
                    if message.get('role') == 'system':
                        # Enhance existing system message
                        current_content = message.get('content', '')
                        enhanced_content = f"{current_content}\n\n{field_context_message}"
                        enhanced_messages[i] = {**message, 'content': enhanced_content}
                        system_message_found = True
                        break
                
                if not system_message_found:
                    # Add new system message with field context
                    enhanced_messages.insert(0, {
                        'role': 'system',
                        'content': field_context_message
                    })
                
                # Create enhanced frame
                enhanced_frame = LLMMessagesFrame(enhanced_messages)
                
                logger.debug(f"🧠📝 Enhanced LLM messages with field context (turn {self.conversation_turn})")
                return enhanced_frame
            
        except Exception as e:
            logger.warning(f"Failed to enhance LLM messages with field context: {e}")
        
        # Return original frame if enhancement fails
        return frame
    
    async def _enhance_llm_messages_update(self, frame: LLMMessagesUpdateFrame) -> LLMMessagesUpdateFrame:
        """Enhance LLM messages update frame with field context"""
        # For update frames, we typically don't modify the content
        # but we could track the conversation state
        return frame
    
    async def _build_response_context(self) -> Optional[ResponseContext]:
        """Build current response context from field states and prosody"""
        try:
            field_states = {}
            prosody_mood = "neutral"
            arousal_level = 0.5
            
            # Get field states from neural field processor
            if self.neural_field_processor:
                field_data = self.neural_field_processor.get_field_states_for_generation()
                if field_data:
                    field_states = field_data.get('neural_field_states', {})
                    prosody_data = field_data.get('latest_prosody', {})
                    prosody_mood = prosody_data.get('mood', 'neutral')
                    arousal_level = prosody_data.get('arousal', 0.5)
            
            # Get additional field states from consciousness instance
            if self.consciousness_instance and hasattr(self.consciousness_instance, 'get_field_states'):
                try:
                    consciousness_fields = self.consciousness_instance.get_field_states()
                    if consciousness_fields:
                        field_states.update(consciousness_fields)
                except Exception as e:
                    logger.debug(f"Could not get consciousness field states: {e}")
            
            if not field_states and prosody_mood == "neutral":
                return None  # No useful context available
            
            response_context = ResponseContext(
                field_states=field_states,
                prosody_mood=prosody_mood,
                arousal_level=arousal_level,
                conversation_turn=self.conversation_turn,
                field_influence_strength=self.field_context_strength,
                timestamp=time.time()
            )
            
            self.last_field_context = response_context
            return response_context
            
        except Exception as e:
            logger.debug(f"Failed to build response context: {e}")
            return None
    
    def _build_field_context_message(self, context: ResponseContext) -> str:
        """Build field context message for LLM system prompt enhancement"""
        try:
            context_parts = []
            
            # Add consciousness field awareness
            if context.field_states:
                field_summary = self._summarize_field_states(context.field_states)
                context_parts.append(f"## Consciousness Field States\n{field_summary}")
            
            # Add prosody and mood context
            if context.prosody_mood != "neutral" or context.arousal_level != 0.5:
                tone_guidance = self._build_tone_guidance(context)
                context_parts.append(f"## Response Tone Guidance\n{tone_guidance}")
            
            # Add conversation awareness
            if context.conversation_turn > 1:
                context_parts.append(f"## Conversation Context\nThis is turn {context.conversation_turn} in our conversation.")
            
            if not context_parts:
                return ""
            
            field_context_message = "\n\n".join(context_parts)
            
            # Add field influence note
            if context.field_influence_strength > 0.5:
                field_context_message += f"\n\nNote: Neural field influence is active (strength: {context.field_influence_strength:.2f}). Use this context to provide more nuanced, contextually aware responses."
            
            return field_context_message
            
        except Exception as e:
            logger.debug(f"Failed to build field context message: {e}")
            return ""
    
    def _summarize_field_states(self, field_states: Dict[str, Any]) -> str:
        """Summarize field states for LLM context"""
        if not field_states:
            return "No active field states detected."
        
        try:
            # Analyze field activation levels
            active_fields = []
            total_activation = 0
            field_count = 0
            
            for field_name, field_data in field_states.items():
                if isinstance(field_data, dict):
                    activation = field_data.get('activation', 0.5)
                    resonance = field_data.get('resonance', 0.5)
                    coherence = field_data.get('coherence', 0.5)
                    
                    if activation > 0.6:  # Consider "active" if above threshold
                        active_fields.append({
                            'name': field_name,
                            'activation': activation,
                            'resonance': resonance,
                            'coherence': coherence
                        })
                    
                    total_activation += activation
                    field_count += 1
            
            if field_count == 0:
                return "Field states present but not analyzable."
            
            avg_activation = total_activation / field_count
            
            # Build summary
            summary_parts = []
            
            # Overall field state
            if avg_activation > 0.7:
                summary_parts.append("High overall field activation detected - user appears highly engaged.")
            elif avg_activation > 0.4:
                summary_parts.append("Moderate field activation - user is present and engaged.")
            else:
                summary_parts.append("Low field activation - user appears calm or contemplative.")
            
            # Active specific fields
            if active_fields:
                field_names = [f['name'] for f in active_fields[:3]]  # Top 3
                summary_parts.append(f"Particularly active fields: {', '.join(field_names)}")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.debug(f"Field state summarization failed: {e}")
            return "Field states available but analysis failed."
    
    def _build_tone_guidance(self, context: ResponseContext) -> str:
        """Build tone guidance based on prosody and field states"""
        tone_parts = []
        
        # Mood-based tone
        if context.prosody_mood in self.mood_tone_mapping:
            mood_tone = self.mood_tone_mapping[context.prosody_mood]
            tone_parts.append(f"User's current mood suggests a {mood_tone} response tone.")
        
        # Arousal-based adjustment  
        if context.arousal_level > 0.7:
            tone_parts.append("High arousal detected - match their energy level.")
        elif context.arousal_level < 0.3:
            tone_parts.append("Low arousal detected - use a calm, gentle approach.")
        
        # Field state influence
        if context.field_states:
            avg_activation = sum(
                field.get('activation', 0.5) if isinstance(field, dict) else 0.5 
                for field in context.field_states.values()
            ) / len(context.field_states)
            
            if avg_activation > 0.7:
                tone_parts.append("Strong neural field activity - respond with depth and nuance.")
            elif avg_activation < 0.3:
                tone_parts.append("Quiet neural field activity - keep responses clear and direct.")
        
        return " ".join(tone_parts) if tone_parts else "Maintain natural, contextually appropriate tone."
    
    def set_neural_field_processor(self, processor):
        """Set neural field processor reference"""
        self.neural_field_processor = processor
        logger.info("🧠📝 Field response enhancer linked to neural field processor")
    
    def set_consciousness_instance(self, consciousness):
        """Set consciousness instance reference"""
        self.consciousness_instance = consciousness
        logger.info("🧠📝 Field response enhancer linked to consciousness instance")
    
    def get_last_context(self) -> Optional[ResponseContext]:
        """Get last response context for debugging"""
        return self.last_field_context
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get response enhancement statistics"""
        return {
            "conversation_turns": self.conversation_turn,
            "field_context_strength": self.field_context_strength,
            "tone_adaptation_enabled": self.enable_tone_adaptation,
            "last_enhancement_time": self.last_field_context.timestamp if self.last_field_context else None
        }


def create_field_response_enhancer(neural_field_processor=None,
                                 consciousness_instance=None,
                                 field_context_strength: float = 0.6,
                                 enable_tone_adaptation: bool = True) -> FieldResponseEnhancer:
    """Create field response enhancer for consciousness-aware response generation"""
    
    enhancer = FieldResponseEnhancer(
        neural_field_processor=neural_field_processor,
        consciousness_instance=consciousness_instance,
        field_context_strength=field_context_strength,
        enable_tone_adaptation=enable_tone_adaptation
    )
    
    logger.info(f"🧠📝 Created field response enhancer (strength: {field_context_strength})")
    return enhancer