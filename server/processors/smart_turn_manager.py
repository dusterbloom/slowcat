"""
Smart Turn Manager - Enhanced turn-taking logic with context awareness

This processor adds intelligent turn-taking logic to reduce assistant interruptions
during natural conversation pauses. It works by:

1. Context Analysis: Detects questions, incomplete sentences, and conversation patterns
2. Adaptive Timing: Adjusts wait times based on context and user's speaking rhythm  
3. Progressive Patience: Waits longer on first pause, progressively reduces for subsequent pauses
4. Learning: Adapts to user's typical pause patterns over time

Key Features:
- Questions get 1.5x longer wait time (user may be thinking)
- Incomplete sentences get 2.0x longer wait time
- Learns user's typical pause duration and adapts accordingly
- Progressive reduction for multiple consecutive pauses
- Interruption tracking and adaptive learning

Configuration:
- Set ENABLE_SMART_TURN_MANAGEMENT=false to disable
- Base wait time comes from VAD config (vad_stop_secs)
- Multipliers and reduction factors configurable in constructor

Integration:
- Automatically used by VADEventBridge when enabled
- Works transparently with existing voice recognition pipeline
- Fallback to standard VAD behavior if disabled
"""
import asyncio
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from pipecat.frames.frames import Frame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger


@dataclass
class ConversationContext:
    """Track conversation context for intelligent turn-taking"""
    last_user_text: str = ""
    last_assistant_text: str = ""
    consecutive_pauses: int = 0
    is_question: bool = False
    is_mid_sentence: bool = False
    user_speaking_start_time: Optional[float] = None
    avg_pause_duration: float = 0.3  # User's typical pause duration
    pause_history: list = field(default_factory=list)
    interruption_count: int = 0
    
    def update_pause_history(self, pause_duration: float):
        """Track user's pause patterns to adapt timing"""
        self.pause_history.append(pause_duration)
        # Keep only last 20 pauses for adaptive learning
        if len(self.pause_history) > 20:
            self.pause_history.pop(0)
        
        # Update average pause duration
        if self.pause_history:
            self.avg_pause_duration = sum(self.pause_history) / len(self.pause_history)
    
    def detect_question(self, text: str) -> bool:
        """Detect if user input is likely a question"""
        question_indicators = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you', 
            'could you', 'would you', 'do you', 'did you', 'will you', 'is', 'are'
        ]
        text_lower = text.lower().strip()
        return (
            text_lower.endswith('?') or 
            any(text_lower.startswith(q) for q in question_indicators) or
            'tell me' in text_lower or
            'explain' in text_lower
        )
    
    def detect_incomplete_sentence(self, text: str) -> bool:
        """Detect if sentence seems incomplete (mid-thought)"""
        incomplete_indicators = [
            'and', 'but', 'or', 'because', 'since', 'while', 'if', 'when',
            'although', 'however', 'therefore', 'so', 'then'
        ]
        text_lower = text.lower().strip()
        
        # Ends with conjunction/connector
        ends_with_connector = any(text_lower.endswith(' ' + ind) for ind in incomplete_indicators)
        
        # Seems like partial thought (no punctuation, short, starts mid-sentence)
        looks_incomplete = (
            not text_lower.endswith('.') and 
            not text_lower.endswith('!') and 
            not text_lower.endswith('?') and
            len(text_lower.split()) < 8  # Short phrases often incomplete
        )
        
        return ends_with_connector or looks_incomplete


class SmartTurnManager:
    """
    Enhanced turn manager that adds intelligence to VAD-based turn detection.
    Provides context-aware turn-taking with adaptive timing.
    Note: This is NOT a FrameProcessor - it's a helper class used by VADEventBridge.
    """
    
    def __init__(self, 
                 base_wait_time: float = 0.4,    # Base wait time (from VAD config)
                 question_wait_multiplier: float = 1.5,  # Wait longer for questions
                 incomplete_wait_multiplier: float = 2.0,  # Wait longer for incomplete sentences
                 progressive_reduction: float = 0.7):     # Reduce wait time on repeated pauses
        
        self.base_wait_time = base_wait_time
        self.question_wait_multiplier = question_wait_multiplier
        self.incomplete_wait_multiplier = incomplete_wait_multiplier
        self.progressive_reduction = progressive_reduction
        
        self.context = ConversationContext()
        self._turn_timer: Optional[asyncio.Task] = None
        self._on_turn_complete: Optional[Callable] = None
        self._user_speaking = False
        self._pause_start_time: Optional[float] = None
    
    def set_turn_complete_callback(self, callback: Callable):
        """Set callback for when turn is determined to be complete"""
        self._on_turn_complete = callback
    
    async def handle_frame(self, frame: Frame):
        """Handle VAD frames for intelligent turn decisions"""
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking()
        
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking()
    
    async def _handle_user_started_speaking(self):
        """Handle when user starts speaking"""
        if not self._user_speaking:
            self._user_speaking = True
            self.context.user_speaking_start_time = time.time()
            self.context.consecutive_pauses = 0
            
            # Cancel any pending turn timer
            if self._turn_timer:
                self._turn_timer.cancel()
                self._turn_timer = None
            
            logger.debug("🎤 User started speaking - smart turn manager")
    
    async def _handle_user_stopped_speaking(self):
        """Handle when user stops speaking - start intelligent wait"""
        if self._user_speaking:
            self._user_speaking = False
            self._pause_start_time = time.time()
            self.context.consecutive_pauses += 1
            
            # Calculate intelligent wait time based on context
            wait_time = self._calculate_wait_time()
            
            logger.debug(f"🤔 User paused - waiting {wait_time:.2f}s (pause #{self.context.consecutive_pauses})")
            
            # Start turn timer
            self._turn_timer = asyncio.create_task(self._wait_and_check_turn(wait_time))
    
    def _calculate_wait_time(self) -> float:
        """Calculate intelligent wait time based on conversation context"""
        base_time = self.base_wait_time
        
        # Adjust based on context
        if self.context.is_question:
            base_time *= self.question_wait_multiplier
            logger.debug("📝 Question detected - extending wait time")
        
        if self.context.is_mid_sentence:
            base_time *= self.incomplete_wait_multiplier
            logger.debug("💬 Incomplete sentence detected - extending wait time")
        
        # Progressive reduction for multiple pauses
        if self.context.consecutive_pauses > 1:
            reduction = self.progressive_reduction ** (self.context.consecutive_pauses - 1)
            base_time *= reduction
            logger.debug(f"⏰ Progressive reduction applied: {reduction:.2f}x")
        
        # Adapt to user's typical speaking pattern
        if self.context.avg_pause_duration > 0.2:  # If user typically pauses longer
            adaptation = min(self.context.avg_pause_duration / 0.3, 1.5)  # Cap at 1.5x
            base_time *= adaptation
            logger.debug(f"🎯 Adapted to user pattern: {adaptation:.2f}x")
        
        # Ensure minimum and maximum bounds
        return max(0.15, min(base_time, 2.0))  # Between 150ms and 2s
    
    async def _wait_and_check_turn(self, wait_time: float):
        """Wait for specified time and then signal turn complete if user hasn't resumed"""
        try:
            await asyncio.sleep(wait_time)
            
            # If we reach here, user hasn't resumed speaking
            if not self._user_speaking:
                # Record pause duration for learning
                if self._pause_start_time:
                    pause_duration = time.time() - self._pause_start_time
                    self.context.update_pause_history(pause_duration)
                
                # Signal turn complete
                if self._on_turn_complete:
                    await self._on_turn_complete()
                
                logger.debug("✅ Turn determined complete after smart analysis")
            
        except asyncio.CancelledError:
            # Timer was cancelled because user resumed speaking
            logger.debug("🔄 Turn wait cancelled - user resumed speaking")
    
    def update_conversation_context(self, user_text: str = "", assistant_text: str = ""):
        """Update conversation context for better turn decisions"""
        if user_text:
            self.context.last_user_text = user_text
            self.context.is_question = self.context.detect_question(user_text)
            self.context.is_mid_sentence = self.context.detect_incomplete_sentence(user_text)
            
            logger.debug(f"📄 Context updated - Q:{self.context.is_question}, Incomplete:{self.context.is_mid_sentence}")
        
        if assistant_text:
            self.context.last_assistant_text = assistant_text
    
    def report_interruption(self):
        """Report that an interruption occurred (for learning)"""
        self.context.interruption_count += 1
        logger.warning(f"🚨 Interruption reported (total: {self.context.interruption_count})")
        
        # Increase base wait time slightly after interruptions
        self.base_wait_time = min(self.base_wait_time * 1.1, 0.8)  # Cap at 800ms