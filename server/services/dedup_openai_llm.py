"""
Custom OpenAI LLM service with Realtime Beta's duplication filtering approach
Works with LM Studio while preventing context duplication
"""

from typing import Optional
from pipecat.services.openai.llm import OpenAILLMService, OpenAIContextAggregatorPair
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams, 
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import Frame, LLMTextFrame, TranscriptionFrame, InterimTranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from loguru import logger


class DedupAssistantContextAggregator(LLMAssistantContextAggregator):
    """
    Assistant context aggregator that filters LLMTextFrames like OpenAI Realtime Beta.
    Prevents streaming text duplication while allowing TTS streaming.
    """
    
    def __init__(self, context, *, params: LLMAssistantAggregatorParams = None):
        super().__init__(context, params=params)
        self._last_tts_text = ""
        logger.info("🚫 Dedup assistant aggregator initialized - filtering streaming frames")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Filter frames and add only final complete response to context"""
        from pipecat.frames.frames import TTSTextFrame, LLMFullResponseEndFrame, TextFrame
        
        # Only log text-related frames to reduce noise
        if hasattr(frame, 'text') or 'Text' in type(frame).__name__:
            logger.debug(f"🔍 Assistant aggregator: {type(frame).__name__} - '{getattr(frame, 'text', 'N/A')}'")
        
        # Block streaming LLM frames 
        if isinstance(frame, (LLMTextFrame, TranscriptionFrame, InterimTranscriptionFrame)):
            logger.debug(f"🚫 BLOCKED {type(frame).__name__} from context")
            return
        
        # For TTSTextFrames, track the final complete one
        if isinstance(frame, TTSTextFrame):
            current_text = frame.text
            # Keep track of the longest/most complete text
            if len(current_text) > len(self._last_tts_text):
                self._last_tts_text = current_text
                logger.debug(f"📝 Updated final TTS text: '{current_text}'")
            # Block TTSTextFrames from direct context addition
            return
        
        # When LLM response ends, add the final complete text to context
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._last_tts_text.strip():
                logger.info(f"✅ Adding final response to context: '{self._last_tts_text}'")
                # Create a clean TextFrame with the final complete response
                final_text_frame = TextFrame(self._last_tts_text.strip())
                await super().process_frame(final_text_frame, direction)
            # Reset for next response
            self._last_tts_text = ""
            # Don't pass through the end frame to context
            return
        
        await super().process_frame(frame, direction)


class DedupOpenAILLMService(OpenAILLMService):
    """
    Custom OpenAI LLM service that uses deduplication filtering for context aggregation.
    Compatible with LM Studio while preventing streaming text duplication.
    """
    
    def __init__(self, **kwargs):
        """Initialize with custom configuration for local LLM endpoints."""
        super().__init__(**kwargs)
        
        # Check if we're using a local endpoint (LM Studio)
        base_url = str(getattr(self._client, 'base_url', ''))
        self._is_local_llm = ("localhost" in base_url or "127.0.0.1" in base_url)
        
        self._accumulated_response = ""
        self._in_streaming_response = False
        
        if self._is_local_llm:
            logger.info("🔧 Custom dedup OpenAI service for LM Studio - filtering enabled")
        else:
            logger.info("🔧 Custom dedup OpenAI service for cloud - filtering enabled")
    
    # Removed push_frame override - let streaming work normally for TTS
    # The assistant aggregator filtering will handle context deduplication
    
    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = None,
        assistant_params: LLMAssistantAggregatorParams = None,
    ) -> OpenAIContextAggregatorPair:
        """
        Create context aggregators with deduplication filtering for assistant.
        Uses custom DedupAssistantContextAggregator that filters LLMTextFrames.
        """
        # Use defaults if not provided
        if user_params is None:
            user_params = LLMUserAggregatorParams()
            
        if assistant_params is None:
            if self._is_local_llm:
                # For local LLMs, use expect_stripped_words=False 
                assistant_params = LLMAssistantAggregatorParams(expect_stripped_words=False)
                logger.info("🎯 Using expect_stripped_words=False for local LLM")
            else:
                # For cloud APIs, use default behavior
                assistant_params = LLMAssistantAggregatorParams(expect_stripped_words=True)
        
        # Create user aggregator normally
        from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator
        user_aggregator = LLMUserContextAggregator(context, params=user_params)
        
        # Create CUSTOM assistant aggregator with filtering
        assistant_aggregator = DedupAssistantContextAggregator(context, params=assistant_params)
        
        logger.info("✅ Created dedup context aggregators - LLMTextFrames will be filtered")
        return OpenAIContextAggregatorPair(user_aggregator, assistant_aggregator)