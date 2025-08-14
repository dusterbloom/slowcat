"""Custom OpenAI LLM service with proper aggregator configuration for LM Studio."""

from typing import Optional
from pipecat.services.openai.llm import OpenAILLMService, OpenAIContextAggregatorPair
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams, 
    LLMAssistantAggregatorParams
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from loguru import logger


class CustomOpenAILLMService(OpenAILLMService):
    """
    Custom OpenAI LLM service that fixes LM Studio streaming duplication issues.
    
    LM Studio sends cumulative chunks (each chunk contains all previous text),
    but we need delta chunks (each chunk contains only new text).
    """
    
    def __init__(self, **kwargs):
        """Initialize with custom configuration for local LLM endpoints."""
        super().__init__(**kwargs)
        
        # Check if we're using a local endpoint (LM Studio)
        base_url = str(getattr(self._client, 'base_url', ''))
        self._is_local_llm = ("localhost" in base_url or "127.0.0.1" in base_url)
        
        # Track accumulated response to compute deltas
        self._last_response = ""
        
        if self._is_local_llm:
            logger.info("🔧 Detected local LLM endpoint - will fix streaming duplication issues")
    
    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = None,
        assistant_params: LLMAssistantAggregatorParams = None,
    ) -> OpenAIContextAggregatorPair:
        """
        Create context aggregators with proper configuration for local LLMs.
        
        For local LLM endpoints (like LM Studio), we set expect_stripped_words=False
        because they already provide properly formatted tokens with correct spacing.
        """
        # Use defaults if not provided
        if user_params is None:
            user_params = LLMUserAggregatorParams()
            
        if assistant_params is None:
            if self._is_local_llm:
                # FINAL FIX: For local LLMs, use expect_stripped_words=False 
                # Local LLMs provide properly formatted tokens, aggregator shouldn't modify them
                assistant_params = LLMAssistantAggregatorParams(expect_stripped_words=False)
                logger.info("🎯 FINAL FIX: Using expect_stripped_words=False for local LLM clean aggregation")
            else:
                # For cloud APIs, use default behavior
                assistant_params = LLMAssistantAggregatorParams(expect_stripped_words=True)
        
        return super().create_context_aggregator(
            context,
            user_params=user_params,
            assistant_params=assistant_params
        )
    
    async def push_frame(self, frame, direction=None):
        """Override push_frame to fix LM Studio cumulative chunks"""
        if self._is_local_llm and hasattr(frame, 'text'):
            # This is a text frame from LM Studio with potentially cumulative content
            cumulative_text = frame.text
            
            # Compute delta (new text only)  
            if cumulative_text.startswith(self._last_response):
                delta_text = cumulative_text[len(self._last_response):]
                if delta_text:  # Only send if there's new content
                    logger.debug(f"🔧 Delta fix: '{cumulative_text}' -> '{delta_text}'")
                    self._last_response = cumulative_text
                    
                    # Create new frame with delta text only
                    from pipecat.frames.frames import TextFrame
                    fixed_frame = TextFrame(delta_text)
                    await super().push_frame(fixed_frame, direction)
                # If no new content, don't push anything
                return
            else:
                # Reset tracking if sequence breaks
                logger.debug(f"🔄 Resetting delta tracking due to sequence break")
                self._last_response = cumulative_text
        
        # For non-text frames or cloud APIs, use default behavior
        await super().push_frame(frame, direction)