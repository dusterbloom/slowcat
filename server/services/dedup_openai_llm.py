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
import os


class DedupAssistantContextAggregator(LLMAssistantContextAggregator):
    """
    Assistant context aggregator that filters LLMTextFrames like OpenAI Realtime Beta.
    Prevents streaming text duplication while allowing TTS streaming.
    """
    
    def __init__(self, context, *, params: LLMAssistantAggregatorParams = None):
        super().__init__(context, params=params)
        self._last_tts_text = ""
        logger.info("🚫 Dedup assistant aggregator initialized - filtering streaming frames")
        logger.info("🚨 DEDUP ASSISTANT AGGREGATOR IS ACTIVE!")
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Filter frames and add only final complete response to context"""
        from pipecat.frames.frames import TTSTextFrame, LLMFullResponseEndFrame, TextFrame
        
        # Only log text-related frames to reduce noise
        if hasattr(frame, 'text') or 'Text' in type(frame).__name__:
            logger.debug(f"🔍 Assistant aggregator: {type(frame).__name__} - '{getattr(frame, 'text', '')[:30]}...'")
        
        # CRITICAL FIX: Block streaming LLMTextFrames from reaching context
        if isinstance(frame, LLMTextFrame):
            logger.debug(f"🚫 Blocking streaming LLMTextFrame: '{frame.text[:30]}...'")
            return  # Don't add streaming chunks to context
            
        # Allow complete responses through
        if isinstance(frame, (TextFrame, TTSTextFrame)):
            logger.debug(f"✅ Allowing complete frame: {type(frame).__name__}")
            await super().process_frame(frame, direction)
            return
            
        # Handle response end - could add accumulated response here if needed
        if isinstance(frame, LLMFullResponseEndFrame):
            logger.debug("✅ LLM response ended - allowing end frame")
            await super().process_frame(frame, direction)
            return
            
        # Pass through all other frames normally
        await super().process_frame(frame, direction)


class DedupOpenAILLMService(OpenAILLMService):
    """
    Custom OpenAI LLM service that prevents context duplication.
    Uses DedupAssistantContextAggregator to filter streaming frames.
    """
    
    def __init__(self, *, base_url: Optional[str] = None, **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        
        # Detect local LLM endpoints
        self._is_local_llm = False
        if base_url:
            self._is_local_llm = any(host in base_url for host in ['localhost', '127.0.0.1', '0.0.0.0'])
            
        logger.info(f"🔧 DedupOpenAILLMService initialized - Local LLM: {self._is_local_llm}")
        logger.info("🚨 DEDUP SERVICE IS ACTIVE - Should prevent context corruption!")
        
    def create_context_aggregator(self, context: OpenAILLMContext) -> OpenAIContextAggregatorPair:
        """Create context aggregator with deduplication for assistant responses"""
        
        logger.info(f"🔍 DedupOpenAILLMService.create_context_aggregator called with context type: {type(context)}")
        
        # Set the LLM adapter like the parent class does
        context.set_llm_adapter(self.get_llm_adapter())
        
        # Check if this is a memory-aware context and create appropriate aggregator
        user_params = LLMUserAggregatorParams()
        
        # Import memory aggregator
        try:
            from processors.memory_context_aggregator import MemoryAwareOpenAILLMContext, MemoryUserContextAggregator
            
            logger.info(f"🔍 Checking if context is MemoryAwareOpenAILLMContext: {isinstance(context, MemoryAwareOpenAILLMContext)}")
            
            if isinstance(context, MemoryAwareOpenAILLMContext):
                logger.info("🧠 Creating memory-aware user aggregator")
                user_aggregator = MemoryUserContextAggregator(context, params=user_params)
            else:
                logger.info("📝 Creating standard user aggregator")
                from pipecat.services.openai.llm import OpenAIUserContextAggregator
                user_aggregator = OpenAIUserContextAggregator(context, params=user_params)
                
        except ImportError as e:
            logger.warning(f"⚠️ Memory aggregator not available: {e}, using standard")
            from pipecat.services.openai.llm import OpenAIUserContextAggregator
            user_aggregator = OpenAIUserContextAggregator(context, params=user_params)
        
        # Create DEDUP assistant aggregator (custom) - does have expect_stripped_words
        assistant_params = LLMAssistantAggregatorParams(
            expect_stripped_words=not self._is_local_llm
        )
        assistant_aggregator = DedupAssistantContextAggregator(context, params=assistant_params)
        
        logger.info("🚫 Created deduplication context aggregators")
        return OpenAIContextAggregatorPair(
            _user=user_aggregator,
            _assistant=assistant_aggregator
        )

    def _get_completion_kwargs(self, context) -> dict:
        """Augment parent kwargs with LM Studio speculative decoding when enabled.

        If `ENABLE_SPECULATIVE=true` and `SPECULATIVE_DRAFT_MODEL` is set in env,
        attach `speculative={"model": <draft>}` to the request. This is compatible
        with LM Studio's OpenAI-compatible API for draft/speculative decoding.
        """
        kwargs = super()._get_completion_kwargs(context)

        try:
            if os.getenv("ENABLE_SPECULATIVE", "false").lower() == "true":
                draft_model = os.getenv("SPECULATIVE_DRAFT_MODEL")
                if draft_model:
                    # Safety: only enable for local LM Studio endpoints
                    if not getattr(self, "_is_local_llm", False):
                        logger.info("⛔ Speculative requested but base_url not local; skipping")
                        return kwargs
                    speculative = {"model": draft_model}
                    # Optional tuning via env
                    max_draft = os.getenv("SPECULATIVE_MAX_DRAFT_TOKENS")
                    min_draft = os.getenv("SPECULATIVE_MIN_DRAFT_TOKENS")
                    if max_draft and max_draft.isdigit():
                        speculative["max_draft_tokens"] = int(max_draft)
                    if min_draft and min_draft.isdigit():
                        speculative["min_draft_tokens"] = int(min_draft)

                    kwargs["speculative"] = speculative
                    logger.info(f"⚡ Speculative decoding enabled (draft='{draft_model}')")
        except Exception as e:
            logger.warning(f"Failed to set speculative params: {e}")

        # Log roles being sent for visibility
        try:
            msgs = getattr(context, 'messages', None)
            if isinstance(msgs, list):
                roles = [m.get('role', '?') for m in msgs if isinstance(m, dict)]
                logger.debug(f"[LLM] Roles in request: {roles}")
        except Exception:
            pass

        return kwargs
