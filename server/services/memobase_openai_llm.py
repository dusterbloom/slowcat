"""
MemoBase-enabled OpenAI LLM service that uses the patched client for automatic memory
"""

from typing import Optional, Dict, Any, Callable, Awaitable
from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
import asyncio


class MemoBaseOpenAILLMService(OpenAILLMService):
    """
    OpenAI LLM service that uses MemoBase patched client for automatic memory handling.
    Replaces the standard OpenAI client with the MemoBase patched version.
    """
    
    def __init__(self, patched_client=None, user_id: str = "default_user", **kwargs):
        """Initialize with MemoBase patched client"""
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Replace the OpenAI client with our patched version
        if patched_client:
            self._sync_client = patched_client
            logger.info(f"🧠 MemoBase patched client set for automatic memory")
        else:
            logger.warning("⚠️ No patched client provided, falling back to standard OpenAI client")
            self._sync_client = None
        
        # Store user_id for memory operations
        self._user_id = user_id
        logger.info(f"🔍 Using user_id for memory: '{self._user_id}'")
    
    async def _stream_chat_completions(self, context):
        """
        Override the streaming method to use MemoBase patched client with user_id.
        This triggers automatic memory handling by the MemoBase patch.
        """
        if not self._sync_client:
            logger.warning("⚠️ No MemoBase patched client available, falling back to parent")
            return await super()._stream_chat_completions(context)
        
        try:
            # Extract messages from context
            messages = context.get_messages()
            tools = context.get_tools() if hasattr(context, 'get_tools') else None
            
            # Prepare the call parameters for MemoBase patched client
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": getattr(self, '_max_tokens', 4096),
                "temperature": getattr(self, '_temperature', 0.6),
                "user_id": self._user_id,  # This triggers MemoBase automatic memory
                "stream": True,  # Enable streaming
            }
            
            # Add tools if available
            if tools:
                params["tools"] = tools
            
            # Add other optional parameters
            if hasattr(self, '_seed') and self._seed is not None:
                params["seed"] = self._seed
            
            if hasattr(self, '_frequency_penalty') and self._frequency_penalty is not None:
                params["frequency_penalty"] = self._frequency_penalty
            
            if hasattr(self, '_presence_penalty') and self._presence_penalty is not None:
                params["presence_penalty"] = self._presence_penalty
            
            if hasattr(self, '_top_p') and self._top_p is not None:
                params["top_p"] = self._top_p
            
            logger.info(f"🧠 Making MemoBase LLM call with user_id='{self._user_id}' for automatic memory")
            logger.info(f"🧠 Using model: {self.model_name}")
            logger.info(f"🧠 Patched client type: {type(self._sync_client)}")
            
            # Convert sync generator to async stream for pipeline compatibility
            def _sync_call():
                return self._sync_client.chat.completions.create(**params)
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_call)
            
            logger.debug(f"✅ MemoBase LLM call completed - memory automatically handled")
            
            # Convert sync generator to async iterator if needed
            if hasattr(response, '__iter__') and not hasattr(response, '__aiter__'):
                # Create an async wrapper for the sync generator
                async def async_generator_wrapper():
                    for chunk in response:
                        yield chunk
                return async_generator_wrapper()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in MemoBase LLM call: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            logger.error(f"❌ Falling back to parent implementation without memory")
            # Fall back to parent implementation
            return await super()._stream_chat_completions(context)
    
    def set_user_id(self, user_id: str):
        """Update the user_id for memory operations"""
        old_user_id = self._user_id
        self._user_id = user_id
        logger.info(f"🔄 Updated MemoBase user_id from '{old_user_id}' to '{user_id}'")