"""
Kokoro-optimized text chunker for natural streaming TTS
Based on research from Kokoro-Conversational and StreamingKokoroJS
"""

import asyncio
import time
from typing import List, Optional
from loguru import logger
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


class KokoroSmartChunker(BaseTextAggregator):
    """
    Smart text chunker optimized for Kokoro TTS streaming.
    
    Uses the approach from Kokoro-Conversational:
    - Chunks on natural breaks (commas, conjunctions)  
    - Minimum chunk size to avoid robotic speech
    - Maximum chunk size to prevent long delays
    - Timeout-based chunking for very long sentences
    """
    
    def __init__(
        self,
        chunk_triggers: List[str] = None,
        min_chunk_length: int = 20,  # Proper word groups
        max_chunk_length: int = 150, # Reasonable sentence chunks  
        force_chunk_timeout: float = 0.8,  # Allow natural pauses
        start_filler_words: List[str] = None
    ):
        """
        Initialize the Kokoro smart chunker.
        
        Args:
            chunk_triggers: Punctuation/phrases that trigger chunking
            min_chunk_length: Minimum characters before allowing chunk
            max_chunk_length: Maximum characters before forcing chunk  
            force_chunk_timeout: Seconds to wait before forcing chunk
            start_filler_words: Words to prioritize for first chunk (reduces latency)
        """
        super().__init__()
        
        self._chunk_triggers = chunk_triggers or [
            ".", "!", "?",  # Primary sentence boundaries (like Kokoro)
            ";", ": ",      # Secondary boundaries  
            ", and ", ", but ", ", so ", ", however "  # Natural speech breaks
        ]
        self._min_chunk_length = min_chunk_length
        self._max_chunk_length = max_chunk_length
        self._force_chunk_timeout = force_chunk_timeout
        self._start_filler_words = start_filler_words or [
            "umm", "well", "so", "ok", "right", "yes", "no", "ah", "oh"
        ]
        
        self._buffer = ""
        self._last_chunk_time = time.time()
    
    @property
    def text(self) -> str:
        """Get the currently aggregated text."""
        return self._buffer
        
    async def aggregate(self, text: str) -> Optional[str]:
        """
        Aggregate new text and return chunk if ready.
        
        Args:
            text: New text to add
            
        Returns:
            Text chunk if ready, None otherwise  
        """
        self._buffer += text
        logger.debug(f"📝 Buffer now: '{self._buffer}'")
        
        # Check for immediate filler word optimization (first chunk only)
        if len(self._buffer.strip()) < 50:  # Only for early text
            for filler in self._start_filler_words:
                if filler.lower() in self._buffer.lower() and len(self._buffer) >= 5:
                    chunk = self._buffer.strip()
                    self._buffer = ""
                    self._last_chunk_time = time.time()
                    logger.info(f"🚀 Fast filler chunk: '{chunk}'")
                    return chunk
        
        # Check length-based chunking
        if len(self._buffer) >= self._max_chunk_length:
            return await self._force_chunk("max length reached")
            
        # Check timeout-based chunking
        if time.time() - self._last_chunk_time >= self._force_chunk_timeout:
            if len(self._buffer.strip()) >= self._min_chunk_length:
                return await self._force_chunk("timeout reached")
        
        # Check trigger-based chunking  
        for trigger in self._chunk_triggers:
            if trigger in self._buffer and len(self._buffer) >= self._min_chunk_length:
                # Find the position after the trigger
                trigger_pos = self._buffer.rfind(trigger)
                if trigger_pos != -1:
                    chunk_end = trigger_pos + len(trigger)
                    chunk = self._buffer[:chunk_end].strip()
                    self._buffer = self._buffer[chunk_end:].lstrip()
                    self._last_chunk_time = time.time()
                    
                    if chunk:  # Only return non-empty chunks
                        logger.info(f"🎯 Trigger chunk on '{trigger}': '{chunk}'")
                        return chunk
        
        # No chunking conditions met
        return None
    
    async def _force_chunk(self, reason: str) -> Optional[str]:
        """Force a chunk regardless of triggers."""
        if self._buffer.strip():
            chunk = self._buffer.strip()
            self._buffer = ""
            self._last_chunk_time = time.time()
            logger.info(f"⚡ Forced chunk ({reason}): '{chunk}'")
            return chunk
        return None
    
    async def handle_interruption(self):
        """Handle interruptions by clearing current buffer."""
        logger.info("🛑 Kokoro chunker handling interruption")
        self._buffer = ""
        self._last_chunk_time = time.time()
    
    async def reset(self):
        """Reset the chunker state."""
        self._buffer = ""
        self._last_chunk_time = time.time()
        logger.debug("🔄 Kokoro chunker reset")