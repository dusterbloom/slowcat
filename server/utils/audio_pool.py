"""
Audio buffer pool for zero-allocation optimization
Reduces memory allocation latency by pre-allocating common buffer sizes
"""
import queue
from typing import Optional
from loguru import logger


class AudioBufferPool:
    """Pre-allocated audio buffer pool for latency optimization"""
    
    def __init__(self, pool_size: int = 10):
        """Initialize pool with common audio buffer sizes"""
        self._pool_size = pool_size
        self._buffers = queue.Queue(maxsize=pool_size)
        
        # Pre-allocate common buffer sizes (1 second at 16kHz)
        self._buffer_size = 16000
        
        # Fill pool with pre-allocated buffers
        for _ in range(pool_size):
            buffer = bytearray(self._buffer_size)
            self._buffers.put(buffer)
            
        logger.info(f"🎵 Audio buffer pool initialized with {pool_size} buffers of {self._buffer_size} bytes each")
    
    def get_buffer(self) -> bytearray:
        """Get a pre-allocated buffer from pool"""
        try:
            buffer = self._buffers.get_nowait()
            # Clear the buffer for reuse
            buffer[:] = b'\x00' * len(buffer)
            return buffer
        except queue.Empty:
            # Fallback: create new buffer if pool is exhausted
            logger.debug("Audio buffer pool exhausted, creating new buffer")
            return bytearray(self._buffer_size)
    
    def return_buffer(self, buffer: bytearray) -> None:
        """Return buffer to pool for reuse"""
        try:
            # Only return if we have space and buffer is correct size
            if len(buffer) == self._buffer_size:
                self._buffers.put_nowait(buffer)
        except queue.Full:
            # Pool is full, let buffer be garbage collected
            pass
    
    def get_stats(self) -> dict:
        """Get pool statistics"""
        return {
            'pool_size': self._pool_size,
            'available_buffers': self._buffers.qsize(),
            'buffer_size': self._buffer_size
        }


# Global instance for the server
GLOBAL_AUDIO_POOL: Optional[AudioBufferPool] = None


def get_audio_pool() -> AudioBufferPool:
    """Get global audio buffer pool instance"""
    global GLOBAL_AUDIO_POOL
    if GLOBAL_AUDIO_POOL is None:
        GLOBAL_AUDIO_POOL = AudioBufferPool()
    return GLOBAL_AUDIO_POOL


def initialize_audio_pool(pool_size: int = 10) -> None:
    """Initialize global audio pool with specified size"""
    global GLOBAL_AUDIO_POOL
    GLOBAL_AUDIO_POOL = AudioBufferPool(pool_size)