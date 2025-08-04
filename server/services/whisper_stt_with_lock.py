"""
WhisperSTT service wrapper that uses MLX_GLOBAL_LOCK for thread safety
"""

import asyncio
from typing import AsyncGenerator
import numpy as np
from loguru import logger

from pipecat.services.whisper.stt import WhisperSTTServiceMLX as BaseWhisperSTTServiceMLX
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.utils.time import time_now_iso8601

# Import the global MLX lock
from utils.mlx_lock import MLX_GLOBAL_LOCK

try:
    import mlx_whisper
except ImportError:
    logger.error("mlx_whisper not found. Please install it with: pip install mlx-whisper")
    raise


class WhisperSTTServiceMLX(BaseWhisperSTTServiceMLX):
    """
    WhisperSTT service that uses MLX_GLOBAL_LOCK to prevent Metal GPU conflicts
    """
    
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Transcribe audio data using MLX Whisper with proper lock protection.
        
        Args:
            audio: Raw audio bytes in 16-bit PCM format.
            
        Yields:
            TranscriptionFrame: Transcribed text frames
        """
        await self.start_ttfb_metrics()
        
        # Convert audio bytes to numpy array
        audio_np = np.frombuffer(audio, dtype=np.int16)
        
        # Normalize to float32 [-1, 1]
        audio_float = audio_np.astype(np.float32) / np.iinfo(np.int16).max
        
        # Get language setting
        whisper_lang = self.language_to_service_language(self._settings["language"])
        
        logger.debug(f"WhisperSTT: Transcribing audio (length: {len(audio_float)} samples)")
        
        try:
            # Run transcription in thread pool with MLX lock protection
            def _transcribe_with_lock():
                with MLX_GLOBAL_LOCK:
                    logger.debug("WhisperSTT: Acquired MLX lock for transcription")
                    result = mlx_whisper.transcribe(
                        audio_float,
                        path_or_hf_repo=self.model_name,
                        temperature=self._temperature,
                        language=whisper_lang,
                    )
                    logger.debug("WhisperSTT: Released MLX lock")
                    return result
            
            # Execute in thread pool
            chunk = await asyncio.to_thread(_transcribe_with_lock)
            
            # Check no_speech_prob if available
            if hasattr(chunk, 'get') and chunk.get("no_speech_prob", 0) >= self._no_speech_prob:
                logger.debug(f"Ignoring chunk: no_speech_prob={chunk.get('no_speech_prob', 0)}")
                await self.stop_ttfb_metrics()
                return
            
            # Get transcribed text
            text = chunk.get("text", "").strip() if hasattr(chunk, 'get') else str(chunk).strip()
            
            if text:
                logger.debug(f"Transcription: {text}")
                await self.stop_ttfb_metrics()
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._settings["language"],
                )
            else:
                await self.stop_ttfb_metrics()
                
        except Exception as e:
            logger.error(f"Error in WhisperSTT transcription: {e}")
            await self.stop_ttfb_metrics()
            raise