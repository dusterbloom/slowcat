# server/services/sherpa_streaming_stt.py
import asyncio
import os
from pathlib import Path
from typing import Optional, AsyncGenerator
import threading
from queue import Queue, Empty
import time

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, InterimTranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

# Lazy import so import-time doesn't pull ONNX into every codepath
_sherpa = None

def _require_sherpa():
    global _sherpa
    if _sherpa is None:
        import sherpa_onnx as _sherpa  # type: ignore
    return _sherpa


class SherpaStreamingSTTService(STTService):
    """
    🔥 STREAMING Sherpa-ONNX STT Service with InterimTranscriptionFrame support!
    
    Processes audio in overlapping windows and emits partial results for ultra-low latency.
    No more waiting for VAD - continuous transcription FTW!
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "auto",
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search",
        provider: Optional[str] = None,
        pool_size: int = 2,
        # 🚀 NEW STREAMING PARAMS
        chunk_duration_ms: int = 500,    # Process every 500ms
        overlap_duration_ms: int = 200,  # 200ms overlap for context
        min_confidence: float = 0.3,     # Emit interim results above this confidence
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.model_dir = Path(model_dir)
        self.language = language
        self.decoding_method = decoding_method
        self.provider = provider or "cpu"
        self.pool_size = pool_size
        
        # 🔥 STREAMING CONFIG
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_duration_ms = overlap_duration_ms
        self.min_confidence = min_confidence
        
        # Calculate chunk sizes in samples
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.overlap_size = int(sample_rate * overlap_duration_ms / 1000)
        
        # Recognizer pool for performance
        self._recognizer_pool = Queue()
        self._pool_lock = threading.Lock()
        
        # Streaming audio buffer
        self._audio_buffer = bytearray()
        self._last_result = ""
        self._processing = False

        if not self.model_dir.exists():
            raise FileNotFoundError(f"SHERPA_ONNX_MODEL_DIR does not exist: {self.model_dir}")

        self._init_recognizer_pool()
        logger.info(f"🚀 Sherpa Streaming STT initialized - chunk: {chunk_duration_ms}ms, overlap: {overlap_duration_ms}ms")

    def _init_recognizer_pool(self):
        """Initialize pool of recognizers for better performance"""
        logger.info(f"🏊 Initializing Sherpa recognizer pool (size: {self.pool_size})")
        
        for i in range(self.pool_size):
            recognizer = self._create_single_recognizer()
            self._recognizer_pool.put(recognizer)
        
        logger.info(f"✅ Sherpa streaming recognizer pool ready")

    def _create_single_recognizer(self):
        sherpa = _require_sherpa()
        tokens = self.model_dir / "tokens.txt"
        model_file = self.model_dir / "model.onnx"

        if not tokens.exists() or not model_file.exists():
            raise FileNotFoundError(f"Model files not found in {self.model_dir}")

        # Use manual configuration instead of factory method to avoid sample rate issues
        try:
            # Create configuration manually
            nemo_ctc_config = sherpa.OfflineNemoEncDecCtcModelConfig(
                model=str(model_file),
            )
            
            model_config = sherpa.OfflineModelConfig(
                nemo_ctc=nemo_ctc_config,
                tokens=str(tokens),
                num_threads=1,
                provider=self.provider,
                debug=False,
                model_type="nemo_ctc",
            )
            
            recognizer_config = sherpa.OfflineRecognizerConfig(
                feat_config=sherpa.FeatureConfig(
                    sample_rate=self.sample_rate,
                    feature_dim=80,
                ),
                model_config=model_config,
                decoding_config=sherpa.OfflineCtcDecodingConfig(
                    decoding_method=self.decoding_method,
                ),
            )
            
            recognizer = sherpa.OfflineRecognizer(recognizer_config)
            logger.info(f"✅ Sherpa recognizer created with manual config (sample_rate={self.sample_rate})")
            return recognizer
            
        except Exception as e:
            logger.error(f"Failed to create Sherpa recognizer with manual config: {e}")
            
            # Fallback to factory method with different sample rates
            for alt_rate in [16000, 8000]:
                try:
                    logger.info(f"Trying factory method with sample rate: {alt_rate}")
                    recognizer = sherpa.OfflineRecognizer.from_nemo_ctc(
                        model=str(model_file),
                        tokens=str(tokens),
                        num_threads=1,
                        sample_rate=alt_rate,
                        feature_dim=80,
                        decoding_method=self.decoding_method,
                        provider=self.provider,
                        debug=False
                    )
                    logger.warning(f"✅ Using factory method with sample rate: {alt_rate}")
                    return recognizer
                except Exception as e2:
                    logger.warning(f"Factory method with {alt_rate}Hz failed: {e2}")
                    continue
            
            # If all fails, raise the original error
            raise e

    def _get_recognizer(self):
        """Get a recognizer from the pool"""
        try:
            return self._recognizer_pool.get(timeout=1.0)
        except Empty:
            logger.warning("⚠️ Recognizer pool exhausted, creating temporary recognizer")
            return self._create_single_recognizer()

    def _return_recognizer(self, recognizer):
        """Return a recognizer to the pool"""
        try:
            self._recognizer_pool.put_nowait(recognizer)
        except:
            pass  # Pool might be full

    def _decode_audio_chunk(self, audio_samples: np.ndarray) -> str:
        """Decode a chunk of audio samples"""
        if audio_samples is None or len(audio_samples) == 0:
            logger.debug("Empty or None audio samples")
            return ""
        
        recognizer = None
        try:
            recognizer = self._get_recognizer()
            stream = recognizer.create_stream()
            
            # Ensure minimum length for model (at least 100ms)
            min_samples = max(8 * 80, int(self.sample_rate * 0.1))  
            if len(audio_samples) < min_samples:
                logger.debug(f"Padding audio from {len(audio_samples)} to {min_samples} samples")
                padding = np.zeros(min_samples - len(audio_samples), dtype=np.float32)
                audio_samples = np.concatenate([audio_samples, padding])
            
            # Validate audio samples
            if np.any(np.isnan(audio_samples)) or np.any(np.isinf(audio_samples)):
                logger.warning("Invalid audio samples (NaN or inf), skipping")
                return ""
            
            # Ensure reasonable amplitude range
            audio_samples = np.clip(audio_samples, -1.0, 1.0)
            
            stream.accept_waveform(self.sample_rate, audio_samples.tolist())
            recognizer.decode_stream(stream)
            result = stream.result.text or ""
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Sherpa decode error: {e}")
            logger.debug(f"Audio info: shape={audio_samples.shape if audio_samples is not None else None}, "
                        f"dtype={audio_samples.dtype if audio_samples is not None else None}")
            return ""
        finally:
            if recognizer:
                self._return_recognizer(recognizer)

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        🔥 STREAMING STT: Process audio chunks and emit interim results!
        """
        if not audio or len(audio) == 0:
            return
            
        await self.start_processing_metrics()
        
        # Add to streaming buffer
        self._audio_buffer.extend(audio)
        
        # Convert to samples - ensure we have valid data
        if len(self._audio_buffer) < 2:  # Need at least 1 int16 sample
            await self.stop_processing_metrics()
            return
            
        # Convert bytes to samples
        try:
            pcm_data = np.frombuffer(self._audio_buffer, dtype=np.int16)
            if pcm_data.size == 0:
                logger.debug("Empty PCM data, skipping")
                await self.stop_processing_metrics()
                return
                
            audio_samples = (pcm_data.astype(np.float32)) / 32768.0
        except Exception as e:
            logger.error(f"Failed to convert audio buffer: {e}")
            await self.stop_processing_metrics()
            return
        
        # Process if we have enough audio (minimum chunk size)
        if len(audio_samples) >= max(self.chunk_size, 1600):  # At least 100ms at 16kHz
            
            # Extract chunk with overlap
            chunk_samples = audio_samples[:self.chunk_size + self.overlap_size]
            
            # Run transcription in thread pool for non-blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._decode_audio_chunk, chunk_samples
            )
            
            # Emit frames based on result quality with better filtering
            if result and result != self._last_result:
                # Clean up the result
                result = result.strip()
                
                # Skip very short or nonsensical results
                if len(result) < 2 or result in ['mm', 'hmm', 'uh', 'um', 'ah']:
                    logger.debug(f"🚫 Skipping noise: '{result}'")
                    return
                
                # Skip results with mostly punctuation or gibberish
                word_chars = sum(c.isalnum() for c in result)
                if word_chars < len(result) * 0.6:  # At least 60% word characters
                    logger.debug(f"🚫 Skipping gibberish: '{result}'")
                    return
                
                # Determine if this is interim or final based on content quality
                words = result.split()
                has_punctuation = result.endswith(('.', '!', '?'))
                is_complete_thought = len(words) >= 2 and has_punctuation
                
                # Only emit interim if it looks like a real partial sentence
                if not is_complete_thought and len(words) >= 2:
                    # 🚀 INTERIM TRANSCRIPTION - Quality partial result
                    logger.debug(f"🔄 Sherpa interim: '{result}' ({len(words)} words)")
                    yield InterimTranscriptionFrame(
                        text=result,
                        user_id="default_user",
                        timestamp=time_now_iso8601(),
                        language=self.language,
                    )
                elif is_complete_thought:
                    # Final transcription - complete sentence
                    logger.info(f"✅ Sherpa final: '{result}' ({len(words)} words)")
                    await self.stop_ttfb_metrics()
                    yield TranscriptionFrame(
                        text=result,
                        user_id="default_user", 
                        timestamp=time_now_iso8601(),
                        language=self.language,
                    )
                else:
                    logger.debug(f"🔄 Partial too short: '{result}' ({len(words)} words)")
                
                self._last_result = result
            
            # Slide the buffer window (keep overlap)
            overlap_bytes = self.overlap_size * 2  # 2 bytes per int16 sample
            self._audio_buffer = self._audio_buffer[self.chunk_size * 2 - overlap_bytes:]
        
        await self.stop_processing_metrics()

    async def cleanup(self):
        """Cleanup recognizer pool"""
        logger.info("🧹 Cleaning up Sherpa streaming recognizer pool")
        while not self._recognizer_pool.empty():
            try:
                recognizer = self._recognizer_pool.get_nowait()
                del recognizer
            except Empty:
                break
        logger.info("✅ Sherpa streaming pool cleaned up")

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass