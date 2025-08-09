# server/services/sherpa_stt.py
import asyncio
import os
from pathlib import Path
from typing import Optional
import threading
from queue import Queue, Empty

import numpy as np
from loguru import logger

from pipecat.frames.frames import TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService

# Lazy import so import-time doesn't pull ONNX into every codepath
_sherpa = None


def _require_sherpa():
    global _sherpa
    if _sherpa is None:
        import sherpa_onnx as _sherpa  # type: ignore
    return _sherpa


class SherpaONNXSTTService(SegmentedSTTService):
    """
    Utterance-level STT using Sherpa-ONNX OfflineRecognizer.
    Works with Slowcat's existing VAD/turn-taking. Low CPU on Apple Silicon.
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "auto",
        sample_rate: int = 16000,
        decoding_method: str = "greedy_search",
        provider: Optional[str] = None,  # None -> let sherpa choose; "cpu" recommended
        hotwords_file: Optional[str] = None,
        hotwords_score: float = 1.5,
        pool_size: int = 3,  # Number of recognizers to pre-create
    ):
        super().__init__(sample_rate=sample_rate)
        self.model_dir = Path(model_dir)
        self.language = language
        self.decoding_method = decoding_method
        self.provider = provider or "cpu"
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score
        self.pool_size = pool_size

        # Recognizer pool for performance
        self._recognizer_pool = Queue()
        self._pool_lock = threading.Lock()

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"SHERPA_ONNX_MODEL_DIR does not exist: {self.model_dir}"
            )

        self._init_recognizer_pool()

    # ---------- Sherpa init ----------
    def _init_recognizer_pool(self):
        """Initialize pool of recognizers for better performance"""
        logger.info(f"🏊 Initializing Sherpa recognizer pool (size: {self.pool_size})")
        
        for i in range(self.pool_size):
            logger.info(f"  Creating recognizer {i+1}/{self.pool_size}")
            recognizer = self._create_single_recognizer()
            self._recognizer_pool.put(recognizer)
        
        logger.info(f"✅ Sherpa recognizer pool ready with {self.pool_size} instances")
    
    def _create_single_recognizer(self):
        sherpa = _require_sherpa()

        tokens = self.model_dir / "tokens.txt"
        model_file = self.model_dir / "model.onnx"

        if not tokens.exists():
            raise FileNotFoundError(f"tokens.txt not found in {self.model_dir}")
        if not model_file.exists():
            raise FileNotFoundError(f"model.onnx not found in {self.model_dir}")

        # Back to factory method but try different sample rates
        # The model metadata shows this is a hybrid model with subsampling_factor=8
        # Maybe it expects a different sample rate
        
        try:
            # First try with the expected 16000 Hz
            recognizer = sherpa.OfflineRecognizer.from_nemo_ctc(
                model=str(model_file),
                tokens=str(tokens),
                num_threads=1,
                sample_rate=16000,  # Model trained on 16kHz
                feature_dim=80,
                decoding_method=self.decoding_method,
                provider="cpu",
                debug=False  # Disable debug for cleaner logs
            )
        except Exception as e:
            logger.warning(f"Failed with 16kHz, trying 8kHz: {e}")
            # Try with 8000 Hz as fallback
            recognizer = sherpa.OfflineRecognizer.from_nemo_ctc(
                model=str(model_file),
                tokens=str(tokens),
                num_threads=1,
                sample_rate=8000,  # Try lower sample rate
                feature_dim=80,
                decoding_method=self.decoding_method,
                provider="cpu",
                debug=False
            )
        
        return recognizer

    def _get_recognizer(self):
        """Get a recognizer from the pool (blocking)"""
        try:
            return self._recognizer_pool.get(timeout=5.0)
        except Empty:
            logger.warning("⚠️ Recognizer pool exhausted, creating temporary recognizer")
            return self._create_single_recognizer()
    
    def _return_recognizer(self, recognizer):
        """Return a recognizer to the pool"""
        try:
            self._recognizer_pool.put_nowait(recognizer)
        except:
            logger.warning("⚠️ Could not return recognizer to pool (full)")

    # ---------- SegmentedSTTService interface ----------
    async def run_stt(self, audio: bytes):
        """
        Process complete audio segment and yield transcription frames.
        SegmentedSTTService handles VAD events and provides complete utterances.
        """
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        
        logger.debug(f"Sherpa: Processing {len(audio)} bytes of segmented audio")
        
        try:
            # Run transcription in thread pool
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._decode_buffer_data, audio
            )
            
            logger.debug(f"Sherpa: Transcription result: '{text}'")
            
            if text.strip():
                from pipecat.utils.time import time_now_iso8601
                logger.info(f"Sherpa: Transcribed: '{text.strip()}'")
                
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
                
                yield TranscriptionFrame(
                    text=text.strip(),
                    user_id="default_user", 
                    timestamp=time_now_iso8601(),
                    language=self.language,
                )
            else:
                logger.debug("Sherpa: No transcription result (empty or silence)")
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
                
        except Exception as e:
            logger.exception("Sherpa decode failed")
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
            raise

    # ---------- Helpers ----------
    def _decode_buffer_data(self, buffer_data: bytes) -> str:
        if not buffer_data:
            logger.debug("Sherpa: Empty buffer data")
            return ""

        # Convert PCM16 bytes -> float32 normalized audio
        pcm = np.frombuffer(buffer_data, dtype=np.int16)
        if pcm.size == 0:
            logger.debug("Sherpa: No PCM data after conversion")
            return ""
        
        samples = (pcm.astype(np.float32)) / 32768.0
        logger.debug(f"Sherpa: Converted {len(buffer_data)} bytes -> {len(samples)} samples ({len(samples)/self.sample_rate:.2f}s)")
        
        # Ensure we have reasonable audio length - model has subsampling_factor=8
        min_samples = 8 * 80  # Minimum for 8x subsampling with 80-dim features
        if len(samples) < min_samples:
            logger.debug(f"Sherpa: Audio too short ({len(samples)} < {min_samples}), padding with zeros")
            padding = np.zeros(min_samples - len(samples), dtype=np.float32)
            samples = np.concatenate([samples, padding])

        # Use recognizer from pool for better performance
        recognizer = None
        try:
            logger.debug("Sherpa: Getting recognizer from pool")
            recognizer = self._get_recognizer()
            
            # Check if we have valid audio samples
            if len(samples) == 0:
                logger.warning("Sherpa: Empty audio samples, skipping decode")
                return ""
            
            stream = recognizer.create_stream()
            logger.debug(f"Sherpa: Created stream, accepting waveform with sample_rate={self.sample_rate}, samples={len(samples)}")
            
            # The issue might be with how accept_waveform handles the data
            # Let's ensure we're passing the right format
            stream.accept_waveform(self.sample_rate, samples.tolist() if hasattr(samples, 'tolist') else samples)
            logger.debug("Sherpa: Waveform accepted, decoding...")
            recognizer.decode_stream(stream)
            result = stream.result.text or ""
            logger.debug(f"Sherpa: Decode complete, result: '{result}'")
            return result
        except Exception as e:
            logger.error(f"Sherpa: Decode error: {e}")
            logger.error(f"Sherpa: Audio samples info - type: {type(samples)}, len: {len(samples) if hasattr(samples, '__len__') else 'unknown'}, dtype: {getattr(samples, 'dtype', 'unknown')}")
            raise
        finally:
            # Always return recognizer to pool
            if recognizer:
                logger.debug("Sherpa: Returning recognizer to pool")
                self._return_recognizer(recognizer)

    def cleanup(self):
        """Cleanup recognizer pool"""
        logger.info("🧹 Cleaning up Sherpa recognizer pool")
        while not self._recognizer_pool.empty():
            try:
                recognizer = self._recognizer_pool.get_nowait()
                del recognizer
            except Empty:
                break
        logger.info("✅ Sherpa recognizer pool cleaned up")

    def __del__(self):
        """Cleanup when service is destroyed"""
        try:
            self.cleanup()
        except:
            pass