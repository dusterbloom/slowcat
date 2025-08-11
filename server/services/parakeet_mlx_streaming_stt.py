"""
Parakeet-MLX Streaming STT Service optimized for Apple Silicon

Provides real-time streaming transcription using parakeet-mlx with:
- Native MLX optimization for M-series chips
- True streaming with configurable context windows
- Low-latency incremental processing
- Draft and finalized token handling
- Memory-efficient local attention modes
"""

import asyncio
import os
from typing import Optional, AsyncGenerator, Tuple
import threading
from queue import Queue, Empty
import time

import numpy as np
from loguru import logger
import mlx.core as mx

from pipecat.frames.frames import Frame, TranscriptionFrame, InterimTranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

# Import the global MLX lock for thread safety
from utils.mlx_lock import MLX_GLOBAL_LOCK

# Lazy import to avoid global state issues
def _require_parakeet_mlx():
    """Import parakeet_mlx safely without global state"""
    try:
        from parakeet_mlx import from_pretrained
        return from_pretrained
    except ImportError as e:
        raise RuntimeError(f"parakeet-mlx not available: {e}. Install with: pip install parakeet-mlx") from e


class ParakeetMLXStreamingSTTService(STTService):
    """
    🚀 Parakeet-MLX Streaming STT Service for Apple Silicon
    
    Features:
    - Real-time streaming transcription
    - Configurable context windows for memory efficiency
    - Local and full attention modes
    - Draft and finalized token handling
    - MLX optimization for M-series chips
    """

    def __init__(
        self,
        model_name: str = "mlx-community/parakeet-tdt-0.6b-v2",
        context_size: Tuple[int, int] = (256, 256),
        attention_mode: str = "local",  # "local" or "full"
        precision: str = "bf16",  # "bf16" or "fp32"
        language: str = "en",
        sample_rate: int = 16000,
        chunk_size_ms: int = 100,  # Default chunk size in ms
        emit_partial_results: bool = True,
        min_confidence: float = 0.1,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._parakeet_model_name = model_name
        self.context_size = context_size
        self.attention_mode = attention_mode
        self.precision = precision
        self._parakeet_language = language
        self.emit_partial_results = emit_partial_results
        self.min_confidence = min_confidence
        
        # Calculate chunk size in samples - Metal GPU requires minimum buffer size
        # Note: Parakeet needs significant buffering to produce finalized tokens
        # With context_size=(256,256) and depth=1, it needs 256 frames before finalizing
        self.min_chunk_size = 1600  # 100ms at 16kHz - minimum for Metal stability
        self.chunk_size = max(int(sample_rate * chunk_size_ms / 1000), self.min_chunk_size)
        
        # Small chunk accumulator for production audio (304 sample chunks)
        self._small_chunk_buffer = []
        self._small_chunk_samples = 0
        
        # Streaming state
        self._model = None  # Will be initialized lazily
        self._transcriber = None
        self._audio_buffer = bytearray()
        self._last_result = ""
        self._processing_lock = threading.RLock()
        self._initialization_attempted = False
        self._stream_active = False
        self._chunks_processed = 0  # Track chunks for periodic reset
        
        # Audio processing queue (larger to handle throughput)
        self._audio_queue = Queue(maxsize=50)  # Increased from 10
        self._result_queue = Queue(maxsize=100)  # Increased from 20
        self._processing_task = None
        
        logger.info(
            f"🚀 Parakeet-MLX STT initialized - model: {self._parakeet_model_name}, "
            f"context: {context_size}, attention: {attention_mode}, "
            f"chunk: {chunk_size_ms}ms, partial: {emit_partial_results}"
        )

    def _init_model(self):
        """Initialize the Parakeet-MLX model with proper configuration"""
        if self._model is not None:
            return
            
        try:
            from_pretrained = _require_parakeet_mlx()
            logger.debug("✅ parakeet-mlx imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import parakeet-mlx: {e}")
            raise RuntimeError(f"Failed to import parakeet-mlx: {e}") from e
        
        try:
            with MLX_GLOBAL_LOCK:
                logger.info(f"🔄 Loading Parakeet-MLX model: {self._parakeet_model_name}")
                
                # Load model with MLX optimization
                self._model = from_pretrained(self._parakeet_model_name)
                
                logger.info(f"✅ Parakeet-MLX model loaded: {self._parakeet_model_name}")
                logger.info(f"🎯 Context size: {self.context_size}, Attention: {self.attention_mode}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Parakeet-MLX model: {e}")
            raise RuntimeError(f"Failed to load model {self._parakeet_model_name}: {e}") from e

    def _start_streaming(self):
        """Initialize streaming transcriber"""
        if self._transcriber is not None:
            self._stop_streaming()  # Stop existing transcriber first
            
        try:
            with MLX_GLOBAL_LOCK:
                logger.debug("🔄 Starting Parakeet-MLX streaming transcriber")
                
                # Create streaming transcriber with requested context size
                logger.info(f"🔧 Using context size: {self.context_size}")
                self._transcriber = self._model.transcribe_stream(
                    context_size=self.context_size
                )
                self._transcriber.__enter__()
                self._stream_active = True
                self._chunks_processed = 0  # Track chunks for periodic reset
                
                logger.debug("✅ Parakeet-MLX streaming transcriber started")
                
        except Exception as e:
            logger.error(f"❌ Failed to start streaming: {e}")
            raise RuntimeError(f"Failed to start streaming: {e}") from e

    def _stop_streaming(self):
        """Stop streaming transcriber and cleanup"""
        if self._transcriber is not None:
            try:
                with MLX_GLOBAL_LOCK:
                    # Get final result before stopping
                    if hasattr(self._transcriber, 'result'):
                        final_result = self._transcriber.result
                        if final_result and hasattr(final_result, 'text'):
                            logger.debug(f"Final text before stop: {final_result.text}")
                    
                    self._transcriber.__exit__(None, None, None)
                    self._transcriber = None
                    self._stream_active = False
                    logger.debug("🛑 Parakeet-MLX streaming transcriber stopped")
            except Exception as e:
                logger.warning(f"Error stopping transcriber: {e}")
    
    def _reset_streaming_context(self):
        """Reset the streaming context to prevent memory buildup"""
        logger.debug("♻️ Resetting streaming context to prevent memory buildup")
        self._stop_streaming()
        self._start_streaming()

    async def _process_audio_chunks(self):
        """Background task to process buffered audio chunks"""
        
        while self._stream_active:
            try:
                # Get buffered chunk (already combined to safe size)
                try:
                    audio_chunk = self._audio_queue.get(timeout=0.1)
                    if audio_chunk is None:  # Shutdown signal
                        self._stream_active = False
                        break
                    logger.debug(f"Processing buffered chunk: {len(audio_chunk)} samples, dtype: {audio_chunk.dtype}")
                except Empty:
                    await asyncio.sleep(0.01)
                    continue
                    
                # Process the buffered chunk directly in async context - NO THREADING for Metal/MLX!
                try:
                    # Ensure the audio is the right shape and type
                    if audio_chunk.ndim != 1:
                        logger.warning(f"Chunk has wrong dimensions: {audio_chunk.ndim}")
                        continue
                    
                    # Convert numpy to MLX array
                    mlx_audio = mx.array(audio_chunk)
                    logger.debug(f"MLX array created: shape={mlx_audio.shape}, dtype={mlx_audio.dtype}")
                    
                    # Add to transcriber - this should work now with buffered chunks >= 1600 samples
                    self._transcriber.add_audio(mlx_audio)
                    self._chunks_processed += 1
                    logger.debug(f"Added buffered chunk to transcriber: {len(audio_chunk)} samples (chunk #{self._chunks_processed})")
                    
                    # Get current result after processing
                    result = self._transcriber.result
                    if result:
                        # Log more details about the result
                        if hasattr(result, 'text'):
                            text = result.text.strip()
                        else:
                            text = str(result).strip()
                        
                        # Check for both draft and finalized tokens
                        draft = getattr(self._transcriber, 'draft_tokens', [])
                        finalized = getattr(self._transcriber, 'finalized_tokens', [])
                        
                        logger.debug(f"📊 Transcriber state - Text: '{text}', Draft: {len(draft)} tokens, Finalized: {len(finalized)} tokens")
                        
                        # Build text from finalized tokens (these are confirmed)
                        finalized_text = ""
                        if finalized:
                            finalized_sentences = []
                            current_sentence = []
                            for token in finalized:
                                if hasattr(token, 'text'):
                                    current_sentence.append(token.text)
                            if current_sentence:
                                finalized_text = "".join(current_sentence).strip()
                        
                        # Build text from draft tokens (these are tentative)
                        draft_text = ""
                        if draft:
                            draft_sentences = []
                            current_sentence = []
                            for token in draft:
                                if hasattr(token, 'text'):
                                    current_sentence.append(token.text)
                            if current_sentence:
                                draft_text = "".join(current_sentence).strip()
                        
                        # Combine finalized and draft text
                        combined_text = finalized_text
                        if draft_text:
                            if combined_text:
                                combined_text += " " + draft_text
                            else:
                                combined_text = draft_text
                        
                        # Reset context if too many draft tokens accumulate (prevents memory buildup)
                        if len(draft) > 30:  # Increased threshold
                            logger.warning(f"⚠️ Too many draft tokens ({len(draft)}), resetting context")
                            if combined_text and len(combined_text) > len(self._last_result):
                                # Save the result before reset
                                self._result_queue.put((combined_text, True, 1.0))
                                self._last_result = combined_text
                            self._reset_streaming_context()
                            continue
                        
                        # Emit results based on what we have
                        if combined_text and len(combined_text) > len(self._last_result):
                            # Determine if this is final based on whether we have finalized tokens
                            is_final = len(finalized) > 0
                            confidence = 1.0 if is_final else 0.8
                            
                            self._result_queue.put((combined_text, is_final, confidence))
                            self._last_result = combined_text
                            logger.info(f"{'✅ Final' if is_final else '🔄 Draft'} transcription: '{combined_text}'")
                    
                    # Periodic reset to prevent memory issues (every 50 chunks = ~5 seconds)
                    if self._chunks_processed > 50:
                        logger.info(f"📊 Periodic reset after {self._chunks_processed} chunks")
                        self._reset_streaming_context()
                            
                except Exception as process_error:
                    logger.error(f"Error in MLX processing: {process_error}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Cleanup MLX memory after processing batch
                try:
                    mx.clear_cache()
                except:
                    pass
                                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                # Don't break - keep trying to process
                await asyncio.sleep(0.1)  # Brief pause before retrying

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Process audio data for streaming transcription
        
        Args:
            audio: Raw audio bytes in 16-bit PCM format
            
        Yields:
            TranscriptionFrame or InterimTranscriptionFrame
        """
        await self.start_processing_metrics()
        
        # Initialize model if not already done
        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                self._init_model()
                self._start_streaming()
                
                # Start background processing task
                if self._processing_task is None:
                    self._processing_task = asyncio.create_task(self._process_audio_chunks())
                    
            except Exception as e:
                logger.error(f"Failed to initialize Parakeet-MLX: {e}")
                await self.stop_processing_metrics()
                return
        
        # Convert audio bytes to numpy array
        try:
            audio_np = np.frombuffer(audio, dtype=np.int16)
            # Normalize to float32 [-1, 1]
            audio_float = audio_np.astype(np.float32) / np.iinfo(np.int16).max
            
            # Debug: Check audio statistics
            if len(audio_float) > 0:
                audio_rms = np.sqrt(np.mean(audio_float**2))
                audio_max = np.max(np.abs(audio_float))
                if self._chunks_processed % 10 == 0:  # Log every 10th chunk
                    logger.debug(f"📊 Audio stats - RMS: {audio_rms:.4f}, Max: {audio_max:.4f}, Samples: {len(audio_float)}")
            
            # Accumulate small chunks until we have enough for Parakeet
            self._small_chunk_buffer.append(audio_float)
            self._small_chunk_samples += len(audio_float)
            
            # Process when we have enough samples - we need MORE buffering for finalization
            # With subsampling_factor=4, 256 frames = 1024 mel frames = ~16384 audio samples
            # But we'll process at 3200 samples (200ms) for better incremental results
            if self._small_chunk_samples >= 3200:
                # Concatenate all buffered chunks
                combined_audio = np.concatenate(self._small_chunk_buffer)
                
                # Send the combined chunk for processing
                if not self._audio_queue.full():
                    self._audio_queue.put(combined_audio)
                    logger.debug(f"ParakeetMLX: Sent buffered chunk ({len(combined_audio)} samples)")
                else:
                    # Queue is full - drop oldest chunks and add new one
                    try:
                        self._audio_queue.get_nowait()  # Drop oldest
                        self._audio_queue.put(combined_audio, block=False)
                        logger.debug(f"Queue overflow: replaced old chunk with new buffered chunk")
                    except:
                        logger.warning("Failed to add buffered chunk")
                
                # Clear buffer
                self._small_chunk_buffer = []
                self._small_chunk_samples = 0
            else:
                logger.debug(f"ParakeetMLX: Buffering ({self._small_chunk_samples}/3200 samples)")
            
            # Check for results with small delay to allow processing
            await asyncio.sleep(0.01)  # Allow processing time
            
            results_processed = 0
            max_wait_time = 2.0  # Maximum wait time for results
            wait_start = asyncio.get_event_loop().time()
            
            # Wait for results with timeout
            while results_processed < 5 and (asyncio.get_event_loop().time() - wait_start) < max_wait_time:
                try:
                    text, is_final, confidence = self._result_queue.get_nowait()
                    logger.debug(f"📤 Got result from queue: '{text}' (final: {is_final})")
                    
                    if text:
                        await self.start_ttfb_metrics()
                        
                        if is_final:
                            logger.debug(f"ParakeetMLX final: {text} (conf: {confidence:.2f})")
                            yield TranscriptionFrame(
                                text,
                                self._user_id,
                                time_now_iso8601(),
                                self._parakeet_language,
                            )
                        elif self.emit_partial_results:
                            logger.debug(f"ParakeetMLX partial: {text} (conf: {confidence:.2f})")
                            yield InterimTranscriptionFrame(
                                text,
                                self._user_id,
                                time_now_iso8601(),
                                self._parakeet_language,
                            )
                        
                        await self.stop_ttfb_metrics()
                        results_processed += 1
                        
                except Empty:
                    # No results available yet, wait a bit more
                    if results_processed == 0:
                        await asyncio.sleep(0.1)
                    else:
                        break
                    
        except Exception as e:
            logger.error(f"Error in ParakeetMLX transcription: {e}")
            raise
        finally:
            await self.stop_processing_metrics()

    async def flush(self):
        """Flush any remaining audio and get final results"""
        try:
            # First, process any remaining buffered audio
            if self._small_chunk_buffer and self._small_chunk_samples > 0:
                logger.debug(f"🔄 Flushing buffered audio: {self._small_chunk_samples} samples")
                
                # Pad the buffer to minimum size if needed
                combined_audio = np.concatenate(self._small_chunk_buffer)
                if len(combined_audio) < 3200:
                    # Pad with silence to meet minimum requirement for better results
                    padding_needed = 3200 - len(combined_audio)
                    padding = np.zeros(padding_needed, dtype=np.float32)
                    combined_audio = np.concatenate([combined_audio, padding])
                    logger.debug(f"📊 Padded audio with {padding_needed} samples of silence")
                
                # Process the final chunk
                try:
                    self._audio_queue.put(combined_audio)
                    # Wait for processing
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Error processing final buffered audio: {e}")
                
                # Clear the buffer
                self._small_chunk_buffer = []
                self._small_chunk_samples = 0
            
            if self._transcriber and self._stream_active:
                logger.debug("🔄 Flushing remaining results from queue")
                
                # Wait a bit for any final processing
                await asyncio.sleep(0.5)
                
                # Yield any remaining results in the queue
                final_results = []
                while not self._result_queue.empty():
                    try:
                        text, _, _ = self._result_queue.get_nowait()
                        if text and text.strip():
                            final_results.append(text.strip())
                            logger.debug(f"📤 Flush found result: '{text}'")
                    except Empty:
                        break
                
                if final_results:
                    combined_text = " ".join(final_results)
                    logger.info(f"🎯 Final flush result: '{combined_text}'")
                    return combined_text
                else:
                    logger.debug("🔍 No additional results found during flush")
                            
        except Exception as e:
            logger.error(f"Error during flush: {e}")
        
        return None

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop processing task
            if self._processing_task:
                self._stream_active = False
                self._audio_queue.put(None)  # Shutdown signal
                await self._processing_task
                self._processing_task = None
            
            # Stop streaming
            self._stop_streaming()
            
            # Clear queues
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except Empty:
                    break
                    
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except Empty:
                    break
            
            logger.info("🧹 Parakeet-MLX STT cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Ensure cleanup on destruction"""
        try:
            if self._transcriber:
                self._stop_streaming()
        except:
            pass