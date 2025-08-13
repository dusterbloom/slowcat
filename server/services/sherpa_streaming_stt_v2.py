# server/services/sherpa_streaming_stt_v2.py
"""
🔥 PROPER Sherpa-ONNX Streaming STT Service using OnlineRecognizer!
This is the REAL streaming implementation - no more OfflineRecognizer hacks!
"""

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

# Lazy import function to avoid global state issues with multiprocessing
def _require_sherpa():
    """Import sherpa_onnx safely without global state"""
    try:
        import sherpa_onnx
        return sherpa_onnx
    except ImportError as e:
        raise RuntimeError(f"sherpa_onnx not available: {e}") from e


class SherpaOnlineSTTService(STTService):
    """
    🚀 TRUE Sherpa-ONNX Streaming STT Service using OnlineRecognizer!
    
    Uses the proper streaming API with:
    - OnlineRecognizer for real-time processing
    - Stream.accept_waveform() for continuous audio feeding
    - Endpoint detection for natural speech boundaries
    - Partial results during speech
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "en",
        sample_rate: int = 16000,
        enable_endpoint_detection: bool = True,
        max_active_paths: int = 4,
        decoding_method: str = "greedy_search",
        provider: str = "cpu",
        num_threads: int = 1,
        # Streaming specific params  
        chunk_size_ms: int = 200,  # Process every 200ms 
        emit_partial_results: bool = False,
        hotwords_file: str = "",
        hotwords_score: float = 1.5,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.model_dir = Path(model_dir)
        self.language = language
        self.enable_endpoint_detection = enable_endpoint_detection
        self.max_active_paths = max_active_paths
        self.decoding_method = decoding_method
        self.provider = provider
        self.num_threads = num_threads
        self.emit_partial_results = emit_partial_results
        self.hotwords_file = hotwords_file
        self.hotwords_score = hotwords_score
        
        # Calculate chunk size in samples (ensure minimum size)
        self.chunk_size = max(int(sample_rate * chunk_size_ms / 1000), 1600)  # At least 100ms
        
        # Streaming state
        self._recognizer = None  # Will be initialized lazily
        self._stream = None
        self._audio_buffer = bytearray()
        self._last_result = ""
        self._processing_lock = threading.RLock()  # Use RLock for nested locking
        self._initialization_attempted = False
        self._metrics_started = False  # Track if metrics are currently active

        # Validate model directory
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

        # Initialize recognizer lazily to avoid import-time issues
        logger.info(f"🚀 Sherpa OnlineRecognizer ready - chunk: {chunk_size_ms}ms, endpoint_detection: {enable_endpoint_detection}")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sherpa service supports metrics generation.
        """
        return True

    def _init_recognizer(self):
        """Initialize the OnlineRecognizer with proper model configuration"""
        try:
            sherpa = _require_sherpa()
            logger.debug("✅ sherpa_onnx imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import sherpa-onnx: {e}")
            raise RuntimeError(f"Failed to import sherpa-onnx: {e}") from e
        
        # Check for different model types in the directory
        model_files = list(self.model_dir.glob("*.onnx"))
        tokens_file = self.model_dir / "tokens.txt"
        
        if not tokens_file.exists():
            raise FileNotFoundError(f"tokens.txt not found in {self.model_dir}")
        
        if not model_files:
            raise FileNotFoundError(f"No ONNX model files found in {self.model_dir}")
        
        logger.info(f"🔍 Found model files: {[f.name for f in model_files]}")
        
        try:
            # Find model files with pattern matching (handles different naming conventions)
            encoder_files = list(self.model_dir.glob("encoder*.onnx"))
            decoder_files = list(self.model_dir.glob("decoder*.onnx"))
            joiner_files = list(self.model_dir.glob("joiner*.onnx"))
            model_file = self.model_dir / "model.onnx"
            
            # Prefer int8 versions for stability (they're often more stable)
            def prefer_quantized(files):
                int8_files = [f for f in files if 'int8' in f.name]
                non_int8 = [f for f in files if 'int8' not in f.name]
                # Try int8 first for stability, fallback to full precision
                return int8_files[0] if int8_files else (non_int8[0] if non_int8 else None)
            
            encoder_file = prefer_quantized(encoder_files)
            decoder_file = prefer_quantized(decoder_files) 
            joiner_file = prefer_quantized(joiner_files)
            
            logger.info(f"🔍 Found transducer files: encoder={encoder_file.name if encoder_file else None}, decoder={decoder_file.name if decoder_file else None}, joiner={joiner_file.name if joiner_file else None}")
            
            if encoder_file and decoder_file and joiner_file:
                logger.info("🎯 Creating transducer OnlineRecognizer")
                
                # Use the factory method with proper parameters from API docs
                logger.info("🎯 Creating transducer OnlineRecognizer with factory method")
                self._recognizer = sherpa.OnlineRecognizer.from_transducer(
                    tokens=str(tokens_file),
                    encoder=str(encoder_file),
                    decoder=str(decoder_file),
                    joiner=str(joiner_file),
                    num_threads=1,  # Single thread for stability
                    sample_rate=self.sample_rate,
                    feature_dim=80,
                    # Endpoint detection parameters (optimized)
                    enable_endpoint_detection=self.enable_endpoint_detection,
                    rule1_min_trailing_silence=1.5,
                    rule2_min_trailing_silence=0.8, 
                    rule3_min_utterance_length=300,
                    # Decoding parameters  
                    decoding_method=self.decoding_method,
                    max_active_paths=min(self.max_active_paths, 8),  # Allow more paths for better accuracy
                    blank_penalty=0.0,
                    temperature_scale=2.0,
                    hotwords_score=self.hotwords_score,
                    hotwords_file=self.hotwords_file,
                    # Skip modeling_unit due to known segfault issue #1536 with bilingual models
                    # Stability settings
                    provider="cpu",  # Force CPU for stability
                    debug=False,
                )
            elif model_file.exists():
                # 🔥 NEMO CTC MODEL - Use direct configuration approach!
                logger.info("🎯 Creating Nemo CTC OnlineRecognizer")
                
                # Create configuration for Nemo CTC model
                nemo_ctc_config = sherpa.OnlineNemoEncDecCtcModelConfig(
                    model=str(model_file),
                )
                
                model_config = sherpa.OnlineModelConfig(
                    nemo_ctc=nemo_ctc_config,
                    tokens=str(tokens_file),
                    num_threads=self.num_threads,
                    provider=self.provider,
                    model_type="nemo_ctc",
                    debug=False,
                )
                
                feat_config = sherpa.FeatureConfig(
                    sample_rate=self.sample_rate,
                    feature_dim=80,
                )
                
                recognizer_config = sherpa.OnlineRecognizerConfig(
                    feat_config=feat_config,
                    model_config=model_config,
                    enable_endpoint_detection=self.enable_endpoint_detection,
                    max_active_paths=self.max_active_paths,
                    decoding_method=self.decoding_method,
                )
                
                self._recognizer = sherpa.OnlineRecognizer(recognizer_config)
                
            else:
                raise RuntimeError(f"Could not determine model type. Found files: {[f.name for f in model_files]}")
            
            logger.info("✅ Sherpa OnlineRecognizer created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create OnlineRecognizer: {e}")
            raise

    def _create_stream(self):
        """Create a new recognition stream with error handling"""
        if not self._recognizer:
            raise RuntimeError("Recognizer not initialized")
        try:
            stream = self._recognizer.create_stream()
            return stream
        except Exception as e:
            logger.error(f"Failed to create stream: {e}")
            raise RuntimeError(f"Stream creation failed: {e}")

    def _process_audio_chunk(self, audio_samples: np.ndarray) -> tuple[str, bool]:
        """
        Process audio chunk and return (partial_result, is_speech_endpoint)
        
        Returns:
            tuple: (transcription_text, is_speech_endpoint)
        """
        if not self._recognizer or not self._stream:
            logger.warning("Recognizer or stream not available")
            return "", False
        
        try:
            # Validate audio samples
            if audio_samples is None or len(audio_samples) == 0:
                return "", False
                
            # Ensure reasonable sample count (at least 10ms worth)
            min_samples = int(self.sample_rate * 0.01)  # 10ms minimum
            if len(audio_samples) < min_samples:
                logger.debug(f"Audio chunk too short: {len(audio_samples)} samples")
                return "", False
            
            # Validate audio range and clean
            audio_samples = np.clip(audio_samples, -1.0, 1.0)
            if np.any(np.isnan(audio_samples)) or np.any(np.isinf(audio_samples)):
                logger.warning("Invalid audio samples, skipping")
                return "", False
            
            # Convert to list safely
            try:
                audio_list = audio_samples.tolist()
            except Exception as e:
                logger.error(f"Failed to convert audio to list: {e}")
                return "", False
            
            # Accept waveform data 
            self._stream.accept_waveform(self.sample_rate, audio_list)
            
            # Process available data
            if self._recognizer.is_ready(self._stream):
                self._recognizer.decode_streams([self._stream])
            
            # Check if this is an endpoint
            is_endpoint = self._recognizer.is_endpoint(self._stream)
            
            # Get current result (handle both string and object types)
            result_obj = self._recognizer.get_result(self._stream)
            result = result_obj.text if hasattr(result_obj, 'text') else str(result_obj)
            
            # Log and handle endpoint
            if is_endpoint:
                if result.strip():
                    logger.debug(f"🎯 Sherpa FINAL: '{result}' (endpoint detected)")
                self._recognizer.reset(self._stream)
                # logger.debug("🔄 Stream reset after endpoint")
            elif result:
                logger.debug(f"🔄 Sherpa interim: '{result}'")
            
            return result.strip(), is_endpoint
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            # Don't try to recover here - let upper level handle it
            return "", False

    def _ensure_recognizer_initialized(self):
        """Lazy initialization of recognizer to avoid import-time issues"""
        if not self._recognizer:
            try:
                logger.info("🔄 Initializing Sherpa OnlineRecognizer...")
                self._init_recognizer()
                logger.info("✅ Sherpa OnlineRecognizer initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize recognizer: {e}")
                # Don't keep trying if it fails
                self._recognizer = "FAILED"
                raise RuntimeError(f"Sherpa initialization failed: {e}") from e
        elif self._recognizer == "FAILED":
            # Reset and retry
            logger.info("🔄 Retrying failed recognizer initialization...")
            self._recognizer = None
            return self._ensure_recognizer_initialized()

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        🚀 TRUE STREAMING STT: Process audio continuously with OnlineRecognizer!
        """
        if not audio or len(audio) == 0:
            return

        with self._processing_lock:
            # Lazy initialization
            try:
                self._ensure_recognizer_initialized()
            except Exception as e:
                logger.error(f"Failed to initialize recognizer: {e}")
                return
            
            # Initialize stream if needed (with retry)
            if not self._stream:
                try:
                    self._stream = self._create_stream()
                except Exception as e:
                    logger.error(f"Failed to create initial stream: {e}")
                    await self.stop_processing_metrics()
                    return
            
            # Add audio to buffer
            self._audio_buffer.extend(audio)
            
            # Process if we have enough audio (but not too much to prevent overflow)
            max_buffer_size = self.chunk_size * 10 * 2  # 10 chunks max
            if len(self._audio_buffer) > max_buffer_size:
                # Truncate buffer if it gets too large
                self._audio_buffer = self._audio_buffer[-max_buffer_size:]
                logger.warning("Audio buffer truncated to prevent overflow")
            
            if len(self._audio_buffer) >= self.chunk_size * 2:  # 2 bytes per int16 sample
                
                # Convert bytes to samples
                try:
                    samples_needed = self.chunk_size
                    pcm_data = np.frombuffer(
                        self._audio_buffer[:samples_needed * 2], 
                        dtype=np.int16
                    )
                    audio_samples = (pcm_data.astype(np.float32)) / 32768.0
                    
                    # Remove processed samples from buffer  
                    self._audio_buffer = self._audio_buffer[samples_needed * 2:]
                    
                except Exception as e:
                    logger.error(f"Failed to convert audio: {e}")
                    await self.stop_processing_metrics()
                    return
                
                # Start metrics if not already started
                if not self._metrics_started:
                    await self.start_processing_metrics()
                    await self.start_ttfb_metrics()
                    self._metrics_started = True
                
                # Process the audio chunk synchronously (thread-safe)
                try:
                    result_text, is_endpoint = self._process_audio_chunk(audio_samples)
                except Exception as e:
                    logger.error(f"Audio processing failed: {e}")
                    # Reset stream on error to prevent corruption
                    try:
                        if self._stream:
                            self._stream = self._create_stream()
                    except:
                        pass
                    if self._metrics_started:
                        await self.stop_ttfb_metrics()
                        await self.stop_processing_metrics()
                        self._metrics_started = False
                    return
                
                # Emit frames based on results
                if result_text and is_endpoint:
                    # 🎯 FINAL TRANSCRIPTION - Always emit on endpoint
                    logger.info(f"✅ Sherpa final: '{result_text}' (endpoint detected)")
                    
                    yield TranscriptionFrame(
                        text=result_text,
                        user_id="default_user",
                        timestamp=time_now_iso8601(),
                        language=self.language,
                    )
                    self._last_result = result_text
                    
                    # Stop metrics after successful transcription
                    if self._metrics_started:
                        await self.stop_ttfb_metrics()
                        await self.stop_processing_metrics()
                        self._metrics_started = False
                        
                elif result_text and self.emit_partial_results and result_text != self._last_result and len(result_text.split()) >= 1:
                    # 🔄 INTERIM TRANSCRIPTION - Only if text changed
                    logger.debug(f"🔄 Sherpa interim: '{result_text}'")
                    
                    yield InterimTranscriptionFrame(
                        text=result_text,
                        user_id="default_user",
                        timestamp=time_now_iso8601(),
                        language=self.language,
                    )
                    self._last_result = result_text
                    # Keep metrics running for interim results

    def cleanup(self):
        """Cleanup recognizer resources safely"""
        logger.info("🧹 Cleaning up Sherpa OnlineRecognizer")
        with self._processing_lock:
            # Stop metrics if they're running
            if self._metrics_started:
                try:
                    # These are async methods but we're in sync context
                    # The metrics system should handle this gracefully
                    self._metrics_started = False
                except:
                    pass
            
            # Clear buffer first
            self._audio_buffer.clear()
            
            # Cleanup stream
            if self._stream:
                try:
                    del self._stream
                except Exception as e:
                    logger.debug(f"Stream cleanup warning: {e}")
                finally:
                    self._stream = None
            
            # Cleanup recognizer
            if self._recognizer and self._recognizer != "FAILED":
                try:
                    del self._recognizer
                except Exception as e:
                    logger.debug(f"Recognizer cleanup warning: {e}")
                finally:
                    self._recognizer = None
            
            self._initialization_attempted = False
        logger.info("✅ Sherpa OnlineRecognizer cleaned up")

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass