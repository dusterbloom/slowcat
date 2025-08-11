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

        # Validate model directory
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

        # Initialize recognizer lazily to avoid import-time issues
        logger.info(f"🚀 Sherpa OnlineRecognizer ready - chunk: {chunk_size_ms}ms, endpoint_detection: {enable_endpoint_detection}")

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
            
            # Check if this is a SenseVoice model
            is_sense_voice = any(
                'sense-voice' in f.name.lower() or 
                'sensevoice' in f.name.lower() or
                'sense_voice' in f.name.lower()
                for f in model_files
            ) or 'sense-voice' in str(self.model_dir).lower()
            
            # Check if this is a Nemo CTC model (single model file)
            # Expanded detection for various CTC model naming conventions
            is_nemo_ctc = any(
                'nemo' in f.name.lower() or 
                'ctc' in f.name.lower() or 
                'parakeet' in f.name.lower() or
                'model' in f.name.lower()  # Generic model files are often CTC
                for f in model_files
            )
            
            if is_sense_voice and model_files:
                # Handle SenseVoice models - these are non-streaming CTC models with excellent accuracy
                main_model_file = prefer_quantized(model_files)
                if not main_model_file:
                    main_model_file = model_files[0]
                    
                logger.info(f"🎯 Creating SenseVoice OnlineRecognizer with model file: {main_model_file.name}")
                logger.info("📈 SenseVoice provides excellent accuracy for URLs, technical terms, and proper names")
                
                # SenseVoice models are CTC-based, so we use from_nemo_ctc
                # but they may need different parameters
                try:
                    self._recognizer = sherpa.OnlineRecognizer.from_nemo_ctc(
                        tokens=str(tokens_file),
                        model=str(main_model_file),
                        num_threads=self.num_threads,
                        sample_rate=self.sample_rate,
                        feature_dim=80,
                        enable_endpoint_detection=self.enable_endpoint_detection,
                        decoding_method=self.decoding_method,
                        provider=self.provider,
                        debug=False,
                    )
                    logger.info("✅ SenseVoice model initialized successfully with CTC method")
                    
                except Exception as sense_error:
                    logger.warning(f"⚠️ SenseVoice CTC initialization failed: {sense_error}")
                    # Try alternative initialization methods for SenseVoice
                    logger.info("🔄 Trying alternative initialization methods for SenseVoice...")
                    
                    # Try Paraformer method as SenseVoice might be compatible
                    try:
                        logger.info("🔄 Trying Paraformer initialization...")
                        self._recognizer = sherpa.OnlineRecognizer.from_paraformer(
                            tokens=str(tokens_file),
                            model=str(main_model_file),
                            num_threads=self.num_threads,
                            sample_rate=self.sample_rate,
                            feature_dim=80,
                            enable_endpoint_detection=self.enable_endpoint_detection,
                            provider=self.provider,
                            debug=False,
                        )
                        logger.info("✅ SenseVoice initialized successfully with Paraformer method")
                        
                    except Exception as para_error:
                        logger.warning(f"⚠️ Paraformer method also failed: {para_error}")
                        
                        # Final fallback: treat as generic CTC model
                        logger.info("🔄 Final fallback: using generic CTC initialization...")
                        try:
                            self._recognizer = sherpa.OnlineRecognizer.from_zipformer2_ctc(
                                tokens=str(tokens_file),
                                model=str(main_model_file),
                                num_threads=self.num_threads,
                                sample_rate=self.sample_rate,
                                feature_dim=80,
                                enable_endpoint_detection=self.enable_endpoint_detection,
                                decoding_method=self.decoding_method,
                                provider=self.provider,
                                debug=False,
                            )
                            logger.info("✅ SenseVoice initialized with Zipformer2 CTC method")
                        except Exception as zip_error:
                            logger.error(f"❌ All SenseVoice initialization methods failed")
                            logger.error(f"   CTC error: {sense_error}")
                            logger.error(f"   Paraformer error: {para_error}")
                            logger.error(f"   Zipformer2 error: {zip_error}")
                            raise RuntimeError(f"Failed to initialize SenseVoice model with any method") from sense_error
                        
            elif encoder_file and decoder_file and joiner_file:
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
            elif is_nemo_ctc and model_files:
                # Handle Nemo CTC models (including Parakeet models)
                # Find the main model file (prefer int8 version)
                main_model_file = prefer_quantized(model_files)
                if not main_model_file:
                    main_model_file = model_files[0]
                    
                logger.info(f"🎯 Creating Nemo CTC OnlineRecognizer with model file: {main_model_file.name}")
                
                # Use from_nemo_ctc for Nemo CTC models
                # Try different initialization methods if metadata is missing
                try:
                    self._recognizer = sherpa.OnlineRecognizer.from_nemo_ctc(
                        tokens=str(tokens_file),
                        model=str(main_model_file),
                        num_threads=self.num_threads,
                        sample_rate=self.sample_rate,
                        feature_dim=80,
                        enable_endpoint_detection=self.enable_endpoint_detection,
                        decoding_method=self.decoding_method,
                        provider=self.provider,
                        debug=False,
                    )
                except Exception as nemo_error:
                    error_msg = str(nemo_error)
                    if "window_size" in error_msg or "vocab_size" in error_msg or "context_size" in error_msg:
                        logger.warning(f"⚠️ Nemo CTC model missing metadata, trying fallback initialization: {error_msg}")
                        
                        # Try using OfflineRecognizer as a fallback for models with missing metadata
                        # This works in test scripts but may have limitations for streaming
                        try:
                            logger.info("🔄 Attempting fallback with OfflineRecognizer wrapper...")
                            
                            # Import OfflineRecognizer if not already imported
                            if not hasattr(sherpa, 'OfflineRecognizer'):
                                logger.error("OfflineRecognizer not available in sherpa_onnx")
                                raise nemo_error
                            
                            # Create an offline recognizer first
                            offline_recognizer = sherpa.OfflineRecognizer.from_nemo_ctc(
                                tokens=str(tokens_file),
                                model=str(main_model_file),
                                num_threads=self.num_threads,
                                sample_rate=self.sample_rate,
                                feature_dim=80,
                                decoding_method=self.decoding_method,
                                provider=self.provider,
                                debug=False,
                            )
                            
                            # Wrap it for streaming (this is a workaround)
                            logger.warning("⚠️ Using OfflineRecognizer in streaming mode - may have higher latency")
                            logger.warning("For better streaming performance, consider using Zipformer or SenseVoice models")
                            
                            # Create a simple wrapper that mimics OnlineRecognizer interface
                            class OfflineToOnlineWrapper:
                                def __init__(self, offline_rec, sample_rate):
                                    self.offline_rec = offline_rec
                                    self.sample_rate = sample_rate
                                    self.accumulated_samples = []
                                
                                def create_stream(self):
                                    return self
                                
                                def accept_waveform(self, sample_rate, samples):
                                    self.accumulated_samples.extend(samples)
                                
                                def is_ready(self, stream):
                                    # Process when we have enough samples (e.g., 1 second)
                                    return len(self.accumulated_samples) >= self.sample_rate
                                
                                def decode_streams(self, streams):
                                    pass  # Processing happens in get_result
                                
                                def is_endpoint(self, stream):
                                    # Simple endpoint detection based on accumulated samples
                                    return len(self.accumulated_samples) >= self.sample_rate * 2
                                
                                def get_result(self, stream):
                                    if not self.accumulated_samples:
                                        return type('Result', (), {'text': ''})()
                                    
                                    # Process accumulated samples with offline recognizer
                                    import numpy as np
                                    samples_array = np.array(self.accumulated_samples, dtype=np.float32)
                                    
                                    # Create offline stream and process
                                    offline_stream = self.offline_rec.create_stream()
                                    offline_stream.accept_waveform(self.sample_rate, samples_array.tolist())
                                    self.offline_rec.decode_stream(offline_stream)
                                    result = self.offline_rec.get_result(offline_stream)
                                    return result
                                
                                def reset(self, stream):
                                    self.accumulated_samples = []
                            
                            self._recognizer = OfflineToOnlineWrapper(offline_recognizer, self.sample_rate)
                            logger.info("✅ Fallback initialization successful using OfflineRecognizer wrapper")
                            
                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback initialization also failed: {fallback_error}")
                            logger.error("Solutions:")
                            logger.error("  1. Use SenseVoice model for better accuracy:")
                            logger.error("     - Download: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
                            logger.error("  2. Use Zipformer model (already installed):")
                            logger.error("     - sherpa-onnx-streaming-zipformer-en-2023-06-26")
                            logger.error("  3. Re-export the Parakeet model with proper metadata")
                            raise RuntimeError(f"Failed to create Nemo CTC recognizer: {nemo_error}") from nemo_error
                    else:
                        raise RuntimeError(f"Failed to create Nemo CTC recognizer: {nemo_error}") from nemo_error
            elif model_files:
                # Handle any single model file (fallback for models that don't match other criteria)
                # Find the main model file (prefer int8 version)
                main_model_file = prefer_quantized(model_files)
                if not main_model_file:
                    main_model_file = model_files[0]
                    
                logger.info(f"🎯 Creating generic OnlineRecognizer with model file: {main_model_file.name}")
                
                # Try different factory methods based on model name patterns
                model_name = main_model_file.name.lower()
                
                try:
                    if 'nemo' in model_name or 'ctc' in model_name or 'parakeet' in model_name or 'model' in model_name:
                        # Try Nemo CTC first for models that match these patterns
                        logger.info("Trying from_nemo_ctc for generic model")
                        self._recognizer = sherpa.OnlineRecognizer.from_nemo_ctc(
                            tokens=str(tokens_file),
                            model=str(main_model_file),
                            num_threads=self.num_threads,
                            sample_rate=self.sample_rate,
                            feature_dim=80,
                            enable_endpoint_detection=self.enable_endpoint_detection,
                            decoding_method=self.decoding_method,
                            provider=self.provider,
                            debug=False,
                        )
                    elif 'paraformer' in model_name:
                        # Try Paraformer
                        logger.info("Trying from_paraformer for generic model")
                        self._recognizer = sherpa.OnlineRecognizer.from_paraformer(
                            tokens=str(tokens_file),
                            model=str(main_model_file),
                            num_threads=self.num_threads,
                            sample_rate=self.sample_rate,
                            feature_dim=80,
                            enable_endpoint_detection=self.enable_endpoint_detection,
                            provider=self.provider,
                            debug=False,
                        )
                    elif 'wenet' in model_name:
                        # Try WeNet CTC
                        logger.info("Trying from_wenet_ctc for generic model")
                        self._recognizer = sherpa.OnlineRecognizer.from_wenet_ctc(
                            tokens=str(tokens_file),
                            model=str(main_model_file),
                            num_threads=self.num_threads,
                            sample_rate=self.sample_rate,
                            feature_dim=80,
                            enable_endpoint_detection=self.enable_endpoint_detection,
                            decoding_method=self.decoding_method,
                            provider=self.provider,
                            debug=False,
                        )
                    elif 'zipformer' in model_name:
                        # Try Zipformer2 CTC
                        logger.info("Trying from_zipformer2_ctc for generic model")
                        self._recognizer = sherpa.OnlineRecognizer.from_zipformer2_ctc(
                            tokens=str(tokens_file),
                            model=str(main_model_file),
                            num_threads=self.num_threads,
                            sample_rate=self.sample_rate,
                            feature_dim=80,
                            enable_endpoint_detection=self.enable_endpoint_detection,
                            decoding_method=self.decoding_method,
                            provider=self.provider,
                            debug=False,
                        )
                    else:
                        # Fall back to Nemo CTC for any other model
                        logger.info("Falling back to from_nemo_ctc for generic model")
                        self._recognizer = sherpa.OnlineRecognizer.from_nemo_ctc(
                            tokens=str(tokens_file),
                            model=str(main_model_file),
                            num_threads=self.num_threads,
                            sample_rate=self.sample_rate,
                            feature_dim=80,
                            enable_endpoint_detection=self.enable_endpoint_detection,
                            decoding_method=self.decoding_method,
                            provider=self.provider,
                            debug=False,
                        )
                except Exception as e:
                    error_msg = str(e)
                    if "window_size" in error_msg:
                        logger.error(f"❌ Model missing required metadata: {error_msg}")
                        logger.error("This model was not exported with all required metadata.")
                        logger.error("Solutions:")
                        logger.error("  1. Use a different model type such as Zipformer:")
                        logger.error("     - sherpa-onnx-streaming-zipformer-en-2023-06-26")
                        logger.error("  2. Update your .env file to point to a working model directory")
                        logger.error("  3. Re-export the model with proper metadata")
                        logger.error("")
                        logger.error("To fix this issue, update your server/.env file:")
                        logger.error("Change: SHERPA_ONNX_MODEL_DIR=./models/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8")
                        logger.error("To:     SHERPA_ONNX_MODEL_DIR=./models/sherpa-onnx-streaming-zipformer-en-2023-06-26")
                    raise RuntimeError(f"Could not create OnlineRecognizer for model {main_model_file.name}: {e}") from e
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
            await self.start_processing_metrics()
            
            # Lazy initialization
            try:
                self._ensure_recognizer_initialized()
            except Exception as e:
                logger.error(f"Failed to initialize recognizer: {e}")
                await self.stop_processing_metrics()
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
                    await self.stop_processing_metrics()
                    return
                
                # Emit frames based on results
                if result_text:
                    if is_endpoint:
                        # 🎯 FINAL TRANSCRIPTION - Always emit on endpoint
                        logger.info(f"✅ Sherpa final: '{result_text}' (endpoint detected)")
                        await self.stop_ttfb_metrics()
                        
                        yield TranscriptionFrame(
                            text=result_text,
                            user_id="default_user",
                            timestamp=time_now_iso8601(),
                            language=self.language,
                        )
                        self._last_result = result_text
                        
                    elif self.emit_partial_results and result_text != self._last_result and len(result_text.split()) >= 1:
                        # 🔄 INTERIM TRANSCRIPTION - Only if text changed
                        logger.debug(f"🔄 Sherpa interim: '{result_text}'")
                        
                        yield InterimTranscriptionFrame(
                            text=result_text,
                            user_id="default_user",
                            timestamp=time_now_iso8601(),
                            language=self.language,
                        )
                        self._last_result = result_text
            
            await self.stop_processing_metrics()

    async def cleanup(self):
        """Cleanup recognizer resources safely"""
        logger.info("🧹 Cleaning up Sherpa OnlineRecognizer")
        with self._processing_lock:
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