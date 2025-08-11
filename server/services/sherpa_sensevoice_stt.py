# server/services/sherpa_sensevoice_stt.py
"""
🔥 SenseVoice STT Service using OfflineRecognizer!
This service is specifically designed for SenseVoice models which provide
excellent accuracy for URLs, technical terms, and proper names.
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


class SherpaVoiceSenseSTTService(STTService):
    """
    🚀 SenseVoice STT Service using OfflineRecognizer!
    
    Uses SenseVoice models for excellent accuracy with:
    - URLs and technical terms
    - Proper names and domain-specific vocabulary
    - Multi-language support (EN, ZH, JA, KO, YUE)
    - High accuracy speech recognition
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "auto",  # SenseVoice supports auto language detection
        sample_rate: int = 16000,
        num_threads: int = 2,
        use_itn: bool = False,  # Inverse Text Normalization
        # Smart adaptive batching params
        chunk_size_ms: int = 800,   # Smaller chunks for better responsiveness
        min_silence_duration_ms: int = 250,  # Faster silence detection
        min_processing_duration_ms: int = 600,  # Minimum before processing (0.6s)
        max_processing_duration_ms: int = 2500, # Maximum accumulation (2.5s)
        speech_end_silence_ms: int = 300,       # Silence to detect speech end
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.model_dir = Path(model_dir)
        self.language = language
        self.num_threads = num_threads
        self.use_itn = use_itn
        self.chunk_size_ms = chunk_size_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_processing_duration_ms = min_processing_duration_ms
        self.max_processing_duration_ms = max_processing_duration_ms
        self.speech_end_silence_ms = speech_end_silence_ms
        
        # Calculate sample counts for adaptive batching
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        self.min_processing_samples = int(sample_rate * min_processing_duration_ms / 1000)
        self.max_processing_samples = int(sample_rate * max_processing_duration_ms / 1000)
        self.speech_end_silence_samples = int(sample_rate * speech_end_silence_ms / 1000)
        
        # Streaming state
        self._recognizer = None  # Will be initialized lazily
        self._audio_buffer = bytearray()
        self._last_result = ""
        self._processing_lock = threading.RLock()
        self._initialization_attempted = False
        self._silence_counter = 0
        self._accumulated_audio = []  # Store audio samples for batching
        self._speech_detected = False  # Track if we're in speech
        self._last_speech_time = 0    # Track when speech last occurred
        self._processing_in_progress = False  # Prevent concurrent processing

        # Validate model directory and files
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")
        
        # Check for required files
        model_file = self.model_dir / "model.onnx"
        model_int8_file = self.model_dir / "model.int8.onnx"
        tokens_file = self.model_dir / "tokens.txt"
        
        if not tokens_file.exists():
            raise FileNotFoundError(f"tokens.txt not found in {self.model_dir}")
        
        if not (model_file.exists() or model_int8_file.exists()):
            raise FileNotFoundError(f"No model files found in {self.model_dir}")

        logger.info(f"🚀 SenseVoice STT ready - chunk: {chunk_size_ms}ms, language: {language}")

    def _init_recognizer(self):
        """Initialize the OfflineRecognizer with SenseVoice model"""
        try:
            sherpa = _require_sherpa()
            logger.debug("✅ sherpa_onnx imported successfully")
        except Exception as e:
            logger.error(f"❌ Failed to import sherpa-onnx: {e}")
            raise RuntimeError(f"Failed to import sherpa-onnx: {e}") from e
        
        # Find model files (prefer int8 for speed)
        model_int8_file = self.model_dir / "model.int8.onnx"
        model_file = self.model_dir / "model.onnx"
        tokens_file = self.model_dir / "tokens.txt"
        
        # Use int8 model if available, otherwise full precision
        if model_int8_file.exists():
            model_path = model_int8_file
            logger.info(f"🎯 Using quantized model: {model_path.name}")
        else:
            model_path = model_file
            logger.info(f"🎯 Using full precision model: {model_path.name}")
        
        try:
            logger.info("🔄 Initializing SenseVoice OfflineRecognizer...")
            
            # Initialize SenseVoice OfflineRecognizer using the correct factory method
            self._recognizer = sherpa.OfflineRecognizer.from_sense_voice(
                model=str(model_path),
                tokens=str(tokens_file),
                num_threads=self.num_threads,
                use_itn=self.use_itn,
                language=self.language,
                debug=False,
            )
            
            logger.info("✅ SenseVoice OfflineRecognizer initialized successfully")
            logger.info("📈 Ready for high-accuracy recognition of URLs, technical terms, and proper names")
            
        except Exception as e:
            logger.error(f"❌ Failed to create SenseVoice OfflineRecognizer: {e}")
            raise

    def _process_audio_chunk(self, audio_samples: np.ndarray) -> tuple[str, bool]:
        """
        Process audio chunk with OfflineRecognizer
        
        Returns:
            tuple: (transcription_text, is_final)
        """
        if not self._recognizer:
            logger.warning("Recognizer not available")
            return "", False
        
        try:
            # Validate audio samples
            if audio_samples is None or len(audio_samples) == 0:
                return "", False
                
            # Ensure reasonable sample count
            if len(audio_samples) < self.min_silence_samples:
                return "", False
            
            # Clean audio samples
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
            
            # Create stream and process with OfflineRecognizer
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio_list)
            
            # Process the stream
            self._recognizer.decode_stream(stream)
            
            # Get result
            result = stream.result
            result_text = result.text if hasattr(result, 'text') else str(result)
            
            # Check if this is significantly different from last result
            is_final = len(result_text.strip()) > 0 and result_text != self._last_result
            
            if is_final:
                logger.info(f"🎯 SenseVoice result: '{result_text}'")
                self._last_result = result_text
            elif result_text:
                logger.debug(f"🔄 SenseVoice processing: '{result_text}'")
            
            return result_text.strip(), is_final
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return "", False

    def _detect_silence(self, audio_samples: np.ndarray) -> bool:
        """Enhanced silence detection to prevent hallucination"""
        if len(audio_samples) == 0:
            return True
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_samples ** 2))
        
        # Much more aggressive silence threshold to prevent hallucination
        silence_threshold = 0.008  # Higher threshold - less sensitive to noise
        is_silence = rms < silence_threshold
        
        # Also check for very low peak amplitude
        peak_amplitude = np.max(np.abs(audio_samples))
        is_very_quiet = peak_amplitude < 0.02  # Very quiet audio
        
        # Check for consistent low energy (typical of background noise)
        energy_variance = np.var(audio_samples ** 2)
        is_consistent_noise = energy_variance < 0.0001  # Very low variance = consistent noise
        
        # Combine all silence indicators
        is_likely_silence = is_silence or (is_very_quiet and is_consistent_noise)
        
        if is_likely_silence:
            self._silence_counter += 1
        else:
            self._silence_counter = 0
        
        # Smart adaptive silence detection (faster response)
        silence_threshold_count = 2  # Only 2 consecutive silent chunks needed
        is_prolonged_silence = self._silence_counter >= silence_threshold_count
        
        # Track speech state for adaptive processing
        if not is_likely_silence:
            self._speech_detected = True
            self._last_speech_time = time.time()
        
        if is_likely_silence:
            # logger.debug(f"🔇 Silence detected: RMS={rms:.6f}, peak={peak_amplitude:.4f}, var={energy_variance:.8f}, count={self._silence_counter}")
            pass  # Comment-only if statement needs pass
        
        return is_prolonged_silence

    def _is_likely_real_speech(self, text: str, audio_duration: float) -> bool:
        """Filter out likely hallucinations based on text characteristics"""
        if not text or not text.strip():
            return False
        
        text = text.strip().lower()
        
        # Filter very short single characters or common hallucination patterns
        if len(text) <= 3:
            # Single characters or very short text is often hallucination  
            common_hallucinations = {'i', 'i.', 'a', 'a.', 'ok', 'the', 'you', 'it', 'so', 'um', 'uh', 'ah', 'oh', 'and', 'but', 'to', 'of', 'in', 'on', 'at', 'is', 'be', 'or', 'as', 'up', 'an', 'my', 'we', 'he', 'me', 'us', 'no', 'go', 'do'}
            if text in common_hallucinations:
                return False
        
        # Filter repetitive patterns and word duplications (often hallucination)
        words = text.split()
        if len(words) > 1:
            # Check for excessive repetition
            unique_words = set(words)
            if len(unique_words) == 1:  # All words are the same
                return False
            
            # Check for word duplications like "latestatest"
            for word in words:
                if len(word) > 6:  # Only check longer words
                    # Simple duplication check (word repeated within itself)
                    mid = len(word) // 2
                    if word[:mid] and word[:mid] in word[mid:]:
                        return False
        
        # Require reasonable speech rate (not too fast, not too slow for amount of audio)
        words_per_second = len(words) / max(audio_duration, 0.1)
        if words_per_second > 10 or (len(words) == 1 and audio_duration > 3):
            # Too fast speech or single word from too much audio = likely hallucination
            return False
        
        # Minimum meaningful length for longer audio
        if audio_duration > 2 and len(text) < 3:
            return False
        
        logger.debug(f"✅ Speech validation passed: '{text}' ({len(words)} words, {words_per_second:.1f} wps)")
        return True
    
    def _should_process_now(self, current_duration: float) -> bool:
        """🚀 Smart decision making: Should we process audio now?"""
        current_samples = len(self._accumulated_audio)
        
        # 1. Minimum duration check (0.6s minimum)
        if current_samples < self.min_processing_samples:
            return False
            
        # 2. Maximum duration check (2.5s maximum - only process if there's actual speech)
        if current_samples >= self.max_processing_samples:
            # Only process on max duration if we detected actual speech
            full_audio = np.array(self._accumulated_audio, dtype=np.float32)
            if not self._detect_silence(full_audio) and self._speech_detected:
                logger.debug(f"⏰ Max duration with speech ({current_duration:.1f}s) - processing")
                return True
            else:
                # Clear silent audio that's been accumulating too long
                # logger.debug(f"🧹 Clearing {current_duration:.1f}s of accumulated silent audio")
                self._accumulated_audio = self._accumulated_audio[-self.sample_rate:]  # Keep last 1s
                self._speech_detected = False
                return False
            
        # 3. Speech-end detection (silence after speech) - ONLY if we actually had speech
        if self._speech_detected and current_samples >= self.min_processing_samples:
            # Check if we have recent silence after speech
            recent_audio = np.array(self._accumulated_audio[-self.speech_end_silence_samples:], dtype=np.float32)
            if len(recent_audio) > 0 and self._detect_silence(recent_audio):
                # Double-check the entire audio has actual speech content
                full_audio = np.array(self._accumulated_audio, dtype=np.float32)
                if not self._detect_silence(full_audio):  # Only if full audio isn't silent
                    logger.debug(f"🎤 Speech end detected ({current_duration:.1f}s) - processing")
                    return True
        
        # 4. Natural pause detection (good chunk size with some speech)
        if (current_samples >= int(self.min_processing_samples * 1.5) and 
            self._speech_detected and 
            current_duration >= 1.0):
            logger.debug(f"⏸️ Natural pause processing ({current_duration:.1f}s)")
            return True
            
        return False

    def _ensure_recognizer_initialized(self):
        """Lazy initialization of recognizer to avoid import-time issues"""
        if not self._recognizer:
            try:
                logger.info("🔄 Initializing SenseVoice OfflineRecognizer...")
                self._init_recognizer()
                logger.info("✅ SenseVoice OfflineRecognizer initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize recognizer: {e}")
                # Don't keep trying if it fails
                self._recognizer = "FAILED"
                raise RuntimeError(f"SenseVoice initialization failed: {e}") from e
        elif self._recognizer == "FAILED":
            # Reset and retry
            logger.info("🔄 Retrying failed recognizer initialization...")
            self._recognizer = None
            return self._ensure_recognizer_initialized()

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        🚀 SenseVoice STT: Process audio with excellent accuracy!
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
            
            # Add audio to buffer
            self._audio_buffer.extend(audio)
            
            # Accumulate audio samples for batch processing
            # OfflineRecognizer works better with longer audio segments
            try:
                # Convert available bytes to samples
                available_samples = len(self._audio_buffer) // 2  # 2 bytes per int16
                if available_samples > 0:
                    pcm_data = np.frombuffer(
                        self._audio_buffer[:available_samples * 2], 
                        dtype=np.int16
                    )
                    audio_samples = (pcm_data.astype(np.float32)) / 32768.0
                    
                    # Add to accumulated audio
                    self._accumulated_audio.extend(audio_samples.tolist())
                    
                    # Clear processed audio from buffer
                    self._audio_buffer = self._audio_buffer[available_samples * 2:]
                    
            except Exception as e:
                logger.error(f"Failed to convert audio: {e}")
                await self.stop_processing_metrics()
                return
            
            # 🚀 SMART ADAPTIVE BATCHING: Process based on speech patterns, not just time
            current_audio_duration = len(self._accumulated_audio) / self.sample_rate
            
            # Multiple triggers for responsive processing:
            should_process = self._should_process_now(current_audio_duration)
            
            if should_process and not self._processing_in_progress:
                self._processing_in_progress = True
                logger.debug(f"🎤 Processing {current_audio_duration:.1f}s of audio")
                
                # Convert accumulated audio to numpy array
                audio_array = np.array(self._accumulated_audio, dtype=np.float32)
                
                # Process the audio directly (smart logic already decided we should process)
                try:
                    result_text, is_final = self._process_audio_chunk(audio_array)
                    
                    # Smart buffer: Keep small overlap for context continuity
                    overlap = min(len(self._accumulated_audio) // 5, int(self.sample_rate * 0.3))  # 20% or 0.3s max
                    if len(self._accumulated_audio) > overlap:
                        self._accumulated_audio = self._accumulated_audio[-overlap:]
                    else:
                        self._accumulated_audio = []
                    
                    self._silence_counter = 0
                    self._speech_detected = False
                    
                    # Filter out likely hallucinations
                    if result_text and is_final:
                        # Additional filtering to prevent hallucination
                        is_likely_real = self._is_likely_real_speech(result_text, len(audio_array)/self.sample_rate)
                        
                        if is_likely_real:
                            # 🎯 FINAL TRANSCRIPTION - High accuracy result
                            logger.info(f"✅ SenseVoice final: '{result_text}' (processed {len(audio_array)/self.sample_rate:.1f}s)")
                            await self.stop_ttfb_metrics()
                            
                            yield TranscriptionFrame(
                                text=result_text,
                                user_id="default_user",
                                timestamp=time_now_iso8601(),
                                language=self.language if self.language != "auto" else "en",
                            )
                        else:
                            logger.debug(f"🚫 Filtered likely hallucination: '{result_text}' (too short/suspicious)")
                    elif result_text:
                        logger.debug(f"🔄 SenseVoice partial: '{result_text}'")
                        
                except Exception as e:
                    logger.error(f"Audio processing failed: {e}")
                    # Clear accumulated audio on error
                    self._accumulated_audio = []
                finally:
                    self._processing_in_progress = False
                    await self.stop_processing_metrics()
                    return
            
            await self.stop_processing_metrics()

    async def cleanup(self):
        """Cleanup recognizer resources safely"""
        logger.info("🧹 Cleaning up SenseVoice OfflineRecognizer")
        with self._processing_lock:
            # Clear buffers first
            self._audio_buffer.clear()
            self._accumulated_audio.clear()
            
            # Cleanup recognizer
            if self._recognizer and self._recognizer != "FAILED":
                try:
                    del self._recognizer
                except Exception as e:
                    logger.debug(f"Recognizer cleanup warning: {e}")
                finally:
                    self._recognizer = None
            
            self._initialization_attempted = False
        logger.info("✅ SenseVoice OfflineRecognizer cleaned up")

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass