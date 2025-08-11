import asyncio
import concurrent.futures
import threading
from typing import AsyncGenerator, Optional
import queue

import numpy as np
from loguru import logger

import mlx.core as mx
from mlx_audio.tts.utils import load_model

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.transcriptions.language import Language

# Import the global MLX lock
from utils.mlx_lock import MLX_GLOBAL_LOCK # NEW IMPORT

class KokoroTTSService(TTSService):
    """Kokoro TTS service implementation using MLX Audio.

    Provides text-to-speech synthesis using Kokoro models running locally
    on Apple Silicon through the mlx-audio library. Uses a separate thread
    for audio generation to avoid blocking the pipeline.
    """

    def _get_voice_language_mapping(self):
        """Map voice names to their respective languages"""
        return {
            # Italian
            'if_sara': 'it', 'im_nicola': 'it',
            # French
            'ff_siwis': 'fr',
            # English (US)
            'af_bella': 'en', 'af_sarah': 'en', 'af_sky': 'en', 'af_alloy': 'en', 
            'af_nova': 'en', 'af_heart': 'en', 'af_jessica': 'en', 'af_kore': 'en',
            'af_nicole': 'en', 'af_river': 'en', 'af_aoede': 'en',
            'am_adam': 'en', 'am_echo': 'en', 'am_michael': 'en', 'am_liam': 'en',
            'am_eric': 'en', 'am_fenrir': 'en', 'am_onyx': 'en', 'am_puck': 'en',
            # English (UK)
            'bf_alice': 'en', 'bf_emma': 'en', 'bf_isabella': 'en', 'bf_lily': 'en',
            'bm_daniel': 'en', 'bm_george': 'en', 'bm_lewis': 'en', 'bm_fable': 'en',
            # Japanese
            'jf_alpha': 'ja', 'jf_gongitsune': 'ja', 'jf_nezumi': 'ja', 'jf_tebukuro': 'ja',
            'jm_kumo': 'ja',
            # Chinese
            'zf_xiaobei': 'zh', 'zf_xiaoni': 'zh', 'zf_xiaoxiao': 'zh', 'zf_xiaoyi': 'zh',
            'zm_yunxi': 'zh', 'zm_yunxia': 'zh', 'zm_yunyang': 'zh', 'zm_yunjian': 'zh',
            # Spanish
            'ef_dora': 'es', 'em_alex': 'es', 'em_santa': 'es',
            # Portuguese
            'pf_dora': 'pt', 'pm_alex': 'pt', 'pm_santa': 'pt'
        }

    def _get_language_from_voice(self, voice_name):
        """Get the language code for a given voice"""
        voice_lang_map = self._get_voice_language_mapping()
        return voice_lang_map.get(voice_name, 'en')  # Default to English

    def _get_kokoro_language_code(self, lang_code):
        """Convert our language codes to Kokoro's expected language codes"""
        # Based on Kokoro pipeline code - uses single letter codes
        kokoro_lang_map = {
            'en': 'a',      # American English 
            'it': 'i',      # Italian
            'fr': 'f',      # French (fr-fr)
            'es': 'e',      # Spanish
            'ja': 'j',      # Japanese
            'zh': 'z',      # Mandarin Chinese
            'pt': 'p',      # Portuguese (Brazilian)
            'de': 'a',      # German not available, fallback to English
        }
        return kokoro_lang_map.get(lang_code, 'a')  # Default to American English

    def __init__(
        self,
        *,
        model: str = "prince-canuma/Kokoro-82M",
        voice: str = "af_heart",
        device: Optional[str] = None,
        sample_rate: int = 24000,
        max_workers: int = 1,
        language: Optional[Language] = None,
        **kwargs,
    ):
        """Initialize the Kokoro TTS service.

        Args:
            model: The Kokoro model to use (default: "prince-canuma/Kokoro-82M").
            voice: The voice to use for synthesis (default: "af_heart").
            device: The device to run on (None for default MLX device).
            sample_rate: Output sample rate (default: 24000).
            max_workers: Number of threads for audio generation (default: 1).
                         For Metal safety, this is effectively forced to 1.
            language: The language for synthesis (optional, auto-detected from voice).
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._device = device
        # Ensure max_workers is always 1 for Metal safety with this approach
        if max_workers != 1:
            logger.warning(f"KokoroTTSService: max_workers set to {max_workers}, but forcing to 1 for Metal safety.")
            max_workers = 1
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        # self._generation_lock = threading.Lock() # No longer strictly needed with global lock and max_workers=1
        
        # Auto-detect language from voice if not provided
        if language is None:
            self._language_code = self._get_language_from_voice(voice)
        else:
            # Convert Language enum to string code
            lang_map = {
                Language.EN: 'en', Language.IT: 'it', Language.FR: 'fr',
                Language.ES: 'es', Language.JA: 'ja', Language.ZH: 'zh',
                Language.PT: 'pt', Language.DE: 'de'
            }
            self._language_code = lang_map.get(language, 'en')
        
        self._kokoro_language = self._get_kokoro_language_code(self._language_code)

        # Model will be initialized on first use
        self._model = None
        self._init_future = None

        self._settings = {
            "model": model,
            "voice": voice,
            "sample_rate": sample_rate,
            "language": self._language_code,
            "kokoro_language": self._kokoro_language,
        }

    def _initialize_model(self):
        """Initialize the Kokoro model. This runs in a separate thread."""
        try:
            logger.debug(f"Loading Kokoro model: {self._model_name}")
            # Acquire the global MLX lock for model loading
            with MLX_GLOBAL_LOCK: # NEW: Use global lock
                self._model = load_model(self._model_name)
                mx.eval(mx.array([0])) # Ensure Metal commands are flushed
            logger.debug("Kokoro model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
            raise

    def can_generate_metrics(self) -> bool:
        return True

    def _generate_audio_streaming(self, text: str, chunk_queue: queue.Queue):
        """Generate audio in streaming fashion, putting chunks in the queue."""
        try:
            # Initialize model on first use if needed
            if self._model is None:
                self._initialize_model()

            logger.debug(f"Starting streaming audio generation for: {text}")

            # Acquire the global MLX lock for audio generation
            with MLX_GLOBAL_LOCK:
                mx.synchronize()
                
                chunk_count = 0
                for result in self._model.generate(
                    text=text,
                    voice=self._voice,
                    speed=1.0,
                    lang_code=self._kokoro_language,
                ):
                    # Process each chunk as it's generated
                    audio_array = result.audio
                    mx.eval(audio_array)
                    mx.synchronize()
                    
                    # Convert to numpy and then to bytes
                    audio_np = np.array(audio_array, copy=False)
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    
                    # Put chunk in queue for async consumption
                    chunk_queue.put(audio_bytes)
                    chunk_count += 1
                    logger.debug(f"Generated audio chunk {chunk_count}")
                
                if chunk_count == 0:
                    logger.warning(f"No audio chunks generated for text: '{text}' (len={len(text)})")
                    raise ValueError("No audio generated")
            
            # Signal end of stream
            chunk_queue.put(None)
            logger.debug(f"Streaming generation complete, generated {chunk_count} chunks")

        except Exception as e:
            logger.error(f"Error in streaming audio generation: {e}")
            # Signal error condition
            chunk_queue.put(e)
            raise

    def _generate_audio_sync(self, text: str) -> bytes:
        """Fallback non-streaming generation for compatibility."""
        # Use streaming internally but collect all chunks
        chunk_queue = queue.Queue()
        self._generate_audio_streaming(text, chunk_queue)
        
        audio_chunks = []
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            audio_chunks.append(chunk)
        
        return b''.join(audio_chunks)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        # Sanitize text for TTS to remove emojis and special characters
        try:
            from tools.text_formatter import sanitize_for_voice
            sanitized_text = sanitize_for_voice(text)
            
            # Check if sanitization resulted in empty or near-empty text - skip TTS silently
            if not sanitized_text or len(sanitized_text.strip()) <= 2:
                logger.info(f"🔇 Skipping TTS for empty/emoji-only text: '{text[:50]}...' -> '{sanitized_text}'")
                # Skip TTS entirely for empty content - don't send any frames
                return
            
            # Log sanitization if text was changed
            if sanitized_text != text:
                logger.warning(f"🧹 TTS SANITIZATION: '{text[:50]}...' -> '{sanitized_text[:50]}...'")
            else:
                logger.info(f"✅ TTS text already clean: '{text[:50]}...'")
        except Exception as e:
            logger.error(f"❌ TTS sanitization failed: {e}")
            sanitized_text = text
        
        logger.debug(f"{self}: Generating TTS [{sanitized_text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(sanitized_text)

            yield TTSStartedFrame()

            # Create a queue for streaming chunks
            chunk_queue = queue.Queue()
            
            # Start generation in background thread
            loop = asyncio.get_event_loop()
            generation_task = loop.run_in_executor(
                self._executor, self._generate_audio_streaming, sanitized_text, chunk_queue
            )
            
            # Add error callback to handle unhandled exceptions
            def _handle_generation_exception(future):
                try:
                    future.result()  # This will raise if there was an exception
                except Exception as e:
                    logger.error(f"Background TTS generation failed: {e}")
                    # Put the exception in the queue so it can be handled by the main loop
                    try:
                        chunk_queue.put(e)
                    except:
                        pass  # Queue might be full or closed
            
            generation_task.add_done_callback(_handle_generation_exception)

            # Stream chunks as they become available
            first_chunk = True
            CHUNK_SIZE = self.chunk_size
            
            # Text streaming setup - split text into chunks for progressive display
            words = sanitized_text.split()
            text_chunks_sent = 0
            total_text_chunks = len(words)
            audio_chunks_processed = 0
            
            while True:
                try:
                    # Non-blocking check for chunk with small timeout
                    chunk = await loop.run_in_executor(None, chunk_queue.get, True, 0.01)
                    
                    if chunk is None:
                        # End of stream
                        break
                    
                    if isinstance(chunk, Exception):
                        raise chunk
                    
                    # Stop TTFB metrics on first chunk
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False
                    
                    # Stream the chunk
                    for i in range(0, len(chunk), CHUNK_SIZE):
                        sub_chunk = chunk[i : i + CHUNK_SIZE]
                        if len(sub_chunk) > 0:
                            yield TTSAudioRawFrame(sub_chunk, self.sample_rate, 1)
                            
                            # Emit text progressively synchronized with audio chunks
                            audio_chunks_processed += 1
                            if total_text_chunks > 0:
                                # Calculate how many text chunks should be sent based on audio progress
                                # More conservative approach - only send text every few audio chunks
                                target_text_chunks = min(total_text_chunks, 
                                    max(1, int((audio_chunks_processed / 15) * total_text_chunks)))
                                
                                # Send more text chunks if needed, but only if we have meaningful progress
                                while text_chunks_sent < target_text_chunks and text_chunks_sent < total_text_chunks:
                                    # Send only the new word, not cumulative text
                                    if text_chunks_sent < len(words):
                                        new_word = words[text_chunks_sent]
                                        # Add space after word (except for last word)
                                        text_to_send = new_word + (" " if text_chunks_sent < len(words) - 1 else "")
                                        yield TTSTextFrame(text=text_to_send)
                                    text_chunks_sent += 1
                            
                            # Small delay to prevent overwhelming the pipeline
                            await asyncio.sleep(0.001)
                            
                except queue.Empty:
                    # Check if generation is still running
                    if generation_task.done():
                        # Get any exception that might have occurred
                        await generation_task
                        break
                    # Otherwise continue waiting for chunks
                    await asyncio.sleep(0.01)

            # Ensure generation task completes
            await generation_task
            
            # Send any remaining text chunks when TTS is complete
            while text_chunks_sent < total_text_chunks:
                if text_chunks_sent < len(words):
                    new_word = words[text_chunks_sent]
                    # Add space after word (except for last word)
                    text_to_send = new_word + (" " if text_chunks_sent < len(words) - 1 else "")
                    yield TTSTextFrame(text=text_to_send)
                text_chunks_sent += 1

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            # For empty text errors, provide a gentler response
            if "No audio generated" in str(e):
                logger.info("Gracefully handling empty TTS content")
                yield TTSStoppedFrame()
            else:
                yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{sanitized_text}]")
            await self.stop_processing_metrics()
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        await super().__aenter__()
        # Ensure model is initialized
        if self._model is None and self._init_future:
            await asyncio.get_event_loop().run_in_executor(None, self._init_future.result)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self._executor.shutdown(wait=True)
        await super().__aexit__(exc_type, exc_val, exc_tb)
