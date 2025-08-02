#
# Kokoro TTS service for Pipecat using MLX Audio
#

import asyncio
import concurrent.futures
from typing import AsyncGenerator, Optional

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
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.transcriptions.language import Language


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
            language: The language for synthesis (optional, auto-detected from voice).
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._model_name = model
        self._voice = voice
        self._device = device
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
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

        # Initialize model in a separate thread to avoid blocking
        self._model = None
        self._init_future = self._executor.submit(self._initialize_model)

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
            self._model = load_model(self._model_name)
            logger.debug("Kokoro model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro model: {e}")
            raise

    def can_generate_metrics(self) -> bool:
        return True

    def _generate_audio_sync(self, text: str) -> bytes:
        """Synchronously generate audio from text. This runs in a separate thread."""
        try:
            if self._init_future:
                self._init_future.result()  # Wait for initialization
                self._init_future = None

            logger.debug(f"Generating audio for: {text}")

            audio_segments = []
            for result in self._model.generate(
                text=text,
                voice=self._voice,
                speed=1.0,
                lang_code=self._kokoro_language,
            ):
                audio_segments.append(result.audio)

            if len(audio_segments) == 0:
                raise ValueError("No audio generated")
            elif len(audio_segments) == 1:
                audio_array = audio_segments[0]
            else:
                audio_array = mx.concatenate(audio_segments, axis=0)

            # Convert MLX array to NumPy array
            audio_np = np.array(audio_array, copy=False)

            # Convert to raw PCM bytes (16-bit signed integer)
            # MLX audio returns float32 normalized audio
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            return audio_bytes

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Kokoro.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Run audio generation in executor (separate thread) to avoid blocking
            loop = asyncio.get_event_loop()
            audio_bytes = await loop.run_in_executor(
                self._executor, self._generate_audio_sync, text
            )

            # Chunk the audio data for streaming
            CHUNK_SIZE = self.chunk_size

            await self.stop_ttfb_metrics()

            # Stream the audio in chunks
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i : i + CHUNK_SIZE]
                if len(chunk) > 0:
                    yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    # Small delay to prevent overwhelming the pipeline
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
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
