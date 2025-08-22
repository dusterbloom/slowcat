"""
Mood Analyzer - estimates paralinguistic mood features from user voice

Collects audio during a user utterance and computes lightweight features:
- Energy (RMS), zero-crossing rate (ZCR)
- Pitch (autocorrelation-based F0 estimate), pitch variability
- Duration; optional speaking rate if provided externally

Stores a compact mood summary into TapeStore.entry_meta bound to the most
recent user tape entry after STT writes the text.

Integration sketch:
- Register as an audio consumer with AudioTeeProcessor
- Hook to VAD start/stop via VADEventBridge callbacks
- On stop: compute metrics and call tape_store.add_entry_meta(ts, meta)
"""
from __future__ import annotations

import time
import math
import numpy as np
from typing import Optional, Dict, Any, Callable
from loguru import logger


class MoodAnalyzer:
    def __init__(self,
                 tape_store,
                 sample_rate: int = 16000,
                 max_buffer_seconds: float = 30.0):
        self.tape_store = tape_store
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_buffer_seconds)
        self._buf = bytearray()
        self._start_ts: Optional[float] = None
        self._stop_ts: Optional[float] = None

    # AudioTee consumer signature
    async def process_audio_frame(self, frame) -> None:
        try:
            audio_bytes: bytes = bytes(getattr(frame, 'audio', b''))
            if not audio_bytes:
                return
            # Cap buffer size
            if len(self._buf) + len(audio_bytes) > self.max_samples * 2:
                # Drop oldest
                excess = (len(self._buf) + len(audio_bytes)) - self.max_samples * 2
                if excess > 0:
                    del self._buf[:excess]
            self._buf.extend(audio_bytes)
        except Exception as e:
            logger.debug(f"MoodAnalyzer process_audio_frame error: {e}")

    # VAD callbacks
    async def on_user_started(self):
        self._buf.clear()
        self._start_ts = time.time()

    async def on_user_stopped(self):
        self._stop_ts = time.time()
        try:
            meta = self._analyze_buffer()
            # Attach to the most recent user entry (best-effort)
            try:
                recent = self.tape_store.get_recent(limit=1)
                if recent:
                    ts = getattr(recent[0], 'ts', None) if not isinstance(recent[0], dict) else recent[0].get('ts')
                    if ts:
                        self.tape_store.add_entry_meta(ts, meta)
                        logger.info(f"🧩 MoodAnalyzer stored meta for ts={ts}: {meta.get('mood','?')}")
            except Exception as e:
                logger.debug(f"MoodAnalyzer could not attach meta: {e}")
        except Exception as e:
            logger.debug(f"MoodAnalyzer analysis failed: {e}")

    def _analyze_buffer(self) -> Dict[str, Any]:
        # Convert to float32 mono [-1,1]
        if not self._buf:
            return {}
        audio = np.frombuffer(self._buf, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = len(audio) / float(self.sample_rate)
        if duration_s <= 0.1:
            return {}

        # Frame params
        win = int(0.040 * self.sample_rate)  # 40ms
        hop = int(0.020 * self.sample_rate)  # 20ms
        if win <= 0 or hop <= 0:
            return {}
        frames = []
        for i in range(0, len(audio) - win + 1, hop):
            frames.append(audio[i:i+win])
        if not frames:
            return {}
        frames = np.stack(frames, axis=0)

        # Energy (RMS per frame)
        rms = np.sqrt(np.maximum(1e-10, np.mean(frames**2, axis=1)))
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # Zero-crossing rate per frame
        signs = np.sign(frames)
        zc = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1) / float(frames.shape[1])
        zcr_mean = float(np.mean(zc))

        # Pitch via autocorrelation on voiced frames
        voiced_mask = rms > (np.median(rms) * 1.2)
        f0_list = []
        min_f0, max_f0 = 50.0, 400.0
        for fr in frames[voiced_mask]:
            f0 = self._pitch_autocorr(fr, self.sample_rate, min_f0, max_f0)
            if f0 is not None:
                f0_list.append(f0)
        pitch_mean = float(np.median(f0_list)) if f0_list else 0.0
        pitch_std = float(np.std(f0_list)) if f0_list else 0.0

        # Arousal heuristic: combine normalized energy + pitch variability
        # Scale to [0,1] approximately
        arousal = float(np.clip((energy_mean * 3.0) + (pitch_std / 200.0), 0.0, 1.0))

        mood = self._classify_mood(arousal, energy_mean, pitch_std, zcr_mean)

        return {
            'mood': mood,
            'arousal': arousal,
            'valence': None,  # Reserved for future (lexical + prosody fusion)
            'pitch_mean_hz': pitch_mean,
            'pitch_std_hz': pitch_std,
            'energy_rms': energy_mean,
            'energy_std': energy_std,
            'zcr_mean': zcr_mean,
            'duration_s': duration_s,
        }

    @staticmethod
    def _pitch_autocorr(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> Optional[float]:
        # Simple autocorrelation-based F0 estimate
        try:
            frame = frame - np.mean(frame)
            if np.max(np.abs(frame)) < 1e-4:
                return None
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            # Define lags for plausible F0
            min_lag = int(sr / fmax)
            max_lag = int(sr / fmin)
            if max_lag <= min_lag + 1 or max_lag >= len(corr):
                return None
            # Find peak in range
            lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
            if lag <= 0:
                return None
            f0 = sr / lag
            if f0 < fmin or f0 > fmax:
                return None
            return float(f0)
        except Exception:
            return None

    @staticmethod
    def _classify_mood(arousal: float, energy_mean: float, pitch_std: float, zcr_mean: float) -> str:
        # Very lightweight rules; safe defaults
        if arousal > 0.7 and pitch_std > 40:
            return 'excited'
        if arousal > 0.6 and zcr_mean > 0.15:
            return 'engaged'
        if arousal < 0.3 and pitch_std < 20:
            return 'calm'
        # Potential stress indicator (rapid changes)
        if pitch_std > 60 and zcr_mean > 0.2:
            return 'stressed'
        return 'neutral'


# Helper to wire with AudioTee + VAD bridge
def attach_mood_analyzer(tee, vad_bridge, tape_store, sample_rate: int = 16000) -> MoodAnalyzer:
    """Attach MoodAnalyzer to AudioTee and VADEventBridge.

    tee: AudioTeeProcessor
    vad_bridge: VADEventBridge
    """
    analyzer = MoodAnalyzer(tape_store, sample_rate=sample_rate)
    try:
        # Register as audio consumer
        tee.register_audio_consumer(analyzer)
    except Exception as e:
        logger.warning(f"Could not register mood analyzer with audio tee: {e}")
    try:
        # Attach VAD callbacks
        vad_bridge.set_callbacks(on_started=analyzer.on_user_started, on_stopped=analyzer.on_user_stopped)
    except Exception as e:
        logger.warning(f"Could not attach mood analyzer to VAD: {e}")
    return analyzer

