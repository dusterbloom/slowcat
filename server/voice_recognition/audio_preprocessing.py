"""Enhanced audio preprocessing for voice recognition"""
import numpy as np
import logging
from typing import Optional, Tuple

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some audio preprocessing features will be disabled.")

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Advanced audio preprocessing for improved voice recognition"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def apply_pre_emphasis(self, audio: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to enhance high frequencies"""
        return np.append(audio[0], audio[1:] - coefficient * audio[:-1])
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio signal"""
        return audio - np.mean(audio)
    
    def apply_bandpass_filter(self, audio: np.ndarray, lowcut: int = 80, highcut: int = 8000) -> np.ndarray:
        """Apply bandpass filter to focus on speech frequencies"""
        nyquist = self.sample_rate / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure frequencies are within valid range (0 < Wn < 1)
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        # Ensure low < high
        if low >= high:
            logger.warning(f"Invalid filter frequencies: low={low}, high={high}. Skipping filter.")
            return audio
        
        if not SCIPY_AVAILABLE:
            return audio
            
        try:
            # Design Butterworth bandpass filter
            sos = signal.butter(5, [low, high], btype='band', output='sos')
            filtered = signal.sosfiltfilt(sos, audio)
            return filtered
        except Exception as e:
            logger.error(f"Bandpass filter failed: {e}. Returning original audio.")
            return audio
    
    def adaptive_silence_removal(self, audio: np.ndarray, 
                               frame_duration: float = 0.02,
                               energy_percentile: int = 10) -> np.ndarray:
        """Remove silence with adaptive thresholding"""
        frame_length = int(frame_duration * self.sample_rate)
        hop_length = frame_length // 2
        
        # Calculate frame energies
        energies = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sum(frame ** 2)
            energies.append(energy)
        
        if not energies:
            return audio
            
        # Adaptive threshold based on energy distribution
        energy_threshold = np.percentile(energies, energy_percentile)
        
        # Find voice activity regions
        voice_regions = []
        in_voice = False
        start_idx = 0
        
        for i, energy in enumerate(energies):
            frame_start = i * hop_length
            
            if energy > energy_threshold and not in_voice:
                in_voice = True
                start_idx = frame_start
            elif energy <= energy_threshold and in_voice:
                in_voice = False
                # Add some context before and after
                context_samples = int(0.1 * self.sample_rate)  # 100ms context
                start = max(0, start_idx - context_samples)
                end = min(len(audio), frame_start + frame_length + context_samples)
                voice_regions.append((start, end))
        
        # Handle case where audio ends while still in voice
        if in_voice:
            context_samples = int(0.1 * self.sample_rate)
            start = max(0, start_idx - context_samples)
            voice_regions.append((start, len(audio)))
        
        # Merge overlapping regions
        merged_regions = []
        for start, end in sorted(voice_regions):
            if merged_regions and start <= merged_regions[-1][1]:
                merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], end))
            else:
                merged_regions.append((start, end))
        
        # Concatenate voice regions
        if merged_regions:
            voice_audio = np.concatenate([audio[start:end] for start, end in merged_regions])
            return voice_audio
        else:
            return audio
    
    def spectral_subtraction(self, audio: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
        """Apply spectral subtraction for noise reduction"""
        if not SCIPY_AVAILABLE:
            return audio
            
        # Estimate noise from the first 0.5 seconds or 10% of audio
        noise_samples = min(int(0.5 * self.sample_rate), len(audio) // 10)
        
        if noise_samples < 512:
            return audio
            
        try:
            # Compute noise spectrum
            noise_segment = audio[:noise_samples]
            _, _, noise_spectrum = signal.spectrogram(
                noise_segment,
                fs=self.sample_rate,
                window='hann',
                nperseg=512,
                noverlap=256
            )
            noise_profile = np.mean(np.abs(noise_spectrum), axis=1)
            
            # Process full audio
            f, t, Sxx = signal.spectrogram(
                audio,
                fs=self.sample_rate,
                window='hann',
                nperseg=512,
                noverlap=256
            )
            
            # Spectral subtraction
            magnitude = np.abs(Sxx)
            phase = np.angle(Sxx)
            
            # Subtract noise profile
            cleaned_magnitude = magnitude - noise_factor * noise_profile[:, np.newaxis]
            cleaned_magnitude = np.maximum(cleaned_magnitude, 0)
            
            # Reconstruct signal
            cleaned_spectrum = cleaned_magnitude * np.exp(1j * phase)
            _, cleaned_audio = signal.istft(
                cleaned_spectrum,
                fs=self.sample_rate,
                window='hann',
                nperseg=512,
                noverlap=256
            )
            
            # Ensure same length as input
            if len(cleaned_audio) > len(audio):
                cleaned_audio = cleaned_audio[:len(audio)]
            elif len(cleaned_audio) < len(audio):
                cleaned_audio = np.pad(cleaned_audio, (0, len(audio) - len(cleaned_audio)))
                
            return cleaned_audio
        except Exception as e:
            logger.error(f"Spectral subtraction failed: {e}")
            return audio
    
    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.7) -> np.ndarray:
        """Normalize audio to target level"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio
    
    def process(self, audio: np.ndarray, config) -> Tuple[np.ndarray, dict]:
        """
        Apply full preprocessing pipeline
        Returns: (processed_audio, preprocessing_info)
        """
        info = {
            'original_length': len(audio),
            'original_max': float(np.max(np.abs(audio)))
        }
        
        # 1. Remove DC offset
        audio = self.remove_dc_offset(audio)
        
        # 2. Apply bandpass filter
        audio = self.apply_bandpass_filter(
            audio, 
            lowcut=config.highpass_cutoff_hz,
            highcut=config.lowpass_cutoff_hz
        )
        
        # 3. Spectral subtraction for noise reduction
        audio = self.spectral_subtraction(audio)
        
        # 4. Pre-emphasis
        audio = self.apply_pre_emphasis(audio, config.pre_emphasis_coefficient)
        
        # 5. Adaptive silence removal
        audio = self.adaptive_silence_removal(audio)
        
        # 6. Normalize
        audio = self.normalize_audio(audio)
        
        info['processed_length'] = len(audio)
        info['processed_max'] = float(np.max(np.abs(audio)))
        info['length_ratio'] = info['processed_length'] / info['original_length']
        
        logger.debug(f"Audio preprocessing complete: {info}")
        
        return audio, info