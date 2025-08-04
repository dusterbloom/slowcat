"""Audio quality validation for voice recognition"""
import numpy as np
import logging
from typing import Tuple, Optional

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some audio quality features will be disabled.")

logger = logging.getLogger(__name__)


class AudioQualityValidator:
    """Validates audio quality before voice recognition processing"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        # Simple energy-based SNR estimation
        # Split audio into frames
        frame_length = int(0.02 * self.sample_rate)  # 20ms frames
        hop_length = frame_length // 2
        
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frames.append(audio[i:i + frame_length])
        
        if not frames:
            return 0.0
            
        # Calculate energy for each frame
        energies = [np.sum(frame ** 2) for frame in frames]
        energies = np.array(energies)
        
        # Estimate noise as the lowest 20% of energies
        sorted_energies = np.sort(energies)
        noise_frames = int(len(sorted_energies) * 0.2)
        noise_energy = np.mean(sorted_energies[:noise_frames]) if noise_frames > 0 else 1e-10
        
        # Signal energy is the mean of all energies above noise
        signal_energy = np.mean(energies[energies > noise_energy * 2])
        
        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_energy / noise_energy) if noise_energy > 0 else 0
        
        return float(snr_db)
    
    def calculate_voice_activity_ratio(self, audio: np.ndarray, energy_threshold_ratio: float = 0.01) -> float:
        """Calculate the ratio of voice activity in the audio"""
        energy = np.abs(audio)
        
        # Use a more sophisticated threshold calculation
        # Based on percentile rather than just max * ratio
        threshold = np.percentile(energy, 20) * 2  # Use 20th percentile * 2 as threshold
        
        # Alternative threshold based on max if percentile is too low
        max_threshold = np.max(energy) * energy_threshold_ratio
        threshold = max(threshold, max_threshold)
        
        voice_samples = np.sum(energy > threshold)
        total_samples = len(audio)
        
        return voice_samples / total_samples if total_samples > 0 else 0.0
    
    def calculate_spectral_features(self, audio: np.ndarray) -> dict:
        """Calculate spectral features for voice detection"""
        if not SCIPY_AVAILABLE:
            # Return default values if scipy not available
            return {
                'speech_power_ratio': 0.8,
                'mean_spectral_centroid': 1000.0
            }
            
        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            audio, 
            fs=self.sample_rate,
            window='hann',
            nperseg=512,
            noverlap=256
        )
        
        # Focus on speech frequency range (80Hz - 8kHz)
        speech_freq_mask = (frequencies >= 80) & (frequencies <= 8000)
        speech_power = np.mean(Sxx[speech_freq_mask, :])
        total_power = np.mean(Sxx)
        
        # Calculate spectral centroid (brightness indicator)
        spectral_centroid = np.sum(frequencies[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)
        mean_centroid = np.mean(spectral_centroid[~np.isnan(spectral_centroid)])
        
        return {
            'speech_power_ratio': float(speech_power / total_power) if total_power > 0 else 0.0,
            'mean_spectral_centroid': float(mean_centroid) if not np.isnan(mean_centroid) else 0.0
        }
    
    def calculate_quality_score(self, audio: np.ndarray, config) -> Tuple[float, dict]:
        """
        Calculate overall audio quality score
        Returns: (quality_score, metrics_dict)
        """
        metrics = {}
        
        # 1. SNR check
        snr_db = self.calculate_snr(audio)
        metrics['snr_db'] = snr_db
        snr_score = min(1.0, max(0.0, (snr_db - 5) / 20))  # Map 5-25 dB to 0-1
        
        # 2. Voice activity ratio
        voice_ratio = self.calculate_voice_activity_ratio(audio, config.energy_threshold_ratio)
        metrics['voice_activity_ratio'] = voice_ratio
        voice_score = min(1.0, voice_ratio * 2)  # Expect at least 50% voice activity
        
        # 3. Spectral features
        spectral_features = self.calculate_spectral_features(audio)
        metrics.update(spectral_features)
        
        # Speech frequency ratio should be high for voice
        speech_freq_score = spectral_features['speech_power_ratio']
        
        # Spectral centroid for voice typically between 500-2000 Hz
        centroid = spectral_features['mean_spectral_centroid']
        centroid_score = 1.0 if 500 <= centroid <= 2000 else 0.5
        
        # Combined quality score
        quality_score = (
            0.3 * snr_score +
            0.3 * voice_score +
            0.2 * speech_freq_score +
            0.2 * centroid_score
        )
        
        metrics['quality_score'] = quality_score
        
        logger.debug(f"Audio quality metrics: {metrics}")
        
        return quality_score, metrics
    
    def is_valid_for_recognition(self, audio: np.ndarray, config) -> Tuple[bool, Optional[str], dict]:
        """
        Check if audio is valid for voice recognition
        Returns: (is_valid, rejection_reason, metrics)
        """
        quality_score, metrics = self.calculate_quality_score(audio, config)
        
        # Check minimum requirements
        if metrics['snr_db'] < config.min_snr_db:
            return False, f"SNR too low ({metrics['snr_db']:.1f} dB < {config.min_snr_db} dB)", metrics
            
        if quality_score < config.audio_quality_threshold:
            return False, f"Quality score too low ({quality_score:.2f} < {config.audio_quality_threshold})", metrics
            
        if metrics['voice_activity_ratio'] < 0.2:  # Reduced from 0.3 to 0.2 for more tolerance
            return False, f"Insufficient voice activity ({metrics['voice_activity_ratio']:.2f} < 0.2)", metrics
            
        return True, None, metrics