#!/usr/bin/env python3
"""
MLX-Whisper vs Sherpa-ONNX Comparison Test

Compare MLX-Whisper against the best sherpa-onnx model on 5-minute sustained load:
1. Performance (RTF, latency)
2. Memory usage and stability  
3. Accuracy on longer audio
4. Resource efficiency
5. Streaming vs batch processing differences
"""

import argparse
import time
import wave
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import logging
import psutil
import threading
import gc
import tracemalloc
import sys

# Add server root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
except ImportError:
    logger.error("sherpa_onnx not found. Please install with: pip install sherpa-onnx")

try:
    import mlx_whisper
except ImportError:
    logger.error("mlx_whisper not found. Please install with: pip install mlx-whisper")

@dataclass
class ComparisonResult:
    """Results comparing two models"""
    model_name: str
    model_type: str  # 'sherpa' or 'mlx-whisper'
    audio_duration_sec: float
    
    # Performance metrics
    model_load_time_ms: float
    total_processing_time_sec: float
    real_time_factor: float
    time_to_first_result_ms: Optional[float]
    
    # Memory metrics
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    
    # CPU metrics
    avg_cpu_percent: float
    peak_cpu_percent: float
    
    # Results
    final_transcript: str
    word_count: int
    
    # Streaming specific (sherpa only)
    segments_produced: int = 0
    avg_segment_length: float = 0.0
    
    # Errors
    errors_encountered: List[str] = None
    
    def __post_init__(self):
        if self.errors_encountered is None:
            self.errors_encountered = []

class ResourceMonitor:
    """Monitor CPU and memory usage"""
    
    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.thread = None
        self.process = psutil.Process()
        
    def start(self):
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def get_stats(self) -> Dict[str, float]:
        if not self.cpu_samples:
            return {'initial_memory': 0, 'peak_memory': 0, 'final_memory': 0, 'avg_cpu': 0, 'peak_cpu': 0}
            
        return {
            'initial_memory': self.memory_samples[0] if self.memory_samples else 0,
            'peak_memory': max(self.memory_samples),
            'final_memory': self.memory_samples[-1] if self.memory_samples else 0,
            'avg_cpu': np.mean(self.cpu_samples),
            'peak_cpu': max(self.cpu_samples)
        }
    
    def _monitor(self):
        while self.monitoring:
            try:
                cpu = self.process.cpu_percent()
                memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                break

def create_synthetic_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Create synthetic speech-like audio for testing"""
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, False)
    
    # Create speech-like pattern with varying frequencies and natural pauses
    # Base frequency modulation (speech formants)
    f1 = 200 + 100 * np.sin(2 * np.pi * 0.5 * t)  # Fundamental
    f2 = 800 + 200 * np.sin(2 * np.pi * 0.3 * t)  # First formant
    f3 = 2400 + 400 * np.sin(2 * np.pi * 0.2 * t) # Second formant
    
    # Amplitude modulation (speech rhythm with natural pauses)
    amplitude = 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    
    # Add periodic longer pauses (every 10-15 seconds)
    for i in range(0, int(duration_sec), 12):
        start_idx = int(i * sample_rate)
        end_idx = int(min((i + 1) * sample_rate, len(t)))
        if start_idx < len(amplitude):
            amplitude[start_idx:end_idx] *= 0.1  # Quiet pause
    
    # Combine frequencies
    audio = amplitude * (
        0.6 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add realistic noise
    noise = 0.01 * np.random.normal(0, 1, total_samples)
    audio = (audio + noise).astype(np.float32)
    
    # Ensure valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio

def create_sherpa_recognizer(model_dir: Path) -> sherpa_onnx.OnlineRecognizer:
    """Create sherpa-onnx recognizer - optimized configuration"""
    
    tokens_file = model_dir / "tokens.txt"
    if not tokens_file.exists():
        raise FileNotFoundError(f"tokens.txt not found in {model_dir}")
    
    # Find model files
    encoder_files = list(model_dir.glob("encoder*.onnx"))
    decoder_files = list(model_dir.glob("decoder*.onnx"))
    joiner_files = list(model_dir.glob("joiner*.onnx"))
    
    if not (encoder_files and decoder_files and joiner_files):
        raise FileNotFoundError(f"Missing model files in {model_dir}")
    
    # Prefer int8 quantized versions
    def get_best_model(files):
        int8_files = [f for f in files if 'int8' in f.name]
        return int8_files[0] if int8_files else files[0]
    
    encoder_file = get_best_model(encoder_files)
    decoder_file = get_best_model(decoder_files)
    joiner_file = get_best_model(joiner_files)
    
    logger.info(f"Sherpa using: {encoder_file.name}")
    
    # Create recognizer with optimized settings for comparison
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(tokens_file),
        encoder=str(encoder_file),
        decoder=str(decoder_file),
        joiner=str(joiner_file),
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method='greedy_search',
        max_active_paths=4,
        blank_penalty=0.0,
        temperature_scale=1.0,
    )
    
    return recognizer

def test_sherpa_streaming(audio_data: np.ndarray, duration_sec: float, 
                         model_dir: Path) -> ComparisonResult:
    """Test sherpa-onnx with streaming processing"""
    
    logger.info("Testing Sherpa-ONNX (streaming)...")
    
    # Start monitoring
    monitor = ResourceMonitor()
    monitor.start()
    tracemalloc.start()
    
    errors = []
    segments = []
    first_result_time = None
    
    # Load model
    load_start = time.time()
    try:
        recognizer = create_sherpa_recognizer(model_dir)
        load_time = (time.time() - load_start) * 1000
    except Exception as e:
        error_msg = f"Failed to load Sherpa model: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        monitor.stop()
        return ComparisonResult(
            model_name="sherpa-onnx-FAILED",
            model_type="sherpa",
            audio_duration_sec=duration_sec,
            model_load_time_ms=0,
            total_processing_time_sec=0,
            real_time_factor=float('inf'),
            time_to_first_result_ms=None,
            initial_memory_mb=0,
            peak_memory_mb=0,
            final_memory_mb=0,
            memory_growth_mb=0,
            avg_cpu_percent=0,
            peak_cpu_percent=0,
            final_transcript="",
            word_count=0,
            errors_encountered=errors
        )
    
    # Process audio in streaming fashion
    process_start = time.time()
    
    try:
        stream = recognizer.create_stream()
        chunk_size_ms = 200
        chunk_size_samples = int(16000 * chunk_size_ms / 1000)
        
        for i in range(0, len(audio_data), chunk_size_samples):
            chunk = audio_data[i:i + chunk_size_samples]
            
            # Feed audio
            stream.accept_waveform(16000, chunk.tolist())
            
            # Decode if ready
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            
            # Check for endpoint
            is_endpoint = recognizer.is_endpoint(stream)
            result = recognizer.get_result(stream)
            result_text = result if isinstance(result, str) else getattr(result, 'text', str(result))
            
            if is_endpoint and result_text.strip():
                if first_result_time is None:
                    first_result_time = (time.time() - process_start) * 1000
                
                segments.append(result_text.strip())
                recognizer.reset(stream)
            
            # Small delay to simulate real-time
            time.sleep(0.001)
        
        # Get final result
        result = recognizer.get_result(stream)
        result_text = result if isinstance(result, str) else getattr(result, 'text', str(result))
        if result_text.strip() and (not segments or result_text.strip() != segments[-1]):
            if first_result_time is None:
                first_result_time = (time.time() - process_start) * 1000
            segments.append(result_text.strip())
            
    except Exception as e:
        error_msg = f"Sherpa processing error: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
    
    processing_time = time.time() - process_start
    
    # Stop monitoring
    monitor.stop()
    stats = monitor.get_stats()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Combine results
    final_transcript = " ".join(segments)
    word_count = len(final_transcript.split()) if final_transcript else 0
    avg_segment_len = duration_sec / len(segments) if segments else 0
    
    result = ComparisonResult(
        model_name="sherpa-onnx-zipformer-en-2023-06-26",
        model_type="sherpa",
        audio_duration_sec=duration_sec,
        model_load_time_ms=load_time,
        total_processing_time_sec=processing_time,
        real_time_factor=processing_time / duration_sec,
        time_to_first_result_ms=first_result_time,
        initial_memory_mb=stats['initial_memory'],
        peak_memory_mb=stats['peak_memory'],
        final_memory_mb=stats['final_memory'],
        memory_growth_mb=stats['final_memory'] - stats['initial_memory'],
        avg_cpu_percent=stats['avg_cpu'],
        peak_cpu_percent=stats['peak_cpu'],
        final_transcript=final_transcript,
        word_count=word_count,
        segments_produced=len(segments),
        avg_segment_length=avg_segment_len,
        errors_encountered=errors
    )
    
    logger.info(f"Sherpa completed: RTF={result.real_time_factor:.3f}, Segments={len(segments)}, Words={word_count}")
    
    return result

def test_mlx_whisper_batch(audio_data: np.ndarray, duration_sec: float) -> ComparisonResult:
    """Test MLX-Whisper with batch processing"""
    
    logger.info("Testing MLX-Whisper (batch)...")
    
    # Start monitoring
    monitor = ResourceMonitor()
    monitor.start()
    tracemalloc.start()
    
    errors = []
    first_result_time = None
    final_transcript = ""
    
    # Load model and process
    load_start = time.time()
    process_start = None
    
    try:
        # MLX-Whisper processes the entire audio at once
        process_start = time.time()
        
        # Use cached model like in whisper_stt_with_lock.py
        # Set cache directory explicitly for consistent behavior
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.environ["HF_HOME"] = cache_dir
        
        # Use the model reference from the codebase
        model_name = "mlx-community/whisper-large-v3-turbo"
        
        # MLX-Whisper expects the audio to be in the right format
        result = mlx_whisper.transcribe(
            audio_data, 
            path_or_hf_repo=model_name,
            verbose=False
        )
        
        load_and_process_time = (time.time() - load_start) * 1000
        processing_time = time.time() - process_start
        first_result_time = processing_time * 1000  # First and only result
        
        final_transcript = result["text"].strip() if result and "text" in result else ""
        
    except Exception as e:
        error_msg = f"MLX-Whisper processing error: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        load_and_process_time = (time.time() - load_start) * 1000 if load_start else 0
        processing_time = time.time() - process_start if process_start else 0
        first_result_time = None
    
    # Stop monitoring
    monitor.stop()
    stats = monitor.get_stats()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    word_count = len(final_transcript.split()) if final_transcript else 0
    
    result = ComparisonResult(
        model_name="mlx-whisper-large-v3-turbo",
        model_type="mlx-whisper",
        audio_duration_sec=duration_sec,
        model_load_time_ms=load_and_process_time,  # MLX-Whisper loads and processes together
        total_processing_time_sec=processing_time,
        real_time_factor=processing_time / duration_sec,
        time_to_first_result_ms=first_result_time,
        initial_memory_mb=stats['initial_memory'],
        peak_memory_mb=stats['peak_memory'],
        final_memory_mb=stats['final_memory'],
        memory_growth_mb=stats['final_memory'] - stats['initial_memory'],
        avg_cpu_percent=stats['avg_cpu'],
        peak_cpu_percent=stats['peak_cpu'],
        final_transcript=final_transcript,
        word_count=word_count,
        errors_encountered=errors
    )
    
    logger.info(f"MLX-Whisper completed: RTF={result.real_time_factor:.3f}, Words={word_count}")
    
    return result

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Simple edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def main():
    parser = argparse.ArgumentParser(description='Compare MLX-Whisper vs Sherpa-ONNX')
    parser.add_argument('--sherpa-model-dir', type=Path, required=True,
                       help='Path to sherpa-onnx model directory')
    parser.add_argument('--duration', type=float, default=300,
                       help='Test duration in seconds (default: 300s = 5min)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--save-audio', action='store_true',
                       help='Save generated test audio')
    
    args = parser.parse_args()
    
    if not args.sherpa_model_dir.exists():
        logger.error(f"Sherpa model directory not found: {args.sherpa_model_dir}")
        return
    
    logger.info(f"Comparing MLX-Whisper vs Sherpa-ONNX on {args.duration}s audio")
    
    # Generate test audio
    logger.info(f"Generating {args.duration}s of synthetic speech...")
    audio_data = create_synthetic_audio(args.duration)
    
    # Save audio if requested
    if args.save_audio:
        audio_dir = Path(__file__).parent / "test_audio" / "comparison"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        audio_file = audio_dir / f"comparison_{args.duration}s.wav"
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(audio_file), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        
        logger.info(f"Saved test audio: {audio_file}")
    
    # Run tests
    results = []
    
    print("\n" + "="*80)
    print("MLX-WHISPER vs SHERPA-ONNX COMPARISON")
    print("="*80)
    
    # Test Sherpa-ONNX
    sherpa_result = test_sherpa_streaming(audio_data, args.duration, args.sherpa_model_dir)
    results.append(sherpa_result)
    
    # Force garbage collection between tests
    gc.collect()
    time.sleep(2)
    
    # Test MLX-Whisper (if available)
    if 'mlx_whisper' in globals():
        mlx_result = test_mlx_whisper_batch(audio_data, args.duration)
        results.append(mlx_result)
    else:
        logger.warning("MLX-Whisper not available - skipping")
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / "results" / f"comparison_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
    
    # Prepare for JSON serialization
    json_results = []
    for result in results:
        result_dict = asdict(result)
        json_results.append(result_dict)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print comparison
    print(f"\n📊 COMPARISON RESULTS ({args.duration}s audio)")
    print("-" * 60)
    
    for result in results:
        print(f"\n🤖 {result.model_name} ({result.model_type})")
        print(f"  Load Time:      {result.model_load_time_ms:.1f}ms")
        print(f"  Processing:     {result.total_processing_time_sec:.2f}s")
        print(f"  RTF:            {result.real_time_factor:.3f}")
        print(f"  Memory Growth:  {result.memory_growth_mb:+.1f}MB")
        print(f"  Peak Memory:    {result.peak_memory_mb:.1f}MB")
        print(f"  CPU (avg/peak): {result.avg_cpu_percent:.1f}% / {result.peak_cpu_percent:.1f}%")
        print(f"  Words:          {result.word_count}")
        
        if result.model_type == "sherpa":
            print(f"  Segments:       {result.segments_produced}")
            print(f"  Avg Seg Length: {result.avg_segment_length:.1f}s")
        
        if result.time_to_first_result_ms:
            print(f"  First Result:   {result.time_to_first_result_ms:.1f}ms")
        
        if result.errors_encountered:
            print(f"  Errors:         {len(result.errors_encountered)}")
        
        # Show sample of transcript
        sample = result.final_transcript[:100] + "..." if len(result.final_transcript) > 100 else result.final_transcript
        print(f"  Transcript:     '{sample}'")
    
    # Head-to-head comparison
    if len(results) == 2:
        sherpa, mlx = results
        
        print(f"\n🏆 HEAD-TO-HEAD COMPARISON")
        print("-" * 40)
        print(f"Speed Winner:     {'Sherpa' if sherpa.real_time_factor < mlx.real_time_factor else 'MLX-Whisper'}")
        print(f"Memory Winner:    {'Sherpa' if sherpa.memory_growth_mb < mlx.memory_growth_mb else 'MLX-Whisper'}")
        print(f"CPU Winner:       {'Sherpa' if sherpa.avg_cpu_percent < mlx.avg_cpu_percent else 'MLX-Whisper'}")
        
        if sherpa.time_to_first_result_ms and mlx.time_to_first_result_ms:
            print(f"Latency Winner:   {'Sherpa' if sherpa.time_to_first_result_ms < mlx.time_to_first_result_ms else 'MLX-Whisper'}")
        
        # Calculate relative performance
        rtf_improvement = ((mlx.real_time_factor - sherpa.real_time_factor) / mlx.real_time_factor) * 100
        memory_improvement = ((mlx.memory_growth_mb - sherpa.memory_growth_mb) / max(mlx.memory_growth_mb, 1)) * 100
        
        print(f"\nPerformance Differences:")
        print(f"  RTF:     Sherpa is {rtf_improvement:+.1f}% vs MLX-Whisper")
        print(f"  Memory:  Sherpa uses {memory_improvement:+.1f}% vs MLX-Whisper")
        
        # Streaming vs Batch characteristics
        print(f"\nStreaming Characteristics:")
        print(f"  Sherpa:      {sherpa.segments_produced} segments, streaming output")
        print(f"  MLX-Whisper: 1 result, batch processing")

if __name__ == "__main__":
    main()