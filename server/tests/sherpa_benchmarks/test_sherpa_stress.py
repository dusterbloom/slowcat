#!/usr/bin/env python3
"""
Sherpa-ONNX Stress Test for Longer Audio Files

Tests sherpa-onnx performance with longer audio durations to evaluate:
1. Memory usage over time
2. Processing stability with sustained load
3. Endpoint detection with longer speech segments
4. Real-time factor consistency
5. Memory leaks and resource cleanup

Duration targets: 10s, 30s, 1min, 5min, 10min+
"""

import argparse
import time
import wave
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import json
import logging
import psutil
import threading
import gc
import tracemalloc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
except ImportError:
    logger.error("sherpa_onnx not found. Please install with: pip install sherpa-onnx")
    exit(1)

@dataclass
class StressTestResult:
    """Results for stress test on longer audio"""
    model_name: str
    audio_duration_sec: float
    chunk_size_ms: int
    
    # Performance metrics
    total_processing_time_sec: float
    real_time_factor: float
    model_load_time_ms: float
    
    # Memory metrics over time
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_mb: float
    memory_samples: List[float]  # Memory usage over time
    
    # CPU metrics
    avg_cpu_percent: float
    peak_cpu_percent: float
    cpu_samples: List[float]
    
    # Streaming metrics
    total_segments: int
    avg_segment_length_sec: float
    endpoint_detections: int
    processing_gaps_ms: List[float]  # Time between processing chunks
    
    # Accuracy (if reference available)
    final_transcript: str
    errors_encountered: List[str]
    chunk_processing_times: List[float]  # Time to process each chunk
    
    # Fields with defaults
    reference_text: str = ""
    word_error_rate: float = 0.0
    memory_leak_detected: bool = False
    performance_degradation: bool = False
    
class DetailedResourceMonitor:
    """Enhanced resource monitoring for stress testing"""
    
    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        self.monitoring = False
        self.thread = None
        self.process = psutil.Process()
        
    def start(self):
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join()
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.cpu_samples:
            return {
                'initial_memory': 0, 'peak_memory': 0, 'final_memory': 0,
                'avg_cpu': 0, 'peak_cpu': 0, 'memory_samples': [],
                'cpu_samples': []
            }
            
        return {
            'initial_memory': self.memory_samples[0] if self.memory_samples else 0,
            'peak_memory': max(self.memory_samples),
            'final_memory': self.memory_samples[-1] if self.memory_samples else 0,
            'avg_cpu': np.mean(self.cpu_samples),
            'peak_cpu': max(self.cpu_samples),
            'memory_samples': self.memory_samples.copy(),
            'cpu_samples': self.cpu_samples.copy()
        }
    
    def _monitor(self):
        while self.monitoring:
            try:
                cpu = self.process.cpu_percent()
                memory = self.process.memory_info().rss / 1024 / 1024  # MB
                timestamp = time.time()
                
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
                self.timestamps.append(timestamp)
                
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                break

def create_synthetic_audio(duration_sec: float, sample_rate: int = 16000, 
                          content_type: str = "speech") -> np.ndarray:
    """Create synthetic audio for stress testing"""
    
    total_samples = int(duration_sec * sample_rate)
    
    if content_type == "silence":
        # Pure silence
        audio = np.zeros(total_samples, dtype=np.float32)
    
    elif content_type == "tone":
        # Simple sine wave tone
        frequency = 440  # A4 note
        t = np.linspace(0, duration_sec, total_samples, False)
        audio = 0.1 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    elif content_type == "speech":
        # Synthetic speech-like pattern (varying frequencies and amplitudes)
        t = np.linspace(0, duration_sec, total_samples, False)
        
        # Base frequency modulation (speech formants)
        f1 = 200 + 100 * np.sin(2 * np.pi * 0.5 * t)  # Fundamental
        f2 = 800 + 200 * np.sin(2 * np.pi * 0.3 * t)  # First formant
        f3 = 2400 + 400 * np.sin(2 * np.pi * 0.2 * t) # Second formant
        
        # Amplitude modulation (speech rhythm)
        amplitude = 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
        
        # Combine frequencies
        audio = amplitude * (
            0.6 * np.sin(2 * np.pi * f1 * t) +
            0.3 * np.sin(2 * np.pi * f2 * t) +
            0.1 * np.sin(2 * np.pi * f3 * t)
        )
        
        # Add some noise for realism
        noise = 0.01 * np.random.normal(0, 1, total_samples)
        audio = (audio + noise).astype(np.float32)
        
        # Add speech-like pauses
        pause_probability = 0.02  # 2% chance of pause per sample
        pauses = np.random.random(total_samples) < pause_probability
        audio[pauses] *= 0.1  # Reduce amplitude during pauses
    
    elif content_type == "mixed":
        # Mixed content: speech + pauses + background
        audio = create_synthetic_audio(duration_sec, sample_rate, "speech")
        
        # Add longer pauses every 5-10 seconds
        pause_interval = 7  # seconds
        pause_duration = 1  # seconds
        
        for i in range(0, int(duration_sec), pause_interval):
            start_idx = int(i * sample_rate)
            end_idx = int(min((i + pause_duration) * sample_rate, len(audio)))
            if start_idx < len(audio):
                audio[start_idx:end_idx] *= 0.05  # Nearly silent
    
    # Ensure audio is in valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio

def save_synthetic_audio(audio: np.ndarray, filepath: Path, sample_rate: int = 16000):
    """Save synthetic audio to WAV file"""
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

def create_recognizer(model_dir: Path, **kwargs) -> sherpa_onnx.OnlineRecognizer:
    """Create sherpa-onnx OnlineRecognizer - reused from test_sherpa_proper.py"""
    
    # Find model files
    tokens_file = model_dir / "tokens.txt"
    if not tokens_file.exists():
        raise FileNotFoundError(f"tokens.txt not found in {model_dir}")
    
    # Look for transducer model files
    encoder_files = list(model_dir.glob("encoder*.onnx"))
    decoder_files = list(model_dir.glob("decoder*.onnx"))
    joiner_files = list(model_dir.glob("joiner*.onnx"))
    
    if encoder_files and decoder_files and joiner_files:
        # Prefer int8 quantized versions for stability
        def get_best_model(files):
            int8_files = [f for f in files if 'int8' in f.name]
            return int8_files[0] if int8_files else files[0]
        
        encoder_file = get_best_model(encoder_files)
        decoder_file = get_best_model(decoder_files)
        joiner_file = get_best_model(joiner_files)
        
        logger.info(f"Using transducer model: {encoder_file.name}")
        
        # Create recognizer using official API
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens_file),
            encoder=str(encoder_file),
            decoder=str(decoder_file),
            joiner=str(joiner_file),
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=kwargs.get('enable_endpoint_detection', True),
            rule1_min_trailing_silence=kwargs.get('rule1_min_trailing_silence', 2.4),
            rule2_min_trailing_silence=kwargs.get('rule2_min_trailing_silence', 1.2),
            rule3_min_utterance_length=kwargs.get('rule3_min_utterance_length', 300),
            decoding_method=kwargs.get('decoding_method', 'greedy_search'),
            max_active_paths=kwargs.get('max_active_paths', 4),
            hotwords_file=kwargs.get('hotwords_file', ''),
            hotwords_score=kwargs.get('hotwords_score', 1.5),
            blank_penalty=0.0,
            temperature_scale=kwargs.get('temperature_scale', 1.0),
        )
        
        return recognizer
    
    raise FileNotFoundError(f"No suitable model files found in {model_dir}")

def stress_test_audio(recognizer: sherpa_onnx.OnlineRecognizer,
                     audio_data: np.ndarray,
                     duration_sec: float,
                     chunk_size_ms: int = 200,
                     sample_rate: int = 16000) -> StressTestResult:
    """Run stress test on audio data"""
    
    logger.info(f"Starting stress test: {duration_sec:.1f}s audio, {chunk_size_ms}ms chunks")
    
    # Initialize monitoring
    monitor = DetailedResourceMonitor(sample_interval=0.5)
    monitor.start()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Create stream
    stream = recognizer.create_stream()
    
    # Processing state
    segments = []
    endpoint_count = 0
    errors = []
    chunk_times = []
    processing_gaps = []
    
    chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
    total_chunks = len(audio_data) // chunk_size_samples
    
    logger.info(f"Processing {total_chunks} chunks of {chunk_size_samples} samples each")
    
    start_time = time.time()
    last_chunk_time = start_time
    
    try:
        for i in range(0, len(audio_data), chunk_size_samples):
            chunk_start = time.time()
            
            # Calculate gap since last chunk
            gap_ms = (chunk_start - last_chunk_time) * 1000
            processing_gaps.append(gap_ms)
            
            # Get audio chunk
            chunk = audio_data[i:i + chunk_size_samples]
            
            try:
                # Feed audio to recognizer
                stream.accept_waveform(sample_rate, chunk.tolist())
                
                # Decode if ready
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                
                # Check for endpoint
                is_endpoint = recognizer.is_endpoint(stream)
                result = recognizer.get_result(stream)
                
                # Handle result
                result_text = result if isinstance(result, str) else getattr(result, 'text', str(result))
                
                if is_endpoint and result_text.strip():
                    segments.append(result_text.strip())
                    endpoint_count += 1
                    recognizer.reset(stream)
                    
                    if len(segments) % 10 == 0:  # Log every 10 segments
                        logger.info(f"Processed {len(segments)} segments, chunk {i//chunk_size_samples}/{total_chunks}")
                
            except Exception as e:
                error_msg = f"Chunk {i//chunk_size_samples}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
            
            # Record chunk processing time
            chunk_time = (time.time() - chunk_start) * 1000  # ms
            chunk_times.append(chunk_time)
            last_chunk_time = time.time()
            
            # Small delay to simulate real-time
            time.sleep(0.001)
            
            # Force garbage collection every 100 chunks to test stability
            if i % (100 * chunk_size_samples) == 0 and i > 0:
                gc.collect()
                
    except Exception as e:
        error_msg = f"Critical error during processing: {str(e)}"
        errors.append(error_msg)
        logger.error(error_msg)
    
    # Get final result
    try:
        result = recognizer.get_result(stream)
        result_text = result if isinstance(result, str) else getattr(result, 'text', str(result))
        if result_text.strip() and (not segments or result_text.strip() != segments[-1]):
            segments.append(result_text.strip())
    except Exception as e:
        errors.append(f"Final result error: {str(e)}")
    
    total_processing_time = time.time() - start_time
    
    # Stop monitoring
    monitor.stop()
    stats = monitor.get_stats()
    
    # Stop memory tracking
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Analyze results
    final_transcript = " ".join(segments)
    rtf = total_processing_time / duration_sec
    
    # Detect memory leak (>50MB growth sustained)
    memory_growth = stats['final_memory'] - stats['initial_memory']
    memory_leak = memory_growth > 50  # MB
    
    # Detect performance degradation (chunk times increasing over time)
    performance_degradation = False
    if len(chunk_times) > 100:
        early_times = np.mean(chunk_times[:50])
        late_times = np.mean(chunk_times[-50:])
        performance_degradation = late_times > early_times * 1.5  # 50% slower
    
    # Calculate average segment length
    avg_segment_length = duration_sec / len(segments) if segments else 0
    
    result = StressTestResult(
        model_name=f"stress-test-{duration_sec}s",
        audio_duration_sec=duration_sec,
        chunk_size_ms=chunk_size_ms,
        total_processing_time_sec=total_processing_time,
        real_time_factor=rtf,
        model_load_time_ms=0,  # Not measured in stress test
        initial_memory_mb=stats['initial_memory'],
        peak_memory_mb=stats['peak_memory'],
        final_memory_mb=stats['final_memory'],
        memory_growth_mb=memory_growth,
        memory_samples=stats['memory_samples'],
        avg_cpu_percent=stats['avg_cpu'],
        peak_cpu_percent=stats['peak_cpu'],
        cpu_samples=stats['cpu_samples'],
        total_segments=len(segments),
        avg_segment_length_sec=avg_segment_length,
        endpoint_detections=endpoint_count,
        processing_gaps_ms=processing_gaps,
        final_transcript=final_transcript,
        errors_encountered=errors,
        memory_leak_detected=memory_leak,
        performance_degradation=performance_degradation,
        chunk_processing_times=chunk_times
    )
    
    logger.info(f"Stress test completed:")
    logger.info(f"  Duration: {duration_sec:.1f}s")
    logger.info(f"  RTF: {rtf:.3f}")
    logger.info(f"  Segments: {len(segments)}")
    logger.info(f"  Memory growth: {memory_growth:.1f}MB")
    logger.info(f"  Errors: {len(errors)}")
    logger.info(f"  Memory leak detected: {memory_leak}")
    logger.info(f"  Performance degradation: {performance_degradation}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Sherpa-ONNX Stress Testing')
    parser.add_argument('--model-dir', type=Path, required=True,
                       help='Path to sherpa-onnx model directory')
    parser.add_argument('--durations', nargs='+', type=float, 
                       default=[10, 30, 60, 300],
                       help='Test durations in seconds (default: 10 30 60 300)')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Audio chunk size in milliseconds')
    parser.add_argument('--content-type', choices=['silence', 'tone', 'speech', 'mixed'],
                       default='mixed',
                       help='Type of synthetic audio to generate')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--save-audio', action='store_true',
                       help='Save generated synthetic audio files')
    
    args = parser.parse_args()
    
    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        return
    
    # Create recognizer once for all tests
    logger.info("Loading sherpa-onnx model...")
    load_start = time.time()
    
    try:
        recognizer = create_recognizer(args.model_dir)
        load_time = (time.time() - load_start) * 1000
        logger.info(f"Model loaded in {load_time:.1f}ms")
    except Exception as e:
        logger.error(f"Failed to create recognizer: {e}")
        return
    
    # Run stress tests for each duration
    results = []
    
    for duration in args.durations:
        logger.info(f"\n{'='*60}")
        logger.info(f"STRESS TEST: {duration}s duration")
        logger.info(f"{'='*60}")
        
        try:
            # Generate synthetic audio
            logger.info(f"Generating {duration}s of {args.content_type} audio...")
            audio_data = create_synthetic_audio(duration, content_type=args.content_type)
            
            # Save audio if requested
            if args.save_audio:
                audio_dir = Path(__file__).parent / "test_audio" / "stress"
                audio_dir.mkdir(exist_ok=True)
                audio_file = audio_dir / f"stress_{args.content_type}_{duration}s.wav"
                save_synthetic_audio(audio_data, audio_file)
                logger.info(f"Saved audio: {audio_file}")
            
            # Run stress test
            result = stress_test_audio(
                recognizer, audio_data, duration, args.chunk_size
            )
            result.model_name = f"sherpa-onnx-{args.model_dir.name}"
            result.model_load_time_ms = load_time
            results.append(result)
            
        except KeyboardInterrupt:
            logger.info("Stress test interrupted by user")
            break
        except Exception as e:
            logger.error(f"Stress test failed for {duration}s: {e}")
            continue
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / "results" / f"stress_test_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
    
    # Prepare results for JSON serialization (convert numpy arrays)
    json_results = []
    for result in results:
        result_dict = asdict(result)
        # Convert numpy arrays to lists and ensure all values are JSON serializable
        for key, value in result_dict.items():
            if hasattr(value, 'tolist'):
                result_dict[key] = value.tolist()
            elif isinstance(value, np.bool_):
                result_dict[key] = bool(value)
            elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                result_dict[key] = int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                result_dict[key] = float(value)
        json_results.append(result_dict)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("SHERPA-ONNX STRESS TEST SUMMARY")
        print("="*80)
        
        for result in results:
            print(f"\nDuration: {result.audio_duration_sec:.1f}s")
            print(f"  RTF: {result.real_time_factor:.3f}")
            print(f"  Memory: {result.initial_memory_mb:.1f} -> {result.final_memory_mb:.1f} MB (Δ{result.memory_growth_mb:+.1f})")
            print(f"  CPU: {result.avg_cpu_percent:.1f}% avg, {result.peak_cpu_percent:.1f}% peak")
            print(f"  Segments: {result.total_segments}")
            print(f"  Errors: {len(result.errors_encountered)}")
            
            # Warnings
            warnings = []
            if result.memory_leak_detected:
                warnings.append("⚠️  MEMORY LEAK DETECTED")
            if result.performance_degradation:
                warnings.append("⚠️  PERFORMANCE DEGRADATION")
            if result.errors_encountered:
                warnings.append(f"⚠️  {len(result.errors_encountered)} ERRORS")
            
            if warnings:
                print(f"  Issues: {' | '.join(warnings)}")
            else:
                print(f"  Status: ✅ STABLE")
        
        # Overall assessment
        print(f"\nOVERALL ASSESSMENT:")
        max_duration = max(r.audio_duration_sec for r in results)
        has_issues = any(r.memory_leak_detected or r.performance_degradation or r.errors_encountered for r in results)
        
        if has_issues:
            print("⚠️  Issues detected - review individual test results")
        else:
            print(f"✅ Stable performance up to {max_duration:.1f}s duration")
        
        avg_rtf = np.mean([r.real_time_factor for r in results])
        print(f"Average RTF: {avg_rtf:.3f} ({'REAL-TIME' if avg_rtf < 1.0 else 'SLOWER THAN REAL-TIME'})")

if __name__ == "__main__":
    main()