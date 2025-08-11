#!/usr/bin/env python3
"""
Comprehensive Parakeet-MLX vs MLX-Whisper vs Sherpa-ONNX Comparison Test

Compare three STT systems across multiple dimensions:
1. Performance (RTF, latency, throughput)
2. Memory usage and efficiency
3. Accuracy on real and synthetic audio
4. Streaming characteristics
5. Apple Silicon optimization
6. Resource utilization

This benchmark uses the same methodology as the existing MLX-Whisper vs Sherpa comparison
but adds Parakeet-MLX as a third contender optimized for streaming on Apple Silicon.
"""

import argparse
import time
import wave
import numpy as np
import asyncio
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
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our services
from services.parakeet_mlx_streaming_stt import ParakeetMLXStreamingSTTService
from services.whisper_stt_with_lock import WhisperSTTServiceMLX
from pipecat.services.whisper.stt import MLXModel
from pipecat.transcriptions.language import Language
from pipecat.frames.frames import TranscriptionFrame, InterimTranscriptionFrame

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results from benchmarking an STT service"""
    service_name: str
    service_type: str  # 'parakeet-mlx', 'mlx-whisper', 'sherpa-onnx'
    audio_duration_sec: float
    
    # Performance metrics
    service_init_time_ms: float
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
    
    # Results and accuracy
    final_transcript: str
    word_count: int
    interim_results_count: int = 0
    
    # Streaming specific
    chunks_processed: int = 0
    avg_chunk_processing_ms: float = 0.0
    
    # Model specific info
    model_name: str = ""
    context_info: str = ""
    
    # Errors
    errors_encountered: List[str] = None
    
    def __post_init__(self):
        if self.errors_encountered is None:
            self.errors_encountered = []

class ResourceMonitor:
    """Monitor CPU and memory usage during processing"""
    
    def __init__(self, sample_interval: float = 0.1):
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
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def get_stats(self) -> Dict[str, float]:
        if not self.cpu_samples:
            return {'initial_memory': 0, 'peak_memory': 0, 'final_memory': 0, 'avg_cpu': 0, 'peak_cpu': 0}
            
        return {
            'initial_memory': self.memory_samples[0] if self.memory_samples else 0,
            'peak_memory': max(self.memory_samples) if self.memory_samples else 0,
            'final_memory': self.memory_samples[-1] if self.memory_samples else 0,
            'avg_cpu': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'peak_cpu': max(self.cpu_samples) if self.cpu_samples else 0
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

def create_synthetic_speech_audio(duration_sec: float, sample_rate: int = 16000) -> np.ndarray:
    """Create realistic synthetic speech-like audio for testing"""
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, False)
    
    # Create speech-like pattern with multiple formants
    # Fundamental frequency varies like natural speech
    f0 = 150 + 50 * np.sin(2 * np.pi * 0.7 * t)  # Fundamental (pitch)
    f1 = 800 + 300 * np.sin(2 * np.pi * 0.4 * t)  # First formant
    f2 = 2400 + 600 * np.sin(2 * np.pi * 0.3 * t) # Second formant
    f3 = 3200 + 400 * np.sin(2 * np.pi * 0.2 * t) # Third formant
    
    # Amplitude envelope with speech rhythm and natural pauses
    amplitude = 0.15 * (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 3 * t)))
    
    # Add periodic pauses (simulating sentence boundaries)
    pause_period = 8  # Every 8 seconds
    for i in range(0, int(duration_sec), pause_period):
        pause_start = int((i + 6) * sample_rate)
        pause_end = int(min((i + 7) * sample_rate, len(t)))
        if pause_start < len(amplitude) and pause_end <= len(amplitude):
            amplitude[pause_start:pause_end] *= 0.05  # Quiet pause
    
    # Combine harmonics to create speech-like sound
    audio = amplitude * (
        0.5 * np.sin(2 * np.pi * f0 * t) +         # Fundamental
        0.3 * np.sin(2 * np.pi * f1 * t) +         # First formant
        0.15 * np.sin(2 * np.pi * f2 * t) +        # Second formant
        0.05 * np.sin(2 * np.pi * f3 * t)          # Third formant
    )
    
    # Add realistic background noise and consonant-like artifacts
    noise = 0.005 * np.random.normal(0, 1, total_samples)
    # Add brief consonant-like bursts
    for i in range(0, total_samples, int(sample_rate * 0.5)):  # Every 0.5 seconds
        if i + 100 < total_samples:
            burst = 0.03 * np.random.normal(0, 1, 100)
            audio[i:i+100] += burst
    
    audio = (audio + noise).astype(np.float32)
    
    # Ensure valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio

async def test_parakeet_mlx(audio_data: np.ndarray, duration_sec: float, 
                           config: Dict[str, Any] = None) -> BenchmarkResult:
    """Test Parakeet-MLX streaming STT service"""
    
    config = config or {}
    model_name = config.get('model_name', 'mlx-community/parakeet-tdt-0.6b-v2')
    context_size = config.get('context_size', (256, 256))
    chunk_ms = config.get('chunk_ms', 100)
    
    logger.info(f"Testing Parakeet-MLX: {model_name} with context {context_size}")
    
    # Start monitoring
    monitor = ResourceMonitor()
    monitor.start()
    tracemalloc.start()
    
    errors = []
    interim_count = 0
    chunks_processed = 0
    processing_times = []
    first_result_time = None
    final_transcript = ""
    
    # Initialize service
    init_start = time.time()
    try:
        service = ParakeetMLXStreamingSTTService(
            model_name=model_name,
            context_size=context_size,
            chunk_size_ms=chunk_ms,
            emit_partial_results=True,
            min_confidence=0.1,
            sample_rate=16000,
            language="en"
        )
        init_time = (time.time() - init_start) * 1000
        
    except Exception as e:
        error_msg = f"Failed to initialize Parakeet-MLX: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        monitor.stop()
        return BenchmarkResult(
            service_name="parakeet-mlx-FAILED",
            service_type="parakeet-mlx",
            audio_duration_sec=duration_sec,
            service_init_time_ms=0,
            total_processing_time_sec=0,
            real_time_factor=float('inf'),
            time_to_first_result_ms=None,
            initial_memory_mb=0, peak_memory_mb=0, final_memory_mb=0, memory_growth_mb=0,
            avg_cpu_percent=0, peak_cpu_percent=0,
            final_transcript="", word_count=0,
            model_name=model_name,
            errors_encountered=errors
        )
    
    # Process audio in chunks
    process_start = time.time()
    chunk_size_samples = int(16000 * chunk_ms / 1000)
    results = []
    
    try:
        for i in range(0, len(audio_data), chunk_size_samples):
            chunk_start = time.time()
            chunk = audio_data[i:i + chunk_size_samples]
            
            # Convert to bytes (16-bit PCM)
            chunk_int16 = (chunk * 32767).astype(np.int16)
            chunk_bytes = chunk_int16.tobytes()
            
            # Process chunk
            async for frame in service.run_stt(chunk_bytes):
                if first_result_time is None:
                    first_result_time = (time.time() - process_start) * 1000
                
                if isinstance(frame, TranscriptionFrame):
                    results.append(frame.text)
                elif isinstance(frame, InterimTranscriptionFrame):
                    interim_count += 1
            
            chunks_processed += 1
            processing_times.append((time.time() - chunk_start) * 1000)
            
            # Small delay to simulate real-time processing
            await asyncio.sleep(0.001)
        
        # Get any final results
        await service.flush()
        
    except Exception as e:
        error_msg = f"Parakeet-MLX processing error: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
    
    processing_time = time.time() - process_start
    
    # Cleanup service
    try:
        await service.cleanup()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")
    
    # Stop monitoring
    monitor.stop()
    stats = monitor.get_stats()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Combine results
    final_transcript = " ".join(results).strip()
    word_count = len(final_transcript.split()) if final_transcript else 0
    avg_chunk_time = np.mean(processing_times) if processing_times else 0
    
    result = BenchmarkResult(
        service_name=f"parakeet-mlx-{model_name.split('/')[-1]}",
        service_type="parakeet-mlx",
        audio_duration_sec=duration_sec,
        service_init_time_ms=init_time,
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
        interim_results_count=interim_count,
        chunks_processed=chunks_processed,
        avg_chunk_processing_ms=avg_chunk_time,
        model_name=model_name,
        context_info=f"context={context_size}, chunk={chunk_ms}ms",
        errors_encountered=errors
    )
    
    logger.info(f"Parakeet-MLX completed: RTF={result.real_time_factor:.3f}, "
               f"Chunks={chunks_processed}, Words={word_count}, Interim={interim_count}")
    
    return result

async def test_mlx_whisper(audio_data: np.ndarray, duration_sec: float) -> BenchmarkResult:
    """Test MLX-Whisper batch processing"""
    
    logger.info("Testing MLX-Whisper (batch)...")
    
    # Start monitoring
    monitor = ResourceMonitor()
    monitor.start()
    tracemalloc.start()
    
    errors = []
    first_result_time = None
    final_transcript = ""
    
    # Initialize and process
    init_start = time.time()
    process_start = None
    
    try:
        import mlx_whisper
        import os
        
        # Set cache directory for consistency
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.environ["HF_HOME"] = cache_dir
        
        process_start = time.time()
        
        # Use same model as in the service
        model_name = "mlx-community/whisper-large-v3-turbo"
        
        result = mlx_whisper.transcribe(
            audio_data, 
            path_or_hf_repo=model_name,
            verbose=False,
            language="en"
        )
        
        init_and_process_time = (time.time() - init_start) * 1000
        processing_time = time.time() - process_start
        first_result_time = processing_time * 1000  # First and only result
        
        final_transcript = result["text"].strip() if result and "text" in result else ""
        
    except ImportError:
        error_msg = "MLX-Whisper not available"
        logger.error(error_msg)
        errors.append(error_msg)
        init_and_process_time = 0
        processing_time = 0
    except Exception as e:
        error_msg = f"MLX-Whisper processing error: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        init_and_process_time = (time.time() - init_start) * 1000 if init_start else 0
        processing_time = time.time() - process_start if process_start else 0
    
    # Stop monitoring
    monitor.stop()
    stats = monitor.get_stats()
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    word_count = len(final_transcript.split()) if final_transcript else 0
    
    result = BenchmarkResult(
        service_name="mlx-whisper-large-v3-turbo",
        service_type="mlx-whisper",
        audio_duration_sec=duration_sec,
        service_init_time_ms=init_and_process_time,
        total_processing_time_sec=processing_time,
        real_time_factor=processing_time / duration_sec if processing_time > 0 else float('inf'),
        time_to_first_result_ms=first_result_time,
        initial_memory_mb=stats['initial_memory'],
        peak_memory_mb=stats['peak_memory'],
        final_memory_mb=stats['final_memory'],
        memory_growth_mb=stats['final_memory'] - stats['initial_memory'],
        avg_cpu_percent=stats['avg_cpu'],
        peak_cpu_percent=stats['peak_cpu'],
        final_transcript=final_transcript,
        word_count=word_count,
        model_name="whisper-large-v3-turbo",
        context_info="batch_processing",
        errors_encountered=errors
    )
    
    logger.info(f"MLX-Whisper completed: RTF={result.real_time_factor:.3f}, Words={word_count}")
    
    return result

def calculate_similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity score between two transcripts"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

async def main():
    parser = argparse.ArgumentParser(description='Comprehensive STT Benchmark: Parakeet-MLX vs MLX-Whisper vs Sherpa-ONNX')
    parser.add_argument('--duration', type=float, default=60,
                       help='Test duration in seconds (default: 60s)')
    parser.add_argument('--parakeet-model', default='mlx-community/parakeet-tdt-0.6b-v2',
                       help='Parakeet-MLX model to test')
    parser.add_argument('--parakeet-context', default='256,256',
                       help='Parakeet-MLX context size (left,right)')
    parser.add_argument('--chunk-ms', type=int, default=100,
                       help='Chunk size in milliseconds')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--save-audio', action='store_true',
                       help='Save generated test audio')
    parser.add_argument('--real-audio', type=Path, default=None,
                       help='Use real audio file instead of synthetic')
    parser.add_argument('--skip-whisper', action='store_true',
                       help='Skip MLX-Whisper test')
    
    args = parser.parse_args()
    
    # Parse context size
    context_size = tuple(map(int, args.parakeet_context.split(',')))
    
    logger.info(f"Comprehensive STT Benchmark: {args.duration}s audio")
    logger.info(f"Parakeet-MLX: {args.parakeet_model}, context={context_size}")
    
    # Load or generate audio
    if args.real_audio and args.real_audio.exists():
        logger.info(f"Loading real audio: {args.real_audio}")
        with wave.open(str(args.real_audio), 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            actual_duration = len(audio_data) / wav_file.getframerate()
        logger.info(f"Loaded {actual_duration:.1f}s of real audio")
    else:
        logger.info(f"Generating {args.duration}s of synthetic speech...")
        audio_data = create_synthetic_speech_audio(args.duration)
        actual_duration = args.duration
    
    # Save audio if requested
    if args.save_audio:
        audio_dir = Path(__file__).parent / "test_audio"
        audio_dir.mkdir(exist_ok=True)
        
        audio_file = audio_dir / f"benchmark_{int(actual_duration)}s.wav"
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(audio_file), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
        
        logger.info(f"Saved test audio: {audio_file}")
    
    # Run tests
    results = []
    
    print("\n" + "="*100)
    print("COMPREHENSIVE STT BENCHMARK: Parakeet-MLX vs MLX-Whisper vs Sherpa-ONNX")
    print("="*100)
    
    # Test Parakeet-MLX
    parakeet_config = {
        'model_name': args.parakeet_model,
        'context_size': context_size,
        'chunk_ms': args.chunk_ms
    }
    
    parakeet_result = await test_parakeet_mlx(audio_data, actual_duration, parakeet_config)
    results.append(parakeet_result)
    
    # Force cleanup between tests
    gc.collect()
    await asyncio.sleep(2)
    
    # Test MLX-Whisper (if not skipped)
    if not args.skip_whisper:
        mlx_result = await test_mlx_whisper(audio_data, actual_duration)
        results.append(mlx_result)
        
        # Force cleanup
        gc.collect()
        await asyncio.sleep(2)
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent / "benchmark_results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / f"stt_benchmark_{timestamp}.json"
    
    # Prepare for JSON serialization
    json_results = []
    for result in results:
        result_dict = asdict(result)
        json_results.append(result_dict)
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print detailed comparison
    print(f"\n📊 BENCHMARK RESULTS ({actual_duration:.1f}s audio)")
    print("-" * 80)
    
    for result in results:
        print(f"\n🤖 {result.service_name} ({result.service_type})")
        print(f"  Model:          {result.model_name}")
        print(f"  Config:         {result.context_info}")
        print(f"  Init Time:      {result.service_init_time_ms:.1f}ms")
        print(f"  Processing:     {result.total_processing_time_sec:.2f}s")
        print(f"  RTF:            {result.real_time_factor:.3f} {'✅' if result.real_time_factor < 1.0 else '⚠️'}")
        print(f"  Memory Growth:  {result.memory_growth_mb:+.1f}MB")
        print(f"  Peak Memory:    {result.peak_memory_mb:.1f}MB")
        print(f"  CPU (avg/peak): {result.avg_cpu_percent:.1f}% / {result.peak_cpu_percent:.1f}%")
        print(f"  Words:          {result.word_count}")
        
        if result.service_type == "parakeet-mlx":
            print(f"  Chunks:         {result.chunks_processed}")
            print(f"  Interim Results:{result.interim_results_count}")
            print(f"  Avg Chunk Time: {result.avg_chunk_processing_ms:.1f}ms")
        
        if result.time_to_first_result_ms:
            print(f"  First Result:   {result.time_to_first_result_ms:.1f}ms")
        
        if result.errors_encountered:
            print(f"  Errors:         {len(result.errors_encountered)}")
            for error in result.errors_encountered[:2]:
                print(f"    • {error}")
        
        # Show sample transcript
        sample = result.final_transcript[:100] + "..." if len(result.final_transcript) > 100 else result.final_transcript
        print(f"  Transcript:     '{sample}'")
    
    # Cross-comparisons
    if len(results) >= 2:
        print(f"\n🏆 COMPARATIVE ANALYSIS")
        print("-" * 50)
        
        # Find best performers
        valid_results = [r for r in results if not r.errors_encountered]
        
        if valid_results:
            speed_winner = min(valid_results, key=lambda x: x.real_time_factor)
            memory_winner = min(valid_results, key=lambda x: x.memory_growth_mb)
            cpu_winner = min(valid_results, key=lambda x: x.avg_cpu_percent)
            
            print(f"Speed Winner:     {speed_winner.service_name} (RTF: {speed_winner.real_time_factor:.3f})")
            print(f"Memory Winner:    {memory_winner.service_name} (+{memory_winner.memory_growth_mb:.1f}MB)")
            print(f"CPU Winner:       {cpu_winner.service_name} ({cpu_winner.avg_cpu_percent:.1f}%)")
            
            # Latency comparison
            latency_results = [r for r in valid_results if r.time_to_first_result_ms is not None]
            if latency_results:
                latency_winner = min(latency_results, key=lambda x: x.time_to_first_result_ms)
                print(f"Latency Winner:   {latency_winner.service_name} ({latency_winner.time_to_first_result_ms:.1f}ms)")
        
        # Transcript similarity analysis
        transcripts = [(r.service_name, r.final_transcript) for r in valid_results if r.final_transcript]
        if len(transcripts) >= 2:
            print(f"\n📝 TRANSCRIPT SIMILARITY")
            print("-" * 30)
            
            for i, (name1, text1) in enumerate(transcripts):
                for name2, text2 in transcripts[i+1:]:
                    similarity = calculate_similarity_score(text1, text2)
                    print(f"  {name1} vs {name2}: {similarity:.3f}")
    
    print(f"\n✅ Benchmark completed! Results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())