#!/usr/bin/env python3
"""
Sherpa-ONNX Streaming STT Benchmarking Test

Comprehensive benchmarking framework to evaluate sherpa-onnx models for:
1. Streaming STT accuracy (especially names/URLs)
2. Latency performance (model load time, TTFT, end-to-end)
3. Resource consumption (CPU, memory, processing speed)

Usage:
    python test_sherpa_streaming_benchmark.py --model all --test-suite standard
    python test_sherpa_streaming_benchmark.py --model zipformer-en-2023-06-26 --test-suite entities
"""

import argparse
import asyncio
import json
import time
import traceback
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import psutil
import threading
from queue import Queue, Empty
import numpy as np
import os
import sys

# Add server root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.sherpa_streaming_stt_v2 import SherpaOnlineSTTService
from services.whisper_stt_with_lock import WhisperSTTServiceMLX

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics"""
    model_name: str
    model_load_time_ms: float
    total_audio_duration_ms: float
    processing_time_ms: float
    real_time_factor: float  # processing_time / audio_duration
    
    # Latency metrics
    time_to_first_token_ms: Optional[float] = None
    time_to_final_transcript_ms: Optional[float] = None
    
    # Accuracy metrics
    reference_text: str = ""
    predicted_text: str = ""
    word_error_rate: float = 0.0
    character_error_rate: float = 0.0
    entity_accuracy: float = 0.0  # For names/URLs
    
    # Resource metrics
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Streaming metrics
    num_partial_results: int = 0
    num_final_results: int = 0
    chunk_size_ms: int = 0
    
    error_message: str = ""

class ResourceMonitor:
    """Monitor CPU and memory usage during testing"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        if not self.cpu_samples:
            return 0.0, 0.0, 0.0, 0.0
            
        return (
            max(self.cpu_samples),
            np.mean(self.cpu_samples),
            max(self.memory_samples),
            np.mean(self.memory_samples)
        )
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        while self.monitoring:
            try:
                cpu = process.cpu_percent()
                memory = process.memory_info().rss / 1024 / 1024  # MB
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
                time.sleep(0.1)  # Sample every 100ms
            except:
                break

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Simple Levenshtein distance for words
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate"""
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    
    # Simple Levenshtein distance for characters
    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
    
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j
        
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    
    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

def extract_entities(text: str) -> List[str]:
    """Extract names, URLs, and other entities from text"""
    import re
    entities = []
    
    # URLs
    url_pattern = r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    entities.extend(re.findall(url_pattern, text))
    
    # Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities.extend(re.findall(email_pattern, text))
    
    # Proper names (capitalized words)
    name_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
    entities.extend(re.findall(name_pattern, text))
    
    return entities

def calculate_entity_accuracy(reference: str, hypothesis: str) -> float:
    """Calculate accuracy for extracted entities"""
    ref_entities = set(extract_entities(reference))
    hyp_entities = set(extract_entities(hypothesis))
    
    if len(ref_entities) == 0:
        return 1.0 if len(hyp_entities) == 0 else 0.0
    
    correct = len(ref_entities & hyp_entities)
    return correct / len(ref_entities)

def load_audio_file(filepath: Path) -> Tuple[np.ndarray, int]:
    """Load audio file (WAV/MP3) and return audio data and sample rate"""
    if filepath.suffix.lower() == '.wav':
        with wave.open(str(filepath), 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(frames, dtype=np.int32)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Handle stereo by taking first channel
            if channels == 2:
                audio_data = audio_data[::2]
            
            return audio_data, sample_rate
    
    elif filepath.suffix.lower() in ['.mp3', '.m4a']:
        # For MP3/M4A files, we need additional libraries
        try:
            import librosa
            # Load with librosa (handles MP3/M4A)
            audio_data, sample_rate = librosa.load(str(filepath), sr=None, mono=True)
            # Convert to int16
            audio_data = (audio_data * 32767).astype(np.int16)
            return audio_data, int(sample_rate)
        except ImportError:
            try:
                import subprocess
                import tempfile
                
                # Fallback: use ffmpeg to convert to WAV
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    subprocess.run([
                        'ffmpeg', '-i', str(filepath), 
                        '-acodec', 'pcm_s16le', 
                        '-ac', '1',  # mono
                        '-y',  # overwrite
                        temp_wav.name
                    ], check=True, capture_output=True)
                    
                    # Load the converted WAV file
                    result = load_audio_file(Path(temp_wav.name))
                    
                    # Clean up
                    Path(temp_wav.name).unlink()
                    return result
                    
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise ValueError(f"Cannot load {filepath.suffix} file: {filepath}. Install librosa or ffmpeg.")
    
    else:
        raise ValueError(f"Unsupported audio format: {filepath.suffix}")

def chunk_audio(audio_data: np.ndarray, sample_rate: int, chunk_size_ms: int) -> List[np.ndarray]:
    """Split audio into chunks for streaming simulation"""
    chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
    chunks = []
    
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples]
        chunks.append(chunk)
    
    return chunks

class StreamingSimulator:
    """Simulate streaming by feeding audio chunks to STT service"""
    
    def __init__(self, stt_service, chunk_size_ms: int = 200):
        self.stt_service = stt_service
        self.chunk_size_ms = chunk_size_ms
        self.partial_results = []
        self.final_results = []
        self.first_result_time = None
        self.final_result_time = None
        
    async def process_audio_file(self, filepath: Path) -> str:
        """Process audio file in streaming chunks"""
        logger.info(f"Processing audio file: {filepath}")
        
        # Load audio
        audio_data, sample_rate = load_audio_file(filepath)
        
        # For sherpa services, assume 16kHz target sample rate
        target_sample_rate = 16000  # Sherpa models typically use 16kHz
        
        # Convert to required sample rate if needed
        if sample_rate != target_sample_rate:
            logger.warning(f"Resampling from {sample_rate} to {target_sample_rate}")
            
            # Simple resampling (for testing - use proper resampling in production)
            ratio = target_sample_rate / sample_rate
            audio_data = np.interp(
                np.linspace(0, len(audio_data), int(len(audio_data) * ratio)),
                np.arange(len(audio_data)),
                audio_data
            ).astype(audio_data.dtype)
            sample_rate = target_sample_rate
        
        # Split into chunks
        chunks = chunk_audio(audio_data, sample_rate, self.chunk_size_ms)
        logger.info(f"Split audio into {len(chunks)} chunks of {self.chunk_size_ms}ms")
        
        # Process chunks
        start_time = time.time()
        self.partial_results = []
        self.final_results = []
        self.first_result_time = None
        self.final_result_time = None
        
        for i, chunk in enumerate(chunks):
            # Convert to bytes (16-bit PCM)
            chunk_bytes = chunk.astype(np.int16).tobytes()
            
            # Process chunk
            async for frame in self.stt_service.run_stt(chunk_bytes):
                current_time = time.time()
                
                if hasattr(frame, 'text') and frame.text.strip():
                    if self.first_result_time is None:
                        self.first_result_time = current_time - start_time
                    
                    # Check if this is a final result
                    if hasattr(frame, '__class__') and 'TranscriptionFrame' in frame.__class__.__name__:
                        self.final_results.append(frame.text)
                        self.final_result_time = current_time - start_time
                        logger.info(f"Final result {len(self.final_results)}: '{frame.text}'")
                    else:
                        self.partial_results.append(frame.text)
                        logger.debug(f"Partial result {len(self.partial_results)}: '{frame.text}'")
            
            # Small delay to simulate real-time streaming
            await asyncio.sleep(0.01)
        
        # Combine final results
        final_text = ' '.join(self.final_results)
        
        return final_text
    
    def get_streaming_metrics(self) -> Tuple[int, int, Optional[float], Optional[float]]:
        """Get streaming-specific metrics"""
        return (
            len(self.partial_results),
            len(self.final_results),
            self.first_result_time,
            self.final_result_time
        )

async def benchmark_model(
    model_config: Dict[str, Any],
    test_files: List[Tuple[Path, str]],  # (filepath, reference_text)
    chunk_size_ms: int = 200
) -> List[BenchmarkMetrics]:
    """Benchmark a single model configuration"""
    model_name = model_config.get('name', 'unknown')
    logger.info(f"Benchmarking model: {model_name}")
    
    results = []
    stt_service = None
    
    try:
        # Initialize model and measure load time
        load_start = time.time()
        
        if model_config['type'] == 'sherpa':
            stt_service = SherpaOnlineSTTService(
                model_dir=model_config['model_dir'],
                chunk_size_ms=chunk_size_ms,
                sample_rate=16000,  # Explicitly set sample rate
                **model_config.get('params', {})
            )
            logger.info(f"STT service sample_rate after init: {getattr(stt_service, 'sample_rate', 'unknown')}")
        elif model_config['type'] == 'whisper':
            stt_service = WhisperSTTServiceMLX(
                model=model_config.get('model', 'LARGE_V3_TURBO_Q4')
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        load_time = (time.time() - load_start) * 1000  # Convert to ms
        
        simulator = StreamingSimulator(stt_service, chunk_size_ms)
        
        # Process each test file
        for filepath, reference_text in test_files:
            logger.info(f"Testing {model_name} on {filepath.name}")
            
            # Get audio duration
            audio_data, sample_rate = load_audio_file(filepath)
            duration_ms = len(audio_data) / sample_rate * 1000
            
            # Start resource monitoring
            monitor = ResourceMonitor()
            monitor.start_monitoring()
            
            try:
                # Process file
                process_start = time.time()
                predicted_text = await simulator.process_audio_file(filepath)
                process_time = (time.time() - process_start) * 1000
                
                # Stop monitoring
                peak_cpu, avg_cpu, peak_mem, avg_mem = monitor.stop_monitoring()
                
                # Get streaming metrics
                num_partial, num_final, ttft, ttfr = simulator.get_streaming_metrics()
                
                # Calculate accuracy metrics
                wer = calculate_wer(reference_text, predicted_text)
                cer = calculate_cer(reference_text, predicted_text)
                entity_acc = calculate_entity_accuracy(reference_text, predicted_text)
                
                # Create metrics object
                metrics = BenchmarkMetrics(
                    model_name=model_name,
                    model_load_time_ms=load_time,
                    total_audio_duration_ms=duration_ms,
                    processing_time_ms=process_time,
                    real_time_factor=process_time / duration_ms,
                    time_to_first_token_ms=ttft * 1000 if ttft else None,
                    time_to_final_transcript_ms=ttfr * 1000 if ttfr else None,
                    reference_text=reference_text,
                    predicted_text=predicted_text,
                    word_error_rate=wer,
                    character_error_rate=cer,
                    entity_accuracy=entity_acc,
                    peak_cpu_percent=peak_cpu,
                    avg_cpu_percent=avg_cpu,
                    peak_memory_mb=peak_mem,
                    avg_memory_mb=avg_mem,
                    num_partial_results=num_partial,
                    num_final_results=num_final,
                    chunk_size_ms=chunk_size_ms
                )
                
                results.append(metrics)
                
                logger.info(f"Results for {filepath.name}:")
                logger.info(f"  WER: {wer:.3f}, CER: {cer:.3f}, Entity Acc: {entity_acc:.3f}")
                logger.info(f"  RTF: {metrics.real_time_factor:.3f}, TTFT: {ttft:.3f}s" + 
                          f", Load time: {load_time:.1f}ms")
                
            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}")
                monitor.stop_monitoring()
                
                # Create error metrics
                metrics = BenchmarkMetrics(
                    model_name=model_name,
                    model_load_time_ms=load_time,
                    total_audio_duration_ms=duration_ms,
                    processing_time_ms=0,
                    real_time_factor=0,
                    reference_text=reference_text,
                    predicted_text="",
                    word_error_rate=1.0,
                    chunk_size_ms=chunk_size_ms,
                    error_message=str(e)
                )
                results.append(metrics)
        
    except Exception as e:
        logger.error(f"Failed to initialize model {model_name}: {e}")
        traceback.print_exc()
        
        # Create error metrics for initialization failure
        metrics = BenchmarkMetrics(
            model_name=model_name,
            model_load_time_ms=0,
            total_audio_duration_ms=0,
            processing_time_ms=0,
            real_time_factor=0,
            chunk_size_ms=chunk_size_ms,
            error_message=f"Model initialization failed: {str(e)}"
        )
        results.append(metrics)
    
    finally:
        # Cleanup
        if stt_service and hasattr(stt_service, 'cleanup'):
            stt_service.cleanup()
    
    return results

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get available sherpa and whisper models"""
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    models = {
        # Sherpa models
        'zipformer-en-2023-06-26': {
            'name': 'zipformer-en-2023-06-26',
            'type': 'sherpa',
            'model_dir': str(models_dir / "sherpa-onnx-streaming-zipformer-en-2023-06-26"),
            'params': {
                'enable_endpoint_detection': True,
                'max_active_paths': 4,
                'emit_partial_results': False
            }
        },
        'zipformer-en-2023-06-26-int8': {
            'name': 'zipformer-en-2023-06-26-int8',
            'type': 'sherpa',
            'model_dir': str(models_dir / "sherpa-onnx-streaming-zipformer-en-2023-06-26"),
            'params': {
                'enable_endpoint_detection': True,
                'max_active_paths': 4,
                'emit_partial_results': False,
                # int8 models are auto-detected by preferring quantized versions
            }
        },
        'zipformer-en-20M': {
            'name': 'zipformer-en-20M',
            'type': 'sherpa',
            'model_dir': str(models_dir / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"),
            'params': {
                'enable_endpoint_detection': True,
                'max_active_paths': 4,
                'emit_partial_results': False
            }
        },
        'nemo-10lang': {
            'name': 'nemo-10lang',
            'type': 'sherpa',
            'model_dir': str(models_dir / "sherpa-nemo-10lang"),
            'params': {
                'enable_endpoint_detection': True,
                'max_active_paths': 4,
                'emit_partial_results': False
            }
        },
        'sherpa-en': {
            'name': 'sherpa-en',
            'type': 'sherpa',
            'model_dir': str(models_dir / "sherpa-en"),
            'params': {
                'enable_endpoint_detection': True,
                'max_active_paths': 4,
                'emit_partial_results': False
            }
        },
        
        # Whisper baseline
        'whisper-large-v3-turbo': {
            'name': 'whisper-large-v3-turbo',
            'type': 'whisper',
            'model': 'LARGE_V3_TURBO_Q4'
        }
    }
    
    # Filter to only available models
    available_models = {}
    for name, config in models.items():
        if config['type'] == 'sherpa':
            if Path(config['model_dir']).exists():
                available_models[name] = config
            else:
                logger.warning(f"Model directory not found: {config['model_dir']}")
        elif config['type'] == 'whisper':
            available_models[name] = config
    
    return available_models

def load_test_suite(suite_name: str) -> List[Tuple[Path, str]]:
    """Load test files for a given test suite"""
    test_dir = Path(__file__).parent / "test_audio"
    
    if suite_name == "standard":
        # Use existing sherpa test files
        sherpa_test_dir = Path(__file__).parent.parent.parent / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26" / "test_wavs"
        
        test_files = []
        if sherpa_test_dir.exists():
            trans_file = sherpa_test_dir / "trans.txt"
            if trans_file.exists():
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            wav_file = sherpa_test_dir / parts[0]
                            if wav_file.exists():
                                test_files.append((wav_file, parts[1]))
        
        return test_files
    
    elif suite_name == "entities":
        # Entity-focused test files
        entities_dir = test_dir / "entities"
        test_files = []
        
        # Load from references file if it exists
        references_file = entities_dir / "references.txt"
        if references_file.exists():
            with open(references_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        audio_filename, reference_text = parts
                        audio_file = entities_dir / audio_filename
                        
                        # Check for both WAV and MP3 files
                        if audio_file.exists():
                            test_files.append((audio_file, reference_text))
                        else:
                            # Try with different extensions
                            for ext in ['.wav', '.mp3', '.m4a']:
                                alt_file = entities_dir / (audio_filename.replace('.wav', ext))
                                if alt_file.exists():
                                    test_files.append((alt_file, reference_text))
                                    break
        
        # Also scan for any additional audio files
        for audio_file in entities_dir.glob("*.wav"):
            if not any(test_file[0].name == audio_file.name for test_file in test_files):
                # Use filename as reference text if no reference available
                test_files.append((audio_file, f"[No reference for {audio_file.name}]"))
        
        for audio_file in entities_dir.glob("*.mp3"):
            if not any(test_file[0].name == audio_file.name for test_file in test_files):
                test_files.append((audio_file, f"[No reference for {audio_file.name}]"))
        
        if not test_files:
            logger.warning("No entity test files found - using standard suite")
            return load_test_suite("standard")
        
        return test_files
    
    elif suite_name == "realworld":
        # Real-world test files
        realworld_dir = test_dir / "realworld"
        test_files = []
        
        # Load from references file if it exists
        references_file = realworld_dir / "references.txt"
        if references_file.exists():
            with open(references_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        audio_filename, reference_text = parts
                        audio_file = realworld_dir / audio_filename
                        
                        # Check for both WAV and MP3 files
                        if audio_file.exists():
                            test_files.append((audio_file, reference_text))
                        else:
                            # Try with different extensions
                            for ext in ['.wav', '.mp3', '.m4a']:
                                alt_file = realworld_dir / (audio_filename.replace('.wav', ext))
                                if alt_file.exists():
                                    test_files.append((alt_file, reference_text))
                                    break
        
        # Also scan for any additional audio files
        for audio_file in realworld_dir.glob("*.wav"):
            if not any(test_file[0].name == audio_file.name for test_file in test_files):
                test_files.append((audio_file, f"[No reference for {audio_file.name}]"))
        
        for audio_file in realworld_dir.glob("*.mp3"):
            if not any(test_file[0].name == audio_file.name for test_file in test_files):
                test_files.append((audio_file, f"[No reference for {audio_file.name}]"))
        
        if not test_files:
            logger.warning("No real-world test files found - using standard suite")
            return load_test_suite("standard")
        
        return test_files
    
    else:
        raise ValueError(f"Unknown test suite: {suite_name}")

def save_results(results: List[BenchmarkMetrics], output_file: Path):
    """Save benchmark results to JSON file"""
    results_dict = [asdict(result) for result in results]
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")

def print_summary(results: List[BenchmarkMetrics]):
    """Print summary of benchmark results"""
    print("\n" + "="*80)
    print("SHERPA-ONNX STREAMING STT BENCHMARK RESULTS")
    print("="*80)
    
    # Group by model
    model_results = {}
    for result in results:
        if result.model_name not in model_results:
            model_results[result.model_name] = []
        model_results[result.model_name].append(result)
    
    # Print results by model
    for model_name, model_metrics in model_results.items():
        print(f"\n📊 {model_name}")
        print("-" * 60)
        
        if any(m.error_message for m in model_metrics):
            print(f"❌ ERRORS: {[m.error_message for m in model_metrics if m.error_message]}")
            continue
        
        # Calculate averages
        avg_wer = np.mean([m.word_error_rate for m in model_metrics])
        avg_cer = np.mean([m.character_error_rate for m in model_metrics])
        avg_entity_acc = np.mean([m.entity_accuracy for m in model_metrics])
        avg_rtf = np.mean([m.real_time_factor for m in model_metrics])
        avg_load_time = np.mean([m.model_load_time_ms for m in model_metrics])
        avg_ttft = np.mean([m.time_to_first_token_ms for m in model_metrics if m.time_to_first_token_ms])
        avg_cpu = np.mean([m.peak_cpu_percent for m in model_metrics])
        avg_memory = np.mean([m.peak_memory_mb for m in model_metrics])
        
        print(f"Accuracy:  WER: {avg_wer:.3f} | CER: {avg_cer:.3f} | Entity: {avg_entity_acc:.3f}")
        print(f"Latency:   Load: {avg_load_time:.1f}ms | TTFT: {avg_ttft:.1f}ms | RTF: {avg_rtf:.3f}")
        print(f"Resources: CPU: {avg_cpu:.1f}% | Memory: {avg_memory:.1f}MB")
        
        # Show per-file details
        for metrics in model_metrics:
            print(f"  📄 File: WER={metrics.word_error_rate:.3f}, Entity={metrics.entity_accuracy:.3f}")
            if metrics.predicted_text:
                print(f"     Predicted: '{metrics.predicted_text[:100]}{'...' if len(metrics.predicted_text) > 100 else ''}'")

async def main():
    parser = argparse.ArgumentParser(description='Sherpa-ONNX Streaming STT Benchmark')
    parser.add_argument('--model', 
                       choices=['all'] + list(get_available_models().keys()),
                       default='all',
                       help='Model to benchmark')
    parser.add_argument('--test-suite',
                       choices=['standard', 'entities', 'realworld'],
                       default='standard',
                       help='Test suite to run')
    parser.add_argument('--chunk-size',
                       type=int,
                       default=200,
                       help='Chunk size for streaming simulation (ms)')
    parser.add_argument('--output',
                       type=Path,
                       default=None,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Load models and test files
    available_models = get_available_models()
    test_files = load_test_suite(args.test_suite)
    
    if not test_files:
        logger.error(f"No test files found for suite: {args.test_suite}")
        return
    
    logger.info(f"Loaded {len(test_files)} test files for suite: {args.test_suite}")
    
    # Select models to test
    if args.model == 'all':
        models_to_test = available_models
    else:
        if args.model not in available_models:
            logger.error(f"Model not available: {args.model}")
            return
        models_to_test = {args.model: available_models[args.model]}
    
    logger.info(f"Testing {len(models_to_test)} models: {list(models_to_test.keys())}")
    
    # Run benchmarks
    all_results = []
    for model_name, model_config in models_to_test.items():
        try:
            results = await benchmark_model(model_config, test_files, args.chunk_size)
            all_results.extend(results)
        except KeyboardInterrupt:
            logger.info("Benchmark interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
    
    # Save and display results
    if args.output:
        output_file = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / "results" / f"benchmark_{args.test_suite}_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
    
    save_results(all_results, output_file)
    print_summary(all_results)

if __name__ == "__main__":
    asyncio.run(main())