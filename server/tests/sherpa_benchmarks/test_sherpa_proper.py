#!/usr/bin/env python3
"""
Proper Sherpa-ONNX Benchmarking Test using Official API

Based on the official sherpa-onnx documentation:
https://k2-fsa.github.io/sherpa/onnx/python/real-time-speech-recongition-from-a-microphone.html

This test properly evaluates sherpa-onnx models for streaming STT performance,
focusing on accuracy for names, URLs, and technical terms.
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import sherpa_onnx
except ImportError:
    logger.error("sherpa_onnx not found. Please install with: pip install sherpa-onnx")
    exit(1)

@dataclass
class BenchmarkResult:
    """Results for a single test"""
    model_name: str
    test_file: str
    reference_text: str
    predicted_text: str
    
    # Performance metrics
    model_load_time_ms: float
    processing_time_ms: float
    audio_duration_ms: float
    real_time_factor: float
    
    # Accuracy metrics
    word_error_rate: float
    character_error_rate: float
    
    # Sherpa-specific metrics
    num_segments: int
    avg_segment_confidence: float
    endpoint_detection_count: int
    
    # Resource usage
    peak_memory_mb: float
    avg_cpu_percent: float

class ResourceMonitor:
    """Monitor CPU and memory usage during processing"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.thread = None
    
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
        
        if not self.cpu_samples:
            return 0.0, 0.0
        
        return max(self.memory_samples), np.mean(self.cpu_samples)
    
    def _monitor(self):
        process = psutil.Process()
        while self.monitoring:
            try:
                cpu = process.cpu_percent()
                memory = process.memory_info().rss / 1024 / 1024  # MB
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
                time.sleep(0.1)
            except:
                break

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate using simple edit distance"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    
    # Simple edit distance for words
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
    
    # Simple edit distance for characters
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

def load_audio_file(filepath: Path) -> Tuple[np.ndarray, int]:
    """Load WAV file and return audio data as float32"""
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
        
        # Convert to float32 [-1, 1]
        if sample_width == 2:
            audio_data = audio_data.astype(np.float32) / 32768.0
        else:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        return audio_data, sample_rate

def create_recognizer(model_dir: Path, **kwargs) -> sherpa_onnx.OnlineRecognizer:
    """Create sherpa-onnx OnlineRecognizer using proper API"""
    
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
        
        logger.info(f"Using transducer model:")
        logger.info(f"  Encoder: {encoder_file.name}")
        logger.info(f"  Decoder: {decoder_file.name}")
        logger.info(f"  Joiner: {joiner_file.name}")
        
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
    
    # Check for CTC model
    model_file = model_dir / "model.onnx"
    if model_file.exists():
        logger.info(f"Using CTC model: {model_file.name}")
        
        # Create CTC recognizer config manually
        nemo_ctc_config = sherpa_onnx.OnlineNemoEncDecCtcModelConfig(
            model=str(model_file)
        )
        
        model_config = sherpa_onnx.OnlineModelConfig(
            nemo_ctc=nemo_ctc_config,
            tokens=str(tokens_file),
            num_threads=1,
            provider="cpu",
            model_type="nemo_ctc"
        )
        
        feat_config = sherpa_onnx.FeatureConfig(
            sample_rate=16000,
            feature_dim=80
        )
        
        recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
            feat_config=feat_config,
            model_config=model_config,
            enable_endpoint_detection=kwargs.get('enable_endpoint_detection', True),
            max_active_paths=kwargs.get('max_active_paths', 4),
            decoding_method=kwargs.get('decoding_method', 'greedy_search')
        )
        
        return sherpa_onnx.OnlineRecognizer(recognizer_config)
    
    raise FileNotFoundError(f"No suitable model files found in {model_dir}")

def process_audio_file(recognizer: sherpa_onnx.OnlineRecognizer, 
                      audio_file: Path, 
                      reference_text: str = "",
                      chunk_size_ms: int = 200) -> BenchmarkResult:
    """Process audio file with sherpa-onnx recognizer"""
    
    logger.info(f"Processing: {audio_file.name}")
    
    # Load audio
    audio_data, sample_rate = load_audio_file(audio_file)
    audio_duration_ms = len(audio_data) / sample_rate * 1000
    
    # Resample if needed (sherpa-onnx expects 16kHz)
    if sample_rate != 16000:
        logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
        # Simple linear interpolation resampling
        ratio = 16000 / sample_rate
        new_length = int(len(audio_data) * ratio)
        audio_data = np.interp(
            np.linspace(0, len(audio_data), new_length),
            np.arange(len(audio_data)),
            audio_data
        )
        sample_rate = 16000
    
    # Create stream
    stream = recognizer.create_stream()
    
    # Process in chunks to simulate streaming
    chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
    segments = []
    endpoint_count = 0
    
    # Start monitoring
    monitor = ResourceMonitor()
    monitor.start()
    
    start_time = time.time()
    
    for i in range(0, len(audio_data), chunk_size_samples):
        chunk = audio_data[i:i + chunk_size_samples]
        
        # Feed audio to recognizer
        stream.accept_waveform(sample_rate, chunk.tolist())
        
        # Decode if ready
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        
        # Check for endpoint
        is_endpoint = recognizer.is_endpoint(stream)
        result = recognizer.get_result(stream)
        
        # Handle different result types (string vs object)
        result_text = result if isinstance(result, str) else getattr(result, 'text', str(result))
        
        if is_endpoint and result_text.strip():
            segments.append(result_text.strip())
            logger.info(f"Segment {len(segments)}: '{result_text.strip()}'")
            recognizer.reset(stream)
            endpoint_count += 1
        
        # Small delay to simulate real-time
        time.sleep(0.001)
    
    # Get final result if any
    result = recognizer.get_result(stream)
    result_text = result if isinstance(result, str) else getattr(result, 'text', str(result))
    if result_text.strip() and (not segments or result_text.strip() != segments[-1]):
        segments.append(result_text.strip())
        logger.info(f"Final segment: '{result_text.strip()}'")
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Stop monitoring
    peak_memory, avg_cpu = monitor.stop()
    
    # Combine segments
    predicted_text = " ".join(segments)
    
    # Calculate metrics
    wer = calculate_wer(reference_text, predicted_text) if reference_text else 0.0
    cer = calculate_cer(reference_text, predicted_text) if reference_text else 0.0
    rtf = processing_time_ms / audio_duration_ms
    
    # Calculate average confidence (placeholder - sherpa-onnx doesn't expose confidence scores easily)
    avg_confidence = 1.0  # Placeholder
    
    result = BenchmarkResult(
        model_name="sherpa-onnx",
        test_file=audio_file.name,
        reference_text=reference_text,
        predicted_text=predicted_text,
        model_load_time_ms=0,  # Not measured separately
        processing_time_ms=processing_time_ms,
        audio_duration_ms=audio_duration_ms,
        real_time_factor=rtf,
        word_error_rate=wer,
        character_error_rate=cer,
        num_segments=len(segments),
        avg_segment_confidence=avg_confidence,
        endpoint_detection_count=endpoint_count,
        peak_memory_mb=peak_memory,
        avg_cpu_percent=avg_cpu
    )
    
    logger.info(f"Results: WER={wer:.3f}, CER={cer:.3f}, RTF={rtf:.3f}, Segments={len(segments)}")
    logger.info(f"Predicted: '{predicted_text}'")
    if reference_text:
        logger.info(f"Reference: '{reference_text}'")
    
    return result

def load_test_files(test_suite: str) -> List[Tuple[Path, str]]:
    """Load test files with reference transcriptions"""
    test_dir = Path(__file__).parent / "test_audio"
    
    if test_suite == "standard":
        # Use sherpa model's test files
        model_dir = Path(__file__).parent.parent.parent / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
        test_wavs_dir = model_dir / "test_wavs"
        
        test_files = []
        if test_wavs_dir.exists():
            trans_file = test_wavs_dir / "trans.txt"
            if trans_file.exists():
                with open(trans_file) as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            wav_file = test_wavs_dir / parts[0]
                            if wav_file.exists():
                                test_files.append((wav_file, parts[1]))
        return test_files
        
    elif test_suite == "realworld":
        realworld_dir = test_dir / "realworld"
        test_files = []
        
        # Load any wav files
        for wav_file in realworld_dir.glob("*.wav"):
            # Use filename as reference if no specific reference
            test_files.append((wav_file, f"[Audio file: {wav_file.name}]"))
        
        return test_files
    
    elif test_suite == "entities":
        # Create entity-focused test cases
        entities_dir = test_dir / "entities"
        
        # For this test, we'll create some synthetic test cases focusing on the
        # "bbb.com/news" issue you mentioned
        test_cases = [
            ("Please visit bbb.com/news for the latest updates", "entity_urls.wav"),
            ("Send email to john.doe@company.com", "entity_email.wav"), 
            ("Check the GitHub repository at github.com/microsoft/project", "entity_github.wav"),
            ("The API endpoint is at api.example.com/v1/users", "entity_api.wav")
        ]
        
        # For now, return empty since we'd need actual audio files
        logger.warning("Entity test suite requires audio files to be generated first")
        return []
    
    else:
        raise ValueError(f"Unknown test suite: {test_suite}")

def main():
    parser = argparse.ArgumentParser(description='Proper Sherpa-ONNX Benchmarking')
    parser.add_argument('--model-dir', type=Path, required=True,
                       help='Path to sherpa-onnx model directory')
    parser.add_argument('--test-suite', 
                       choices=['standard', 'realworld', 'entities'],
                       default='standard',
                       help='Test suite to run')
    parser.add_argument('--chunk-size', type=int, default=200,
                       help='Audio chunk size in milliseconds')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON file for results')
    
    # Sherpa configuration options
    parser.add_argument('--enable-endpoint-detection', action='store_true', default=True)
    parser.add_argument('--max-active-paths', type=int, default=4)
    parser.add_argument('--decoding-method', default='greedy_search')
    parser.add_argument('--hotwords-file', default='')
    parser.add_argument('--hotwords-score', type=float, default=1.5)
    
    args = parser.parse_args()
    
    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        return
    
    # Load test files
    test_files = load_test_files(args.test_suite)
    if not test_files:
        logger.error(f"No test files found for suite: {args.test_suite}")
        return
    
    logger.info(f"Loaded {len(test_files)} test files")
    
    # Create recognizer
    logger.info("Creating sherpa-onnx recognizer...")
    load_start = time.time()
    
    try:
        recognizer = create_recognizer(
            args.model_dir,
            enable_endpoint_detection=args.enable_endpoint_detection,
            max_active_paths=args.max_active_paths,
            decoding_method=args.decoding_method,
            hotwords_file=args.hotwords_file,
            hotwords_score=args.hotwords_score
        )
        load_time = (time.time() - load_start) * 1000
        logger.info(f"Model loaded in {load_time:.1f}ms")
        
    except Exception as e:
        logger.error(f"Failed to create recognizer: {e}")
        return
    
    # Process test files
    results = []
    for audio_file, reference_text in test_files:
        try:
            result = process_audio_file(
                recognizer, audio_file, reference_text, args.chunk_size
            )
            result.model_load_time_ms = load_time
            result.model_name = f"sherpa-onnx-{args.model_dir.name}"
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            continue
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = Path(__file__).parent / "results" / f"sherpa_proper_{args.test_suite}_{timestamp}.json"
        output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    if results:
        print("\n" + "="*80)
        print("SHERPA-ONNX PROPER BENCHMARK RESULTS")
        print("="*80)
        
        avg_wer = np.mean([r.word_error_rate for r in results])
        avg_cer = np.mean([r.character_error_rate for r in results])
        avg_rtf = np.mean([r.real_time_factor for r in results])
        avg_segments = np.mean([r.num_segments for r in results])
        
        print(f"Model: {results[0].model_name}")
        print(f"Test Suite: {args.test_suite}")
        print(f"Files Processed: {len(results)}")
        print(f"Average WER: {avg_wer:.3f}")
        print(f"Average CER: {avg_cer:.3f}")
        print(f"Average RTF: {avg_rtf:.3f}")
        print(f"Average Segments per file: {avg_segments:.1f}")
        print(f"Model Load Time: {results[0].model_load_time_ms:.1f}ms")
        
        print("\nPer-file Results:")
        for result in results:
            print(f"  {result.test_file}: WER={result.word_error_rate:.3f}, "
                  f"Segments={result.num_segments}, RTF={result.real_time_factor:.3f}")
            if result.reference_text and not result.reference_text.startswith("["):
                print(f"    Expected: '{result.reference_text}'")
            print(f"    Got:      '{result.predicted_text}'")

if __name__ == "__main__":
    main()