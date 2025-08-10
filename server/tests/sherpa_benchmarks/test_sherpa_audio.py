#!/usr/bin/env python3
"""
Test accuracy enhancement on real audio files using Sherpa-ONNX
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from advanced_accuracy_enhancer import AdvancedAccuracyEnhancer

def get_sherpa_model_config():
    """Get Sherpa-ONNX model configuration"""
    import sherpa_onnx
    
    # Use the model directory from your environment
    model_dir = os.getenv("SHERPA_ONNX_MODEL_DIR", "./models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
    
    if not os.path.exists(model_dir):
        # Try alternative model directories
        alternative_dirs = [
            "./models/sherpa-onnx-streaming-zipformer-en-2023-06-26",
            "../models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
            "../../models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        ]
        
        for alt_dir in alternative_dirs:
            if os.path.exists(alt_dir):
                model_dir = alt_dir
                break
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Sherpa-ONNX model directory not found: {model_dir}")
    
    # Check for different model types
    if os.path.exists(os.path.join(model_dir, "encoder.onnx")):
        # Transducer model
        config = sherpa_onnx.OnlineRecognizerConfig(
            model_config=sherpa_onnx.OnlineModelConfig(
                transducer=sherpa_onnx.OnlineTransducerModelConfig(
                    encoder=os.path.join(model_dir, "encoder.onnx"),
                    decoder=os.path.join(model_dir, "decoder.onnx"),
                    joiner=os.path.join(model_dir, "joiner.onnx"),
                ),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=2,
                provider="cpu",
            ),
            decoding_method="modified_beam_search",
            max_active_paths=4,
        )
    elif os.path.exists(os.path.join(model_dir, "model.onnx")):
        # Paraformer model
        config = sherpa_onnx.OnlineRecognizerConfig(
            model_config=sherpa_onnx.OnlineModelConfig(
                paraformer=sherpa_onnx.OnlineParaformerModelConfig(
                    model=os.path.join(model_dir, "model.onnx"),
                ),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=2,
                provider="cpu",
            ),
        )
    else:
        raise FileNotFoundError(f"No valid model files found in {model_dir}")
    
    return config

def transcribe_audio_file(audio_file_path):
    """Transcribe an audio file using Sherpa-ONNX"""
    try:
        import sherpa_onnx
        import wave
        import numpy as np
    except ImportError:
        print("❌ Sherpa-ONNX or required libraries not available")
        return None
    
    try:
        # Get model configuration
        config = get_sherpa_model_config()
        
        # Initialize recognizer
        recognizer = sherpa_onnx.OnlineRecognizer(config)
        
        # Read audio file
        if audio_file_path.endswith('.mp3'):
            # Convert MP3 to WAV first (requires pydub)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(audio_file_path)
                # Convert to WAV in memory
                import io
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)
                with wave.open(wav_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    samples = np.frombuffer(frames, dtype=np.int16)
                    samples = samples.astype(np.float32) / 32768.0
            except ImportError:
                print("❌ pydub not available for MP3 conversion")
                return None
        else:
            # Read WAV file
            with wave.open(audio_file_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16)
                samples = samples.astype(np.float32) / 32768.0
        
        # Check sample rate
        if sample_rate != recognizer.sample_rate:
            print(f"⚠️  Sample rate mismatch: file={sample_rate}Hz, model={recognizer.sample_rate}Hz")
            # Resample if needed (requires librosa or similar)
            try:
                import librosa
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=recognizer.sample_rate)
                sample_rate = recognizer.sample_rate
            except ImportError:
                print("❌ librosa not available for resampling")
                return None
        
        # Create stream and decode
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        
        # Get result
        result = recognizer.get_result(stream)
        return result.text
    
    except Exception as e:
        print(f"❌ Error transcribing {audio_file_path}: {e}")
        return None

async def test_real_audio_enhancement():
    """Test accuracy enhancement on real audio files"""
    
    # Initialize the accuracy enhancer
    enhancer = AdvancedAccuracyEnhancer()
    
    # Audio files to test
    audio_files = [
        "test_audio/realworld/python_example_test(1).wav",
        "test_audio/comparison/comparison_300.0s.wav",
        "test_audio/standard/01 The Blackmailer.mp3"
    ]
    
    base_path = Path(__file__).parent
    results = []
    
    print("🎙️ Testing accuracy enhancement on real audio files")
    print("=" * 60)
    
    for i, audio_file in enumerate(audio_files, 1):
        full_path = base_path / audio_file
        
        if not full_path.exists():
            print(f"\nTest {i}: {audio_file}")
            print("❌ File not found")
            continue
        
        print(f"\nTest {i}: {audio_file}")
        print(f"Size: {full_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Transcribe audio
        transcription = transcribe_audio_file(str(full_path))
        
        if transcription is None:
            print("❌ Transcription failed")
            continue
        
        print(f"Original transcription: {transcription[:100]}{'...' if len(transcription) > 100 else ''}")
        
        # Apply accuracy enhancement
        result = await enhancer.enhance_accuracy(transcription, confidence=0.7)
        
        print(f"Enhanced transcription:  {result.corrected_text[:100]}{'...' if len(result.corrected_text) > 100 else ''}")
        print(f"Method:                 {result.method_used}")
        print(f"Processing time:        {result.processing_time_ms:.1f}ms")
        
        if result.corrections_applied:
            print("Corrections applied:")
            for corr in result.corrections_applied[:5]:  # Show first 5 corrections
                print(f"  • {corr['original']} → {corr['corrected']} ({corr['method']})")
            if len(result.corrections_applied) > 5:
                print(f"  ... and {len(result.corrections_applied) - 5} more")
        
        results.append({
            'file': audio_file,
            'original': transcription,
            'enhanced': result.corrected_text,
            'method': result.method_used,
            'time': result.processing_time_ms,
            'corrections': len(result.corrections_applied)
        })
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  Corrections: {result['corrections']}")
        print(f"  Time: {result['time']:.1f}ms")
        print(f"  Method: {result['method']}")
    
    print(f"\n🎯 Test completed successfully!")
    return results

if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(Path(__file__).parent)
    asyncio.run(test_real_audio_enhancement())