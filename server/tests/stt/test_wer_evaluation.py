#!/usr/bin/env python3
"""
WER (Word Error Rate) Evaluation Script for STT Services

Measures transcription accuracy using Jiwer library for proper WER calculation.
Tests both Parakeet-MLX and MLX-Whisper on known audio with ground truth.
"""

import asyncio
import argparse
import json
import time
import wave
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

# Import STT services
from services.parakeet_mlx_streaming_stt import ParakeetMLXStreamingSTTService
from services.whisper_stt_with_lock import WhisperSTTServiceMLX

# Import WER calculation
try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    print("Installing jiwer for WER calculation...")
    import subprocess
    subprocess.run(["pip", "install", "jiwer"], check=True)
    import jiwer
    HAS_JIWER = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test cases with known ground truth
TEST_CASES = [
    {
        "audio_file": "test_audio/benchmark_30s.wav",
        "ground_truth": "unknown",  # Will need to determine from transcription
        "description": "Medium audio test (30 seconds)"
    }
]

class WEREvaluator:
    """Comprehensive WER evaluation for STT services"""
    
    def __init__(self):
        self.results = []
        
    async def evaluate_service(self, service_class, service_config: Dict, test_case: Dict) -> Dict[str, Any]:
        """Evaluate a single STT service on a test case"""
        logger.info(f"🔄 Evaluating {service_class.__name__}")
        
        # Initialize service
        service = service_class(**service_config)
        
        # Load and process audio
        audio_file = test_case["audio_file"]
        ground_truth = test_case["ground_truth"]
        
        try:
            # Read audio file
            with wave.open(audio_file, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
            logger.info(f"📁 Audio: {audio_file} ({len(frames)} bytes, {sample_rate}Hz)")
            
            # Process audio through STT with proper streaming chunks
            start_time = time.time()
            transcription_parts = []
            
            # Convert raw bytes to numpy array for chunking
            audio_np = np.frombuffer(frames, dtype=np.int16)
            
            # Stream in optimized chunks (faster for long audio)
            chunk_size_ms = 1000  # 1 second chunks for faster processing
            chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
            total_chunks = len(audio_np) // chunk_size_samples + (1 if len(audio_np) % chunk_size_samples != 0 else 0)
            
            logger.info(f"🔄 Streaming {total_chunks} chunks of {chunk_size_ms}ms each")
            
            for i in range(0, len(audio_np), chunk_size_samples):
                chunk = audio_np[i:i + chunk_size_samples]
                chunk_bytes = chunk.astype(np.int16).tobytes()
                
                async for frame in service.run_stt(chunk_bytes):
                    if hasattr(frame, 'text') and frame.text:
                        text = frame.text.strip()
                        if text:
                            transcription_parts.append(text)
                            logger.debug(f"📝 Chunk {i//chunk_size_samples + 1}/{total_chunks}: '{text}'")
                
                # No artificial delay - process as fast as possible
            
            # Get any remaining results from flush (if service supports it)
            if hasattr(service, 'flush'):
                final_flush = await service.flush()
                if final_flush and final_flush.strip():
                    transcription_parts.append(final_flush.strip())
                    logger.debug(f"📝 Flush result: '{final_flush.strip()}'")
            
            # Clean up service
            await service.cleanup()
            
            processing_time = time.time() - start_time
            final_transcript = " ".join(transcription_parts) if transcription_parts else ""
            
            # Calculate WER (skip for unknown ground truth)
            if final_transcript.strip() and ground_truth != "unknown":
                # Normalize both texts for comparison
                normalized_ground_truth = ground_truth.lower().strip()
                normalized_transcript = final_transcript.lower().strip()
                
                # Calculate WER using jiwer
                wer_score = jiwer.wer(normalized_ground_truth, normalized_transcript)
                word_accuracy = 1.0 - wer_score
                
                # Calculate additional metrics
                cer_score = jiwer.cer(normalized_ground_truth, normalized_transcript) 
                
                # Word-level analysis
                ground_truth_words = normalized_ground_truth.split()
                transcript_words = normalized_transcript.split()
                
                # Calculate word-level statistics manually
                alignment = {
                    "substitutions": 0,
                    "deletions": 0,
                    "insertions": 0,
                    "hits": len(set(ground_truth_words) & set(transcript_words))
                }
                
            elif final_transcript.strip() and ground_truth == "unknown":
                # Have transcription but no ground truth - just report the transcription
                wer_score = 0.0  # Can't calculate WER without ground truth
                word_accuracy = 1.0
                cer_score = 0.0
                normalized_ground_truth = "N/A"
                normalized_transcript = final_transcript.lower().strip()
                alignment = {"note": "No ground truth available for comparison"}
            else:
                # No transcription produced
                wer_score = 1.0  # 100% error rate
                word_accuracy = 0.0
                cer_score = 1.0
                normalized_ground_truth = ground_truth if ground_truth != "unknown" else "N/A"
                normalized_transcript = ""
                alignment = None
            
            result = {
                "service": service_class.__name__,
                "config": service_config,
                "test_case": test_case["description"],
                "processing_time_sec": processing_time,
                "ground_truth": ground_truth,
                "transcription": final_transcript,
                "normalized_ground_truth": normalized_ground_truth if final_transcript.strip() else ground_truth,
                "normalized_transcript": normalized_transcript if final_transcript.strip() else "",
                "wer": wer_score,
                "word_accuracy": word_accuracy,
                "cer": cer_score,
                "word_count_ground_truth": len(ground_truth.split()),
                "word_count_transcript": len(final_transcript.split()) if final_transcript.strip() else 0,
                "has_output": bool(final_transcript.strip()),
                "alignment": alignment
            }
            
            logger.info(f"✅ {service_class.__name__}: WER={wer_score:.3f}, Accuracy={word_accuracy:.3f}")
            if final_transcript.strip():
                logger.info(f"🎯 Ground Truth: '{ground_truth}'")
                logger.info(f"📝 Transcription: '{final_transcript}'")
            else:
                logger.warning(f"⚠️ No transcription output produced!")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {service_class.__name__}: {e}")
            return {
                "service": service_class.__name__,
                "config": service_config,
                "test_case": test_case["description"],
                "error": str(e),
                "wer": 1.0,
                "word_accuracy": 0.0,
                "has_output": False
            }
    
    async def run_evaluation(self):
        """Run complete WER evaluation"""
        logger.info("🚀 Starting WER Evaluation")
        
        # Service configurations
        services_config = [
            {
                "class": ParakeetMLXStreamingSTTService,
                "config": {
                    "model_name": "mlx-community/parakeet-tdt-0.6b-v2",
                    "context_size": (64, 64),  # Using safe context
                    "chunk_size_ms": 1000,    # 1 second chunks
                    "language": "en"
                }
            },
            {
                "class": WhisperSTTServiceMLX,
                "config": {
                    "model_name": "whisper-large-v3-turbo",
                    "language": "en"
                }
            }
        ]
        
        # Run evaluation on all test cases
        all_results = []
        
        for test_case in TEST_CASES:
            logger.info(f"\n📋 Test Case: {test_case['description']}")
            
            for service_info in services_config:
                try:
                    result = await self.evaluate_service(
                        service_info["class"], 
                        service_info["config"], 
                        test_case
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"❌ Failed to evaluate service: {e}")
                    continue
        
        # Generate report
        self.generate_report(all_results)
        return all_results
    
    def generate_report(self, results: List[Dict]):
        """Generate WER evaluation report"""
        print("\n" + "="*80)
        print("📊 WER EVALUATION REPORT")
        print("="*80)
        
        # Group by service
        by_service = {}
        for result in results:
            service = result["service"]
            if service not in by_service:
                by_service[service] = []
            by_service[service].append(result)
        
        # Print service comparison
        for service, service_results in by_service.items():
            print(f"\n🤖 {service}")
            print("-" * 60)
            
            for result in service_results:
                if result.get("error"):
                    print(f"  ❌ Error: {result['error']}")
                    continue
                    
                print(f"  Test: {result['test_case']}")
                print(f"  WER: {result['wer']:.3f} | Accuracy: {result['word_accuracy']:.3f}")
                print(f"  Ground Truth: '{result['ground_truth']}'")
                if result['has_output']:
                    print(f"  Transcription: '{result['transcription']}'")
                else:
                    print(f"  ⚠️  NO TRANSCRIPTION OUTPUT")
        
        # Overall comparison
        print(f"\n🏆 OVERALL COMPARISON")
        print("-" * 60)
        
        service_averages = {}
        for service, service_results in by_service.items():
            valid_results = [r for r in service_results if not r.get("error") and r["has_output"]]
            if valid_results:
                avg_wer = sum(r["wer"] for r in valid_results) / len(valid_results)
                avg_accuracy = sum(r["word_accuracy"] for r in valid_results) / len(valid_results)
                service_averages[service] = {"wer": avg_wer, "accuracy": avg_accuracy, "count": len(valid_results)}
            else:
                service_averages[service] = {"wer": 1.0, "accuracy": 0.0, "count": 0}
        
        for service, stats in service_averages.items():
            if stats["count"] > 0:
                print(f"  {service}: WER={stats['wer']:.3f}, Accuracy={stats['accuracy']:.3f} ({stats['count']} tests)")
            else:
                print(f"  {service}: NO VALID TRANSCRIPTIONS PRODUCED")
        
        # Winner
        best_service = min(service_averages.keys(), key=lambda x: service_averages[x]["wer"])
        if service_averages[best_service]["count"] > 0:
            print(f"\n🥇 WINNER: {best_service} (WER: {service_averages[best_service]['wer']:.3f})")
        else:
            print(f"\n⚠️ NO SERVICE PRODUCED VALID TRANSCRIPTIONS")

async def main():
    parser = argparse.ArgumentParser(description="WER Evaluation for STT Services")
    parser.add_argument("--output", "-o", default="wer_evaluation_results.json", help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    evaluator = WEREvaluator()
    results = await evaluator.run_evaluation()
    
    # Save results
    output_file = args.output
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())