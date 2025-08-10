#!/usr/bin/env python3
"""
Generate audio files using Kokoro TTS for entity-dense benchmarking

This script uses the Kokoro TTS system to generate audio files from 
the text samples, creating proper WAV files for sherpa-onnx testing.
"""

import sys
from pathlib import Path
import asyncio
import logging

# Add server root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kokoro_tts import KokoroTTSService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def generate_audio_files():
    """Generate audio files from text samples using Kokoro TTS"""
    
    # Initialize Kokoro TTS
    logger.info("Initializing Kokoro TTS...")
    tts = KokoroTTSService(
        voice="af_heart",  # English voice
        sample_rate=16000  # Required for sherpa-onnx
    )
    
    # Process entity test files
    entities_dir = Path(__file__).parent / "test_audio" / "entities"
    references_file = entities_dir / "references.txt"
    
    if references_file.exists():
        logger.info("Generating entity-dense audio files...")
        with open(references_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wav_filename, text = parts
                    output_path = entities_dir / wav_filename
                    
                    logger.info(f"Generating: {wav_filename}")
                    logger.info(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                    
                    try:
                        # Generate audio
                        await tts.synthesize_to_file(text, str(output_path))
                        logger.info(f"✅ Generated: {output_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to generate {wav_filename}: {e}")
    
    # Process real-world test files
    realworld_dir = Path(__file__).parent / "test_audio" / "realworld"
    references_file = realworld_dir / "references.txt"
    
    if references_file.exists():
        logger.info("Generating real-world audio files...")
        with open(references_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wav_filename, text = parts
                    output_path = realworld_dir / wav_filename
                    
                    logger.info(f"Generating: {wav_filename}")
                    logger.info(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                    
                    try:
                        # Generate audio
                        await tts.synthesize_to_file(text, str(output_path))
                        logger.info(f"✅ Generated: {output_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to generate {wav_filename}: {e}")
    
    logger.info("Audio generation complete!")

def check_existing_files():
    """Check what audio files already exist"""
    test_dir = Path(__file__).parent / "test_audio"
    
    for subdir in ["entities", "realworld", "standard"]:
        subdir_path = test_dir / subdir
        if subdir_path.exists():
            audio_files = list(subdir_path.glob("*.wav")) + list(subdir_path.glob("*.mp3"))
            if audio_files:
                print(f"\n📁 {subdir}/")
                for audio_file in audio_files:
                    file_size = audio_file.stat().st_size / 1024  # KB
                    print(f"  🎵 {audio_file.name} ({file_size:.1f} KB)")
            else:
                print(f"\n📁 {subdir}/ (no audio files)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate audio files using Kokoro TTS')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check existing files, do not generate')
    
    args = parser.parse_args()
    
    if args.check_only:
        print("Checking existing audio files...")
        check_existing_files()
    else:
        print("Generating audio files with Kokoro TTS...")
        check_existing_files()
        print("\nStarting generation...")
        asyncio.run(generate_audio_files())