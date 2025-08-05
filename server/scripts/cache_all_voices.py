#!/usr/bin/env python3
"""
Script to pre-cache all Kokoro TTS voices for offline use
This ensures all voice models are downloaded before running in offline mode
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger
import mlx.core as mx
from mlx_audio.tts.utils import load_model

# All available Kokoro voices
VOICES = [
    # Italian
    'if_sara', 'im_nicola',
    # French
    'ff_siwis',
    # English (US)
    'af_bella', 'af_sarah', 'af_sky', 'af_alloy', 
    'af_nova', 'af_heart', 'af_jessica', 'af_kore',
    'af_nicole', 'af_river', 'af_aoede',
    'am_adam', 'am_echo', 'am_michael', 'am_liam',
    'am_eric', 'am_fenrir', 'am_onyx', 'am_puck',
    # English (UK)
    'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
    'bm_daniel', 'bm_george', 'bm_lewis', 'bm_fable',
    # Japanese
    'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro',
    'jm_kumo',
    # Chinese
    'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
    'zm_yunxi', 'zm_yunxia', 'zm_yunyang', 'zm_yunjian',
    # Spanish
    'ef_dora', 'em_alex', 'em_santa',
    # Portuguese
    'pf_dora', 'pm_alex', 'pm_santa'
]

def cache_voice(model, voice: str):
    """Cache a single voice by generating a test phrase"""
    try:
        logger.info(f"Caching voice: {voice}")
        
        # Generate a short test phrase to download voice files
        test_text = "Test"
        for result in model.generate(
            text=test_text,
            voice=voice,
            speed=1.0,
            lang_code='a'  # Default to English
        ):
            # Just iterate through to trigger download
            pass
            
        logger.success(f"✅ Cached voice: {voice}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to cache voice {voice}: {e}")
        return False

def main():
    """Main function to cache all voices"""
    logger.info("🎤 Starting Kokoro voice caching...")
    logger.info(f"Total voices to cache: {len(VOICES)}")
    
    # Load the model
    model_name = "prince-canuma/Kokoro-82M"
    logger.info(f"Loading model: {model_name}")
    
    try:
        model = load_model(model_name)
        mx.eval(mx.array([0]))  # Ensure Metal commands are flushed
        logger.success("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Cache each voice
    successful = 0
    failed = 0
    
    for i, voice in enumerate(VOICES, 1):
        logger.info(f"\n[{i}/{len(VOICES)}] Processing {voice}...")
        if cache_voice(model, voice):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info(f"✅ Successfully cached: {successful} voices")
    if failed > 0:
        logger.warning(f"❌ Failed to cache: {failed} voices")
    else:
        logger.success("🎉 All voices cached successfully!")
    
    logger.info("\nYou can now enable offline mode in .env:")
    logger.info("HF_HUB_OFFLINE=1")
    logger.info("TRANSFORMERS_OFFLINE=1")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())