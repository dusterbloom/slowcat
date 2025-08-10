#!/usr/bin/env python3
"""
Test the sherpa service directly without full bot integration
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_sherpa_service():
    """Test the sherpa service"""
    try:
        from services.sherpa_streaming_stt_v2 import SherpaOnlineSTTService
        
        print("🚀 Creating SherpaOnlineSTTService...")
        service = SherpaOnlineSTTService(
            model_dir="./models/sherpa-onnx-streaming-zipformer-en-2023-06-26",
            language="en",
            sample_rate=16000,
            enable_endpoint_detection=True,
            chunk_size_ms=100,
            emit_partial_results=True,
        )
        print("✅ Service created")
        
        # Test with empty audio first
        print("🔄 Testing with empty audio...")
        frames = []
        async for frame in service.run_stt(b''):
            frames.append(frame)
        print(f"✅ Empty audio test passed ({len(frames)} frames)")
        
        # Test with small audio chunk
        print("🔄 Testing with small audio chunk...")
        # 100ms of silence as int16 PCM
        import numpy as np
        samples = np.zeros(1600, dtype=np.int16)  # 100ms at 16kHz
        audio_bytes = samples.tobytes()
        
        frames = []
        async for frame in service.run_stt(audio_bytes):
            frames.append(frame)
            print(f"  Frame: {type(frame).__name__}: {getattr(frame, 'text', '')}")
        
        print(f"✅ Audio processing test passed ({len(frames)} frames)")
        
        # Cleanup
        print("🔄 Testing cleanup...")
        service.cleanup()
        print("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 SHERPA SERVICE TEST")
    print("=" * 50)
    
    success = asyncio.run(test_sherpa_service())
    
    if success:
        print("\n✅ Service test completed successfully!")
    else:
        print("\n❌ Service test failed")