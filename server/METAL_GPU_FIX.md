# Metal GPU Fix for Kokoro TTS Crash

## Problem
The error `[IOGPUMetalCommandBuffer encodeSignalEvent:value:]:426: failed assertion 'encodeSignalEvent:value: with uncommitted encoder'` indicates a Metal command encoder conflict where multiple encoders are being created without properly ending the previous ones.

## Root Cause
This happens when:
1. Multiple MLX operations (Whisper STT and Kokoro TTS) run concurrently
2. Metal command encoders aren't properly synchronized between operations
3. Command buffers are left uncommitted

## Applied Fixes

### 1. Enhanced Synchronization in kokoro_tts.py
Added `mx.synchronize()` calls to ensure Metal command buffers are properly flushed:
- Before starting generation (line 169)
- After generation completion (line 189)

### 2. Existing Safeguards
The codebase already has:
- Thread locks (`self._generation_lock`)
- Single-threaded execution (`max_workers=1`)
- Explicit Metal command flushing (`mx.eval()`)

## Additional Recommendations

### 1. Environment Variable for Metal Debugging
Add to your `.env` or shell:
```bash
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=0
```

### 2. Update MLX Dependencies
```bash
pip install --upgrade mlx mlx-audio mlx-whisper
```

### 3. Alternative: Disable Metal Validation
If the issue persists, you can disable Metal validation (not recommended for production):
```bash
export MTL_DEBUG_LAYER=0
```

### 4. Process Isolation Option
If conflicts continue, consider running STT and TTS in separate processes:
```python
# In bot.py, add process isolation
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

## Testing the Fix

1. Restart the bot:
```bash
./run_bot.sh
```

2. Test with simple interaction first
3. Monitor for Metal errors in the console

## If Issues Persist

1. **Check MLX versions compatibility**:
```bash
pip list | grep mlx
```
Ensure all MLX packages are compatible versions.

2. **Try sequential processing**:
Temporarily disable concurrent STT/TTS by adding delays between operations.

3. **Report to MLX team**:
If the issue continues, it may be a bug in MLX that needs reporting to: https://github.com/ml-explore/mlx/issues