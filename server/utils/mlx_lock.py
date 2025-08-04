# server/utils/mlx_lock.py
import threading

# A global lock to synchronize access to MLX (Metal GPU) operations.
# This ensures that only one MLX-based operation (e.g., Whisper STT, Kokoro TTS)
# can access the Metal GPU at any given time, preventing low-level driver conflicts.
MLX_GLOBAL_LOCK = threading.Lock()