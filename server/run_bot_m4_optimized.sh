#!/bin/bash
# Run script optimized for M4 Pro with 32GB RAM

# Activate virtual environment
source venv/bin/activate

# macOS multiprocessing fixes
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export no_proxy=*

# Metal GPU settings optimized for M4 Pro
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=0

# Allow Metal to use multiple threads on M4
export METAL_SINGLE_THREADED=0
export MLX_METAL_SINGLE_THREADED=0

# MLX settings for high-end hardware
export MLX_USE_METAL=1
export MLX_METAL_BUFFER_SIZE=2GB
export MLX_METAL_MEMORY_LIMIT=16GB

# Disable debugging for performance
export MTL_SHADER_VALIDATION=0
export MTL_DEBUG_LAYER=0
export METAL_GPU_FRAME_CAPTURE_ENABLED=0

# Allow parallel execution
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Python optimizations
export PYTHONUNBUFFERED=1

echo "Starting bot optimized for M4 Pro..."
echo "Configuration:"
echo "  - Hardware: M4 Pro with 32GB RAM"
echo "  - Metal: Multi-threaded enabled"
echo "  - MLX Buffer: 2GB"
echo "  - Memory Limit: 16GB"
echo "Arguments: $@"

# Run without nice to use full performance
python bot.py "$@"