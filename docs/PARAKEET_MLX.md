# Parakeet-MLX Streaming STT Service

Parakeet-MLX is an Apple Silicon-optimized ASR implementation that provides native streaming capabilities with excellent performance characteristics. It complements the existing MLX Whisper service by offering real-time streaming transcription with lower latency.

## Key Features

- **Native Apple Silicon Optimization**: Built specifically for M-series chips using MLX
- **True Streaming**: Real-time processing with configurable context windows
- **Low Latency**: Optimized for incremental processing with draft/finalized tokens
- **Memory Efficient**: Local attention modes reduce memory usage
- **Flexible Output**: Supports word-level timestamps and multiple formats

## Configuration

### Environment Variables

```bash
# Enable Parakeet-MLX STT backend
export STT_BACKEND=parakeet-mlx

# Model configuration
export PARAKEET_MODEL="mlx-community/parakeet-tdt-0.6b-v2"  # Default model
# Alternative models:
# export PARAKEET_MODEL="mlx-community/parakeet-tdt-1.1b"  # Higher accuracy

# Context window configuration (left_context, right_context)
export PARAKEET_CONTEXT_SIZE="256,256"  # Default: balanced memory/accuracy
# export PARAKEET_CONTEXT_SIZE="512,512"  # Higher accuracy, more memory
# export PARAKEET_CONTEXT_SIZE="128,128"  # Lower memory, faster

# Attention mode
export PARAKEET_ATTENTION_MODE="local"  # Default: memory efficient
# export PARAKEET_ATTENTION_MODE="full"  # Higher accuracy, more memory

# Precision mode
export PARAKEET_PRECISION="bf16"  # Default: balanced speed/quality
# export PARAKEET_PRECISION="fp32"  # Higher quality, slower

# Streaming configuration
export PARAKEET_CHUNK_MS="100"          # Processing chunk size (50-200ms recommended)
export PARAKEET_EMIT_PARTIAL="true"     # Enable partial/interim results
export PARAKEET_MIN_CONFIDENCE="0.1"    # Minimum confidence for results
```

### Usage Examples

#### Basic Streaming STT
```bash
cd server/
export STT_BACKEND=parakeet-mlx
python bot_v2.py
```

#### High-Accuracy Configuration
```bash
export STT_BACKEND=parakeet-mlx
export PARAKEET_MODEL="mlx-community/parakeet-tdt-1.1b"
export PARAKEET_CONTEXT_SIZE="512,512"
export PARAKEET_ATTENTION_MODE="full"
export PARAKEET_PRECISION="fp32"
python bot_v2.py
```

#### Low-Latency Configuration
```bash
export STT_BACKEND=parakeet-mlx
export PARAKEET_CONTEXT_SIZE="128,128"
export PARAKEET_CHUNK_MS="50"
export PARAKEET_EMIT_PARTIAL="true"
python bot_v2.py
```

## Model Options

### Available Models

| Model | Size | Accuracy | Speed | Memory |
|-------|------|----------|-------|--------|
| `parakeet-tdt-0.6b-v2` | 600M | Good | Fast | Low |
| `parakeet-tdt-1.1b` | 1.1B | Better | Medium | Medium |
| Custom fine-tuned | Varies | Custom | Varies | Varies |

### Model Selection Guidelines

- **Real-time applications**: Use 0.6b model with local attention
- **High accuracy needs**: Use 1.1b model with full attention
- **Memory constrained**: Use smaller context windows (128,128)
- **Quality critical**: Use fp32 precision and full attention

## Performance Optimization

### Context Window Tuning

- **Small (128,128)**: Fastest, lowest memory, good for short phrases
- **Medium (256,256)**: Balanced performance, recommended default
- **Large (512,512)**: Best accuracy, higher memory usage

### Attention Mode Impact

- **Local Attention**: Memory efficient, good for streaming
- **Full Attention**: Higher accuracy, more memory intensive

### Chunk Size Recommendations

- **50ms**: Ultra-low latency, may impact accuracy
- **100ms**: Good balance of latency and accuracy (recommended)
- **200ms**: Better accuracy, higher latency

## Comparison with Other STT Services

| Feature | MLX Whisper | Sherpa Streaming | Parakeet-MLX |
|---------|-------------|------------------|--------------|
| **Latency** | High (batch) | Medium | Low (streaming) |
| **Accuracy** | Excellent | Good | Very Good |
| **Apple Silicon** | Optimized | CPU/Limited | Native MLX |
| **Streaming** | No | Yes | Yes (native) |
| **Memory** | High | Medium | Low (local attention) |
| **Languages** | Many | Many | English-focused |
| **Setup Complexity** | Low | Medium | Low |

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure `parakeet-mlx` is installed
   ```bash
   pip install parakeet-mlx
   ```

2. **Model Loading Failed**: Check model name and network connectivity
   ```bash
   # Test model loading
   python -c "from parakeet_mlx import from_pretrained; from_pretrained('mlx-community/parakeet-tdt-0.6b-v2')"
   ```

3. **High Memory Usage**: Reduce context size or use local attention
   ```bash
   export PARAKEET_CONTEXT_SIZE="128,128"
   export PARAKEET_ATTENTION_MODE="local"
   ```

4. **MLX Lock Conflicts**: The service uses `MLX_GLOBAL_LOCK` for thread safety

### Debug Logging

Enable verbose logging to debug issues:
```bash
export LOG_LEVEL=DEBUG
python bot_v2.py
```

## Integration Notes

- Compatible with existing VAD and turn detection
- Supports multi-language pipelines (though optimized for English)
- Can be used alongside other STT services
- Automatic cleanup and resource management
- Thread-safe with MLX global lock integration

## Future Enhancements

- Multi-language model variants
- Custom model fine-tuning support
- Advanced streaming optimizations
- Integration with voice activity detection
- Confidence-based hybrid STT switching