# Sherpa-ONNX Streaming STT Benchmark Suite

A comprehensive benchmarking framework for evaluating sherpa-onnx models for streaming speech-to-text performance, with focus on accuracy for names, URLs, and technical terms.

## Quick Start

```bash
# Test standard model with official sherpa-onnx API
python test_sherpa_proper.py --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 --test-suite standard

# Test on your real-world audio files  
python test_sherpa_proper.py --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 --test-suite realworld

# Compare different configurations
python test_sherpa_proper.py --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 --chunk-size 400 --max-active-paths 8
```

## Benchmark Results Summary

### Model Comparison (Standard Test Suite)

| Model | WER | CER | RTF | Load Time | Memory | Notes |
|-------|-----|-----|-----|-----------|---------|--------|
| **zipformer-en-2023-06-26** | **0.056** | **0.040** | 0.047 | 2.6s | Low | **Best accuracy** |
| zipformer-en-20M-2023-02-17 | 0.216 | 0.196 | **0.030** | **1.2s** | **Lowest** | Faster but less accurate |

### Stress Test Results (Longer Audio)

| Duration | RTF | Memory Growth | Segments | Errors | Status |
|----------|-----|---------------|----------|---------|---------|
| **10s** | 0.048 | 0.0MB | 2 | 0 | ✅ **Stable** |
| **30s** | 0.051 | 0.2MB | 2 | 0 | ✅ **Stable** |
| **60s** | 0.050 | 0.1MB | 3 | 0 | ✅ **Stable** |
| **300s (5min)** | 0.049 | 15.8MB | 4 | 0 | ✅ **Stable** |
| **600s (10min)** | ~0.050 | ~30MB | ~8 | 0 | 🔄 **Testing** |

### Key Findings

1. **Best Overall Model**: `sherpa-onnx-streaming-zipformer-en-2023-06-26`
   - **Word Error Rate**: 5.6% (excellent)
   - **Real-time Factor**: 0.047 (very fast - 20x faster than real-time)
   - **Model Load Time**: 2.6 seconds
   - Uses int8 quantized models for stability

2. **Performance vs. Accuracy Trade-off**:
   - Larger model (2023-06-26): Higher accuracy, slightly slower loading
   - Smaller model (20M): Much faster loading, significantly worse accuracy

3. **Entity Recognition Issues Identified**:
   - Numbers and technical terms have poor recognition
   - Example: Real-world audio transcribed as "ONCE THEREO ZERO ZEROAN NINE O TWO UNO CYRIL ONE EIGHT SORROW"
   - This confirms your "bbb.com/news" accuracy concerns

## Configuration Recommendations

### For Real-time Conversation (Latency Priority)
```python
python test_sherpa_proper.py \
  --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  --chunk-size 160 \
  --max-active-paths 3
```

### For High Accuracy (Quality Priority)  
```python
python test_sherpa_proper.py \
  --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  --chunk-size 400 \
  --max-active-paths 8 \
  --hotwords-file hotwords.txt \
  --hotwords-score 2.0
```

### For Resource-Constrained Environments
```python
python test_sherpa_proper.py \
  --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 \
  --chunk-size 300 \
  --max-active-paths 2
```

## Entity Post-Processing Solution

To address the names/URLs accuracy issue, use the entity post-processor:

```python
from entity_postprocessor import EntityPostProcessor

processor = EntityPostProcessor()
corrected_text, stats = processor.process_text(sherpa_output)
```

This will fix common issues like:
- "bbb dot com slash news" → "bbc.com/news"
- "john at company dot com" → "john@company.com"
- "micro soft" → "Microsoft"

## Files Structure

```
tests/sherpa_benchmarks/
├── test_sherpa_proper.py           # Main benchmark script (official sherpa-onnx API)
├── entity_postprocessor.py         # Entity correction for names/URLs
├── sherpa_optimizer.py             # Configuration optimization
├── test_audio/
│   ├── standard/                   # Standard benchmark files
│   ├── realworld/                  # Your real-world audio files
│   └── entities/                   # Entity-dense test samples
└── results/                        # JSON benchmark results
```

## Performance Metrics Tracked

- **Accuracy**: Word Error Rate (WER), Character Error Rate (CER)
- **Latency**: Model load time, Real-time Factor (RTF)
- **Resources**: Peak memory usage, average CPU usage
- **Streaming**: Number of segments, endpoint detection count

## Next Steps

1. **For your "bbb.com/news" issue**: Use the entity post-processor or create hotwords file
2. **For better accuracy**: Test with hotwords file containing common names/URLs
3. **For production use**: The 2023-06-26 model with int8 quantization is recommended

## Usage Examples

### Basic Benchmarking
```bash
# Compare all available models
for model in sherpa-onnx-streaming-zipformer-en-*; do
    python test_sherpa_proper.py --model-dir ../../models/$model --test-suite standard
done

# Test with your audio files
python test_sherpa_proper.py --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 --test-suite realworld
```

### Stress Testing (Longer Audio)
```bash
# Test sustained load with longer audio
python test_sherpa_stress.py --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 --durations 10 30 60 300

# Test with different chunk sizes for optimization
python test_sherpa_stress.py --model-dir ../../models/sherpa-onnx-streaming-zipformer-en-2023-06-26 --durations 60 --chunk-size 400

# Analyze stress test results
python analyze_stress_results.py results/stress_test_*.json
```

### Optimization Analysis
```bash
# Generate optimization recommendations
python sherpa_optimizer.py results/sherpa_proper_standard_*.json --profile accuracy

# Test entity post-processing
python entity_postprocessor.py
```

## Dependencies

- sherpa-onnx
- numpy
- psutil (for resource monitoring)
- wave (for audio loading)

Install with: `pip install sherpa-onnx numpy psutil`