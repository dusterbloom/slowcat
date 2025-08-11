# Kroko-ASR Ultra Low Latency Testing Guide

This guide walks you through testing the new Kroko-ASR model for ultra-low latency streaming ASR in Slowcat.

## About Kroko-ASR

- **Ultra-low latency**: Designed specifically for edge devices and streaming
- **Languages**: English, French, Spanish (more coming)
- **Architecture**: Transducer (encoder/decoder/joiner) - compatible with existing Sherpa service
- **Model sizes**: ~70MB encoder, small decoder/joiner files
- **Goal**: Faster/higher quality than similarly sized Whisper models

## Quick Start

### 1. Download Model

```bash
cd server/
./scripts/download_kroko_model.sh en  # Download English model
./scripts/download_kroko_model.sh fr  # Download French model  
./scripts/download_kroko_model.sh es  # Download Spanish model
```

### 2. Test Model Performance

```bash
# Basic accuracy test
python test_kroko_model.py --language en

# Detailed test with output
python test_kroko_model.py --language en --output kroko_results.json --verbose

# Test different languages
python test_kroko_model.py --language fr --model-dir ./models/kroko-asr-fr
```

### 3. Test Integration

```bash
# Test with existing Sherpa service
python test_kroko_integration.py --language en

# Test all available languages
python test_kroko_integration.py --all-languages
```

### 4. Use in Production

Update your `.env` file:
```bash
SHERPA_ONNX_MODEL_DIR=./models/kroko-asr-en
```

Or for other languages:
```bash
SHERPA_ONNX_MODEL_DIR=./models/kroko-asr-fr  # French
SHERPA_ONNX_MODEL_DIR=./models/kroko-asr-es  # Spanish
```

## Performance Metrics

The test script measures:
- **RTF (Real-Time Factor)**: Lower is better, <1.0 means real-time
- **WER (Word Error Rate)**: Lower is better, accuracy measure
- **Processing Time**: Actual time to process audio
- **Success Rate**: Percentage of successful recognitions

### Target Performance for Ultra-Low Latency:
- RTF < 0.5: Excellent (ultra-low latency)
- RTF < 1.0: Good (real-time)
- WER < 0.05: Excellent accuracy
- WER < 0.15: Good accuracy

## Files Created

- `server/scripts/download_kroko_model.sh` - Download script
- `server/test_kroko_model.py` - Comprehensive accuracy/performance test
- `server/test_kroko_integration.py` - Integration test with existing service
- `docs/KROKO_ASR_TESTING.md` - This guide

## Expected Results

Based on Kroko-ASR being designed for edge devices, you should see:
1. **Lower RTF** than current Zipformer models (better latency)
2. **Comparable or better accuracy** on technical terms, URLs
3. **Seamless integration** with existing Sherpa service
4. **Fast initialization** and streaming performance

## Troubleshooting

### Model Download Issues
```bash
# If script fails, try manual download
huggingface-cli download Banafo/Kroko-ASR en_encoder.onnx --local-dir ./models/kroko-asr-en
huggingface-cli download Banafo/Kroko-ASR en_decoder.onnx --local-dir ./models/kroko-asr-en
huggingface-cli download Banafo/Kroko-ASR en_joiner.onnx --local-dir ./models/kroko-asr-en
huggingface-cli download Banafo/Kroko-ASR en_tokens.txt --local-dir ./models/kroko-asr-en
```

### Integration Issues
- Kroko-ASR uses standard transducer architecture
- Should work with existing `SherpaOnlineSTTService` 
- Check model file naming (may need renaming)

### Performance Issues
- Try different `max_active_paths` settings (2-8)
- Adjust endpoint detection parameters
- Use CPU provider for stability

## Next Steps

If tests show good results:
1. Update service factory to support Kroko models
2. Add model selection in configuration
3. Create comparative benchmarks vs current models
4. Deploy to test environment
5. Measure end-to-end voice-to-voice latency

## Model Comparison Command

```bash
# Compare with existing model
python test_kroko_model.py --model-dir ./models/kroko-asr-en --output kroko.json
python test_existing_model.py --model-dir ./models/sherpa-onnx-streaming-zipformer-en-2023-06-26 --output zipformer.json

# Then compare results manually or create comparison script
```