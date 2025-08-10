# Sherpa-ONNX Accuracy Enhancement Service

## Overview

This service provides language-agnostic accuracy enhancement for Sherpa-ONNX transcriptions, improving recognition of names, URLs, and technical terms without requiring static hotwords files.

## Features

- **Zero Maintenance**: No static hotwords.txt files to manage
- **Language Agnostic**: Works across languages using phonetic algorithms
- **Performance Optimized**: Fast pattern matching with optional LLM enhancement
- **Local LLM Integration**: Uses lightweight local models via LM Studio

## Components

- `AdvancedAccuracyEnhancer`: Main enhancement orchestrator
- `ServiceWrapper`: Integration wrapper for existing Sherpa services
- `Config`: Configuration management

## Usage

### Basic Integration

```python
from services.accuracy_enhancement import AdvancedAccuracyEnhancer

# Initialize enhancer
enhancer = AdvancedAccuracyEnhancer()

# Enhance transcription
result = await enhancer.enhance_accuracy(
    "Please visit github dot com slash user",
    confidence=0.65,
    context="User asking for repository information"
)
print(result.corrected_text)  # "Please visit github.com/user"
```

### Service Wrapper Integration

```python
from services.sherpa_streaming_stt import SherpaStreamingSTTService
from services.accuracy_enhancement.service_wrapper import SherpaWithAccuracyEnhancement

# Create base Sherpa service
base_service = SherpaStreamingSTTService(
    model_dir="./models/sherpa-onnx-streaming-zipformer-en-2023-06-26"
)

# Wrap with accuracy enhancement
enhanced_service = SherpaWithAccuracyEnhancement(
    base_service=base_service,
    enable_accuracy_enhancement=True
)
```

## Configuration

Configure via environment variables:

```bash
# Enable accuracy enhancement
SHERPA_ENABLE_ACCURACY_ENHANCEMENT=true

# LLM model for enhancement
SHERPA_ACCURACY_MODEL=qwen3-1.7b:2

# LLM API endpoint
SHERPA_ACCURACY_LLM_BASE_URL=http://localhost:1234/v1

# Confidence threshold
SHERPA_ACCURACY_CONFIDENCE_THRESHOLD=0.7
```

## Dependencies

See `requirements.txt` for full list of dependencies.

Install spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

## Performance

- Pattern Recognition: <50ms
- Phonetic Matching: 10-100ms
- LLM Enhancement: 500-1500ms (only when needed)