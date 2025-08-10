# Integration Guide: Sherpa-ONNX Accuracy Enhancement

## Overview

This guide describes how to integrate the accuracy enhancement system into the existing Sherpa-ONNX transcription service.

## Current Implementation Location

The prototype implementation is located at:
```
server/tests/sherpa_benchmarks/advanced_accuracy_enhancer.py
```

## Integration Steps

### 1. Move Core Files to Production Location

```bash
# Create production directory
mkdir -p server/services/accuracy_enhancement

# Move core implementation
cp server/tests/sherpa_benchmarks/advanced_accuracy_enhancer.py server/services/accuracy_enhancement/
```

### 2. Update Sherpa-ONNX Service

Modify `server/services/stt/sherpa_service.py` to include accuracy enhancement:

```python
# Add import at top
from services.accuracy_enhancement.advanced_accuracy_enhancer import AdvancedAccuracyEnhancer

class SherpaSTTService:
    def __init__(self):
        # ... existing initialization ...
        self.accuracy_enhancer = AdvancedAccuracyEnhancer() if self.enable_accuracy_enhancement else None
    
    async def transcribe(self, audio_data: bytes, context: str = "") -> str:
        # ... existing transcription ...
        raw_transcription = self.recognizer.get_result(stream).text
        
        # Apply accuracy enhancement if enabled
        if self.accuracy_enhancer and self.enable_accuracy_enhancement:
            enhancement_result = await self.accuracy_enhancer.enhance_accuracy(
                raw_transcription, 
                confidence=0.65,  # Adjust based on Sherpa confidence
                context=context
            )
            return enhancement_result.corrected_text
        
        return raw_transcription
```

### 3. Add Configuration Options

Update `server/config.py`:

```python
@dataclass
class STTConfig:
    # ... existing config ...
    enable_accuracy_enhancement: bool = field(default_factory=lambda: os.getenv("SHERPA_ENABLE_ACCURACY_ENHANCEMENT", "false").lower() == "true")
    accuracy_enhancement_model: str = field(default_factory=lambda: os.getenv("SHERPA_ACCURACY_MODEL", "qwen3-1.7b:2"))
```

### 4. Update Environment Configuration

Add to `.env`:

```bash
# Sherpa-ONNX Accuracy Enhancement
SHERPA_ENABLE_ACCURACY_ENHANCEMENT=true
SHERPA_ACCURACY_MODEL=qwen3-1.7b:2
```

### 5. Add Required Dependencies

Update `server/requirements.txt`:

```txt
# Accuracy Enhancement Dependencies
spacy>=3.0.0
jellyfish>=0.8.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.12.0
editdistance>=0.5.0
metaphone>=0.6.0
phonetics>=0.1.0
openai>=1.0.0  # For LM Studio integration
```

### 6. Install spaCy Model

```bash
# Install lightweight English model
python -m spacy download en_core_web_sm
```

## Integration Points

### Real-time Streaming Integration

For streaming transcriptions, apply enhancement to final results only:

```python
async def stream_transcribe(self, audio_stream, context: str = ""):
    # ... streaming logic ...
    if is_final_segment:  # Only enhance final segments
        if self.accuracy_enhancer:
            result = await self.accuracy_enhancer.enhance_accuracy(
                text, 
                confidence=segment_confidence,
                context=context
            )
            yield result.corrected_text
        else:
            yield text
```

### Batch Processing Integration

For batch processing, enhance entire transcriptions:

```python
async def batch_transcribe(self, audio_files: List[str], context: str = ""):
    results = []
    for audio_file in audio_files:
        raw_text = await self.transcribe_file(audio_file)
        if self.accuracy_enhancer:
            enhanced_result = await self.accuracy_enhancer.enhance_accuracy(
                raw_text,
                confidence=0.7,  # Default confidence for batch
                context=context
            )
            results.append(enhanced_result.corrected_text)
        else:
            results.append(raw_text)
    return results
```

## Performance Considerations

### Asynchronous Processing

The accuracy enhancer is designed to be async-friendly:

```python
# Non-blocking enhancement
enhancement_task = asyncio.create_task(
    self.accuracy_enhancer.enhance_accuracy(text, confidence, context)
)
# Continue processing while enhancement runs
result = await enhancement_task
```

### Caching Strategy

Implement caching for frequent corrections:

```python
# In AdvancedAccuracyEnhancer class
def __init__(self, enable_caching: bool = True):
    self.cache = {} if enable_caching else None
    self.cache_ttl = 3600  # 1 hour
```

## Testing Integration

### Unit Tests

Create `server/tests/test_accuracy_enhancement.py`:

```python
import pytest
from services.accuracy_enhancement.advanced_accuracy_enhancer import AdvancedAccuracyEnhancer

@pytest.mark.asyncio
async def test_url_formatting():
    enhancer = AdvancedAccuracyEnhancer()
    result = await enhancer.enhance_accuracy("Please visit github dot com")
    assert "github.com" in result.corrected_text

@pytest.mark.asyncio  
async def test_technical_terms():
    enhancer = AdvancedAccuracyEnhancer()
    result = await enhancer.enhance_accuracy("The dock her container")
    assert "Docker" in result.corrected_text
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_sherpa_with_enhancement():
    # Test full integration with Sherpa service
    service = SherpaSTTService()
    service.enable_accuracy_enhancement = True
    
    # Mock audio with known issues
    result = await service.transcribe(test_audio_with_urls)
    assert "github.com" in result  # Should be formatted
```

## Monitoring and Metrics

### Add Performance Metrics

```python
import time
from utils.monitoring import record_metric

async def transcribe_with_metrics(self, audio_data, context=""):
    start_time = time.time()
    
    result = await self.transcribe(audio_data, context)
    
    # Record enhancement metrics
    if self.accuracy_enhancer:
        enhancement_time = time.time() - start_time
        record_metric("accuracy_enhancement_time", enhancement_time)
        record_metric("accuracy_improvement", self.calculate_improvement())  # Custom metric
    
    return result
```

## Rollout Strategy

### 1. Feature Flag
Start with feature flag disabled by default:
```python
SHERPA_ENABLE_ACCURACY_ENHANCEMENT=false  # Opt-in initially
```

### 2. Gradual Rollout
- Enable for test users first
- Monitor performance impact
- Gradually increase rollout percentage

### 3. Fallback Mechanism
```python
try:
    enhanced_text = await self.accuracy_enhancer.enhance_accuracy(text, confidence, context)
except Exception as e:
    logger.warning(f"Accuracy enhancement failed: {e}")
    return text  # Fallback to raw transcription
```

## Troubleshooting

### Common Issues

1. **LLM Connection Failures**
   ```python
   # Add connection health check
   if not self.accuracy_enhancer.llm_corrector.client:
       logger.warning("LLM client not available, skipping enhancement")
   ```

2. **Performance Degradation**
   ```python
   # Add timeout for LLM calls
   try:
       result = await asyncio.wait_for(
           self.accuracy_enhancer.enhance_accuracy(text), 
           timeout=5.0  # 5 second timeout
       )
   except asyncio.TimeoutError:
       logger.warning("Accuracy enhancement timed out, using raw text")
       return text
   ```

3. **Memory Usage**
   ```python
   # Implement memory monitoring
   import psutil
   if psutil.virtual_memory().percent > 80:
       logger.warning("High memory usage, skipping accuracy enhancement")
       return raw_text
   ```

## Next Steps

1. **Create Pull Request** with integration changes
2. **Set up CI/CD pipeline** for accuracy enhancement tests
3. **Monitor performance** in staging environment
4. **Gather user feedback** on accuracy improvements
5. **Optimize based on real-world usage patterns**