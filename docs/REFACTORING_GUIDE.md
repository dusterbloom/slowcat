# Phase 1 Refactoring Implementation Guide

## Overview

This guide provides detailed implementation steps for Phase 1 of the Slowcat server refactoring. The goal is to break down the monolithic `bot.py` into focused modules while maintaining complete backward compatibility.

## Architecture Changes

### Before (Current Structure)
```
bot.py (472 lines)
├── ML module loading
├── Service initialization  
├── Pipeline configuration
├── FastAPI server setup
├── WebRTC handling
└── Main execution logic
```

### After (Refactored Structure)
```
core/
├── service_factory.py     # Dependency injection & service creation
├── service_interfaces.py  # Abstract base classes
└── pipeline_builder.py    # Pipeline construction logic

server/
├── app.py                 # FastAPI application
└── webrtc.py             # WebRTC connection management

bot_v2.py                  # New entry point (backward compatible)
tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
└── performance/           # Performance tests
```

## Step-by-Step Implementation

### Step 1: Service Factory Implementation

The `ServiceFactory` provides dependency injection and manages service lifecycles:

```python
from core.service_factory import service_factory

# Get a service (creates if not exists)
stt_service = await service_factory.get_service("stt_service")

# Create language-specific services
services = await service_factory.create_services_for_language("es", "custom-model")

# Services are cached as singletons by default
llm1 = await service_factory.get_service("llm_service") 
llm2 = await service_factory.get_service("llm_service")
assert llm1 is llm2  # Same instance
```

#### Key Features:
- **Lazy Loading**: ML modules load in background
- **Dependency Injection**: Services declare their dependencies
- **Singleton Management**: Services are cached and reused
- **Language Support**: Create services for specific languages

### Step 2: Pipeline Builder Usage

The `PipelineBuilder` constructs complete processing pipelines:

```python
from core.pipeline_builder import PipelineBuilder
from core.service_factory import service_factory

# Create pipeline builder
builder = PipelineBuilder(service_factory)

# Build complete pipeline
pipeline, task = await builder.build_pipeline(
    webrtc_connection=connection,
    language="es", 
    llm_model="custom-model"
)

# Run pipeline
from pipecat.pipeline.runner import PipelineRunner
runner = PipelineRunner(handle_sigint=False)
await runner.run(task)
```

#### Pipeline Components (automatically configured):
1. **Transport Layer**: WebRTC input/output with VAD and turn detection
2. **Audio Processing**: STT, audio tee for voice recognition  
3. **Context Management**: Memory injection, speaker context
4. **LLM Processing**: Context aggregation, tool handling
5. **Response Generation**: TTS, greeting filtering

### Step 3: Server Module Usage

The server module provides clean FastAPI application management:

```python
from server import create_app, run_server

# Create configured application
app = create_app(language="es", llm_model="custom-model")

# Or run server directly
run_server(
    host="0.0.0.0",
    port=8080, 
    language="fr",
    llm_model="gemma-3-12b"
)
```

#### Features:
- **Health Endpoint**: `/health` shows server status
- **WebRTC Management**: Automatic connection lifecycle
- **Background Tasks**: ML loading and pipeline execution
- **Graceful Shutdown**: Proper cleanup on exit

### Step 4: Backward Compatibility

The `bot_v2.py` maintains the exact same interface as `bot.py`:

```bash
# These commands work identically
python bot.py --language es --llm gemma-3-12b --port 8080
python bot_v2.py --language es --llm gemma-3-12b --port 8080

# The original run_bot function is preserved
from bot_v2 import run_bot
await run_bot(webrtc_connection, "es", "custom-model")
```

## Migration Process

### Phase 1A: Create New Modules (Non-Breaking)

1. **Add core modules** without changing existing code:
   ```bash
   # Create new architecture
   mkdir -p core server tests/{unit,integration,performance}
   
   # Copy provided files:
   # - core/service_factory.py
   # - core/service_interfaces.py  
   # - core/pipeline_builder.py
   # - server/app.py
   # - server/webrtc.py
   # - bot_v2.py
   ```

2. **Test new architecture** alongside existing:
   ```bash
   # Run original bot
   python bot.py --language en
   
   # Run refactored bot (should work identically)
   python bot_v2.py --language en
   ```

3. **Validate functionality**:
   ```bash
   # Test all languages
   python bot_v2.py --language es
   python bot_v2.py --language fr --llm custom-model
   
   # Test health endpoint
   curl http://localhost:7860/health
   ```

### Phase 1B: Migrate Tests (Non-Breaking)

1. **Organize existing tests**:
   ```bash
   python tests/migration_plan.py
   ```

2. **Run test suite**:
   ```bash
   python run_tests.py --unit        # Unit tests only
   python run_tests.py --integration # Integration tests
   python run_tests.py --all         # All tests
   python run_tests.py --coverage    # With coverage report
   ```

### Phase 1C: Gradual Adoption (Breaking Changes)

⚠️ **Only after Phase 1A/1B are working perfectly**:

1. **Switch default entry point**:
   ```bash
   # Rename files
   mv bot.py bot_original.py  
   mv bot_v2.py bot.py
   ```

2. **Update documentation and scripts**:
   ```bash
   # Update run_bot.sh if needed
   # Update README.md references
   # Update any deployment scripts
   ```

3. **Remove deprecated code** (after thorough testing):
   ```bash
   # Remove original after validation period
   rm bot_original.py
   ```

## Configuration Examples

### Service Factory Customization

```python
from core.service_factory import ServiceFactory

# Create custom factory
factory = ServiceFactory()

# Register custom service
def create_custom_processor():
    return MyCustomProcessor()

factory.registry.register(
    "custom_processor",
    create_custom_processor,
    dependencies=["memory_service"],
    singleton=True
)

# Use custom service
processor = await factory.get_service("custom_processor")
```

### Pipeline Builder Customization

```python
from core.pipeline_builder import PipelineBuilder

class CustomPipelineBuilder(PipelineBuilder):
    async def _build_pipeline_components(self, transport, services, processors, context_aggregator):
        # Add custom component
        custom_processor = await self.service_factory.get_service("custom_processor")
        
        components = await super()._build_pipeline_components(
            transport, services, processors, context_aggregator
        )
        
        # Insert custom processor at specific position
        components.insert(-2, custom_processor)  # Before greeting filter
        return components

# Use custom builder
builder = CustomPipelineBuilder(service_factory)
```

### Language-Specific Service Configuration

```python
# Create services for specific language
services = await service_factory.create_services_for_language("ja", "custom-japanese-model")

# Services automatically use correct:
# - Whisper model (MEDIUM for non-English)
# - Voice (jf_alpha for Japanese)
# - Language settings (JA)
# - System prompt with language consistency notes
```

## Testing Strategy

### Unit Tests
Focus on individual components in isolation:

```python
# Test service factory
def test_service_registration():
    factory = ServiceFactory()
    factory.registry.register("test", lambda: "value")
    assert "test" in factory.registry.list_services()

# Test pipeline builder
async def test_pipeline_components():
    builder = PipelineBuilder(mock_factory)
    components = await builder._build_pipeline_components(...)
    assert len(components) > 0
```

### Integration Tests  
Test complete workflows:

```python
async def test_full_pipeline():
    builder = PipelineBuilder(service_factory)
    pipeline, task = await builder.build_pipeline(mock_connection, "en")
    
    # Verify pipeline can be created
    assert pipeline is not None
    assert task is not None
```

### Performance Tests
Ensure refactoring doesn't impact performance:

```python
async def test_service_creation_speed():
    start = time.time()
    services = await service_factory.create_services_for_language("en")
    duration = time.time() - start
    
    # Should create services quickly (after ML modules loaded)
    assert duration < 1.0
```

## Monitoring and Rollback

### Health Monitoring
```python
# Check service factory health
healthy_services = []
for service_name in service_factory.registry.list_services():
    try:
        await service_factory.get_service(service_name)
        healthy_services.append(service_name)
    except Exception as e:
        logger.error(f"Service {service_name} unhealthy: {e}")
```

### Rollback Strategy
If issues occur during migration:

1. **Immediate rollback**:
   ```bash
   # Switch back to original
   mv bot.py bot_v2.py
   mv bot_original.py bot.py
   systemctl restart slowcat  # Or your service manager
   ```

2. **Identify issues**:
   ```bash
   # Check logs
   tail -f logs/slowcat.log
   
   # Run specific tests
   python run_tests.py --integration
   ```

3. **Fix and re-deploy**:
   ```bash
   # Fix issues in bot_v2.py
   # Test thoroughly
   python bot_v2.py --language en  # Validate
   
   # Re-deploy when ready
   mv bot.py bot_original.py
   mv bot_v2.py bot.py
   ```

## Benefits Achieved

### Code Organization
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Services are loosely coupled
- **Testability**: Components can be tested in isolation
- **Maintainability**: Changes are localized to specific modules

### Performance
- **Lazy Loading**: ML modules load in background
- **Singleton Services**: Expensive services created once
- **Resource Management**: Clean service lifecycle management

### Developer Experience
- **Clear Interfaces**: Abstract base classes define contracts
- **Easy Testing**: Comprehensive test organization
- **Backward Compatibility**: Gradual migration path
- **Documentation**: Clear examples and migration guide

## Next Steps (Phase 2)

After Phase 1 is complete and stable:

1. **Processor Abstractions**: Create interfaces for all processors
2. **Configuration Management**: Centralized config validation
3. **Plugin System**: Dynamic processor loading
4. **Monitoring Integration**: Health checks and metrics
5. **Performance Optimization**: Further lazy loading and caching

This refactoring provides a solid foundation for future enhancements while maintaining the reliability and performance of the current system.