# M3 Integration Guide

This guide explains how to use the M3 (Multimodal Memory Management) system integrated with Slowcat bot pipeline.

## Overview

The M3 integration replaces the standard SmartContextManager with an intelligent context retrieval system that uses:

- **M3 SimilaritySearch**: Vector-based similarity matching
- **M3 EquivalenceResolver**: Resolves equivalent queries and concepts
- **M3 ContextRetriever**: Intelligent ranking and selection of relevant context
- **SurrealDB**: Graph-based memory storage with temporal relationships

## Features

✅ **Backward Compatible**: Works without M3 when disabled, falls back to standard memory
✅ **Environment Controlled**: All settings configurable via environment variables
✅ **Graceful Degradation**: Automatically falls back to standard memory if SurrealDB unavailable
✅ **Performance Monitoring**: Built-in stats and monitoring
✅ **Token Budget Management**: Maintains fixed token limits while providing relevant context

## Environment Variables

### Core M3 Settings
```bash
ENABLE_M3=true                      # Enable M3 system (default: false)
USE_M3_CONTEXT=true                 # Use M3 for context retrieval (default: false)
```

### SurrealDB Connection
```bash
SURREALDB_HOST=localhost            # SurrealDB host (default: localhost)
SURREALDB_PORT=8000                 # SurrealDB port (default: 8000)
SURREALDB_NAMESPACE=slowcat         # SurrealDB namespace (default: slowcat)
SURREALDB_DATABASE=memory           # SurrealDB database (default: memory)
```

### M3 Retrieval Settings
```bash
M3_MAX_CONTEXT_TOKENS=4096          # Total context limit (default: 4096)
M3_MEMORY_TOKENS=2000               # Tokens for memory context (default: 2000)
M3_SIMILARITY_THRESHOLD=0.7         # Similarity matching threshold (default: 0.7)
M3_MAX_RETRIEVAL_ITEMS=20           # Max items to retrieve (default: 20)
M3_RETRIEVAL_STRATEGY=hybrid        # Strategy: hybrid, similarity_first, entity_first (default: hybrid)
M3_ENABLE_EQUIVALENCE=true          # Enable equivalence resolution (default: true)
```

### Performance Settings
```bash
M3_EMBEDDING_CACHE_SIZE=1000        # Embedding cache size (default: 1000)
M3_CONNECTION_TIMEOUT=10            # Connection timeout seconds (default: 10)
M3_QUERY_TIMEOUT=5                  # Query timeout seconds (default: 5)
```

### Graceful Degradation
```bash
M3_FALLBACK_TO_STANDARD=true       # Fallback to standard memory (default: true)
M3_STARTUP_RETRIES=3                # Connection retry attempts (default: 3)
M3_STARTUP_RETRY_DELAY=2            # Delay between retries (default: 2)
```

## Usage Examples

### Basic M3 Enabled Setup
```bash
# Enable M3 with default settings
export ENABLE_M3=true
export USE_M3_CONTEXT=true
./run_bot.sh
```

### Full M3 Configuration
```bash
# Complete M3 setup with custom settings
export ENABLE_M3=true
export USE_M3_CONTEXT=true
export SURREALDB_HOST=localhost
export SURREALDB_PORT=8000
export M3_MAX_CONTEXT_TOKENS=8192
export M3_MEMORY_TOKENS=4000
export M3_SIMILARITY_THRESHOLD=0.8
export M3_RETRIEVAL_STRATEGY=hybrid
./run_bot.sh
```

### Production Setup with Fallback
```bash
# Production setup with graceful fallback
export ENABLE_M3=true
export USE_M3_CONTEXT=true
export M3_FALLBACK_TO_STANDARD=true
export M3_STARTUP_RETRIES=5
export M3_STARTUP_RETRY_DELAY=3
./run_bot.sh
```

### Disable M3 (Standard Memory)
```bash
# Use standard SmartContextManager
export ENABLE_M3=false
export USE_M3_CONTEXT=false
./run_bot.sh
```

## Architecture

### Pipeline Integration

```
User Input → STT → M3IntegratedContextManager → LLM → TTS → Output
                            ↓
         M3 Context Retrieval (if enabled) OR Standard Memory (fallback)
                            ↓
                   [M3SimilaritySearch]
                            ↓
                 [M3EquivalenceResolver]
                            ↓
                   [M3ContextRetriever]
                            ↓
                     [SurrealDB Graph]
```

### Service Factory Integration

- **M3 Services**: Registered in ServiceFactory for dependency injection
- **Lazy Loading**: M3 components created only when needed
- **Connection Management**: Automatic SurrealDB connection management
- **Error Handling**: Comprehensive error handling with fallbacks

### Configuration Flow

1. **Environment Variables** → M3Config in config.py
2. **M3Config** → ServiceFactory for component creation
3. **ServiceFactory** → PipelineBuilder for context manager creation
4. **PipelineBuilder** → Creates M3IntegratedContextManager or falls back

## Testing

### Integration Tests
```bash
# Run M3 integration tests
source .venv/bin/activate
python test_m3_pipeline_integration.py
```

### Production Tests
```bash
# Run production integration tests
source .venv/bin/activate
python test_m3_production_integration.py
```

### Manual Testing
```bash
# Test with M3 enabled
ENABLE_M3=true USE_M3_CONTEXT=true ./run_bot.sh --help

# Test with M3 disabled
ENABLE_M3=false USE_M3_CONTEXT=false ./run_bot.sh --help
```

## Monitoring

### Performance Stats

The M3IntegratedContextManager provides performance statistics:

```python
# Access stats from the context manager
stats = context_manager.get_stats()
print(f"M3 enabled: {stats['m3_enabled']}")
print(f"Total queries: {stats['total_queries']}")
print(f"M3 queries: {stats['m3_queries']}")
print(f"Standard queries: {stats['standard_queries']}")
print(f"Fallback events: {stats['fallback_events']}")
print(f"Avg retrieval time: {stats['avg_retrieval_time_ms']}ms")
```

### Log Messages

Key log messages to monitor:

- `🧠 Creating M3-integrated context manager...` - M3 context manager created
- `✅ M3 SurrealDB connection established` - SurrealDB connection successful
- `🔄 Falling back to standard memory system` - Fallback to standard memory
- `❌ M3 system initialization failed` - M3 initialization failed

## Troubleshooting

### SurrealDB Connection Issues

**Problem**: `❌ Failed to create M3 connection`
**Solution**: 
1. Ensure SurrealDB is running: `surreal start --log trace memory://`
2. Check host/port configuration
3. Enable fallback: `M3_FALLBACK_TO_STANDARD=true`

### Performance Issues

**Problem**: Slow context retrieval
**Solution**:
1. Reduce max retrieval items: `M3_MAX_RETRIEVAL_ITEMS=10`
2. Increase similarity threshold: `M3_SIMILARITY_THRESHOLD=0.8`
3. Reduce connection timeout: `M3_CONNECTION_TIMEOUT=5`

### Memory Issues

**Problem**: High memory usage
**Solution**:
1. Reduce embedding cache: `M3_EMBEDDING_CACHE_SIZE=500`
2. Reduce context tokens: `M3_MAX_CONTEXT_TOKENS=2048`
3. Reduce memory tokens: `M3_MEMORY_TOKENS=1000`

### Fallback Not Working

**Problem**: M3 fails but doesn't fall back
**Solution**:
1. Enable fallback: `M3_FALLBACK_TO_STANDARD=true`
2. Check standard memory dependencies are available
3. Review error logs for specific issues

## Development

### Adding New M3 Components

1. **Create Component**: Implement new M3 component
2. **Register Service**: Add to ServiceFactory in `core/service_factory.py`
3. **Update Integration**: Modify M3IntegratedContextManager to use component
4. **Add Tests**: Create tests in `tests/test_m3_integration.py`
5. **Update Config**: Add configuration options if needed

### Custom Retrieval Strategies

Implement custom strategies by extending M3ContextRetriever:

```python
class CustomM3ContextRetriever(M3ContextRetriever):
    async def retrieve_context(self, query: str, **kwargs):
        # Custom retrieval logic
        return await super().retrieve_context(query, **kwargs)
```

## Security Considerations

- **Local Processing**: All M3 processing happens locally
- **No Cloud Dependencies**: SurrealDB runs locally
- **Data Isolation**: Each user/session has isolated memory space
- **Connection Security**: Use secure WebSocket connections in production

## Performance Guidelines

### Recommended Settings

For **Development**:
```bash
M3_MAX_CONTEXT_TOKENS=4096
M3_MEMORY_TOKENS=2000
M3_MAX_RETRIEVAL_ITEMS=20
M3_SIMILARITY_THRESHOLD=0.7
```

For **Production**:
```bash
M3_MAX_CONTEXT_TOKENS=8192
M3_MEMORY_TOKENS=4000
M3_MAX_RETRIEVAL_ITEMS=15
M3_SIMILARITY_THRESHOLD=0.8
M3_EMBEDDING_CACHE_SIZE=2000
```

For **Resource Constrained**:
```bash
M3_MAX_CONTEXT_TOKENS=2048
M3_MEMORY_TOKENS=1000
M3_MAX_RETRIEVAL_ITEMS=10
M3_SIMILARITY_THRESHOLD=0.9
M3_EMBEDDING_CACHE_SIZE=500
```

## Future Enhancements

- **Multimodal Support**: Image and audio memory integration
- **Federated Learning**: Distributed memory across multiple instances
- **Advanced Strategies**: ML-based retrieval strategy selection
- **Real-time Analytics**: Live performance dashboards
- **Memory Compression**: Intelligent memory compaction and archiving

## Support

For issues or questions:
1. Check logs for specific error messages
2. Run integration tests to verify setup
3. Review configuration against this guide
4. Test with M3 disabled to isolate issues