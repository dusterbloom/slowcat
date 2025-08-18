# Enhanced Stateless Memory Implementation Plan

Based on your friend's excellent feedback, here's a comprehensive plan to improve the stateless memory system with production-grade features:

## Phase 1: Tokenizer Adapter & Budget Management

### 1.1 Add Proper Tokenizer with Adapter Pattern
**File**: `server/processors/tokenizer_adapter.py` (new)
- Create TokenizerAdapter interface supporting multiple tokenizers
- Implement TiktokenAdapter (for OpenAI models) 
- Implement SimpleTokenizerAdapter (fallback using better heuristics)
- Add character-based fallback for non-English languages
- Cache tokenization results for repeated content

### 1.2 Replace Token Estimation with Hard Budget Trimming
**File**: `server/processors/stateless_memory.py`
- Replace `_estimate_token_count()` with TokenizerAdapter
- Implement hard token budget enforcement with priority ordering:
  1. System messages (highest priority)
  2. Current user message  
  3. Perfect recall window (recent N messages)
  4. Semantically relevant older memories
- Add token counting with overflow protection
- Implement graceful truncation when over budget

## Phase 2: Content Sanitization & Deduplication

### 2.1 Add Content Sanitizer
**File**: `server/processors/memory_sanitizer.py` (new)
- Strip PII (phone numbers, SSNs, credit cards)
- Remove repetitive tokens/phrases
- Normalize whitespace and formatting
- Clean up LLM artifacts (thinking tags, etc.)
- Remove system prompts from stored memories

### 2.2 Implement Deduplication
**Updates**: `server/processors/stateless_memory.py`
- Add semantic hashing with locality-sensitive hashing (LSH)
- Implement fuzzy matching for near-duplicates
- Merge similar memories with timestamp updates
- Track and merge conversation branches

### 2.3 No-Echo Rule
**Updates**: `server/processors/stateless_memory.py`
- Never inject the exact current user message into context
- Filter out memories that are >90% similar to current query
- Prevent context loops and redundancy

## Phase 3: Storage Optimization

### 3.1 Single-Writer Queue Architecture
**File**: `server/processors/memory_queue.py` (new)
- Implement async write queue with single writer thread
- Add write batching for efficiency
- Implement write-ahead log (WAL) for durability
- Add crash recovery mechanism

### 3.2 Tiered Compression (LZ4/Zstd)
**Updates**: `server/processors/stateless_memory.py`
- Hot tier: No compression (last 10 messages)
- Warm tier: LZ4 compression (fast, last 100 messages)
- Cold tier: Zstd compression (high ratio, older messages)
- Add compression ratio monitoring
- Implement adaptive compression based on content type

## Phase 4: Hybrid Retrieval System

### 4.1 Multi-Strategy Retrieval
**File**: `server/processors/hybrid_retriever.py` (new)
- **Recency Strategy**: Last N messages (configurable)
- **Keyword Strategy**: BM25 or TF-IDF based retrieval
- **Entity Strategy**: Named entity recognition and matching
- **Semantic Strategy**: Optional embedding-based search
- **Weighted Fusion**: Combine strategies with learned weights

### 4.2 Smart Ranking & Reranking
**Updates**: `server/processors/stateless_memory.py`
- Initial retrieval gets 3x the needed memories
- Score each memory on multiple dimensions:
  - Recency score (exponential decay)
  - Relevance score (keyword/entity match)
  - Importance score (user flags, access count)
  - Coherence score (context continuity)
- Optional: Use small reranker model for final selection
- Return top-K within token budget

## Phase 5: Robustness & Configuration

### 5.1 Slow Path with Retry Logic
**Updates**: `server/processors/stateless_memory.py`
- Fast path: Try injection with 10ms timeout
- Slow path: If fast path fails, retry with 50ms timeout
- Add circuit breaker pattern for repeated failures
- Implement graceful degradation (return partial context)
- Add telemetry for slow path triggers

### 5.2 Configuration Flags
**Updates**: `server/config.py`
- Add feature flags for each enhancement:
  ```python
  memory_use_tokenizer: bool = True
  memory_sanitize_content: bool = True
  memory_deduplicate: bool = True
  memory_use_compression_tiers: bool = True
  memory_hybrid_retrieval: bool = True
  memory_enable_slow_path: bool = True
  memory_max_fast_path_ms: int = 10
  memory_max_slow_path_ms: int = 50
  ```

## Phase 6: Testing & Monitoring

### 6.1 Enhanced Testing
**File**: `server/tests/test_memory_production.py` (new)
- Test tokenizer accuracy across languages
- Test deduplication with edge cases
- Test compression ratios and speed
- Test hybrid retrieval quality
- Load test with 1000+ conversation turns
- Test slow path triggers and recovery

### 6.2 Monitoring & Metrics
**Updates**: `server/processors/stateless_memory.py`
- Add Prometheus metrics:
  - Token usage histogram
  - Compression ratio gauge
  - Retrieval latency histogram
  - Cache hit rate counter
  - Slow path trigger rate
- Add detailed logging with structured data
- Implement memory usage profiling

## Implementation Order (Fastest Path)

1. **Day 1**: Tokenizer adapter + hard budget (fixes biggest issue)
2. **Day 1**: Sanitizer + dedupe + no-echo (prevents common problems)
3. **Day 2**: Single-writer queue + compression tiers (improves reliability)
4. **Day 2**: Hybrid retrieval basics (better relevance)
5. **Day 3**: Slow path + config flags (production readiness)
6. **Day 3**: Testing + monitoring (observability)

## Key Improvements Over Current Implementation

1. **Accurate token counting** instead of word-based estimates
2. **Content sanitization** to prevent PII leaks
3. **Deduplication** to reduce storage and improve quality
4. **Better compression** with tiered approach
5. **Smarter retrieval** beyond just recency
6. **Production robustness** with retries and fallbacks
7. **Full configurability** for A/B testing

This plan addresses all the critical production concerns while maintaining the core benefits of the stateless architecture.