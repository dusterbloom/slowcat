# Future Work: Enhanced Stateless Memory System

## Context
We've implemented a stateless memory system for Slowcat voice assistant with constant <10ms performance. The basic implementation is complete but needs production-grade enhancements based on expert feedback.

## Current Status
- ✅ Basic stateless memory processor implemented
- ✅ Semantic validation with fallbacks
- ✅ Memory degradation with forgetting curves
- ✅ Tool integration interfaces
- ✅ Error handling and async operations
- 🐛 DEBUGGING NEEDED: Pipecat frame processor integration issues

## Immediate Tasks (Debug Current Implementation)

### 1. Fix Pipecat Frame Processing Issues
**Error**: `StatelessMemoryProcessor#0 Trying to process UserStartedSpeakingFrame#0 but StartFrame not received yet`
**Fix**: Add proper StartFrame handling in process_frame method

**Error**: `TranscriptionFrame.__init__() missing 2 required positional arguments: 'user_id' and 'timestamp'`  
**Fix**: Update test to use correct TranscriptionFrame constructor

### 2. Fix Frame Processor Lifecycle
- Add StartFrame and EndFrame handling
- Ensure proper processor startup sequence
- Fix frame direction handling

### 3. Update Tests for Pipecat Compatibility
- Fix TranscriptionFrame constructor calls
- Add proper frame sequencing
- Test actual pipeline integration

## Next Phase: Production Enhancements (from enhanced_stateless_memory_plan.md)

### Priority 1 (Day 1)
1. **Tokenizer Adapter**: Replace word-based token estimation with proper tokenization
2. **Content Sanitizer**: Add PII removal, deduplication, no-echo rules

### Priority 2 (Day 2)  
3. **Storage Optimization**: Single-writer queue + LZ4/Zstd tiered compression
4. **Hybrid Retrieval**: Multi-strategy retrieval (recency + keywords + entities + optional embeddings)

### Priority 3 (Day 3)
5. **Robustness**: Slow path retry logic, circuit breakers, graceful degradation
6. **Configuration**: Feature flags and monitoring

## Key Files to Work On
- `server/processors/stateless_memory.py` - Main implementation
- `server/processors/tokenizer_adapter.py` - NEW: Proper tokenization
- `server/processors/memory_sanitizer.py` - NEW: Content cleaning
- `server/processors/hybrid_retriever.py` - NEW: Smart retrieval
- `server/config.py` - Add feature flags
- `server/tests/test_memory_production.py` - NEW: Production tests

## Success Criteria
- [ ] No frame processing errors
- [ ] <10ms injection latency maintained
- [ ] Proper tokenization accuracy
- [ ] PII sanitization working
- [ ] Deduplication preventing storage bloat
- [ ] Multi-strategy retrieval improving relevance
- [ ] Robust error handling and recovery
- [ ] Full A/B testing capability

## Current Architecture Strengths to Preserve
- Constant performance regardless of conversation length
- LLM remains completely stateless
- Natural memory degradation prevents infinite growth
- Perfect recall window for user experience
- Tool integration compatibility

Continue from debugging the Pipecat integration, then proceed with the production enhancements plan.