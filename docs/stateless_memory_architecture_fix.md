# Stateless Memory Architecture Fix Plan

## Problem Analysis

The current memory system uses **frame injection** (adding system messages during conversation) instead of true **stateless context reconstruction**. This causes:

1. **Multiple memory injections** - Memory gets injected through `LLMMessagesAppendFrame` every time a transcription happens
2. **Context pollution** - Multiple system messages confuse the LLM 
3. **LLM ignoring memory** - Even when memory contains the answer (like "Potola"), the LLM doesn't use it

## Core Issue

Current flow:
```
Transcription → Memory Search → Inject Frame → LLM → Response
                                    ↓
                              (Happens multiple times)
```

Desired flow:
```
Transcription → Store Memory → LLM Request → Build Context (with memory) → Response
                                                    ↓
                              (Happens once per turn)
```

## Solution: True Stateless Context Building

### 1. Remove Frame Injection from EnhancedStatelessMemoryProcessor
- Stop using `LLMMessagesAppendFrame` in `_inject_memory_for_transcription()`
- Keep memory storage/retrieval but remove injection
- Memory processor becomes purely a storage/retrieval service

### 2. Implement Context Building in MemoryAwareOpenAILLMContext
- Override `get_messages()` to build complete context from scratch each time
- Integrate memories directly into conversation flow, not as system messages
- Format: System prompt → Memory context → Current conversation

### 3. Fix Memory Format in Context
**Current (broken):**
```json
{"role": "system", "content": "IMPORTANT: Use the following conversation history..."}
```

**Proposed (integrated):**
```json
{"role": "system", "content": "You are Slowcat. Previous context: User's dog is named Potola."}
```

Or as conversation pairs:
```json
{"role": "user", "content": "My dog's name is Potola"},
{"role": "assistant", "content": "I'll remember that your dog is named Potola"}
```

### 4. Ensure Single Injection Point
- Context building happens ONLY when LLM service calls `get_messages()`
- Remove all `LLMMessagesAppendFrame` usage
- No frame-based injection during pipeline processing

### 5. Pipeline Changes

**Remove:**
- `_inject_memory_for_transcription()` method
- All `LLMMessagesAppendFrame` creation and pushing

**Add:**
- Context builder that runs at LLM call time
- Direct integration with `OpenAILLMContext.get_messages()`

## Implementation Order

### Phase 1: Fix Architecture (Current Focus)
1. Remove frame injection from enhanced_stateless_memory.py
2. Implement context building in memory_context_aggregator.py
3. Update pipeline_builder.py to use new context aggregator
4. Test with "Potola" example

### Phase 2: Improve Retrieval (Next)
1. Integrate BM25s for better search
2. Add semantic similarity as secondary ranking
3. Implement query-aware retrieval strategies

## Testing Plan

1. **Storage Test**: "My dog's name is Potola" → Verify storage
2. **Retrieval Test**: "What's my dog's name?" → Should retrieve "Potola"
3. **Context Test**: Check LLM receives clean, single context
4. **Response Test**: LLM should answer "Your dog's name is Potola"

## Performance Goals

- **Latency**: < 50ms for memory retrieval
- **Context Size**: Keep under 1024 tokens for memory
- **Accuracy**: 95%+ recall for recent memories
- **Degradation**: Natural aging from hot → warm → cold tiers