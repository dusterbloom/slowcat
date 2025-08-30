# Fact Extraction Analysis & Gemma 3-270M Integration

## Summary

This document analyzes the implementation of LLM-based fact extraction using Gemma 3-270M as a replacement for SpaCy, and explores its potential role in query classification and memory retrieval gating.

---

## Problem Statement

### Original Issues
1. **SpaCy extraction failure**: Simple statements like "My dog is Luna" extracted 0 facts
2. **Overly complex patterns**: SpaCy required constant maintenance of hardcoded patterns
3. **Retrieval over-gating**: Facts rarely included in context due to restrictive heuristics
4. **Assistant never gets facts**: Complex pipeline often failed to include extracted facts in LLM context

### Core Question
Can small LLMs (270M parameters) replace SpaCy for fact extraction while maintaining acceptable speed and reliability?

---

## Methodology

### Test Framework
Created comprehensive comparison test (`test_optimized_fact_extraction.py`) with:
- **Models tested**: Gemma 3-270M, Qwen 2.5-0.5B, SpaCy baseline
- **JSON Schema enforcement**: Structured output for SurrealDB compatibility
- **Persistent connections**: Avoid model reload overhead
- **Speed optimization**: Token limits, temperature=0.0, focused prompts

### Test Cases
Space exploration themed (consistent with project patterns):
1. "My dog is Luna" (original problem case)
2. "Neil Armstrong walked on the moon" 
3. "I work as a software engineer"
4. "The Apollo 11 mission launched in 1969"
5. "Buzz Aldrin's real name is Edwin Eugene Aldrin Jr"

### Performance Metrics
- **Success rate**: JSON parsing success
- **Extraction quality**: Entity/relation count and accuracy
- **Speed**: Cold start vs warm model performance
- **Format compatibility**: SurrealDB schema readiness

---

## Results Analysis

### Final Performance Comparison

| Metric | SpaCy | Gemma 3-270M | Winner |
|--------|-------|--------------|---------|
| **Success Rate** | 100% | 100% | Tie |
| **Speed (avg)** | 371ms | 1252ms | SpaCy (3.4x faster) |
| **"My dog is Luna"** | 0 facts | 5 entities + 1 relation | **Gemma** |
| **Relationship Extraction** | 0 relations | 1+ relations | **Gemma** |
| **Output Format** | Custom | SurrealDB-ready JSON | **Gemma** |
| **Maintenance** | Pattern coding | Prompt engineering | **Gemma** |

### Critical Discovery
- **SpaCy**: Fast but extracts 0 relationships from simple statements
- **Gemma**: 3.4x slower but actually solves the core problem
- **Trade-off**: 371ms → 1252ms for facts that actually exist vs none

### Key Insights from LM Studio Logs
1. **Token truncation**: Initial failures due to 250 token limit (fixed with 400 tokens)
2. **Over-generation**: Model wanted to extract 8+ entities; limited to 5 max
3. **Warm performance**: Consistent ~1.25s after initial model load
4. **JSON schema enforcement**: Works reliably with proper token allocation

---

## Implementation Decision

### ✅ RECOMMENDATION: Deploy Gemma 3-270M for Fact Extraction

**Justification**: 1.25 seconds to extract meaningful facts is infinitely better than 0.37 seconds to extract nothing.

### Production Integration Strategy

1. **Replace SpaCy calls** in `extract_facts_from_text()`
2. **Maintain async execution** - run extraction in background
3. **Use persistent HTTP connections** - keep model warm
4. **Direct SurrealDB storage** - leverage existing schema functions
5. **JSON schema enforcement** - prevent parsing failures

### Code Changes Required
- `memory/spacy_fact_extractor.py` → `memory/gemma_fact_extractor.py`
- Update imports in `SmartContextManager`
- Add LM Studio dependency to deployment

---

## Query Classification & Memory Retrieval Analysis

### Current Retrieval Gating System

Looking at `SmartContextManager._get_relevant_facts()` (lines 1879-1882):

```python
# Heuristic: only search when the user asks a question or uses query-like phrasing
query_starters = ("who", "what", "where", "when", "why", "how", "which", "do ", "did ", "can ", "could ", "would ", "is ", "are ")
mem_keywords = ("remember", "recall", "age", "name", "location", "live", "from", "work", "job")
is_query_like = ("?" in q) or qlow.startswith(query_starters) or any(k in qlow for k in mem_keywords)
```

### Problems with Current Approach
1. **Over-restrictive**: Many valid memory queries don't match these patterns
2. **Language-dependent**: Hardcoded English patterns
3. **Brittle**: Requires constant maintenance as conversation patterns evolve
4. **Context-agnostic**: No understanding of conversation flow

### Gemma 3-270M for Query Classification

#### Potential Benefits
- **Natural language understanding**: Better than keyword matching
- **Context awareness**: Can understand conversation flow
- **Consistent with fact extraction**: Same model, same inference stack
- **Fine-tunable**: Can train on specific classification tasks

#### Classification Schema
```json
{
  "intent": "memory_query" | "casual_conversation" | "factual_statement" | "question_answering",
  "memory_needed": true | false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
```

#### Sample Classification Prompt
```
Text: "{user_input}"
Previous context: "{last_2_exchanges}"

Classify this input:
- Does this need memory retrieval? (facts, personal info, history)
- What type of interaction is this?
- Confidence level?

Be conservative - only retrieve memory when clearly needed.
```

### Speed Considerations for Classification

**Current facts retrieval gating**:
- Keyword matching: ~1ms
- Current solution: Skip retrieval for 80%+ of messages

**Gemma classification**:
- Small prompt: ~200-300ms per classification
- Could be acceptable if it significantly improves retrieval accuracy

### Alternative Approaches

#### 1. **Hybrid Gating**
- Fast keyword pre-filter (current system)
- Gemma classification only for uncertain cases
- Best of both worlds: speed + accuracy

#### 2. **Semantic Similarity**
- Use embeddings to match input against fact types
- Faster than LLM classification
- Already available in your embedding infrastructure

#### 3. **Intent-Aware Context Building**
- Always include top 3-5 facts (minimal overhead)
- Use Gemma to select which facts are relevant
- Moves classification to selection rather than gating

---

## Recommendations

### Phase 1: Fact Extraction (Immediate)
✅ **Deploy Gemma 3-270M for fact extraction**
- Proven 100% success rate
- Solves core user problem
- 1.25s latency acceptable for background processing

### Phase 2: Query Classification (Investigate)
🔍 **Evaluate Gemma for memory retrieval gating**

**Test questions**:
1. Can 200-300ms classification improve retrieval accuracy significantly?
2. What's the false positive/negative rate vs current keyword system?
3. Is hybrid approach (keyword pre-filter + LLM classification) optimal?

**Suggested experiment**:
- Log current retrieval decisions vs what Gemma would decide
- Measure classification accuracy on real conversation data
- Compare retrieval relevance scores

### Phase 3: Optimization (Future)
- **Fine-tune Gemma** on your specific fact patterns
- **Implement INT4 quantization** for faster inference
- **Consider model caching** strategies for repeated classifications

---

## Technical Specifications

### Production Requirements
- **LM Studio**: Running with Gemma 3-270M model loaded
- **Persistent connections**: HTTP keep-alive for warm performance
- **Token allocation**: 400 max tokens for complex extractions
- **JSON schema**: Strict mode for reliable parsing
- **Async execution**: Non-blocking background processing

### Performance Targets
- **Fact extraction**: <1.5s per text snippet
- **Query classification**: <300ms per user input (if implemented)
- **Success rate**: 100% JSON parsing reliability
- **Memory efficiency**: Persistent model loading

### Integration Points
- `SmartContextManager._extract_facts_async()`
- `SmartContextManager._get_relevant_facts()` (potential)
- SurrealDB schema functions for storage

---

## Conclusion

Gemma 3-270M successfully solves the fact extraction problem with acceptable performance trade-offs. The model's design for "high-volume, well-defined tasks like entity extraction" aligns perfectly with our use case.

For query classification, further investigation is needed to determine if the 200-300ms classification overhead provides sufficient value over the current keyword-based gating system.

**Next Steps**: Implement fact extraction replacement and design query classification experiment to measure real-world performance impact.