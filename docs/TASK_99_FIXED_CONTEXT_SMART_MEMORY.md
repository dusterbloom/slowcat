# TASK 99: Fixed Context with Smart Memory System

## Problem Statement

The current Slowcat voice agent has a critical issue where `context_aggregator.user()` in the pipeline accumulates ALL user messages indefinitely, causing:
- Context grows unbounded (50,000+ tokens after 100 turns)
- LLM responses become increasingly slow
- Eventually becomes unusable
- Memory retrieval doesn't actually work due to poor BM25 search on raw text

## Root Cause

Line 528 in `server/core/pipeline_builder.py`:
```python
context_aggregator.user(),  # This accumulates EVERYTHING!
```

The `LLMUserContextAggregator` from Pipecat:
- Receives every `TranscriptionFrame` from STT
- Adds it to context
- Never forgets anything
- Sends entire history to LLM on every turn

## Solution Overview

Replace the accumulating context aggregator with a **Smart Context Manager** that:
1. Maintains FIXED 4096 token context always
2. Extracts facts into a graph structure (subject→predicate→value)
3. Implements natural decay (fidelity levels S4→S0)
4. Generates dynamic system prompts with session metadata
5. Uses tiered retrieval (facts → warm cache → tape)

## Architecture

### Current (Broken) Flow
```
User speaks → STT → TranscriptionFrame → context_aggregator.user() → Accumulates forever → LLM gets huge context → SLOW
```

### New (Fixed) Flow
```
User speaks → STT → TranscriptionFrame → SmartContextManager → Fixed 4096 tokens → LLM always fast
                                              ↓
                                        Extract Facts → Store in Graph
```

## Implementation Components

### 1. Smart Context Manager
**File**: `server/processors/smart_context_manager.py`

Replaces `context_aggregator.user()` with:
- Fixed-size context building (4096 tokens MAX)
- Fact extraction from conversations
- Dynamic prompt generation
- Session metadata tracking
- Budgeted token allocation:
  - Synopsis: 600 tokens
  - Facts: 400 tokens  
  - Recent conversation: 2000 tokens
  - Current turn: 996 tokens
  - Buffer: 100 tokens

### 2. Facts Graph Storage
**File**: `server/memory/facts_graph.py`

Implements friend's proposal:
- SQLite storage with facts table
- Fidelity levels (S4=verbatim → S0=forgotten)
- Natural decay over time
- Reinforcement on re-mention
- Example progression:
  - S4: "my dog name is Potola" (full)
  - S3: "user::pet[name=Potola, species=dog]" (structured)
  - S2: "(user, pet, Potola)" (tuple)
  - S1: "(user —has_pet→ dog)" (edge only)
  - S0: forgotten

### 3. Pipeline Integration
**Modify**: `server/core/pipeline_builder.py`

Line 528, replace:
```python
context_aggregator.user(),
```

With:
```python
SmartContextManager(
    context=context,
    facts_graph=FactsGraph(config.memory.facts_db_path),
    max_tokens=4096
),
```

## Key Benefits

### Performance
- **Before**: Turn 100 = 50,000 tokens → unusable
- **After**: Turn 100 = 4,096 tokens → still fast
- **After**: Turn 1000 = 4,096 tokens → FOREVER fast

### Memory Quality
- Structured facts instead of raw text search
- Natural degradation mimics human memory
- Important facts reinforced and retained
- Trivial details naturally forgotten

### User Experience
- Constant sub-100ms latency
- Can talk forever without degradation
- Remembers important facts
- Forgets irrelevant details

## Implementation Steps

### Phase 1: Core Components
1. Create `SmartContextManager` processor
2. Implement `FactsGraph` with SQLite
3. Add fact extraction heuristics

### Phase 2: Pipeline Integration  
1. Replace `context_aggregator.user()`
2. Update context building logic
3. Test with fixed context size

### Phase 3: Enhancement
1. Add decay mechanisms
2. Implement synopsis generation
3. Add tiered retrieval modes

## Testing Strategy

### Test Case: "My dog name is Potola"
```python
# Turn 1
User: "My dog name is Potola"
→ Extract: Fact(subject="user", predicate="pet", value="Potola", species="dog")
→ Context: 4096 tokens (includes fact)

# Turn 50  
User: "What's my dog's name?"
→ Retrieve: Fact about dog from graph
→ Context: 4096 tokens (same size!)
→ Response: "Your dog's name is Potola"

# Turn 100
Context: STILL 4096 tokens
Performance: STILL <100ms
```

## Success Metrics

1. **Context Size**: Always exactly 4096 tokens (±5%)
2. **Latency**: <100ms response time (p99)
3. **Retrieval Accuracy**: >90% for important facts
4. **Memory Usage**: <200MB regardless of conversation length
5. **LM Studio Compatibility**: Works unchanged with OpenAI API

## Notes for Implementation

- We already have `MemoryAwareOpenAILLMContext` - enhance it rather than replace
- Current BM25 search is broken - facts graph fixes this
- Friend's proposal adds natural memory decay which is brilliant
- Dynamic prompts provide session continuity without growing context
- This solves the fundamental problem: unbounded context growth

## Related Files

- `server/processors/memory_context_aggregator.py` - Existing memory context (enhance this)
- `server/processors/enhanced_stateless_memory.py` - Current broken memory (replace retrieval)
- `server/core/pipeline_builder.py` - Pipeline configuration (modify line 528)
- `server/config.py` - Add facts database configuration

## Open Questions

1. Should we implement the micro-LLM distiller for fact extraction?
2. How aggressive should decay be? (6-hour half-life proposed)
3. Should synopsis be generated continuously or on-demand?
4. Integration with existing speaker recognition system?

## Language-Agnostic Query Classification

### Multi-Signal Classification Architecture

Instead of hardcoding English patterns, the system uses language-agnostic signals:

#### 1. Semantic Vector Classification
- Uses multilingual embeddings (sentence-transformers)
- Defines intents by example sentences in multiple languages
- Computes cosine similarity to intent clusters
- Works immediately with new languages

#### 2. Universal Linguistic Features  
- Uses Universal POS (UPOS) tags that work across languages
- Detects possessives via `PRON + Poss=Yes` tags
- Identifies temporal markers via NER `DATE/TIME` entities
- Questions detected via universal punctuation (?, ？, ؟)

#### 3. LLM-Based Classification (Fallback)
- Small local LLM classifies intent
- Single prompt works for any language
- Returns standardized intent categories
- Graceful fallback when other methods uncertain

#### 4. Hybrid Approach (Production)
```python
class HybridQueryClassifier:
    """Combines multiple signals for robust classification"""
    
    async def classify(self, query: str, context: Dict) -> QueryIntent:
        # Get classifications from multiple sources
        semantic_intent = self.semantic.classify(query)    # 40% weight
        linguistic_intent = self.linguistic.classify(query) # 30% weight  
        context_intent = self.from_context(query, context) # 30% weight
        
        # Weighted voting for final intent
        return highest_confidence_intent
```

### Universal Features Instead of Patterns

| Feature | Detection Method | Works Across Languages |
|---------|-----------------|------------------------|
| Possessive | UPOS tags (PRON + Poss=Yes) | ✓ All languages |
| Personal Reference | UPOS tags (Person=1) | ✓ All languages |
| Temporal Markers | NER entities (DATE, TIME) | ✓ All languages |
| Questions | Punctuation (?, ？, ؟, ¿) | ✓ All languages |
| Named Entities | NER (PERSON, LOC, ORG) | ✓ All languages |
| Sentence Structure | Dependency parsing | ✓ All languages |

### Testing Across Languages

```python
# Same classifier works for all languages without modification
test_cases = [
    ("What's my dog's name?", PERSONAL_FACTS),        # English
    ("¿Cómo se llama mi perro?", PERSONAL_FACTS),     # Spanish  
    ("Comment s'appelle mon chien?", PERSONAL_FACTS),  # French
    ("私の犬の名前は何ですか？", PERSONAL_FACTS),          # Japanese
    ("Wie heißt mein Hund?", PERSONAL_FACTS),         # German
    ("我的狗叫什么名字？", PERSONAL_FACTS),              # Chinese
]
```

### Confidence-Based Routing

```python
if confidence >= 0.8:
    route_to_single_store()      # High confidence
elif confidence >= 0.6:
    route_with_fallback()        # Medium confidence  
else:
    hybrid_search_all_stores()   # Low confidence
```

## Update from Friend

*[Space for friend's update to be added]*

---

**Status**: READY FOR IMPLEMENTATION
**Priority**: CRITICAL - Fixes fundamental performance issue
**Estimated Effort**: 1 week for full implementation