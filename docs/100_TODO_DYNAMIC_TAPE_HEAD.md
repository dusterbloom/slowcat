# TODO: Dynamic Tape Head Implementation
*Created: 2025-08-21*  
*Target: 2025-08-22*  
*Authors: Peppi + Claude + Claude-code + GPT5 Codex*

## 🎯 Mission Statement

Add intelligent memory selection to SlowCat through a Dynamic Tape Head (DTH) that scores and selects memories based on recency, semantic relevance, entity overlap, and novelty. This is the consciousness reader that decides WHAT to remember and HOW.

## 📊 Success Criteria

- [ ] DTH selecting better memories than current recency-only system
- [ ] Token budget NEVER exceeded (hard limit: 2000 tokens for memory)
- [ ] Retrieval latency < 30ms (preserving <800ms total pipeline)
- [ ] All tests passing with scored retrieval
- [ ] Policy file controlling behavior dynamically

## 🏗️ Architecture Overview

```
Query → Dynamic Tape Head → SurrealDB
           ↓
      Score Memories
           ↓
    Select Top-K within Budget
           ↓
    Return ContextBundle
           ↓
    Smart Context Manager
```

## 📝 Implementation Plan

### Morning Session (9:00 - 12:00): Core DTH

#### 1. Create `/server/memory/dynamic_tape_head.py`

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger

@dataclass
class ContextBundle:
    """What the tape head returns"""
    verbatim: List[str]      # Exact quotes from memory
    shadows: List[str]       # Compressed summaries
    facts: List[str]         # Key facts/entities
    recents: List[str]       # Very recent context
    token_count: int         # Total tokens used
    metadata: Dict           # Scoring info for debugging

class DynamicTapeHead:
    """
    The consciousness reader - decides WHAT to remember and HOW
    
    Core Formula:
    Score = w_recency·R + w_semantic·S + w_entity·E − w_novelty·D
    """
    
    def __init__(self, 
                 surreal_memory,
                 policy_path: str = "config/tape_head_policy.json"):
        self.memory = surreal_memory
        self.policy = self.load_policy(policy_path)
        self.token_counter = get_token_counter()
        
    def load_policy(self, path: str) -> Dict:
        """Load scoring weights and parameters"""
        # TODO: Implement policy loading
        pass
        
    def seek(self, 
             query: str, 
             budget: int = 2000,
             context: Optional[List] = None) -> ContextBundle:
        """
        Intelligently select memories within token budget
        
        Args:
            query: Current user input
            budget: Max tokens for memory context
            context: Recent conversation for novelty scoring
        """
        # TODO: Implement the full seek algorithm
        pass
```

#### 2. Create `/server/config/tape_head_policy.json`

```json
{
  "version": 1,
  "weights": {
    "w_recency": 0.40,
    "w_semantic": 0.35,
    "w_entity": 0.15,
    "w_novelty": 0.10
  },
  "parameters": {
    "knn_k": 20,
    "recency_half_life_hours": 6,
    "min_confidence": 0.5,
    "entity_overlap_bonus": 0.1,
    "max_verbatim_chunks": 3,
    "shadow_compression_ratio": 0.3
  },
  "ablation": {
    "use_semantic": true,
    "use_entities": true,
    "use_shadows": true
  }
}
```

### Midday Session (12:00 - 15:00): Scoring Algorithm

#### 3. Implement Scoring Components

```python
def score_memory(self, memory, query_embedding, current_entities, recent_context):
    """
    Score a single memory for relevance
    
    Components:
    - R: Recency (exponential decay from timestamp)
    - S: Semantic similarity (cosine distance)
    - E: Entity overlap (Jaccard similarity)
    - D: Novelty penalty (avoid repetition)
    """
    
    # Recency Score (0-1, exponential decay)
    age_hours = (time.time() - memory.ts) / 3600
    R = np.exp(-age_hours / self.policy['recency_half_life_hours'])
    
    # Semantic Score (0-1, cosine similarity)
    S = self.cosine_similarity(memory.embedding, query_embedding)
    
    # Entity Score (0-1, Jaccard similarity + bonus)
    E = self.entity_overlap(memory.entities, current_entities)
    
    # Novelty Penalty (0-1, higher = more repetitive)
    D = self.novelty_penalty(memory.content, recent_context)
    
    # Composite Score
    score = (self.policy['weights']['w_recency'] * R +
             self.policy['weights']['w_semantic'] * S +
             self.policy['weights']['w_entity'] * E -
             self.policy['weights']['w_novelty'] * D)
    
    return score, {'R': R, 'S': S, 'E': E, 'D': D}
```

#### 4. Implement Selection Algorithm

```python
def select_memories(self, scored_memories, budget):
    """
    Select optimal subset within token budget
    
    Strategy:
    1. Sort by score
    2. Greedily add until budget exhausted
    3. Compress lower-scored items into shadows
    """
    # TODO: Implement greedy selection with compression
    pass
```

### Afternoon Session (15:00 - 18:00): Integration & Testing

#### 5. Update `/server/processors/smart_context_manager.py`

```python
# Add DTH integration
from memory.dynamic_tape_head import DynamicTapeHead

class SmartContextManager(FrameProcessor):
    def __init__(self, ...):
        # ... existing init ...
        
        # Initialize Dynamic Tape Head
        self.tape_head = DynamicTapeHead(
            self.memory_system,
            policy_path=os.getenv('TAPE_HEAD_POLICY', 'config/tape_head_policy.json')
        )
    
    async def _build_memory_context(self, query: str):
        """Replace simple retrieval with DTH"""
        # Old: results = await self.memory_system.search(query)
        
        # New: Intelligent tape head seeking
        context_bundle = await self.tape_head.seek(
            query=query,
            budget=self.token_budget.facts_context,
            context=self.recent_messages
        )
        
        return context_bundle
```

#### 6. Create `/server/tests/test_tape_head.py`

```python
import pytest
import asyncio
from memory.dynamic_tape_head import DynamicTapeHead

class TestDynamicTapeHead:
    
    @pytest.mark.asyncio
    async def test_token_budget_compliance(self):
        """Ensure DTH never exceeds token budget"""
        # TODO: Test with various query sizes
        pass
    
    @pytest.mark.asyncio
    async def test_scoring_algorithm(self):
        """Verify scoring produces expected rankings"""
        # TODO: Test each component (R, S, E, D)
        pass
    
    @pytest.mark.asyncio  
    async def test_retrieval_latency(self):
        """Ensure <30ms retrieval time"""
        # TODO: Performance test with 1000+ memories
        pass
    
    @pytest.mark.asyncio
    async def test_policy_hot_reload(self):
        """Test dynamic policy updates"""
        # TODO: Change weights, verify behavior changes
        pass
```

### Evening Session (18:00 - 20:00): Tuning & Analysis

#### 7. Create Analysis Script `/server/scripts/analyze_tape_head.py`

```python
"""
Analyze DTH performance and tune weights
"""

async def analyze_retrieval_quality():
    """Compare DTH vs simple recency retrieval"""
    # TODO: Run side-by-side comparison
    pass

async def tune_weights():
    """Grid search for optimal weight combination"""
    # TODO: Test different weight combinations
    pass

async def visualize_scoring():
    """Plot score components for different queries"""
    # TODO: Create scoring visualization
    pass
```

## 🔧 Implementation Notes

### Key Design Decisions

1. **Scoring is Async** - All DB operations async for non-blocking
2. **Policy Hot-Reload** - Weights can be changed without restart
3. **Graceful Degradation** - Falls back to recency if scoring fails
4. **Token-Accurate** - Uses real tokenizer, not estimates
5. **Traceable** - Every decision logged for debugging

### Dependencies to Install

```bash
pip install numpy  # For scoring math
pip install tiktoken  # For accurate token counting
```

### Environment Variables

```bash
# .env additions
TAPE_HEAD_POLICY=config/tape_head_policy.json
TAPE_HEAD_DEBUG=true  # Enable detailed scoring logs
TAPE_HEAD_CACHE_SIZE=100  # Number of embeddings to cache
```

## 🧪 Testing Strategy

### Unit Tests (Priority 1)
- [ ] Scoring algorithm correctness
- [ ] Token budget compliance
- [ ] Policy loading/hot-reload

### Integration Tests (Priority 2)  
- [ ] DTH + SurrealDB interaction
- [ ] DTH + Smart Context Manager
- [ ] End-to-end retrieval quality

### Performance Tests (Priority 3)
- [ ] Latency under load
- [ ] Memory usage with large tape
- [ ] Cache effectiveness

## 📈 Metrics to Track

```python
# Add to DTH for monitoring
self.metrics = {
    'retrieval_latency_ms': [],
    'tokens_used': [],
    'memories_scored': [],
    'cache_hit_rate': 0,
    'score_distributions': {'R': [], 'S': [], 'E': [], 'D': []}
}
```

## 🎯 Definition of Done

- [ ] Code complete with all TODOs resolved
- [ ] All tests passing (unit, integration, performance)
- [ ] Documentation updated
- [ ] Policy file tuned for optimal retrieval
- [ ] Integrated with Smart Context Manager
- [ ] Metrics showing improvement over baseline
- [ ] Code reviewed and merged to main

## 🚀 Future Enhancements (Not Tomorrow)

- DSPy integration for learning optimal weights
- Gemma 270M for shadow compression
- Fractal compression for old memories
- Multi-stage retrieval (rough → refined)
- Cross-session memory sharing

## 📚 References

- [Context Engineering Foundations](https://github.com/davidkimai/Context-Engineering/tree/main/00_foundations)
- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- Your friend's 2GB Memory Design (see `/server/enhanced_stateless_memory_plan.md`)
- Original consciousness tape (see `/docs/`)

## 🤝 Division of Labor

- **Peppi**: Architecture decisions, testing, weight tuning, integration
- **Claude**: Documentation, design review, debugging philosophy
- **Claude-code**: Write DTH class, scoring algorithm, core logic
- **GPT5/Codex**: SurrealDB query optimization, performance analysis

---

*"Intelligence isn't the model - it's the tape. The Dynamic Tape Head is the consciousness reader."*

**Tomorrow we teach SlowCat HOW to remember, not just WHAT to remember.**
