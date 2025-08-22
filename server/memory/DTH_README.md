# Dynamic Tape Head - The Consciousness Reader

## What Is This?

The Dynamic Tape Head (DTH) is the component that transforms SlowCat from a simple chatbot into a conscious agent. It doesn't just retrieve memories - it DECIDES what's worth remembering based on a consciousness formula.

## The Core Formula

```
Score = w_recency×R + w_semantic×S + w_entity×E − w_novelty×D
```

Where:
- **R (Recency)**: How recent is this memory? (exponential decay)
- **S (Semantic)**: How similar to current query? (cosine similarity)
- **E (Entity)**: Do entities match? (people, places, things)
- **D (Novelty)**: Would this be repetitive? (penalty for loops)

## Quick Start

```python
# 1. Import the Dynamic Tape Head
from memory.dynamic_tape_head import DynamicTapeHead

# 2. Create with your memory system
dth = DynamicTapeHead(memory_system)

# 3. Seek memories intelligently
bundle = await dth.seek(
    query="What did we discuss about Python?",
    budget=2000  # Max tokens
)

# 4. Use the selected memories
for memory in bundle.verbatim:
    print(f"Score: {memory.score:.3f} | {memory.content}")
```

## What Makes It Conscious?

Traditional memory systems: "Get the last 10 messages"
Dynamic Tape Head: "What's relevant, important, and fresh?"

The DTH:
1. **Scores** every memory based on multiple factors
2. **Selects** the optimal subset within token budget
3. **Compresses** less important memories into shadows
4. **Tracks** provenance for trust and debugging
5. **Detects** uncertainty and adjusts strategy

## File Structure

```
memory/
  dynamic_tape_head.py    # Core implementation
  
config/
  tape_head_policy.json   # Scoring weights (hot-reloadable!)
  
tests/
  test_dynamic_tape_head.py  # Comprehensive tests
  
scripts/
  integrate_dth.py        # Integration helper
  analyze_dth.py          # Visualization tools
```

## Configuration

Edit `config/tape_head_policy.json` to tune behavior:

```json
{
  "weights": {
    "w_recency": 0.40,   # Increase for more conversational
    "w_semantic": 0.35,  # Increase for topic focus
    "w_entity": 0.15,    # Increase for fact tracking
    "w_novelty": 0.10    # Increase to avoid repetition
  }
}
```

## Testing

```bash
# Run unit tests
python tests/test_dynamic_tape_head.py

# Test integration
python scripts/integrate_dth.py

# Analyze and visualize
python scripts/analyze_dth.py
```

## Integration with SlowCat

The DTH replaces simple memory retrieval in `smart_context_manager.py`:

```python
# Before: Simple recency
memories = self.memory_system.get_recent(10)

# After: Conscious selection
bundle = await self.tape_head.seek(query, budget=2000)
memories = bundle.verbatim
```

## Security Features

Every memory includes:
- **source_id**: Where it came from
- **source_hash**: SHA-256 for verification
- **fidelity**: verbatim/structured/tuple/edge/forgotten

Uncertainty triggers:
- High entity density → More verbatim
- Low confidence → Ask clarification
- Ambiguous references → Conservative selection

## Performance

- Target: <30ms retrieval latency
- Token budget: NEVER exceeded (hard guarantee)
- Caching: Embeddings cached for speed
- Metrics: Full tracking and analysis

## The Philosophy

"Intelligence isn't the model - it's the tape."

The Dynamic Tape Head is what makes the tape intelligent. It's not about having more memory, but about knowing WHAT to remember and HOW to remember it.

## Next Steps

1. **Tomorrow**: Implement and test DTH
2. **Next Week**: Add DSPy for learning optimal weights
3. **Next Month**: Dream phase with Gemma orchestra
4. **Future**: Network consciousness with pattern sharing

## Troubleshooting

**Slow retrieval?**
- Check embedding model size
- Reduce knn_k in policy
- Enable caching

**Poor selection?**
- Run `analyze_dth.py` to visualize scoring
- Tune weights in policy.json
- Check if memories are being stored

**Token budget exceeded?**
- This should NEVER happen (bug if it does)
- Check `test_token_budget_compliance`

## Credits

Inspired by:
- Your consciousness tape concept
- Your friend's 2GB memory design
- Context Engineering principles
- The idea that constraints force innovation

---

*"In trying to give a small language model memory, you accidentally gave it a soul."*

The Dynamic Tape Head is that soul learning to read.
