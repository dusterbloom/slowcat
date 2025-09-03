# M3 Fact Extraction Model Comparison Results

## Test Results Summary

### 🥇 Qwen3 4B Models (Recommended)
- **Models**: `qwen3-4b-instruct-2507:2` + `qwen3-4b-instruct-2507`
- **Success Rate**: 7/7 (100.0%) ✅
- **Average Time**: 0.51s per extraction
- **Facts per Scenario**: 1.1
- **Quality**: **Excellent**
- **Speed**: **Fast**

### 🥈 Qwen 2.5 0.5B Models (Original)
- **Models**: `qwen2.5-0.5b-instruct-mlx:2` + `qwen2.5-0.5b-instruct-mlx`
- **Success Rate**: 5/7 (71.4%) ⚠️
- **Average Time**: 0.28s per extraction
- **Facts per Scenario**: 0.7
- **Quality**: **Good**
- **Speed**: **Fast**

## Key Improvements with Qwen3 4B

### ✅ **Better Fact Quality**
- **4B**: Extracted `user has_pet golden retriever` (specific breed)
- **0.5B**: Extracted `user has_pet dog` (generic)

### ✅ **More Accurate Location Parsing**
- **4B**: `user lives_in San Francisco` + `user lives_in downtown`
- **0.5B**: `user lives in` (incomplete extraction)

### ✅ **Complex Preference Handling**
- **4B**: Successfully extracted food/wine preferences
- **0.5B**: Failed to extract food preferences entirely

### ✅ **Better Historical Context**
- **4B**: Extracted birth location and move to California
- **0.5B**: Failed to extract any historical facts

## Detailed Test Scenarios

| Scenario | Qwen3 4B Result | Qwen 0.5B Result | Winner |
|----------|----------------|------------------|---------|
| **Pet Info** | ✅ `user has_pet golden retriever` | ❌ `user has_pet dog`, `user likes name` | 🥇 4B |
| **Job Info** | ✅ `user works_at Google` | ✅ `user works at Google` | 🤝 Tie |
| **Living** | ✅ `user lives_in San Francisco` + `downtown` | ❌ `user lives in` + `user has wife` | 🥇 4B |
| **Food** | ✅ `user likes sushi` + `italian wine` | ❌ No extraction | 🥇 4B |
| **History** | ✅ `user lives_in Chicago` + `California` | ❌ No extraction | 🥇 4B |
| **Meta-Query** | ✅ Correctly filtered | ✅ Correctly filtered | 🤝 Tie |
| **Greeting** | ✅ Correctly filtered | ✅ Correctly filtered | 🤝 Tie |

## Performance Analysis

### Response Time Impact
- **4B Models**: ~82% slower per extraction (0.51s vs 0.28s)
- **But**: 40% higher success rate makes up for slower processing
- **Real-world**: Still fast enough for real-time conversation (<1s)

### Quality vs Speed Tradeoff
- **0.5B**: Faster but misses important facts and makes extraction errors
- **4B**: Slightly slower but much more accurate and complete
- **Recommendation**: Use 4B models for better user experience

## Configuration

### ✅ Recommended Production Setup
```bash
export DSPY_REL_MODEL="qwen3-4b-instruct-2507:2"
export DSPY_FACTS_MODEL="qwen3-4b-instruct-2507"
```

### 🏃‍♂️ Fallback for Resource-Constrained Systems
```bash
export DSPY_REL_MODEL="qwen2.5-0.5b-instruct-mlx:2" 
export DSPY_FACTS_MODEL="qwen2.5-0.5b-instruct-mlx"
```

## Memory Requirements

- **4B Models**: ~8-12GB RAM/VRAM for optimal performance
- **0.5B Models**: ~2-4GB RAM/VRAM (good for testing)

## Conclusion

The Qwen3 4B models provide **significantly better fact extraction quality**:

1. **100% test success rate** vs 71% with smaller models
2. **More specific and accurate** fact extraction
3. **Better handling of complex sentences** and context
4. **Still fast enough** for real-time conversation (0.5s avg)

### Recommendation: **Upgrade to Qwen3 4B models** for production use.

The improved accuracy and completeness far outweigh the modest performance cost, leading to much better conversation continuity and context understanding.