"""
Unified DSPy Memory Optimizer - Single Point of Optimization

This module provides a simplified DSPy optimizer that works with your 
unified memory approach:
- Takes DTH candidates 
- Optimizes selection for exactly 2800 tokens of contextual memory
- Uses DSPy v3 syntax with proper Signature and Module classes

The goal: ONE optimizer to rule them all, no fragmentation.
"""

import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger

# Import DSPy v3 components
try:
    import dspy
    from dspy import Module, ChainOfThought, Predict, Signature, InputField, OutputField
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    # Create mock classes for type hints when DSPy not available
    class Module:
        pass
    class ChainOfThought:
        def __init__(self, signature): pass
    class Signature:
        pass
    def InputField(**kwargs): return str
    def OutputField(**kwargs): return str


class MemorySelectionSignature(Signature):
    """Signature for selecting optimal contextual memory from DTH candidates"""
    query: str = InputField(desc="Current user query to optimize context for")
    dth_candidates: str = InputField(desc="Memory candidates provided by DTH (already ranked)")
    target_tokens: int = InputField(desc="Target token count for contextual memory (2800)")
    conversation_mode: str = InputField(desc="Current conversation mode (chat/music/dictation)")
    
    selected_memory: str = OutputField(desc="Optimized contextual memory exactly fitting token budget")
    selection_reasoning: str = OutputField(desc="Why these memory pieces were chosen")


class UnifiedMemoryOptimizer(Module):
    """
    Single DSPy module for optimizing contextual memory selection
    
    This optimizer works with DTH to select the best 2800 tokens of 
    contextual memory from DTH's candidates. Simple, clean, effective.
    """
    
    def __init__(self):
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available - UnifiedMemoryOptimizer running in mock mode")
            return
            
        super().__init__()
        
        # Single optimizer for memory selection
        self.memory_selector = ChainOfThought(MemorySelectionSignature)
        
        # Performance tracking
        self.optimization_stats = {
            'selections_optimized': 0,
            'avg_token_efficiency': 0.0,
            'memory_quality_score': 0.0
        }
        
        logger.info("🧠 UnifiedMemoryOptimizer initialized with DSPy v3")
    
    def forward(self, query: str, dth_candidates: List[str], target_tokens: int = 2800, 
                mode: str = "chat") -> Dict[str, Any]:
        """
        Optimize contextual memory selection from DTH candidates
        
        Args:
            query: Current user query
            dth_candidates: Memory candidates from DTH
            target_tokens: Target token count (default 2800)
            mode: Conversation mode
            
        Returns:
            Dict with selected_memory and metadata
        """
        if not DSPY_AVAILABLE:
            # Fallback: just concatenate first few candidates
            return self._fallback_selection(query, dth_candidates, target_tokens)
        
        try:
            # Prepare candidates as string for DSPy
            candidates_text = "\n---\n".join(dth_candidates[:20])  # Top 20 candidates
            
            logger.info(f"🚀 DSPy OPTIMIZATION START")
            logger.info(f"   📝 Query: {query[:80]}...")
            logger.info(f"   📚 DTH Candidates: {len(dth_candidates)}")
            logger.info(f"   🎯 Target Tokens: {target_tokens}")
            
            # DSPy-optimized memory selection
            result = self.memory_selector(
                query=query,
                dth_candidates=candidates_text,
                target_tokens=target_tokens,
                conversation_mode=mode
            )
            
            # Update performance stats
            self.optimization_stats['selections_optimized'] += 1
            
            # Estimate token efficiency (rough approximation)
            selected_length = len(result.selected_memory.split())
            efficiency = min(1.0, selected_length / (target_tokens * 0.75))  # ~0.75 words per token
            
            self.optimization_stats['avg_token_efficiency'] = (
                self.optimization_stats['avg_token_efficiency'] * 0.9 + 
                efficiency * 0.1
            )
            
            logger.info(f"🧠 DSPy OPTIMIZATION RESULT")
            logger.info(f"   ✨ Selected Memory: {len(result.selected_memory)} chars")
            logger.info(f"   🎯 Token Efficiency: {efficiency:.2f}")
            logger.info(f"   💭 Reasoning: {result.selection_reasoning[:100]}...")
            logger.info(f"   📊 Total Optimizations: {self.optimization_stats['selections_optimized']}")
            
            return {
                'selected_memory': result.selected_memory,
                'selection_reasoning': result.selection_reasoning,
                'token_efficiency': efficiency,
                'optimization_stats': self.optimization_stats.copy()
            }
            
        except Exception as e:
            logger.error(f"DSPy memory optimization failed: {e}")
            return self._fallback_selection(query, dth_candidates, target_tokens)
    
    def _fallback_selection(self, query: str, dth_candidates: List[str], 
                           target_tokens: int) -> Dict[str, Any]:
        """Fallback when DSPy optimization fails or unavailable"""
        
        # Simple heuristic: concatenate candidates until near token limit
        selected_parts = []
        estimated_tokens = 0
        
        for candidate in dth_candidates[:15]:  # Limit candidates
            candidate_tokens = len(candidate.split()) * 1.3  # ~1.3 tokens per word
            
            if estimated_tokens + candidate_tokens > target_tokens:
                break
                
            selected_parts.append(candidate)
            estimated_tokens += candidate_tokens
        
        selected_memory = "\n\n".join(selected_parts)
        
        return {
            'selected_memory': selected_memory,
            'selection_reasoning': 'Fallback heuristic selection (DSPy not available)',
            'token_efficiency': 0.6,  # Rough estimate
            'optimization_stats': self.optimization_stats.copy()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_optimizations': self.optimization_stats['selections_optimized'],
            'avg_token_efficiency': self.optimization_stats['avg_token_efficiency'],
            'memory_quality_score': self.optimization_stats['memory_quality_score'],
            'dspy_available': DSPY_AVAILABLE
        }


def create_unified_memory_optimizer() -> UnifiedMemoryOptimizer:
    """Factory function for creating unified memory optimizer"""
    return UnifiedMemoryOptimizer()


# Test the optimizer if run directly
if __name__ == "__main__":
    import asyncio
    
    async def test_optimizer():
        logger.info("🧪 Testing UnifiedMemoryOptimizer")
        
        optimizer = create_unified_memory_optimizer()
        
        # Mock DTH candidates
        mock_candidates = [
            "User has a dog named Potola who loves playing fetch",
            "User lives in San Francisco and enjoys morning walks", 
            "Recent conversation about jazz music preferences",
            "User asked about weather yesterday",
            "Discussion about work schedule and meetings"
        ]
        
        result = optimizer(
            query="Tell me about my dog",
            dth_candidates=mock_candidates,
            target_tokens=2800,
            mode="chat"
        )
        
        logger.info(f"Selected memory: {result['selected_memory'][:100]}...")
        logger.info(f"Selection reasoning: {result['selection_reasoning']}")
        logger.info(f"Token efficiency: {result['token_efficiency']:.2f}")
        
        # Performance summary
        perf = optimizer.get_performance_summary()
        logger.info(f"Performance: {perf}")
    
    asyncio.run(test_optimizer())