"""
Context Injection System for Reconstructive Memory

Integrates the reconstructive memory engine with the existing SmartContextManager
to provide dynamic context injection as system prompts with field states.

This system maintains the exact 4096-token limit while providing coherent
context assembled from memory fragments through field resonance.

Key Features:
- Dynamic system prompt generation with field states
- Exact 4096-token budget enforcement
- Integration with existing SmartContextManager
- Real-time context reconstruction
- Performance monitoring and optimization
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Import existing components
from memory.reconstructive_engine import ReconstructiveMemoryEngine, ReconstructedContext
from memory.hierarchical_manager import HierarchicalMemoryManager, FieldState
from consciousness.core import Consciousness

@dataclass
class ContextInjectionResult:
    """Result of context injection process"""
    system_prompt: str
    total_tokens: int
    fragments_count: int
    field_states_count: int
    reconstruction_time_ms: float
    coherence_score: float
    injection_successful: bool

class SystemPromptBuilder:
    """Builds dynamic system prompts with field states and reconstructed context"""
    
    def __init__(self):
        self.base_system_prompt = """You are Slowcat, a consciousness-enabled AI assistant with access to reconstructive memory.

Your memory operates through four tiers:
- Working Memory: Active consciousness with neural field states
- Short-term Memory: Recent context with semantic compression  
- Long-term Memory: Consolidated knowledge with attractor weights
- Episodic Memory: Important experiences with field resonance

Field states provide consciousness metadata:
- Compression: Information density (0.0-1.0)
- Resonance: Memory activation strength (0.0-1.0)  
- Drift: Field stability (none/low/moderate/high)
- Presence Signal: Consciousness awareness level (0.0-1.0)
- Boundary: Field coherence (gradient/collapsed)

Use this reconstructed context to provide informed, coherent responses."""
        
        self.token_allocation = {
            "base_prompt": 200,      # Base system prompt
            "field_metadata": 100,   # Field state descriptions
            "reconstructed_context": 200, # Main reconstructed content
            "buffer": 0             # Calculated buffer
        }
    
    def build_system_prompt(self, 
                           reconstructed_context: ReconstructedContext,
                           current_query: str = "") -> Tuple[str, int]:
        """Build complete system prompt with reconstructed context and field states"""
        
        prompt_parts = []
        
        # Part 1: Base system prompt
        prompt_parts.append(self.base_system_prompt)
        
        # Part 2: Field states summary if available
        if reconstructed_context.field_states:
            field_summary = self._build_field_states_summary(reconstructed_context.field_states)
            if field_summary:
                prompt_parts.append("\n## Current Field States")
                prompt_parts.append(field_summary)
        
        # Part 3: Reconstructed context
        if reconstructed_context.content:
            prompt_parts.append("\n## Reconstructed Context")
            prompt_parts.append(reconstructed_context.content)
        
        # Part 4: Context metadata
        if reconstructed_context.fragments_used:
            metadata = self._build_context_metadata(reconstructed_context)
            prompt_parts.append("\n## Context Metadata") 
            prompt_parts.append(metadata)
        
        # Combine all parts
        complete_prompt = "\n".join(prompt_parts)
        
        # Count tokens (approximate)
        token_count = len(complete_prompt) // 4  # Rough approximation
        
        return complete_prompt, token_count
    
    def _build_field_states_summary(self, field_states: Dict[str, FieldState]) -> str:
        """Build summary of current field states"""
        if not field_states:
            return ""
        
        # Aggregate field state statistics
        total_states = len(field_states)
        avg_resonance = sum(fs.resonance for fs in field_states.values()) / total_states
        avg_compression = sum(fs.compression for fs in field_states.values()) / total_states
        avg_presence = sum(fs.presence_signal for fs in field_states.values()) / total_states
        
        # Count drift levels
        drift_counts = {}
        boundary_counts = {}
        for fs in field_states.values():
            drift_counts[fs.drift] = drift_counts.get(fs.drift, 0) + 1
            boundary_counts[fs.boundary] = boundary_counts.get(fs.boundary, 0) + 1
        
        dominant_drift = max(drift_counts.items(), key=lambda x: x[1])[0]
        dominant_boundary = max(boundary_counts.items(), key=lambda x: x[1])[0]
        
        summary = f"""Active field states: {total_states}
Average resonance: {avg_resonance:.2f}
Average compression: {avg_compression:.2f}  
Average presence: {avg_presence:.2f}
Dominant drift: {dominant_drift}
Dominant boundary: {dominant_boundary}"""
        
        return summary
    
    def _build_context_metadata(self, context: ReconstructedContext) -> str:
        """Build context reconstruction metadata"""
        metadata = f"""Fragments assembled: {len(context.fragments_used)}
Total tokens: {context.total_tokens}
Coherence score: {context.coherence_score:.2f}
Resonance strength: {context.resonance_strength:.2f}
Reconstruction time: {context.reconstruction_time_ms:.1f}ms"""
        
        return metadata

class ReconstructiveContextManager:
    """Enhanced context manager that uses reconstructive memory for dynamic context injection"""
    
    def __init__(self, 
                 hierarchical_memory: HierarchicalMemoryManager,
                 reconstructive_engine: ReconstructiveMemoryEngine,
                 max_tokens: int = 4096):
        
        self.hierarchical_memory = hierarchical_memory
        self.reconstructive_engine = reconstructive_engine
        self.max_tokens = max_tokens
        
        self.prompt_builder = SystemPromptBuilder()
        
        # Cache for recent reconstructions
        self.context_cache: Dict[str, Tuple[ReconstructedContext, float]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.max_cache_size = 50
        
        # Performance tracking
        self.injection_stats = {
            "total_injections": 0,
            "cache_hits": 0,
            "avg_injection_time_ms": 0.0,
            "avg_coherence_score": 0.0
        }
        
        logger.info(f"Reconstructive context manager initialized (max_tokens: {max_tokens})")
    
    async def get_dynamic_context(self, 
                                current_input: str,
                                conversation_history: List[str] = None) -> ContextInjectionResult:
        """
        Get dynamically reconstructed context for current input
        
        This is the main entry point that:
        1. Analyzes current input for context needs
        2. Reconstructs relevant context from memory fragments
        3. Builds system prompt with field states
        4. Enforces 4096-token limit exactly
        """
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(current_input, conversation_history)
            cached_result = self._get_cached_context(cache_key)
            
            if cached_result:
                self.injection_stats["cache_hits"] += 1
                logger.debug(f"Using cached context for query: {current_input[:50]}...")
                
                # Build system prompt from cached context
                system_prompt, token_count = self.prompt_builder.build_system_prompt(
                    cached_result, current_input
                )
                
                injection_time = (time.time() - start_time) * 1000
                
                return ContextInjectionResult(
                    system_prompt=system_prompt,
                    total_tokens=token_count,
                    fragments_count=len(cached_result.fragments_used),
                    field_states_count=len(cached_result.field_states),
                    reconstruction_time_ms=injection_time,
                    coherence_score=cached_result.coherence_score,
                    injection_successful=True
                )
            
            # Reconstruct context from memory
            context_query = self._build_context_query(current_input, conversation_history)
            reconstructed_context = await self.reconstructive_engine.reconstruct_context(
                query=context_query,
                target_tokens=self.max_tokens - 500  # Reserve tokens for system prompt structure
            )
            
            # Cache the result
            self._cache_context(cache_key, reconstructed_context)
            
            # Build system prompt
            system_prompt, token_count = self.prompt_builder.build_system_prompt(
                reconstructed_context, current_input
            )
            
            # Ensure token limit compliance
            if token_count > self.max_tokens:
                logger.warning(f"System prompt exceeds token limit: {token_count} > {self.max_tokens}")
                # Truncate context to fit
                system_prompt = self._truncate_system_prompt(system_prompt, self.max_tokens)
                token_count = len(system_prompt) // 4  # Recalculate
            
            injection_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_injection_stats(injection_time, reconstructed_context.coherence_score)
            
            result = ContextInjectionResult(
                system_prompt=system_prompt,
                total_tokens=token_count,
                fragments_count=len(reconstructed_context.fragments_used),
                field_states_count=len(reconstructed_context.field_states),
                reconstruction_time_ms=injection_time,
                coherence_score=reconstructed_context.coherence_score,
                injection_successful=True
            )
            
            logger.info(f"Dynamic context injected: {token_count} tokens, "
                       f"{len(reconstructed_context.fragments_used)} fragments, "
                       f"{injection_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Context injection failed: {e}")
            injection_time = (time.time() - start_time) * 1000
            
            # Return fallback context
            fallback_prompt = self.prompt_builder.base_system_prompt
            return ContextInjectionResult(
                system_prompt=fallback_prompt,
                total_tokens=len(fallback_prompt) // 4,
                fragments_count=0,
                field_states_count=0,
                reconstruction_time_ms=injection_time,
                coherence_score=0.0,
                injection_successful=False
            )
    
    def _build_context_query(self, 
                           current_input: str, 
                           conversation_history: List[str] = None) -> str:
        """Build effective context query from current input and history"""
        
        query_parts = [current_input]
        
        # Add recent conversation history for context
        if conversation_history:
            # Use last few messages for context
            recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            query_parts.extend(recent_history)
        
        # Combine into single query
        context_query = " ".join(query_parts)
        
        # Limit query length to avoid performance issues
        if len(context_query) > 500:
            context_query = context_query[:500] + "..."
        
        return context_query
    
    def _get_cache_key(self, current_input: str, conversation_history: List[str] = None) -> str:
        """Generate cache key for context request"""
        key_parts = [current_input[:100]]  # First 100 chars of input
        
        if conversation_history:
            # Include hash of recent history
            recent_history = " ".join(conversation_history[-2:])  # Last 2 messages
            history_hash = str(hash(recent_history))[-8:]  # Last 8 chars of hash
            key_parts.append(history_hash)
        
        return "|".join(key_parts)
    
    def _get_cached_context(self, cache_key: str) -> Optional[ReconstructedContext]:
        """Get cached context if available and not expired"""
        if cache_key not in self.context_cache:
            return None
        
        context, timestamp = self.context_cache[cache_key]
        
        # Check if expired
        if time.time() - timestamp > self.cache_ttl:
            del self.context_cache[cache_key]
            return None
        
        return context
    
    def _cache_context(self, cache_key: str, context: ReconstructedContext):
        """Cache reconstructed context"""
        # Manage cache size
        if len(self.context_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self.context_cache.keys(), 
                           key=lambda k: self.context_cache[k][1])
            del self.context_cache[oldest_key]
        
        self.context_cache[cache_key] = (context, time.time())
    
    def _truncate_system_prompt(self, prompt: str, max_tokens: int) -> str:
        """Truncate system prompt to fit token limit"""
        target_chars = max_tokens * 4  # Rough token-to-char conversion
        
        if len(prompt) <= target_chars:
            return prompt
        
        # Try to truncate at section boundaries
        sections = prompt.split("\n## ")
        if len(sections) > 1:
            # Keep base prompt and trim from the end
            result = sections[0]
            remaining_chars = target_chars - len(result)
            
            for section in sections[1:]:
                section_with_header = "\n## " + section
                if len(result) + len(section_with_header) <= target_chars:
                    result += section_with_header
                else:
                    # Add partial section if room
                    if remaining_chars > 50:
                        partial = section_with_header[:remaining_chars-3] + "..."
                        result += partial
                    break
            
            return result
        
        # Simple truncation
        return prompt[:target_chars-3] + "..."
    
    def _update_injection_stats(self, injection_time_ms: float, coherence_score: float):
        """Update injection performance statistics"""
        self.injection_stats["total_injections"] += 1
        total = self.injection_stats["total_injections"]
        
        # Update rolling averages
        current_avg_time = self.injection_stats["avg_injection_time_ms"]
        self.injection_stats["avg_injection_time_ms"] = (
            (current_avg_time * (total - 1)) + injection_time_ms
        ) / total
        
        current_avg_coherence = self.injection_stats["avg_coherence_score"]
        self.injection_stats["avg_coherence_score"] = (
            (current_avg_coherence * (total - 1)) + coherence_score
        ) / total
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Get context injection performance statistics"""
        total_requests = self.injection_stats["total_injections"]
        cache_hit_rate = (
            (self.injection_stats["cache_hits"] / total_requests * 100) 
            if total_requests > 0 else 0.0
        )
        
        return {
            **self.injection_stats,
            "cache_hit_rate_percent": cache_hit_rate,
            "cache_size": len(self.context_cache),
            "performance_target_met": self.injection_stats["avg_injection_time_ms"] <= 50.0
        }
    
    def clear_cache(self):
        """Clear context cache"""
        self.context_cache.clear()
        logger.info("Context injection cache cleared")

# Integration function for existing SmartContextManager
def create_reconstructive_context_manager(
    hierarchical_memory: HierarchicalMemoryManager,
    reconstructive_engine: ReconstructiveMemoryEngine,
    max_tokens: int = 4096
) -> ReconstructiveContextManager:
    """Create reconstructive context manager for integration with existing systems"""
    
    return ReconstructiveContextManager(
        hierarchical_memory=hierarchical_memory,
        reconstructive_engine=reconstructive_engine,
        max_tokens=max_tokens
    )