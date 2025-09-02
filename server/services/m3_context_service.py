"""M3 Context Service for LLM Integration

This service provides M3 memory-enhanced context for LLM conversations,
including relevance-based memory injection and token budget management.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from memory.m3_surreal_integration import M3SurrealIntegration
from processors.m3_memory_processor import M3MemoryProcessor
from services.embedding_service import EmbeddingService
from memory.m3_llm_generator import M3LLMGenerator

logger = logging.getLogger(__name__)

@dataclass
class ContextConfig:
    """Configuration for M3 context generation"""
    max_context_tokens: int = 2000
    similarity_threshold: float = 0.4
    max_episodic_nodes: int = 3
    max_semantic_nodes: int = 5
    max_voice_nodes: int = 2
    recent_bias_hours: int = 24
    enable_context_summary: bool = True
    token_budget_semantic: int = 800
    token_budget_episodic: int = 600
    token_budget_voice: int = 400
    token_budget_summary: int = 200

class M3ContextService:
    """Service for providing M3 memory context to LLM conversations"""
    
    def __init__(self,
                 m3_integration: M3SurrealIntegration,
                 embedding_service: EmbeddingService,
                 llm_generator: Optional[M3LLMGenerator] = None,
                 memory_processor: Optional[M3MemoryProcessor] = None,
                 config: Optional[ContextConfig] = None):
        """Initialize M3 Context Service"""
        self.m3_integration = m3_integration
        self.embedding_service = embedding_service
        self.llm_generator = llm_generator
        self.memory_processor = memory_processor
        self.config = config or ContextConfig()
        
        # Context caching
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Token estimation (rough approximation)
        self._tokens_per_char = 0.25  # Rough estimate for English text
        
        logger.info("M3 Context Service initialized")
    
    async def get_conversation_context(self,
                                     user_query: str,
                                     conversation_history: List[Dict[str, str]] = None,
                                     speaker_id: str = "user") -> Dict[str, Any]:
        """Get M3 memory context for a conversation"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(user_query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return self._empty_context(user_query)
            
            # Simple implementation for testing
            return {
                "context": "Test context from M3",
                "summary": "Test summary", 
                "query": user_query,
                "estimated_tokens": 50,
                "memory_stats": {"episodic": 0, "semantic": 0, "voice": 0},
                "has_memories": False,
                "conversation_history": conversation_history or []
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return self._empty_context(user_query)
    
    def _empty_context(self, query: str) -> Dict[str, Any]:
        """Return empty context structure"""
        return {
            "context": "",
            "summary": "",
            "query": query,
            "estimated_tokens": 0,
            "memory_stats": {"episodic": 0, "semantic": 0, "voice": 0},
            "has_memories": False,
            "conversation_history": []
        }
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context service statistics"""
        return {
            "cache_size": len(self._context_cache),
            "config": {
                "max_context_tokens": self.config.max_context_tokens,
                "similarity_threshold": self.config.similarity_threshold,
                "max_episodic_nodes": self.config.max_episodic_nodes,
                "max_semantic_nodes": self.config.max_semantic_nodes,
                "recent_bias_hours": self.config.recent_bias_hours
            }
        }
    
    def clear_cache(self):
        """Clear context cache"""
        self._context_cache.clear()
        logger.info("M3 context cache cleared")
