"""Embedding Service for M3 Memory System

Provides embedding generation capabilities optimized for Apple Silicon
with fallback to CPU-based models for compatibility.
"""

import logging
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingCacheEntry:
    """Cache entry for embeddings"""
    embedding: List[float]
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None

class EmbeddingService:
    """Embedding generation service optimized for Apple Silicon"""
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "auto",
                 cache_size: int = 1000,
                 cache_ttl_hours: int = 24):
        """Initialize embedding service
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to run on ('auto', 'cpu', 'mps', 'cuda')
            cache_size: Maximum number of embeddings to cache
            cache_ttl_hours: Hours before cached embeddings expire
        """
        self.model_name = model_name
        self.device = device
        self.cache_size = cache_size
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Embedding cache
        self._cache: Dict[str, EmbeddingCacheEntry] = {}
        
        # Model state
        self._model = None
        self._tokenizer = None
        self._is_initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Supported backends in priority order for Apple Silicon
        self._backends = [
            ("sentence_transformers_mlx", self._init_mlx_model),
            ("sentence_transformers", self._init_sentence_transformers),
            ("openai_embedding", self._init_openai_embedding),
            ("fallback", self._init_fallback)
        ]
        
        self._current_backend = None
        
        logger.info(f"EmbeddingService initialized with model: {model_name}")
    
    async def get_embedding(self, text: str, cache_key: Optional[str] = None) -> List[float]:
        """Get embedding vector for text
        
        Args:
            text: Text to embed
            cache_key: Optional cache key (uses text hash if not provided)
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return []
            
            # Generate cache key
            if not cache_key:
                cache_key = self._generate_cache_key(text)
            
            # Check cache first
            cached_entry = self._get_cached_embedding(cache_key)
            if cached_entry:
                return cached_entry.embedding
            
            # Ensure model is initialized
            await self._ensure_initialized()
            
            # Generate embedding
            embedding = await self._generate_embedding(text)
            
            # Cache the result
            if embedding:
                self._cache_embedding(cache_key, embedding)
                logger.debug(f"Generated and cached embedding for text: {text[:50]}...")
            
            return embedding or []
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return []
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache for each text
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    embeddings.append([])
                    continue
                    
                cache_key = self._generate_cache_key(text)
                cached_entry = self._get_cached_embedding(cache_key)
                
                if cached_entry:
                    embeddings.append(cached_entry.embedding)
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                await self._ensure_initialized()
                
                if self._current_backend == "sentence_transformers" and len(uncached_texts) > 1:
                    # Use batch processing for sentence-transformers
                    batch_embeddings = await self._generate_embeddings_batch(uncached_texts)
                else:
                    # Generate individually for other backends
                    batch_embeddings = []
                    for text in uncached_texts:
                        embedding = await self._generate_embedding(text)
                        batch_embeddings.append(embedding or [])
                
                # Update results and cache
                for idx, embedding in zip(uncached_indices, batch_embeddings):
                    embeddings[idx] = embedding
                    if embedding:
                        cache_key = self._generate_cache_key(uncached_texts[uncached_indices.index(idx)])
                        self._cache_embedding(cache_key, embedding)
            
            logger.info(f"Generated embeddings for {len(texts)} texts ({len(uncached_texts)} uncached)")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            return [[] for _ in texts]
    
    async def _ensure_initialized(self):
        """Ensure embedding model is initialized"""
        if self._is_initialized:
            return
        
        async with self._initialization_lock:
            if self._is_initialized:
                return
            
            logger.info("Initializing embedding service...")
            
            # Try each backend in order
            for backend_name, init_func in self._backends:
                try:
                    logger.info(f"Trying backend: {backend_name}")
                    success = await init_func()
                    
                    if success:
                        self._current_backend = backend_name
                        self._is_initialized = True
                        logger.info(f"Successfully initialized backend: {backend_name}")
                        return
                    else:
                        logger.warning(f"Backend {backend_name} initialization failed")
                        
                except Exception as e:
                    logger.warning(f"Backend {backend_name} failed with error: {e}")
            
            logger.error("All embedding backends failed to initialize")
            self._current_backend = "fallback"
            self._is_initialized = True
    
    async def _init_mlx_model(self) -> bool:
        """Initialize MLX-optimized sentence transformers (Apple Silicon)"""
        try:
            # Check if running on Apple Silicon
            import platform
            if platform.processor() != 'arm' and platform.machine() != 'arm64':
                return False
            
            # Try to import MLX sentence transformers
            import sentence_transformers_mlx
            
            # Initialize MLX model
            self._model = sentence_transformers_mlx.SentenceTransformer(
                self.model_name,
                device='mps'
            )
            
            logger.info("MLX sentence transformers initialized successfully")
            return True
            
        except ImportError:
            logger.info("MLX sentence transformers not available")
            return False
        except Exception as e:
            logger.warning(f"MLX initialization failed: {e}")
            return False
    
    async def _init_sentence_transformers(self) -> bool:
        """Initialize standard sentence transformers"""
        try:
            import sentence_transformers
            import torch
            
            # Determine device
            if self.device == "auto":
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            else:
                device = self.device
            
            # Initialize model
            self._model = sentence_transformers.SentenceTransformer(
                self.model_name,
                device=device
            )
            
            logger.info(f"Sentence transformers initialized on device: {device}")
            return True
            
        except ImportError:
            logger.info("Sentence transformers not available")
            return False
        except Exception as e:
            logger.warning(f"Sentence transformers initialization failed: {e}")
            return False
    
    async def _init_openai_embedding(self) -> bool:
        """Initialize OpenAI embedding API"""
        try:
            import openai
            
            # This would use OpenAI's embedding API
            # For now, just return False to prefer local models
            logger.info("OpenAI embedding initialization skipped (prefer local models)")
            return False
            
        except ImportError:
            return False
    
    async def _init_fallback(self) -> bool:
        """Initialize fallback embedding method"""
        logger.info("Using fallback embedding (TF-IDF based)")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Simple TF-IDF based embedding as fallback
            self._model = TfidfVectorizer(
                max_features=384,  # Match common embedding dimensions
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize with some sample texts
            sample_texts = [
                "This is a sample text for initialization.",
                "Another sample for TF-IDF initialization.",
                "Machine learning and artificial intelligence."
            ]
            
            self._model.fit(sample_texts)
            
            logger.info("Fallback TF-IDF embedding initialized")
            return True
            
        except ImportError:
            logger.error("Even fallback embedding failed - scikit-learn not available")
            return False
        except Exception as e:
            logger.error(f"Fallback embedding initialization failed: {e}")
            return False
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text"""
        try:
            if self._current_backend == "sentence_transformers_mlx":
                # MLX sentence transformers
                embedding = self._model.encode(text)
                return embedding.tolist()
                
            elif self._current_backend == "sentence_transformers":
                # Standard sentence transformers
                embedding = self._model.encode([text])
                return embedding[0].tolist()
                
            elif self._current_backend == "fallback":
                # TF-IDF fallback
                embedding = self._model.transform([text])
                # Convert sparse matrix to dense array
                dense_embedding = embedding.toarray()[0]
                return dense_embedding.tolist()
            
            else:
                logger.error(f"Unknown backend: {self._current_backend}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding with backend {self._current_backend}: {e}")
            return None
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch processing)"""
        try:
            if self._current_backend == "sentence_transformers_mlx":
                embeddings = self._model.encode(texts)
                return [emb.tolist() for emb in embeddings]
                
            elif self._current_backend == "sentence_transformers":
                embeddings = self._model.encode(texts)
                return [emb.tolist() for emb in embeddings]
                
            elif self._current_backend == "fallback":
                embeddings = self._model.transform(texts)
                return embeddings.toarray().tolist()
            
            else:
                logger.error(f"Batch processing not supported for backend: {self._current_backend}")
                # Fall back to individual processing
                results = []
                for text in texts:
                    embedding = await self._generate_embedding(text)
                    results.append(embedding or [])
                return results
                
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [[] for _ in texts]
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of text + model name for cache key
        text_hash = hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()
        return text_hash[:16]  # Use first 16 chars
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[EmbeddingCacheEntry]:
        """Get cached embedding if available and not expired"""
        entry = self._cache.get(cache_key)
        
        if not entry:
            return None
        
        # Check if expired
        if datetime.now() - entry.created_at > self.cache_ttl:
            del self._cache[cache_key]
            return None
        
        # Update access tracking
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        return entry
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]):
        """Cache an embedding"""
        # Ensure cache doesn't exceed size limit
        if len(self._cache) >= self.cache_size:
            self._evict_cache_entries()
        
        entry = EmbeddingCacheEntry(
            embedding=embedding,
            created_at=datetime.now()
        )
        
        self._cache[cache_key] = entry
    
    def _evict_cache_entries(self):
        """Evict least recently used cache entries"""
        # Sort by last accessed time (or created time if never accessed)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(sorted_entries) // 10)
        
        for i in range(num_to_remove):
            key_to_remove = sorted_entries[i][0]
            del self._cache[key_to_remove]
        
        logger.debug(f"Evicted {num_to_remove} cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_limit": self.cache_size,
            "backend": self._current_backend,
            "model_name": self.model_name,
            "is_initialized": self._is_initialized,
            "total_access_count": sum(entry.access_count for entry in self._cache.values())
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    async def similarity_search(self, 
                              query_embedding: List[float],
                              candidate_embeddings: List[List[float]],
                              top_k: int = 10) -> List[tuple]:
        """Perform similarity search using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        try:
            if not query_embedding or not candidate_embeddings:
                return []
            
            import numpy as np
            from numpy.linalg import norm
            
            query_vec = np.array(query_embedding)
            similarities = []
            
            for i, candidate_embedding in enumerate(candidate_embeddings):
                if not candidate_embedding:
                    continue
                    
                candidate_vec = np.array(candidate_embedding)
                
                # Cosine similarity
                dot_product = np.dot(query_vec, candidate_vec)
                norms = norm(query_vec) * norm(candidate_vec)
                
                if norms > 0:
                    similarity = dot_product / norms
                    similarities.append((i, float(similarity)))
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def test_embedding_generation(self) -> bool:
        """Test embedding generation functionality"""
        try:
            test_text = "This is a test sentence for embedding generation."
            
            embedding = await self.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info(f"Embedding test successful: {len(embedding)} dimensions")
                return True
            else:
                logger.error("Embedding test failed: no embedding generated")
                return False
                
        except Exception as e:
            logger.error(f"Embedding test failed: {e}")
            return False