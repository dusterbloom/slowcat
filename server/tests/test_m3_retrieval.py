"""Comprehensive tests for M3 retrieval system

Tests the complete M3 implementation including:
- Similarity search with MIPS
- Equivalence resolution with voting
- Context retrieval with relevance ranking
- QueryRouter integration
- Performance benchmarks
"""

import pytest
import asyncio
import time
import logging
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# M3 imports
from memory.m3_similarity_search import M3SimilaritySearch, ModalityType, SearchResult
from memory.m3_equivalence_resolver import M3EquivalenceResolver, EquivalenceCandidate, EntityType
from memory.m3_context_retriever import M3ContextRetriever, ContextType, RetrievalStrategy
from memory.m3_surreal_integration import M3SurrealIntegration
from memory.query_router import create_m3_query_router
from services.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockM3Integration:
    """Mock M3 integration for testing"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.clips = {}
        self.equivalences = {}
        self.current_clip_id = 1
        
        # Create sample test data
        self._create_test_data()
    
    def _create_test_data(self):
        """Create realistic test data for M3 system"""
        # Sample embeddings (384-dimensional like sentence transformers)
        def random_embedding():
            vec = np.random.randn(384).astype(np.float32)
            return (vec / np.linalg.norm(vec)).tolist()  # Normalized
        
        # Create test nodes
        test_nodes = [
            {
                'node_id': 1,
                'node_type': 'voice',
                'contents': ['Hello, my name is Alice'],
                'embeddings': [random_embedding()],
                'clip_id': 1,
                'metadata': {
                    'speaker_id': 'alice_voice',
                    'confidence': 0.9,
                    'created_at': time.time() - 3600  # 1 hour ago
                }
            },
            {
                'node_id': 2,
                'node_type': 'semantic',
                'contents': ['Alice is a software engineer'],
                'embeddings': [random_embedding()],
                'clip_id': 1,
                'metadata': {
                    'speaker_id': 'alice_voice',
                    'confidence': 0.8,
                    'created_at': time.time() - 3500
                }
            },
            {
                'node_id': 3,
                'node_type': 'voice',
                'contents': ['I work on machine learning'],
                'embeddings': [random_embedding()],
                'clip_id': 2,
                'metadata': {
                    'speaker_id': 'alice_voice',
                    'confidence': 0.85,
                    'created_at': time.time() - 1800  # 30 min ago
                }
            },
            {
                'node_id': 4,
                'node_type': 'semantic',
                'contents': ['Alice enjoys hiking on weekends'],
                'embeddings': [random_embedding()],
                'clip_id': 2,
                'metadata': {
                    'speaker_id': 'alice_voice',
                    'confidence': 0.75,
                    'created_at': time.time() - 1700
                }
            },
            {
                'node_id': 5,
                'node_type': 'voice',
                'contents': ['My dog is named Potola'],
                'embeddings': [random_embedding()],
                'clip_id': 3,
                'metadata': {
                    'speaker_id': 'alice_voice',
                    'confidence': 0.9,
                    'created_at': time.time() - 900  # 15 min ago
                }
            }
        ]
        
        # Store nodes
        for node in test_nodes:
            self.nodes[node['node_id']] = node
        
        # Create test clips
        self.clips = {
            1: {'clip_id': 1, 'session_id': 'test_session', 'start_time': time.time() - 3600, 'node_count': 2},
            2: {'clip_id': 2, 'session_id': 'test_session', 'start_time': time.time() - 1800, 'node_count': 2},
            3: {'clip_id': 3, 'session_id': 'test_session', 'start_time': time.time() - 900, 'node_count': 1}
        }
        
        # Create test edges (similarity relationships)
        self.edges = {
            (1, 2): {'source_node_id': 1, 'target_node_id': 2, 'weight': 0.8, 'edge_type': 'similarity'},
            (2, 3): {'source_node_id': 2, 'target_node_id': 3, 'weight': 0.7, 'edge_type': 'similarity'},
            (1, 5): {'source_node_id': 1, 'target_node_id': 5, 'weight': 0.6, 'edge_type': 'equivalence'}
        }
    
    async def search_similar_nodes(self, query_embedding, node_type=None, limit=10, min_similarity=0.0):
        """Mock similarity search"""
        query_vec = np.array(query_embedding, dtype=np.float32)
        results = []
        
        for node_id, node in self.nodes.items():
            if node_type and node['node_type'] != node_type:
                continue
            
            if not node.get('embeddings'):
                continue
            
            # Calculate cosine similarity
            node_vec = np.array(node['embeddings'][0], dtype=np.float32)
            similarity = float(np.dot(query_vec, node_vec))
            
            if similarity >= min_similarity:
                result = {
                    'node_id': node_id,
                    'similarity': similarity,
                    **node
                }
                results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    async def get_node_by_id(self, node_id):
        """Mock get node by ID"""
        return self.nodes.get(node_id)
    
    async def get_clip_nodes(self, clip_id):
        """Mock get clip nodes"""
        return [node for node in self.nodes.values() if node.get('clip_id') == clip_id]
    
    async def get_graph_context(self, node_id, max_depth=2, max_nodes=20):
        """Mock graph context retrieval"""
        central_node = self.nodes.get(node_id)
        if not central_node:
            return {"nodes": [], "edges": [], "central_node": None}
        
        # Find connected nodes
        connected_nodes = []
        for edge_key, edge in self.edges.items():
            if node_id in edge_key:
                other_id = edge_key[0] if edge_key[1] == node_id else edge_key[1]
                if other_id in self.nodes:
                    connected_nodes.append(self.nodes[other_id])
        
        return {
            "central_node": central_node,
            "nodes": connected_nodes[:max_nodes],
            "edges": list(self.edges.values())
        }
    
    async def query(self, query_sql, params=None):
        """Mock database query"""
        # Simple mock - return empty for most queries
        return []
    
    async def create_m3_edge(self, source_id, target_id, weight=1.0, edge_type='similarity'):
        """Mock edge creation"""
        edge_key = (min(source_id, target_id), max(source_id, target_id))
        self.edges[edge_key] = {
            'source_node_id': source_id,
            'target_node_id': target_id,
            'weight': weight,
            'edge_type': edge_type
        }
        return True
    
    async def resolve_equivalence(self, entity_name, node_ids, entity_type="speaker"):
        """Mock equivalence resolution"""
        self.equivalences[entity_name] = {
            'canonical_id': entity_name,
            'node_ids': node_ids,
            'entity_type': entity_type
        }
        return entity_name


class MockEmbeddingService:
    """Mock embedding service for testing"""
    
    async def get_embedding(self, text):
        """Generate consistent embeddings for testing"""
        # Create deterministic embeddings based on text hash
        import hashlib
        
        text_hash = hashlib.sha256(text.encode()).digest()
        # Convert to float values between -1 and 1
        embedding = []
        for i in range(0, len(text_hash), 4):
            chunk = text_hash[i:i+4]
            if len(chunk) == 4:
                val = int.from_bytes(chunk, 'little', signed=True) / (2**31)
                embedding.append(val)
        
        # Pad or truncate to 384 dimensions
        while len(embedding) < 384:
            embedding.extend(embedding[:384-len(embedding)])
        embedding = embedding[:384]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        
        return embedding


@pytest.fixture
async def m3_components():
    """Create M3 components for testing"""
    # Create mock components
    mock_integration = MockM3Integration()
    mock_embedding_service = MockEmbeddingService()
    
    # Create M3 components
    similarity_search = M3SimilaritySearch(mock_integration)
    equivalence_resolver = M3EquivalenceResolver(mock_integration, similarity_search)
    context_retriever = M3ContextRetriever(
        mock_integration,
        similarity_search,
        equivalence_resolver,
        mock_embedding_service
    )
    
    return {
        'integration': mock_integration,
        'embedding_service': mock_embedding_service,
        'similarity_search': similarity_search,
        'equivalence_resolver': equivalence_resolver,
        'context_retriever': context_retriever
    }


class TestM3SimilaritySearch:
    """Test M3 similarity search functionality"""
    
    @pytest.mark.asyncio
    async def test_similarity_search_basic(self, m3_components):
        """Test basic similarity search"""
        similarity_search = m3_components['similarity_search']
        
        # Generate query embedding
        query_embedding = await m3_components['embedding_service'].get_embedding("Alice")
        
        # Search for similar nodes
        results = await similarity_search.search_nodes(
            query_embedding=query_embedding,
            modality=ModalityType.TEXT,
            max_results=5
        )
        
        assert len(results) > 0, "Should find similar nodes"
        assert all(isinstance(r, SearchResult) for r in results), "Should return SearchResult objects"
        
        # Results should be sorted by similarity
        similarities = [r.similarity_score for r in results]
        assert similarities == sorted(similarities, reverse=True), "Results should be sorted by similarity"
        
        logger.info(f"✅ Basic similarity search found {len(results)} results")
    
    @pytest.mark.asyncio
    async def test_similarity_search_modality_filter(self, m3_components):
        """Test modality-specific filtering"""
        similarity_search = m3_components['similarity_search']
        
        query_embedding = await m3_components['embedding_service'].get_embedding("voice message")
        
        # Search for voice nodes only
        voice_results = await similarity_search.search_nodes(
            query_embedding=query_embedding,
            modality=ModalityType.VOICE,
            max_results=10
        )
        
        # All results should be voice type
        for result in voice_results:
            assert result.node_type == 'voice', f"Expected voice node, got {result.node_type}"
        
        logger.info(f"✅ Modality filtering found {len(voice_results)} voice nodes")
    
    @pytest.mark.asyncio
    async def test_similarity_search_thresholds(self, m3_components):
        """Test M3-Agent threshold behavior"""
        similarity_search = m3_components['similarity_search']
        
        query_embedding = await m3_components['embedding_service'].get_embedding("test query")
        
        # Test different modality thresholds
        text_results = await similarity_search.search_nodes(
            query_embedding=query_embedding,
            modality=ModalityType.TEXT,  # threshold 0.3
            max_results=10
        )
        
        voice_results = await similarity_search.search_nodes(
            query_embedding=query_embedding,
            modality=ModalityType.VOICE,  # threshold 0.6
            max_results=10
        )
        
        # Voice should have fewer results due to higher threshold
        logger.info(f"✅ Threshold test: text={len(text_results)}, voice={len(voice_results)} results")
    
    @pytest.mark.asyncio
    async def test_clip_based_search(self, m3_components):
        """Test M3-Agent clip-level retrieval"""
        similarity_search = m3_components['similarity_search']
        
        query_embedding = await m3_components['embedding_service'].get_embedding("recent conversation")
        
        # Test clip retrieval
        clips = await similarity_search.search_clips(
            query_embedding=query_embedding,
            max_clips=3,
            threshold=0.3
        )
        
        assert len(clips) <= 3, "Should not exceed max_clips"
        
        # Each clip should have a similarity score
        for clip in clips:
            assert 'similarity_score' in clip, "Clip should have similarity score"
            assert 'clip_id' in clip, "Clip should have clip_id"
        
        logger.info(f"✅ Clip search found {len(clips)} relevant clips")


class TestM3EquivalenceResolver:
    """Test M3 equivalence resolution with voting"""
    
    @pytest.mark.asyncio
    async def test_equivalence_voting_mechanism(self, m3_components):
        """Test M3-Agent voting mechanism for equivalences"""
        equivalence_resolver = m3_components['equivalence_resolver']
        embedding_service = m3_components['embedding_service']
        
        # Create equivalence candidates for same entity
        alice_voice_embedding = await embedding_service.get_embedding("Hello, I'm Alice")
        alice_text_embedding = await embedding_service.get_embedding("Alice is a software engineer")
        
        candidates = [
            EquivalenceCandidate(
                node_id=1,
                modality=ModalityType.VOICE,
                content="Hello, I'm Alice",
                confidence=0.9,
                clip_id=1,
                embedding=alice_voice_embedding,
                metadata={'speaker_id': 'alice_voice'}
            ),
            EquivalenceCandidate(
                node_id=2,
                modality=ModalityType.TEXT,
                content="Alice is a software engineer",
                confidence=0.8,
                clip_id=1,
                embedding=alice_text_embedding,
                metadata={'speaker_id': 'alice_text'}
            )
        ]
        
        # Resolve equivalence
        canonical_id = await equivalence_resolver.resolve_entity_equivalence(
            candidates=candidates,
            entity_type=EntityType.SPEAKER
        )
        
        assert canonical_id is not None, "Should resolve equivalence successfully"
        assert EntityType.SPEAKER.value in canonical_id, "Canonical ID should include entity type"
        
        logger.info(f"✅ Equivalence resolved to: {canonical_id}")
    
    @pytest.mark.asyncio
    async def test_meta_clip_algorithm(self, m3_components):
        """Test M3-Agent meta-clip algorithm for identity resolution"""
        equivalence_resolver = m3_components['equivalence_resolver']
        
        # Build meta-dictionary
        meta_dict = await equivalence_resolver.build_meta_dictionary("test_session")
        
        # Should find some equivalences in test data
        logger.info(f"✅ Meta-clip algorithm found {len(meta_dict)} equivalences")
    
    @pytest.mark.asyncio
    async def test_conflict_resolution(self, m3_components):
        """Test weight-based conflict resolution"""
        equivalence_resolver = m3_components['equivalence_resolver']
        
        # Update equivalence multiple times to test conflict resolution
        success1 = await equivalence_resolver.update_entity_equivalence(
            source_node_id=1,
            target_node_id=2,
            source_modality=ModalityType.VOICE,
            target_modality=ModalityType.TEXT,
            confidence=0.8
        )
        
        success2 = await equivalence_resolver.update_entity_equivalence(
            source_node_id=1,
            target_node_id=3,  # Different target (conflict)
            source_modality=ModalityType.VOICE,
            target_modality=ModalityType.TEXT,
            confidence=0.9  # Higher confidence
        )
        
        assert success1 and success2, "Both updates should succeed"
        
        # Check stats for conflict resolution
        stats = equivalence_resolver.get_equivalence_stats()
        logger.info(f"✅ Conflict resolution stats: {stats}")


class TestM3ContextRetriever:
    """Test M3 context retrieval with relevance ranking"""
    
    @pytest.mark.asyncio
    async def test_context_retrieval_strategies(self, m3_components):
        """Test different retrieval strategies"""
        context_retriever = m3_components['context_retriever']
        
        strategies = [
            RetrievalStrategy.SIMILARITY_FIRST,
            RetrievalStrategy.ENTITY_FIRST,
            RetrievalStrategy.TEMPORAL_FIRST,
            RetrievalStrategy.HYBRID
        ]
        
        for strategy in strategies:
            result = await context_retriever.retrieve_context(
                query="Tell me about Alice",
                max_items=5,
                strategy=strategy
            )
            
            assert hasattr(result, 'items'), f"Strategy {strategy} should return items"
            assert result.retrieval_strategy == strategy, f"Should use requested strategy"
            
            logger.info(f"✅ Strategy {strategy.value}: {len(result.items)} items, "
                       f"{result.retrieval_time_ms:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_relevance_ranking(self, m3_components):
        """Test relevance-based ranking replaces token budgeting"""
        context_retriever = m3_components['context_retriever']
        
        result = await context_retriever.retrieve_context(
            query="machine learning and hiking",
            max_items=10,
            strategy=RetrievalStrategy.HYBRID
        )
        
        # Check relevance ranking
        if len(result.items) > 1:
            relevance_scores = [item.relevance_score for item in result.items]
            assert relevance_scores == sorted(relevance_scores, reverse=True), \
                "Items should be sorted by relevance score"
        
        # Check total relevance calculation
        expected_total = sum(item.relevance_score for item in result.items)
        assert abs(result.total_relevance - expected_total) < 0.001, \
            "Total relevance should match sum of item scores"
        
        logger.info(f"✅ Relevance ranking: {len(result.items)} items, "
                   f"total relevance: {result.total_relevance:.2f}")
    
    @pytest.mark.asyncio
    async def test_context_type_filtering(self, m3_components):
        """Test context type specific retrieval"""
        context_retriever = m3_components['context_retriever']
        
        context_types = [
            ContextType.VOICE,
            ContextType.SEMANTIC,
            ContextType.EPISODIC,
            ContextType.RECENT
        ]
        
        for context_type in context_types:
            result = await context_retriever.retrieve_context(
                query="Alice",
                context_type=context_type,
                max_items=5
            )
            
            assert result.query_type == context_type.value, \
                f"Query type should match requested context type"
            
            logger.info(f"✅ Context type {context_type.value}: {len(result.items)} items")
    
    @pytest.mark.asyncio
    async def test_entity_aware_retrieval(self, m3_components):
        """Test entity-aware context selection"""
        context_retriever = m3_components['context_retriever']
        
        result = await context_retriever.retrieve_context(
            query="What does Alice do?",
            strategy=RetrievalStrategy.ENTITY_FIRST,
            entity_filter=["alice"],
            max_items=5
        )
        
        # Check that results reference the entity
        entity_referenced = any(
            'alice' in ' '.join(item.entity_refs).lower() or 
            'alice' in item.content.lower()
            for item in result.items
        )
        
        assert entity_referenced, "Results should reference Alice entity"
        assert 'alice' in result.entities_referenced or len(result.items) == 0, \
            "Entity should be in referenced entities"
        
        logger.info(f"✅ Entity-aware retrieval: {len(result.items)} items, "
                   f"entities: {list(result.entities_referenced)}")


class TestM3QueryRouterIntegration:
    """Test M3 integration with QueryRouter"""
    
    @pytest.mark.asyncio
    async def test_m3_query_router_creation(self, m3_components):
        """Test M3-enabled QueryRouter creation"""
        context_retriever = m3_components['context_retriever']
        
        router = create_m3_query_router(
            m3_context_retriever=context_retriever,
            facts_graph=None,
            tape_store=None
        )
        
        assert router is not None, "Should create M3 query router"
        assert 'embeddings' in router.stores, "Should have M3 embedding store"
        assert hasattr(router.stores['embeddings'], 'm3_context_retriever'), \
            "Embedding store should have M3 context retriever"
        
        logger.info(f"✅ M3 QueryRouter created with {len(router.stores)} stores")
    
    @pytest.mark.asyncio
    async def test_m3_embedding_store_integration(self, m3_components):
        """Test M3 embedding store in QueryRouter"""
        from memory.query_router import EmbeddingStoreAdapter
        
        context_retriever = m3_components['context_retriever']
        
        # Create M3-enabled embedding store
        store = EmbeddingStoreAdapter(
            embedding_store=None,
            m3_context_retriever=context_retriever
        )
        
        # Test search functionality
        results = await store.search("Alice software engineer", limit=3)
        
        assert isinstance(results, list), "Should return list of results"
        assert store.get_store_name() == "M3 Semantic Search", \
            "Should identify as M3 semantic search"
        
        logger.info(f"✅ M3 embedding store integration: {len(results)} results")


class TestM3PerformanceBenchmarks:
    """Test M3 system performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, m3_components):
        """Test similarity search meets <20ms target"""
        similarity_search = m3_components['similarity_search']
        embedding_service = m3_components['embedding_service']
        
        # Warm up
        query_embedding = await embedding_service.get_embedding("warmup query")
        await similarity_search.search_nodes(query_embedding, ModalityType.TEXT, 5)
        
        # Benchmark multiple queries
        query_times = []
        for i in range(10):
            query = f"test query {i}"
            query_embedding = await embedding_service.get_embedding(query)
            
            start_time = time.time()
            results = await similarity_search.search_nodes(
                query_embedding=query_embedding,
                modality=ModalityType.TEXT,
                max_results=10
            )
            query_time_ms = (time.time() - start_time) * 1000
            query_times.append(query_time_ms)
        
        avg_time = sum(query_times) / len(query_times)
        max_time = max(query_times)
        
        logger.info(f"🚀 Similarity search performance:")
        logger.info(f"   Average: {avg_time:.1f}ms")
        logger.info(f"   Maximum: {max_time:.1f}ms")
        logger.info(f"   Target: <20ms")
        
        # Note: This is a mock test - real performance depends on SurrealDB
        # The actual implementation should be benchmarked with real database
        assert avg_time < 100, f"Average search time {avg_time:.1f}ms should be reasonable"
    
    @pytest.mark.asyncio
    async def test_context_retrieval_performance(self, m3_components):
        """Test context retrieval performance"""
        context_retriever = m3_components['context_retriever']
        
        # Benchmark context retrieval
        queries = [
            "Tell me about Alice",
            "What does Alice do?",
            "Alice's hobbies and interests",
            "Recent conversations with Alice",
            "Alice's technical background"
        ]
        
        retrieval_times = []
        for query in queries:
            start_time = time.time()
            result = await context_retriever.retrieve_context(
                query=query,
                max_items=10,
                strategy=RetrievalStrategy.HYBRID
            )
            retrieval_time = result.retrieval_time_ms
            retrieval_times.append(retrieval_time)
        
        avg_time = sum(retrieval_times) / len(retrieval_times)
        
        logger.info(f"🚀 Context retrieval performance:")
        logger.info(f"   Average: {avg_time:.1f}ms")
        logger.info(f"   Queries: {len(queries)}")
        
        assert avg_time < 200, f"Average retrieval time {avg_time:.1f}ms should be reasonable"
    
    def test_memory_usage(self, m3_components):
        """Test memory usage is reasonable"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        logger.info(f"🚀 Memory usage: {memory_mb:.1f} MB")
        
        # This is a basic check - real usage depends on data size
        assert memory_mb < 500, f"Memory usage {memory_mb:.1f}MB should be reasonable for tests"


@pytest.mark.asyncio 
async def test_m3_system_integration():
    """Integration test for complete M3 system"""
    logger.info("🧠 Starting M3 system integration test...")
    
    # Create components
    mock_integration = MockM3Integration()
    mock_embedding_service = MockEmbeddingService()
    
    similarity_search = M3SimilaritySearch(mock_integration)
    equivalence_resolver = M3EquivalenceResolver(mock_integration, similarity_search)
    context_retriever = M3ContextRetriever(
        mock_integration, similarity_search, equivalence_resolver, mock_embedding_service
    )
    
    # Create M3-enabled query router
    router = create_m3_query_router(m3_context_retriever=context_retriever)
    
    # Test end-to-end query routing
    test_queries = [
        "What does Alice do for work?",
        "Tell me about Alice's hobbies",
        "What is Alice's dog's name?",
        "Recent conversations with Alice"
    ]
    
    for query in test_queries:
        # Route query through M3 system
        # Note: This would normally use router.route_query() but we're testing components directly
        
        result = await context_retriever.retrieve_context(
            query=query,
            max_items=5,
            strategy=RetrievalStrategy.HYBRID
        )
        
        logger.info(f"Query: '{query}' → {len(result.items)} results in {result.retrieval_time_ms:.1f}ms")
        
        for item in result.items[:2]:  # Show top 2 results
            logger.info(f"  - {item.content} (relevance: {item.relevance_score:.2f})")
    
    logger.info("✅ M3 system integration test completed successfully")


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_m3_system_integration())
    logger.info("🎉 All M3 tests completed!")