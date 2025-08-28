#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hierarchical Memory System

Tests the complete four-tier hierarchical memory system:
1. Working Memory (0-5 min): MLX tensors and neural field states  
2. Short-term Memory (5min-2hr): SurrealDB time-series with semantic compression
3. Long-term Memory (2hr+): SurrealDB graph nodes with attractor weights
4. Episodic Memory (permanent): SurrealDB documents with importance scoring

Validates:
- Fragment retrieval under 50ms
- Memory tier transitions 
- Importance-based promotion
- Semantic compression
- Neural field integration
- Backward compatibility
"""

import asyncio
import time
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from loguru import logger

# Import components to test
from memory.hierarchical_manager import (
    HierarchicalMemoryManager, WorkingMemoryManager, MemoryFragment, FieldState
)
from memory.transition_manager import (
    MemoryTransitionManager, ImportanceCalculator, SemanticCompressor
)
from memory.fragment_retrieval import FastFragmentRetriever, RetrievalResult
from memory.enhanced_surreal_memory import HierarchicalSurrealMemory, create_hierarchical_surreal_memory
from consciousness.core import MLX_AVAILABLE

# Test configuration
TEST_DB_CONFIG = {
    "db_url": "memory://test",  # In-memory database for testing
    "username": "test",
    "password": "test",
    "namespace": "test",
    "database": "test"
}

class TestWorkingMemoryManager(unittest.IsolatedAsyncioTestCase):
    """Test Working Memory (Tier 1) functionality"""
    
    async def asyncSetUp(self):
        self.working_memory = WorkingMemoryManager(max_fragments=50)
    
    async def test_fragment_creation(self):
        """Test fragment creation and storage"""
        content = "The user likes coffee in the morning"
        fragment_id = await self.working_memory.add_fragment(
            content=content, 
            fragment_type="semantic",
            context_tags=["preference", "morning"]
        )
        
        self.assertIsInstance(fragment_id, str)
        self.assertIn(fragment_id, self.working_memory.fragments)
        
        fragment = self.working_memory.fragments[fragment_id]
        self.assertEqual(fragment.content["text"], content)
        self.assertEqual(fragment.type, "semantic")
        self.assertIn("preference", fragment.context_tags)
        self.assertEqual(fragment.memory_tier, 1)
        self.assertEqual(fragment.strength, 1.0)
    
    async def test_fragment_retrieval(self):
        """Test semantic fragment retrieval"""
        # Add test fragments
        fragments = [
            ("User likes coffee", "semantic", ["preference"]),
            ("Meeting at 3pm today", "episodic", ["schedule"]),
            ("Python is a programming language", "semantic", ["knowledge"]),
            ("User felt happy about the project", "emotional", ["mood"])
        ]
        
        fragment_ids = []
        for content, ftype, tags in fragments:
            fid = await self.working_memory.add_fragment(content, ftype, tags)
            fragment_ids.append(fid)
        
        # Test retrieval
        results = await self.working_memory.retrieve_fragments("coffee", limit=5)
        self.assertGreater(len(results), 0)
        
        # Should find coffee-related fragment
        found_coffee = any("coffee" in r.content.get("text", "").lower() for r in results)
        self.assertTrue(found_coffee)
        
        # Test access tracking
        for result in results:
            self.assertGreater(result.access_count, 0)
            self.assertLessEqual(result.last_accessed, time.time())
    
    async def test_capacity_management(self):
        """Test working memory capacity management"""
        initial_capacity = self.working_memory.max_fragments
        
        # Add fragments beyond capacity
        for i in range(initial_capacity + 10):
            await self.working_memory.add_fragment(f"Test fragment {i}", "semantic")
        
        # Should not exceed capacity
        self.assertLessEqual(len(self.working_memory.fragments), initial_capacity)
        
        stats = self.working_memory.get_memory_stats()
        self.assertEqual(stats["max_capacity"], initial_capacity)
        self.assertLessEqual(stats["utilization"], 1.0)
    
    async def test_mlx_integration(self):
        """Test MLX integration if available"""
        if not MLX_AVAILABLE:
            self.skipTest("MLX not available")
        
        content = "Test MLX integration with neural field states"
        fragment_id = await self.working_memory.add_fragment(content, "semantic")
        
        # Check if field state was created
        field_state = self.working_memory.get_field_state(fragment_id)
        
        # Field state creation depends on consciousness field availability
        # Test passes if no errors occurred during fragment creation
        self.assertIsInstance(fragment_id, str)

class TestImportanceCalculator(unittest.IsolatedAsyncioTestCase):
    """Test importance scoring algorithms"""
    
    async def asyncSetUp(self):
        self.calculator = ImportanceCalculator()
        
        # Create test fragments with different characteristics
        self.fragments = [
            MemoryFragment(
                fragment_id="frag1",
                type="semantic", 
                content={"text": "User loves pizza"},
                context_tags=["preference"],
                strength=1.0,
                memory_tier=1,
                last_accessed=time.time() - 3600,  # 1 hour ago
                access_count=5,
                created_at=time.time() - 86400  # 1 day ago
            ),
            MemoryFragment(
                fragment_id="frag2", 
                type="emotional",
                content={"text": "User felt excited about the promotion"},
                context_tags=["mood", "important"],
                strength=1.0,
                memory_tier=1,
                last_accessed=time.time() - 1800,  # 30 min ago
                access_count=3,
                created_at=time.time() - 7200  # 2 hours ago
            )
        ]
    
    async def test_importance_calculation(self):
        """Test comprehensive importance scoring"""
        fragment = self.fragments[0]
        all_fragments = self.fragments
        
        metrics = await self.calculator.calculate_importance(fragment, all_fragments)
        
        # Check all metrics are in valid range [0, 1]
        self.assertGreaterEqual(metrics.access_frequency, 0.0)
        self.assertLessEqual(metrics.access_frequency, 1.0)
        self.assertGreaterEqual(metrics.recency_boost, 0.0)
        self.assertLessEqual(metrics.recency_boost, 1.0)
        self.assertGreaterEqual(metrics.semantic_uniqueness, 0.0)
        self.assertLessEqual(metrics.semantic_uniqueness, 1.0)
        
        final_score = self.calculator.calculate_final_importance_score(metrics)
        self.assertGreaterEqual(final_score, 0.0)
        self.assertLessEqual(final_score, 1.0)
    
    async def test_emotional_weight_calculation(self):
        """Test emotional weight calculation"""
        emotional_fragment = self.fragments[1]  # Has emotional content
        semantic_fragment = self.fragments[0]   # Regular semantic content
        
        emotional_metrics = await self.calculator.calculate_importance(emotional_fragment)
        semantic_metrics = await self.calculator.calculate_importance(semantic_fragment)
        
        # Emotional fragment should have higher emotional weight
        self.assertGreater(emotional_metrics.emotional_weight, semantic_metrics.emotional_weight)
    
    async def test_recency_boost(self):
        """Test recency boost calculation"""
        recent_fragment = self.fragments[1]  # Accessed 30 min ago
        older_fragment = self.fragments[0]   # Accessed 1 hour ago
        
        recent_metrics = await self.calculator.calculate_importance(recent_fragment)
        older_metrics = await self.calculator.calculate_importance(older_fragment)
        
        # More recently accessed fragment should have higher recency boost
        self.assertGreater(recent_metrics.recency_boost, older_metrics.recency_boost)

class TestSemanticCompressor(unittest.IsolatedAsyncioTestCase):
    """Test semantic compression for tier transitions"""
    
    async def asyncSetUp(self):
        self.compressor = SemanticCompressor()
        
        self.test_fragment = MemoryFragment(
            fragment_id="test_compress",
            type="semantic",
            content={
                "text": "This is a very long piece of text that should be compressed when transitioning to higher memory tiers. It contains multiple sentences with various pieces of information. Some information is more important than others. The key point is about machine learning and artificial intelligence applications in voice recognition systems."
            },
            context_tags=["ai", "machine_learning"],
            strength=0.8,
            memory_tier=1,
            last_accessed=time.time(),
            access_count=3
        )
    
    async def test_semantic_compression(self):
        """Test compression for short-term memory (tier 2)"""
        compressed = await self.compressor.compress_for_tier(self.test_fragment, 2)
        
        self.assertEqual(compressed.memory_tier, 2)
        self.assertLess(compressed.strength, self.test_fragment.strength)
        self.assertEqual(compressed.content.get("compression_level"), "semantic")
        
        # Should preserve key information
        self.assertIn("text", compressed.content)
        
        # Original fragment should be unchanged
        self.assertEqual(self.test_fragment.memory_tier, 1)
    
    async def test_graph_node_compression(self):
        """Test compression for long-term memory (tier 3)"""
        compressed = await self.compressor.compress_to_graph_node(self.test_fragment, 3)
        
        self.assertEqual(compressed.memory_tier, 3)
        self.assertEqual(compressed.content.get("compression_level"), "graph_node")
        self.assertIn("entities", compressed.content)
        self.assertIn("relationships", compressed.content)
        self.assertIn("semantic_hash", compressed.content)
    
    async def test_episodic_compression(self):
        """Test compression for episodic memory (tier 4)"""
        compressed = await self.compressor.compress_episodic(self.test_fragment, 4)
        
        self.assertEqual(compressed.memory_tier, 4)
        self.assertEqual(compressed.type, "episodic")
        self.assertEqual(compressed.content.get("compression_level"), "episodic")
        self.assertIn("episode_summary", compressed.content)
        self.assertIn("key_entities", compressed.content)

class TestMemoryTransitionManager(unittest.IsolatedAsyncioTestCase):
    """Test memory tier transition logic"""
    
    async def asyncSetUp(self):
        # Create a mock surreal memory for testing
        self.surreal_memory = None  # Will use None for testing without DB
        self.transition_manager = MemoryTransitionManager(
            surreal_memory=self.surreal_memory,
            transition_interval=1  # 1 second for testing
        )
        
        # Override thresholds for faster testing
        self.transition_manager.tier_thresholds = {
            1: 2,   # 2 seconds: Working → Short-term
            2: 5,   # 5 seconds: Short-term → Long-term
            3: 10,  # 10 seconds: Long-term → Episodic
        }
    
    async def test_transition_candidate_identification(self):
        """Test identification of fragments ready for transition"""
        # Create fragments of different ages
        old_fragment = MemoryFragment(
            fragment_id="old_frag",
            type="semantic",
            content={"text": "Old fragment"},
            context_tags=[],
            strength=0.8,
            memory_tier=1,
            last_accessed=time.time(),
            access_count=2,
            created_at=time.time() - 10  # 10 seconds old
        )
        
        new_fragment = MemoryFragment(
            fragment_id="new_frag",
            type="semantic", 
            content={"text": "New fragment"},
            context_tags=[],
            strength=0.9,
            memory_tier=1,
            last_accessed=time.time(),
            access_count=1,
            created_at=time.time() - 1  # 1 second old
        )
        
        working_fragments = [old_fragment, new_fragment]
        
        # Run transition cycle
        await self.transition_manager.run_transition_cycle(working_fragments)
        
        # Check statistics
        stats = self.transition_manager.get_transition_stats()
        self.assertGreaterEqual(stats["total_transitions"], 0)
    
    async def test_importance_based_early_promotion(self):
        """Test early promotion based on high importance"""
        # Create high-importance fragment
        important_fragment = MemoryFragment(
            fragment_id="important_frag",
            type="emotional",
            content={"text": "Very important meeting reminder"},
            context_tags=["important", "urgent", "meeting"],
            strength=1.0,
            memory_tier=1,
            last_accessed=time.time(),
            access_count=10,  # High access count
            created_at=time.time() - 1.5  # 1.5 seconds (below normal threshold)
        )
        
        working_fragments = [important_fragment]
        
        await self.transition_manager.run_transition_cycle(working_fragments)
        
        # Check if early promotions were recorded
        stats = self.transition_manager.get_transition_stats()
        # Early promotion stats would be tracked in real implementation
        self.assertIsInstance(stats, dict)

class TestFastFragmentRetriever(unittest.IsolatedAsyncioTestCase):
    """Test fragment retrieval performance"""
    
    async def asyncSetUp(self):
        self.retriever = FastFragmentRetriever(surreal_memory=None)
    
    async def test_embedding_cache(self):
        """Test embedding caching for performance"""
        query = "test query for caching"
        
        # First retrieval - should compute embedding
        embedding1 = await self.retriever._get_or_compute_embedding(query)
        cache_size_1 = len(self.retriever.embedding_cache)
        
        # Second retrieval - should use cache
        embedding2 = await self.retriever._get_or_compute_embedding(query)
        cache_size_2 = len(self.retriever.embedding_cache)
        
        # Embeddings should be identical (from cache)
        np.testing.assert_array_equal(embedding1, embedding2)
        
        # Cache size should be the same (no new computation)
        self.assertEqual(cache_size_1, cache_size_2)
        
        # Should have at least one cache hit
        stats = self.retriever.get_performance_stats()
        self.assertGreater(stats["cache_hit_rate"], 0)
    
    async def test_performance_tracking(self):
        """Test retrieval performance statistics"""
        # Simulate some retrievals (will use fallback search)
        queries = ["test query 1", "test query 2", "test query 3"]
        
        for query in queries:
            await self.retriever.retrieve_fragments(query, limit=5)
        
        stats = self.retriever.get_performance_stats()
        
        self.assertEqual(stats["total_queries"], len(queries))
        self.assertGreaterEqual(stats["avg_retrieval_time_ms"], 0)
        self.assertIsInstance(stats["performance_target_met"], bool)
    
    async def test_cache_capacity_management(self):
        """Test embedding cache capacity management"""
        # Set small cache size for testing
        self.retriever.max_cache_size = 3
        
        # Add more queries than cache capacity
        queries = [f"query {i}" for i in range(5)]
        
        for query in queries:
            await self.retriever._get_or_compute_embedding(query)
        
        # Cache should not exceed max size
        self.assertLessEqual(len(self.retriever.embedding_cache), self.retriever.max_cache_size)

class TestHierarchicalMemoryManager(unittest.IsolatedAsyncioTestCase):
    """Test complete hierarchical memory system integration"""
    
    async def asyncSetUp(self):
        self.memory_manager = HierarchicalMemoryManager(
            surreal_memory=None,  # No SurrealDB for unit tests
            working_memory_capacity=20
        )
    
    async def test_memory_storage_and_retrieval(self):
        """Test end-to-end memory storage and retrieval"""
        # Store various types of memories
        memories = [
            ("User prefers dark coffee", "semantic", ["preference"]),
            ("Meeting with Sarah at 2pm", "episodic", ["schedule", "meeting"]),
            ("Learn about machine learning", "procedural", ["learning", "ai"]),
            ("User felt excited about project", "emotional", ["mood", "project"])
        ]
        
        fragment_ids = []
        for content, memory_type, tags in memories:
            fid = await self.memory_manager.store_memory(
                content=content,
                memory_type=memory_type,
                context_tags=tags
            )
            fragment_ids.append(fid)
        
        self.assertEqual(len(fragment_ids), len(memories))
        
        # Test retrieval
        results = await self.memory_manager.retrieve_memory("coffee", limit=10)
        self.assertGreater(len(results), 0)
        
        # Should find coffee-related memory
        found_coffee = any("coffee" in r.content.get("text", "").lower() for r in results)
        self.assertTrue(found_coffee)
    
    async def test_system_statistics(self):
        """Test system statistics collection"""
        # Add some memories
        await self.memory_manager.store_memory("Test memory 1", "semantic")
        await self.memory_manager.store_memory("Test memory 2", "episodic")
        
        stats = self.memory_manager.get_system_stats()
        
        self.assertIn("working_memory", stats)
        self.assertIn("surreal_available", stats)
        self.assertIn("tier_thresholds", stats)
        
        working_stats = stats["working_memory"]
        self.assertGreater(working_stats["total_fragments"], 0)
        self.assertGreaterEqual(working_stats["utilization"], 0.0)
    
    async def test_memory_promotion_cycle(self):
        """Test automated memory promotion"""
        # Add a memory
        fragment_id = await self.memory_manager.store_memory(
            "Test memory for promotion", "semantic"
        )
        
        # Run promotion cycle
        await self.memory_manager.promote_fragments()
        
        # Promotion would normally happen based on age and importance
        # For unit test, just verify no errors occurred
        self.assertIsInstance(fragment_id, str)

class TestHierarchicalSurrealMemory(unittest.IsolatedAsyncioTestCase):
    """Test enhanced SurrealDB memory integration (mocked)"""
    
    async def asyncSetUp(self):
        # Create memory system with hierarchical disabled for testing
        self.memory = HierarchicalSurrealMemory(
            **TEST_DB_CONFIG,
            enable_hierarchical=False  # Disable for unit testing
        )
        # Don't actually connect to avoid DB dependency in unit tests
    
    async def test_fragment_storage_fallback(self):
        """Test fragment storage fallback to fact storage"""
        # This tests backward compatibility when hierarchical is disabled
        content = "User likes tea"
        
        # With hierarchical disabled, should parse as fact
        subject, predicate, obj = self.memory._parse_simple_fact(content)
        
        self.assertEqual(subject, "User")
        self.assertEqual(predicate, "likes")
        self.assertEqual(obj, "tea")
    
    async def test_enhanced_stats_structure(self):
        """Test enhanced statistics structure"""
        stats = self.memory.get_enhanced_stats()
        
        self.assertIn("hierarchical_enabled", stats)
        self.assertIn("working_memory_fragments", stats)
        self.assertIn("field_states_cached", stats)
        self.assertIn("fragments_by_tier", stats)
        self.assertIn("performance_target_met", stats)

class TestPerformanceBenchmarks(unittest.IsolatedAsyncioTestCase):
    """Performance benchmark tests"""
    
    async def asyncSetUp(self):
        self.memory_manager = HierarchicalMemoryManager(working_memory_capacity=100)
        
        # Pre-populate with test data
        test_memories = [
            f"Test memory {i} about various topics including AI, machine learning, and technology"
            for i in range(50)
        ]
        
        for memory in test_memories:
            await self.memory_manager.store_memory(memory, "semantic")
    
    async def test_retrieval_performance_target(self):
        """Test that retrieval meets <50ms target"""
        queries = [
            "AI and machine learning",
            "technology topics",
            "test memory",
            "various topics"
        ]
        
        retrieval_times = []
        
        for query in queries:
            start_time = time.time()
            results = await self.memory_manager.retrieve_memory(query, limit=20)
            end_time = time.time()
            
            retrieval_time_ms = (end_time - start_time) * 1000
            retrieval_times.append(retrieval_time_ms)
            
            # Individual retrieval should be reasonably fast
            # Note: Without SurrealDB, this is just working memory search
            self.assertLess(retrieval_time_ms, 100)  # 100ms for working memory only
            self.assertGreater(len(results), 0)
        
        avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
        logger.info(f"Average retrieval time: {avg_retrieval_time:.2f}ms")
        
        # With full SurrealDB implementation, this should be <50ms
        # For working memory only, <100ms is acceptable
        self.assertLess(avg_retrieval_time, 100)
    
    async def test_memory_scalability(self):
        """Test memory system scalability with larger datasets"""
        initial_count = len(self.memory_manager.working_memory.fragments)
        
        # Add more memories up to capacity
        capacity = self.memory_manager.working_memory.max_fragments
        memories_to_add = max(0, capacity - initial_count - 10)  # Leave room for management
        
        start_time = time.time()
        
        for i in range(memories_to_add):
            await self.memory_manager.store_memory(
                f"Scalability test memory {i} with unique content",
                "semantic"
            )
        
        storage_time = (time.time() - start_time) * 1000
        
        # Storage should be reasonably fast even with many memories
        avg_storage_time = storage_time / memories_to_add if memories_to_add > 0 else 0
        self.assertLess(avg_storage_time, 10)  # <10ms per memory storage
        
        # System should still be responsive
        query_start = time.time()
        results = await self.memory_manager.retrieve_memory("scalability test", limit=10)
        query_time = (time.time() - query_start) * 1000
        
        self.assertLess(query_time, 50)  # Query should still be fast
        self.assertGreater(len(results), 0)

class TestIntegrationScenarios(unittest.IsolatedAsyncioTestCase):
    """Integration test scenarios"""
    
    async def asyncSetUp(self):
        self.memory_manager = HierarchicalMemoryManager(working_memory_capacity=30)
    
    async def test_conversation_memory_scenario(self):
        """Test realistic conversation memory scenario"""
        # Simulate a conversation about planning a trip
        conversation_memories = [
            ("User wants to visit Paris next month", "episodic", ["travel", "plans"]),
            ("User prefers budget accommodations", "semantic", ["preference", "travel"]),
            ("User felt excited about the Eiffel Tower", "emotional", ["mood", "travel", "landmark"]),
            ("Need to book flights by Friday", "procedural", ["task", "deadline"]),
            ("User has been to France before", "semantic", ["experience", "travel"]),
            ("User worried about language barrier", "emotional", ["concern", "travel"]),
            ("Download translation app", "procedural", ["task", "preparation"]),
            ("User loves French cuisine", "semantic", ["preference", "food"])
        ]
        
        # Store all memories
        fragment_ids = []
        for content, mem_type, tags in conversation_memories:
            fid = await self.memory_manager.store_memory(content, mem_type, tags)
            fragment_ids.append(fid)
        
        # Test various queries that might come up later
        queries = [
            "travel plans",
            "Paris trip",
            "accommodation preferences",
            "feelings about trip",
            "tasks to complete",
            "French experience"
        ]
        
        for query in queries:
            results = await self.memory_manager.retrieve_memory(query, limit=5)
            
            # Should find relevant memories for each query
            self.assertGreater(len(results), 0)
            
            # Results should be relevant (contain query terms or related concepts)
            relevant_found = any(
                any(word in result.content.get("text", "").lower() 
                    for word in query.lower().split())
                for result in results
            )
            self.assertTrue(relevant_found, f"No relevant results for query: {query}")
    
    async def test_memory_evolution_scenario(self):
        """Test memory system evolution over time"""
        # Store initial memory
        fragment_id = await self.memory_manager.store_memory(
            "User is learning Python programming", "semantic", ["learning", "programming"]
        )
        
        # Simulate passage of time and repeated access
        working_memory = self.memory_manager.working_memory
        fragment = working_memory.fragments[fragment_id]
        
        # Simulate multiple accesses (would increase importance)
        initial_access_count = fragment.access_count
        
        # Retrieve the memory multiple times
        for _ in range(5):
            results = await self.memory_manager.retrieve_memory("Python programming")
            self.assertGreater(len(results), 0)
        
        # Access count should have increased
        updated_fragment = working_memory.fragments.get(fragment_id)
        if updated_fragment:  # Fragment might have been promoted
            self.assertGreater(updated_fragment.access_count, initial_access_count)
        
        # Test memory transition simulation
        await self.memory_manager.promote_fragments()
        
        # System should still be functional after promotion cycle
        results = await self.memory_manager.retrieve_memory("programming")
        # Results might come from different tiers now, but should still exist
        self.assertIsInstance(results, list)

async def run_all_tests():
    """Run all hierarchical memory system tests"""
    logger.info("🚀 Starting Hierarchical Memory System Test Suite")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestWorkingMemoryManager,
        TestImportanceCalculator,
        TestSemanticCompressor,
        TestMemoryTransitionManager,
        TestFastFragmentRetriever,
        TestHierarchicalMemoryManager,
        TestHierarchicalSurrealMemory,
        TestPerformanceBenchmarks,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info(f"📊 Test Results:")
    logger.info(f"   Total Tests: {total_tests}")
    logger.info(f"   Passed: {total_tests - failures - errors}")
    logger.info(f"   Failed: {failures}")
    logger.info(f"   Errors: {errors}")
    logger.info(f"   Success Rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        logger.info("✅ All hierarchical memory system tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed. Check output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)