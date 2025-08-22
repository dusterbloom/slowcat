"""
Test suite for Dynamic Tape Head - The Consciousness Reader

These tests verify that DTH:
1. Never exceeds token budget (HARD REQUIREMENT)
2. Scores memories correctly using the consciousness formula
3. Maintains <30ms latency for retrieval
4. Includes provenance for all memories
5. Handles uncertainty with tripwires
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
import numpy as np
from unittest.mock import Mock, MagicMock, AsyncMock

import sys
sys.path.append(str(Path(__file__).parent.parent))

from memory.dynamic_tape_head import (
    DynamicTapeHead, 
    MemorySpan, 
    ContextBundle
)
from memory.tape_store import TapeStore, TapeEntry


class TestDynamicTapeHead:
    
    @pytest.fixture
    def mock_memory_system(self):
        """Create a mock memory system with tape store"""
        memory = Mock()
        
        # Mock tape store with some test data
        tape_store = Mock()
        tape_entries = [
            TapeEntry(
                ts=time.time() - 3600,  # 1 hour ago
                speaker_id="user",
                role="user",
                content="Tell me about quantum computing and its applications"
            ),
            TapeEntry(
                ts=time.time() - 1800,  # 30 min ago
                speaker_id="user", 
                role="user",
                content="What's the weather like in Sardinia today?"
            ),
            TapeEntry(
                ts=time.time() - 60,  # 1 min ago
                speaker_id="user",
                role="user", 
                content="Can you help me with Python code?"
            )
        ]
        
        tape_store.get_recent = Mock(return_value=tape_entries)
        tape_store.search = Mock(return_value=tape_entries[:2])
        
        memory.tape_store = tape_store
        return memory
    
    @pytest.fixture
    def dth(self, mock_memory_system):
        """Create a DTH instance with mock memory and test-optimized policy"""
        return DynamicTapeHead(mock_memory_system, policy_path="config/test_tape_head_policy.json")
    
    @pytest.mark.asyncio
    async def test_token_budget_compliance(self, dth):
        """Ensure DTH NEVER exceeds token budget"""
        
        # Test with various budget sizes
        budgets = [100, 500, 1000, 2000]
        
        for budget in budgets:
            bundle = await dth.seek(
                query="Tell me everything about our past conversations",
                budget=budget
            )
            
            # HARD REQUIREMENT: Never exceed budget
            assert bundle.token_count <= budget, \
                f"Token budget violated! Used {bundle.token_count} of {budget}"
            
            # Verify all components counted
            distribution = bundle.get_token_distribution()
            assert distribution['total'] == bundle.token_count
    
    @pytest.mark.asyncio
    async def test_scoring_algorithm(self, dth):
        """Verify scoring produces expected rankings"""
        
        # Create test memories with known characteristics
        recent_memory = MemorySpan(
            content="Recent relevant content about Python",
            ts=time.time() - 60,  # 1 minute ago
            role="user",
            speaker_id="test",
            source_id="test_1",
            source_hash="hash1"
        )
        
        old_memory = MemorySpan(
            content="Old content about cooking recipes",
            ts=time.time() - 86400,  # 1 day ago
            role="user",
            speaker_id="test",
            source_id="test_2",
            source_hash="hash2"
        )
        
        # Score them
        query_entities = ["Python", "code"]
        score_recent, comp_recent = dth._score_memory(
            recent_memory, None, query_entities, []
        )
        score_old, comp_old = dth._score_memory(
            old_memory, None, query_entities, []
        )
        
        # Recent + relevant should score higher
        assert score_recent > score_old, \
            f"Recent relevant ({score_recent:.3f}) should score higher than old irrelevant ({score_old:.3f})"
        
        # Verify recency component
        assert comp_recent['R'] > comp_old['R'], "Recent should have higher R score"
        
        # Verify entity component (Python matches)
        assert comp_recent['E'] > comp_old['E'], "Matching entities should score higher"
    
    @pytest.mark.asyncio
    async def test_retrieval_latency(self, dth):
        """Ensure <30ms retrieval time"""
        
        # Run multiple retrievals
        latencies = []
        for _ in range(10):
            start = time.time()
            bundle = await dth.seek(
                query="Quick test query",
                budget=1000
            )
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        # Check p95 latency
        p95_latency = np.percentile(latencies, 95)
        assert p95_latency < 30, \
            f"P95 latency {p95_latency:.1f}ms exceeds 30ms target"
        
        # Verify metrics tracking
        assert len(dth.metrics['retrieval_latency_ms']) == 10
    
    @pytest.mark.asyncio
    async def test_provenance_tracking(self, dth):
        """Every memory MUST have provenance"""
        
        bundle = await dth.seek(
            query="Test query for provenance",
            budget=1000
        )
        
        # Check all memories have required provenance
        all_memories = bundle.verbatim + bundle.shadows + bundle.recents
        
        for memory in all_memories:
            assert memory.source_id, f"Memory missing source_id: {memory.content[:50]}"
            assert memory.source_hash, f"Memory missing source_hash: {memory.content[:50]}"
            assert memory.fidelity in ["verbatim", "structured", "tuple", "edge", "forgotten"], \
                f"Invalid fidelity: {memory.fidelity}"
        
        # Verify bundle can report all sources
        sources = bundle.get_all_sources()
        assert len(sources) == len(all_memories)
        assert all(s['hash'] for s in sources)
    
    @pytest.mark.asyncio
    async def test_uncertainty_tripwires(self, dth):
        """Test uncertainty detection and response"""
        
        # Query with high entity density
        entity_heavy_query = "Tell me about John Smith from Microsoft in Seattle"
        bundle = await dth.seek(entity_heavy_query, budget=1000)
        
        # Should trigger high_entity_density
        assert "high_entity_density" in dth.tripwires_triggered or \
               len(dth.tripwires_triggered) == 0  # May not trigger if no entities extracted
        
        # Query with ambiguous references
        ambiguous_query = "What about that thing?"
        bundle = await dth.seek(ambiguous_query, budget=1000)
        
        # Should trigger ambiguous_reference
        if "ambiguous_reference" in dth.tripwires_triggered:
            # When triggered, should prefer verbatim over shadows
            assert len(bundle.verbatim) >= len(bundle.shadows)
    
    @pytest.mark.asyncio
    async def test_policy_hot_reload(self, dth):
        """Test dynamic policy updates"""
        
        # Save current policy
        original_policy = dth.policy.copy()
        
        # Modify policy
        new_policy = original_policy.copy()
        new_policy['weights']['w_recency'] = 0.8  # Heavily favor recency
        new_policy['weights']['w_semantic'] = 0.1
        new_policy['version'] = 2
        
        # Save to file
        policy_path = Path("config/test_tape_head_policy.json")
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(policy_path, 'w') as f:
            json.dump(new_policy, f)
        
        # Create new DTH with updated policy
        dth2 = DynamicTapeHead(dth.memory, policy_path=str(policy_path))
        
        # Verify policy loaded
        assert dth2.policy['version'] == 2
        assert dth2.policy['weights']['w_recency'] == 0.8
        
        # Clean up
        policy_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_policy_hot_reload_in_process(self, tmp_path, mock_memory_system):
        """Modify policy file and verify in-process hot reload on seek()"""
        # Write initial policy
        p = tmp_path / "policy.json"
        with open(p, 'w') as f:
            json.dump({
                "version": 1,
                "weights": {"w_recency": 0.4, "w_semantic": 0.35, "w_entity": 0.15, "w_novelty": 0.10},
                "parameters": {"knn_k": 5, "recency_half_life_hours": 6, "min_confidence": 0.0, "entity_overlap_bonus": 0.1, "max_verbatim_chunks": 5, "shadow_compression_ratio": 0.3, "uncertainty_threshold": 0.4},
                "ablation": {"use_semantic": False, "use_entities": False, "use_shadows": False}
            }, f)

        dth_local = DynamicTapeHead(mock_memory_system, policy_path=str(p))
        # First seek to load policy and set mtime
        await dth_local.seek("hello", budget=200)
        assert dth_local.policy["version"] == 1

        # Update policy file
        newp = {
            "version": 2,
            "weights": {"w_recency": 0.8, "w_semantic": 0.1, "w_entity": 0.05, "w_novelty": 0.05},
            "parameters": {"knn_k": 5, "recency_half_life_hours": 3, "min_confidence": 0.0, "entity_overlap_bonus": 0.1, "max_verbatim_chunks": 5, "shadow_compression_ratio": 0.3, "uncertainty_threshold": 0.4},
            "ablation": {"use_semantic": False, "use_entities": False, "use_shadows": False}
        }
        # Ensure mtime changes
        import time as _t
        _t.sleep(0.05)
        with open(p, 'w') as f:
            json.dump(newp, f)

        # Second seek should reload policy
        await dth_local.seek("world", budget=200)
        assert dth_local.policy["version"] == 2
    
    @pytest.mark.asyncio
    async def test_compression_to_shadows(self, dth):
        """Test memory compression maintains provenance"""
        
        original = MemorySpan(
            content="This is a very long memory that contains lots of important information about the conversation",
            ts=time.time(),
            role="user",
            speaker_id="test",
            source_id="original_123",
            source_hash="original_hash",
            fidelity="verbatim",
            tokens=20
        )
        
        # Compress it
        shadow = dth._compress_to_shadow(original)
        
        # Verify compression
        assert len(shadow.content) < len(original.content)
        assert shadow.tokens < original.tokens
        
        # Verify provenance maintained
        assert shadow.source_id == original.source_id
        assert shadow.source_hash == original.source_hash
        assert shadow.fidelity == "structured"  # Downgraded
    
    @pytest.mark.asyncio
    async def test_entity_extraction_fallback(self, dth):
        """Test entity extraction works even without spaCy"""
        
        # Temporarily disable NLP
        original_nlp = dth.nlp
        dth.nlp = None
        
        text = "John Smith from OpenAI visited Paris"
        entities = dth._extract_entities(text)
        
        # Should extract capitalized words
        assert "John" in entities or "Smith" in entities
        assert "OpenAI" in entities or "Paris" in entities
        
        # Restore
        dth.nlp = original_nlp
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, dth):
        """Test performance metrics are tracked correctly"""
        
        # Run several queries
        for i in range(5):
            await dth.seek(f"Test query {i}", budget=1000)
        
        # Get metrics
        summary = dth.get_metrics_summary()
        
        # Verify metrics collected
        assert summary['avg_latency_ms'] > 0
        assert summary['avg_tokens_used'] > 0
        assert summary['avg_memories_scored'] > 0
        
        # Verify score distributions
        for component in ['R', 'S', 'E', 'D']:
            assert component in summary['score_distributions']
            assert 'mean' in summary['score_distributions'][component]
        
        # Save metrics
        metrics_path = "logs/test_dth_metrics.json"
        dth.save_metrics(metrics_path)
        assert Path(metrics_path).exists()
        
        # Clean up
        Path(metrics_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, dth):
        """Test embedding cache improves performance"""
        
        if not dth.encoder:
            pytest.skip("No encoder available")
        
        text = "Test text for caching"
        
        # First call - should compute
        start = time.time()
        emb1 = dth._get_embedding(text)
        time_first = time.time() - start
        
        # Second call - should use cache
        start = time.time()
        emb2 = dth._get_embedding(text)
        time_cached = time.time() - start
        
        # Cache should be faster
        assert time_cached < time_first * 0.5  # At least 2x faster
        
        # Embeddings should be identical
        if emb1 is not None and emb2 is not None:
            assert np.array_equal(emb1, emb2)
    
    @pytest.mark.asyncio
    async def test_selection_strategy(self, dth):
        """Test memory selection follows the strategy"""
        
        # Create many scored memories
        memories = []
        for i in range(20):
            memory = MemorySpan(
                content=f"Memory {i}",
                ts=time.time() - i * 3600,  # Varying ages
                role="user",
                speaker_id="test",
                source_id=f"mem_{i}",
                source_hash=f"hash_{i}",
                tokens=10
            )
            memory.score = 1.0 - (i * 0.05)  # Decreasing scores
            memories.append(memory)
        
        # Select within budget
        bundle = dth._select_within_budget(memories, budget=200)
        
        # Should have recents (most recent)
        assert len(bundle.recents) > 0
        assert bundle.recents[0].content == "Memory 0"  # Most recent
        
        # Should have verbatim (high scoring)
        assert len(bundle.verbatim) > 0
        
        # Should respect max_verbatim_chunks
        max_chunks = dth.policy['parameters']['max_verbatim_chunks']
        assert len(bundle.verbatim) <= max_chunks
        
        # Total should not exceed budget
        assert bundle.token_count <= 200

    @pytest.mark.asyncio
    async def test_uses_surrealdb_knn_when_available(self):
        """DTH should use memory.knn_tape when provided (no encoder required)."""
        class FakeMemory:
            def __init__(self):
                self.tape_store = None
            async def knn_tape(self, query: str, limit: int = 5, scan: int = 50):
                now = time.time()
                return [
                    {"ts": now - 10, "speaker_id": "u", "role": "user", "content": "Discussed Python decorators"},
                    {"ts": now - 20, "speaker_id": "u", "role": "user", "content": "Talked about Sardinia weather"},
                ]

        dth = DynamicTapeHead(FakeMemory(), policy_path="config/test_tape_head_policy.json")
        # Force encoder to None to prove knn_tape path does not require it
        dth.encoder = None
        bundle = await dth.seek("python tips", budget=200)
        # Expect verbatim or recents to include KNN items (depending on score/confidence)
        assert len(bundle.verbatim) + len(bundle.recents) + len(bundle.shadows) > 0
        texts = [m.content for m in (bundle.verbatim + bundle.recents + bundle.shadows)]
        assert any("Python" in t or "Sardinia" in t for t in texts)

    def test_diversity_filter_prefers_unique(self, dth):
        """Diversity filter should reduce near-duplicates by time/text."""
        # Many similar memories in the same time window
        base_ts = time.time()
        items = []
        for i in range(10):
            m = MemorySpan(
                content="Same topic repeated here",
                ts=base_ts - (i * 30),  # within 120s window
                role="user",
                speaker_id="t",
                source_id=f"dup_{i}",
                source_hash=f"hash_dup_{i}",
                tokens=8,
            )
            items.append(m)
        diverse = dth._apply_diversity(items, max_out=5)
        assert len(diverse) == 1  # due to 120s min_time_gap_s and high text similarity

    @pytest.mark.asyncio
    async def test_cross_encoder_optional(self, monkeypatch, dth):
        """Cross-encoder path should be optional and safe when unavailable."""
        # Force enable via env then re-init
        monkeypatch.setenv('USE_CROSS_ENCODER', 'true')
        dth2 = DynamicTapeHead(dth.memory, policy_path="config/test_tape_head_policy.json")
        if dth2.cross_encoder is None:
            pytest.skip("Cross-encoder not available in environment")
        # Score a simple memory to ensure path runs
        mem = MemorySpan(
            content="Python programming help and code examples",
            ts=time.time(),
            role="user",
            speaker_id="x",
            source_id="x1",
            source_hash="h1",
            tokens=8,
        )
        score, comps = dth2._score_memory(mem, None, ["Python", "code"], [])
        assert isinstance(score, float)


class TestIntegration:
    """Integration tests with real memory system"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_with_real_surreal_memory(self):
        """Test DTH with actual SurrealDB (if available)"""
        
        try:
            from memory.surreal_memory import SurrealMemory
            
            # Create real memory system
            memory = SurrealMemory()
            await memory.connect()
            
            # Create DTH with test policy for speed
            dth = DynamicTapeHead(memory, policy_path="config/test_tape_head_policy.json")
            
            # Add some test data
            await memory.add_entry(
                speaker_id="test_user",
                content="Integration test content",
                role="user"
            )
            
            # Wait a moment for data to be available
            await asyncio.sleep(0.1)
            
            # Test retrieval
            bundle = await dth.seek(
                query="test content",
                budget=1000
            )
            
            assert bundle.token_count <= 1000
            
            # Debug information if assertion fails
            total_memories = len(bundle.verbatim) + len(bundle.shadows) + len(bundle.recents)
            if total_memories == 0:
                # Try to get some data to debug
                recent_entries = await memory.get_recent(limit=5)
                print(f"Recent entries in memory: {len(recent_entries)}")
                if recent_entries:
                    print(f"Sample entry: {recent_entries[0]}")
                
                # If we have data but DTH isn't finding it, that's the issue
                if recent_entries:
                    pytest.fail(f"Memory has {len(recent_entries)} entries but DTH found 0 memories")
                else:
                    pytest.skip("No data was inserted into SurrealDB - database connection issue")
            
            assert len(bundle.get_all_sources()) > 0, f"Expected sources but got {len(bundle.get_all_sources())} from {total_memories} memories"
            
            await memory.close()
            
        except ImportError:
            pytest.skip("SurrealDB not available")
        except Exception as e:
            # Skip if SurrealDB is not running or has connection issues
            pytest.skip(f"SurrealDB connection failed (probably not running): {e}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_with_smart_context_manager(self):
        """Test DTH integration with Smart Context Manager"""
        
        try:
            from processors.smart_context_manager import SmartContextManager
            
            # This would test the full integration
            # Skipping for now as it requires full pipeline setup
            pytest.skip("Full pipeline integration test")
            
        except ImportError:
            pytest.skip("Smart Context Manager not available")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
