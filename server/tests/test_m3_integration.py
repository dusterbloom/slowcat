#!/usr/bin/env python3
"""Comprehensive M3 Memory System Integration Tests

Tests the complete M3 memory system including schema migration,
node creation, edge inference, and graph traversal.
"""

import asyncio
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from memory.surreal_connection import SurrealConnectionManager
from memory.m3_surreal_integration import M3SurrealIntegration, M3Node
from processors.m3_memory_processor import M3MemoryProcessor
from services.embedding_service import EmbeddingService
from scripts.migrate_to_m3 import M3Migration

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
async def test_connection():
    """Create test SurrealDB connection"""
    connection = SurrealConnectionManager()
    await connection.connect()
    yield connection
    await connection.close()

@pytest.fixture
async def m3_integration(test_connection):
    """Create M3 integration instance with test connection"""
    m3_integration = M3SurrealIntegration(test_connection)
    
    # Apply M3 schema migration for testing
    migration_file = Path(__file__).parent.parent / "schema" / "m3_migration.surql"
    if migration_file.exists():
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        await test_connection.query(migration_sql)
    
    await m3_integration.initialize()
    yield m3_integration

@pytest.fixture
def test_embeddings():
    """Generate test embeddings for consistency"""
    return {
        "hello world": [0.1, 0.2, 0.3, 0.4, 0.5],
        "good morning": [0.15, 0.25, 0.35, 0.45, 0.55],
        "machine learning": [0.8, 0.7, 0.6, 0.5, 0.4],
        "artificial intelligence": [0.85, 0.75, 0.65, 0.55, 0.45],
        "weather today": [0.3, 0.4, 0.5, 0.6, 0.7]
    }

class TestM3Schema:
    """Test M3 schema creation and migration"""
    
    async def test_schema_migration(self, test_connection):
        """Test M3 schema migration applies successfully"""
        # Apply migration
        migration_file = Path(__file__).parent.parent / "schema" / "m3_migration.surql"
        assert migration_file.exists(), "M3 migration file not found"
        
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        # Execute migration
        result = await test_connection.query(migration_sql)
        assert result is not None, "Migration failed to execute"
        
        # Verify M3 tables exist
        info_result = await test_connection.query("INFO FOR DB")
        tables = info_result[0].get('tb', {})
        
        required_tables = ['m3_nodes', 'm3_edges', 'm3_clips', 'm3_equivalences']
        for table in required_tables:
            assert table in tables, f"Table {table} not created"
        
        logger.info("✅ M3 schema migration test passed")
    
    async def test_migration_verification(self, m3_integration):
        """Test built-in migration verification function"""
        result = await m3_integration.connection.query("RETURN fn::verify_m3_migration()")
        
        assert result is not None, "Verification function failed"
        assert result[0].get('status') == 'success', f"Verification failed: {result[0]}"
        
        logger.info("✅ M3 migration verification test passed")

class TestM3NodeOperations:
    """Test M3 node creation and retrieval"""
    
    async def test_create_voice_node(self, m3_integration, test_embeddings):
        """Test creating voice nodes with embeddings"""
        node_id = await m3_integration.store_m3_node(
            node_type="voice",
            contents=["Hello world, this is a test"],
            embeddings=[test_embeddings["hello world"]],
            speaker_id="test_user",
            extraction_method="test",
            confidence=0.9
        )
        
        assert node_id is not None, "Failed to create voice node"
        assert isinstance(node_id, int), "Node ID should be integer"
        
        # Retrieve and verify node
        node = await m3_integration.get_node_by_id(node_id)
        assert node is not None, "Failed to retrieve created node"
        assert node['node_type'] == 'voice'
        assert node['contents'] == ["Hello world, this is a test"]
        assert node['metadata']['speaker_id'] == 'test_user'
        
        logger.info(f"✅ Voice node creation test passed (node_id: {node_id})")
    
    async def test_create_semantic_node(self, m3_integration, test_embeddings):
        """Test creating semantic nodes with multiple contents"""
        contents = [
            "Machine learning is a subset of AI",
            "Neural networks are inspired by biology"
        ]
        
        node_id = await m3_integration.store_m3_node(
            node_type="semantic",
            contents=contents,
            embeddings=[test_embeddings["machine learning"]],
            speaker_id="assistant",
            extraction_method="fact_extraction",
            confidence=0.95
        )
        
        assert node_id is not None, "Failed to create semantic node"
        
        # Verify node contents
        node = await m3_integration.get_node_by_id(node_id)
        assert node['node_type'] == 'semantic'
        assert len(node['contents']) == 2
        assert node['metadata']['speaker_id'] == 'assistant'
        
        logger.info(f"✅ Semantic node creation test passed (node_id: {node_id})")
    
    async def test_create_episodic_node(self, m3_integration, test_embeddings):
        """Test creating episodic nodes"""
        episode_contents = [
            "User asked about the weather",
            "Assistant provided weather information",
            "User thanked the assistant"
        ]
        
        node_id = await m3_integration.store_m3_node(
            node_type="episodic",
            contents=episode_contents,
            embeddings=[test_embeddings["weather today"]],
            speaker_id="episode_manager",
            extraction_method="episodic_aggregation",
            confidence=0.8
        )
        
        assert node_id is not None, "Failed to create episodic node"
        
        # Verify episode structure
        node = await m3_integration.get_node_by_id(node_id)
        assert node['node_type'] == 'episodic'
        assert len(node['contents']) == 3
        
        logger.info(f"✅ Episodic node creation test passed (node_id: {node_id})")

class TestM3EdgeOperations:
    """Test M3 edge creation and graph relationships"""
    
    async def test_create_explicit_edge(self, m3_integration, test_embeddings):
        """Test explicit edge creation between nodes"""
        # Create two nodes
        node1_id = await m3_integration.store_m3_node(
            node_type="semantic",
            contents=["Machine learning"],
            embeddings=[test_embeddings["machine learning"]],
            speaker_id="test",
            extraction_method="test"
        )
        
        node2_id = await m3_integration.store_m3_node(
            node_type="semantic", 
            contents=["Artificial intelligence"],
            embeddings=[test_embeddings["artificial intelligence"]],
            speaker_id="test",
            extraction_method="test"
        )
        
        # Create edge between them
        success = await m3_integration.create_m3_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            weight=0.9,
            edge_type="similarity"
        )
        
        assert success, "Failed to create edge"
        
        # Verify edge exists
        edges_result = await m3_integration.connection.query("""
            SELECT * FROM m3_edges 
            WHERE source = (SELECT id FROM m3_nodes WHERE node_id = $node1_id)[0]
            AND target = (SELECT id FROM m3_nodes WHERE node_id = $node2_id)[0]
        """, {"node1_id": node1_id, "node2_id": node2_id})
        
        assert len(edges_result) > 0, "Edge not found in database"
        edge = edges_result[0]
        assert edge['weight'] == 0.9
        assert edge['edge_type'] == 'similarity'
        
        logger.info(f"✅ Edge creation test passed ({node1_id} -> {node2_id})")
    
    async def test_automatic_edge_inference(self, m3_integration, test_embeddings):
        """Test automatic edge inference based on similarity"""
        # Create related nodes
        node1_id = await m3_integration.store_m3_node(
            node_type="semantic",
            contents=["Hello world"],
            embeddings=[test_embeddings["hello world"]],
            speaker_id="test",
            extraction_method="test"
        )
        
        node2_id = await m3_integration.store_m3_node(
            node_type="semantic",
            contents=["Good morning"],
            embeddings=[test_embeddings["good morning"]],
            speaker_id="test", 
            extraction_method="test"
        )
        
        # Infer edges for first node
        edges_created = await m3_integration.infer_edges_for_node(
            node1_id,
            similarity_threshold=0.5,
            max_edges=3
        )
        
        assert edges_created > 0, "No edges were automatically created"
        
        logger.info(f"✅ Automatic edge inference test passed ({edges_created} edges created)")

class TestM3Clips:
    """Test M3 temporal clip management"""
    
    async def test_clip_creation(self, m3_integration):
        """Test creating and managing clips"""
        # Create new clip
        clip_id = await m3_integration.create_new_clip("test_session")
        assert clip_id is not None, "Failed to create clip"
        assert isinstance(clip_id, int), "Clip ID should be integer"
        
        # Verify clip exists
        clips = await m3_integration.connection.query(
            "SELECT * FROM m3_clips WHERE clip_id = $clip_id",
            {"clip_id": clip_id}
        )
        
        assert len(clips) > 0, "Clip not found in database"
        clip = clips[0]
        assert clip['session_id'] == 'test_session'
        assert clip['node_count'] == 0  # New clip should be empty
        
        logger.info(f"✅ Clip creation test passed (clip_id: {clip_id})")
    
    async def test_clip_node_association(self, m3_integration, test_embeddings):
        """Test associating nodes with clips"""
        # Create clip
        clip_id = await m3_integration.create_new_clip("test_session")
        
        # Create nodes in clip
        node_ids = []
        for i in range(3):
            node_id = await m3_integration.store_m3_node(
                node_type="voice",
                contents=[f"Test message {i}"],
                embeddings=[test_embeddings["hello world"]],
                clip_id=clip_id,
                speaker_id="test",
                extraction_method="test"
            )
            node_ids.append(node_id)
        
        # Get nodes in clip
        clip_nodes = await m3_integration.get_clip_nodes(clip_id)
        assert len(clip_nodes) == 3, f"Expected 3 nodes in clip, found {len(clip_nodes)}"
        
        # Verify all our nodes are in the clip
        clip_node_ids = [node['node_id'] for node in clip_nodes]
        for node_id in node_ids:
            assert node_id in clip_node_ids, f"Node {node_id} not found in clip"
        
        logger.info(f"✅ Clip node association test passed ({len(clip_nodes)} nodes in clip)")

class TestM3Search:
    """Test M3 similarity search and retrieval"""
    
    async def test_similarity_search(self, m3_integration, test_embeddings):
        """Test vector similarity search"""
        # Create diverse nodes
        test_data = [
            ("hello world", "voice", "greeting"),
            ("good morning", "voice", "greeting"),
            ("machine learning", "semantic", "tech"),
            ("artificial intelligence", "semantic", "tech"),
            ("weather today", "episodic", "weather")
        ]
        
        node_ids = []
        for content, node_type, category in test_data:
            node_id = await m3_integration.store_m3_node(
                node_type=node_type,
                contents=[content],
                embeddings=[test_embeddings[content]],
                speaker_id=category,  # Use category as speaker for testing
                extraction_method="test"
            )
            node_ids.append(node_id)
        
        # Search for greeting-related content
        similar_nodes = await m3_integration.search_similar_nodes(
            query_embedding=test_embeddings["hello world"],
            limit=5,
            min_similarity=0.1
        )
        
        assert len(similar_nodes) > 0, "No similar nodes found"
        
        # Verify results are ordered by similarity
        similarities = [node.get('similarity', 0) for node in similar_nodes]
        assert similarities == sorted(similarities, reverse=True), "Results not ordered by similarity"
        
        # Top result should be the exact match
        top_result = similar_nodes[0]
        assert "hello world" in top_result.get('contents', []), "Top result should be exact match"
        
        logger.info(f"✅ Similarity search test passed ({len(similar_nodes)} results)")
    
    async def test_filtered_search(self, m3_integration, test_embeddings):
        """Test similarity search with node type filtering"""
        # Create nodes of different types
        semantic_id = await m3_integration.store_m3_node(
            node_type="semantic",
            contents=["Machine learning concepts"],
            embeddings=[test_embeddings["machine learning"]],
            speaker_id="test",
            extraction_method="test"
        )
        
        voice_id = await m3_integration.store_m3_node(
            node_type="voice", 
            contents=["Machine learning discussion"],
            embeddings=[test_embeddings["machine learning"]],
            speaker_id="test",
            extraction_method="test"
        )
        
        # Search only for semantic nodes
        semantic_results = await m3_integration.search_similar_nodes(
            query_embedding=test_embeddings["machine learning"],
            node_type="semantic",
            limit=5
        )
        
        # All results should be semantic nodes
        for node in semantic_results:
            assert node.get('node_type') == 'semantic', "Non-semantic node in filtered results"
        
        logger.info(f"✅ Filtered search test passed ({len(semantic_results)} semantic results)")

class TestM3Equivalences:
    """Test M3 equivalence resolution system"""
    
    async def test_equivalence_creation(self, m3_integration, test_embeddings):
        """Test creating equivalence relations"""
        # Create nodes that refer to the same entity
        node_ids = []
        for content in ["John Smith", "John", "Mr. Smith"]:
            node_id = await m3_integration.store_m3_node(
                node_type="semantic",
                contents=[f"Information about {content}"],
                embeddings=[test_embeddings["hello world"]],  # Use consistent embedding
                speaker_id="test",
                extraction_method="test"
            )
            node_ids.append(node_id)
        
        # Create equivalence relation
        canonical_id = await m3_integration.resolve_equivalence(
            entity_name="John Smith",
            node_ids=node_ids,
            entity_type="person"
        )
        
        assert canonical_id == "John Smith", "Failed to resolve equivalence"
        
        # Verify equivalence exists
        equiv_result = await m3_integration.connection.query(
            "SELECT * FROM m3_equivalences WHERE canonical_id = $canonical_id",
            {"canonical_id": "John Smith"}
        )
        
        assert len(equiv_result) > 0, "Equivalence not found in database"
        equiv = equiv_result[0]
        assert len(equiv['node_ids']) == 3, "Not all nodes linked to equivalence"
        assert equiv['entity_type'] == 'person'
        
        logger.info(f"✅ Equivalence creation test passed (canonical_id: {canonical_id})")

class TestM3GraphTraversal:
    """Test M3 graph traversal and context retrieval"""
    
    async def test_graph_context_retrieval(self, m3_integration, test_embeddings):
        """Test getting graph context around a node"""
        # Create connected nodes
        central_node_id = await m3_integration.store_m3_node(
            node_type="semantic",
            contents=["Central topic"],
            embeddings=[test_embeddings["machine learning"]],
            speaker_id="test",
            extraction_method="test"
        )
        
        # Create connected nodes
        connected_ids = []
        for i in range(3):
            node_id = await m3_integration.store_m3_node(
                node_type="semantic",
                contents=[f"Related topic {i}"],
                embeddings=[test_embeddings["artificial intelligence"]],
                speaker_id="test",
                extraction_method="test"
            )
            connected_ids.append(node_id)
            
            # Create edges to central node
            await m3_integration.create_m3_edge(
                source_node_id=central_node_id,
                target_node_id=node_id,
                weight=0.8,
                edge_type="similarity"
            )
        
        # Get graph context
        context = await m3_integration.get_graph_context(
            node_id=central_node_id,
            max_depth=2,
            max_nodes=10
        )
        
        assert context is not None, "Failed to get graph context"
        assert context['central_node'] is not None, "Central node not in context"
        assert len(context['nodes']) > 0, "No connected nodes in context"
        assert len(context['edges']) > 0, "No edges in context"
        
        logger.info(f"✅ Graph context test passed ({len(context['nodes'])} nodes, {len(context['edges'])} edges)")

class TestM3Statistics:
    """Test M3 system statistics and monitoring"""
    
    async def test_system_statistics(self, m3_integration, test_embeddings):
        """Test getting comprehensive system statistics"""
        # Create some test data
        for i in range(5):
            await m3_integration.store_m3_node(
                node_type="semantic",
                contents=[f"Test content {i}"],
                embeddings=[test_embeddings["hello world"]],
                speaker_id="test",
                extraction_method="test"
            )
        
        # Get statistics
        stats = await m3_integration.get_statistics()
        
        assert stats is not None, "Failed to get statistics"
        assert 'total_nodes' in stats, "Missing total_nodes in statistics"
        assert 'total_edges' in stats, "Missing total_edges in statistics"
        assert 'total_clips' in stats, "Missing total_clips in statistics"
        
        # Verify we have at least our test nodes
        assert stats['total_nodes'] >= 5, f"Expected at least 5 nodes, got {stats['total_nodes']}"
        
        logger.info(f"✅ System statistics test passed: {stats}")

class TestM3Migration:
    """Test M3 migration script functionality"""
    
    async def test_migration_script_dry_run(self, test_connection):
        """Test migration script in dry-run mode"""
        migration = M3Migration(
            connection=test_connection,
            dry_run=True
        )
        
        # Test backup in dry-run mode
        backup_success = await migration.backup_existing_data()
        assert backup_success, "Dry-run backup failed"
        
        # Test schema migration in dry-run mode
        schema_success = await migration.apply_schema_migration()
        assert schema_success, "Dry-run schema migration failed"
        
        # Test verification in dry-run mode
        verify_success = await migration.verify_migration()
        assert verify_success, "Dry-run verification failed"
        
        logger.info("✅ Migration dry-run test passed")

class TestM3MemoryProcessor:
    """Test M3 memory processor frame handling"""
    
    @pytest.fixture
    async def m3_processor(self, m3_integration):
        """Create M3 memory processor for testing"""
        # Mock embedding service for testing
        class MockEmbeddingService:
            async def get_embedding(self, text: str) -> List[float]:
                # Simple hash-based embedding for testing
                hash_val = hash(text) % 1000000
                return [float(hash_val % 10) / 10.0 for _ in range(5)]
        
        processor = M3MemoryProcessor(
            m3_integration=m3_integration,
            embedding_service=MockEmbeddingService(),
            similarity_threshold=0.5,
            auto_create_edges=True,
            clip_duration_seconds=10
        )
        
        return processor
    
    async def test_processor_initialization(self, m3_processor):
        """Test M3 processor initializes correctly"""
        # Test processor was created
        assert m3_processor is not None
        assert m3_processor.m3_integration is not None
        assert m3_processor.embedding_service is not None
        
        logger.info("✅ M3 processor initialization test passed")

# Test configuration
@pytest.mark.asyncio
class TestM3IntegrationSuite:
    """Main test suite for M3 integration"""
    
    async def test_full_m3_workflow(self, m3_integration, test_embeddings):
        """Test complete M3 workflow from node creation to retrieval"""
        # 1. Create nodes of different types
        voice_id = await m3_integration.store_m3_node(
            node_type="voice",
            contents=["Hello, how are you today?"],
            embeddings=[test_embeddings["hello world"]],
            speaker_id="user1",
            extraction_method="transcription"
        )
        
        semantic_id = await m3_integration.store_m3_node(
            node_type="semantic", 
            contents=["Weather information", "It's sunny today"],
            embeddings=[test_embeddings["weather today"]],
            speaker_id="assistant",
            extraction_method="fact_extraction"
        )
        
        # 2. Create edges
        await m3_integration.create_m3_edge(voice_id, semantic_id, 0.7, "contextual")
        
        # 3. Test retrieval
        similar_nodes = await m3_integration.search_similar_nodes(
            test_embeddings["hello world"],
            limit=5
        )
        
        # 4. Test graph context
        context = await m3_integration.get_graph_context(voice_id)
        
        # 5. Test statistics
        stats = await m3_integration.get_statistics()
        
        # Verify workflow completed successfully
        assert voice_id is not None
        assert semantic_id is not None
        assert len(similar_nodes) > 0
        assert context['central_node'] is not None
        assert stats['total_nodes'] >= 2
        
        logger.info("✅ Full M3 workflow test passed")

if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v", "--tb=short"])