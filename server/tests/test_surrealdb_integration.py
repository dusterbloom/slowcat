#!/usr/bin/env python3
"""
SurrealDB Integration Tests

Comprehensive test suite for the SurrealDB memory system integration.
Tests authentication, data persistence, memory system compatibility, and performance.
"""

import asyncio
import os
import sys
import time
import pytest
from pathlib import Path

# Add the parent directory to the path so we can import from server
sys.path.insert(0, str(Path(__file__).parent.parent))

# Only run if SurrealDB is available
try:
    from surrealdb import AsyncSurreal
    SURREALDB_AVAILABLE = True
except ImportError:
    SURREALDB_AVAILABLE = False

@pytest.mark.skipif(not SURREALDB_AVAILABLE, reason="SurrealDB client not available")
class TestSurrealDBIntegration:
    """Comprehensive SurrealDB integration tests"""
    
    @pytest.fixture
    async def surreal_memory(self):
        """Create a test SurrealDB memory instance"""
        from memory.surreal_memory import SurrealMemory
        
        memory = SurrealMemory(
            surreal_url="ws://localhost:8000/rpc",
            namespace="test",
            database="memory"
        )
        
        try:
            await memory.connect()
            yield memory
        finally:
            await memory.close()
    
    @pytest.mark.asyncio
    async def test_authentication(self):
        """Test SurrealDB authentication works correctly"""
        from memory.surreal_memory import SurrealMemory
        
        memory = SurrealMemory()
        await memory.connect()
        
        assert memory.connected is True
        await memory.close()
    
    @pytest.mark.asyncio
    async def test_fact_operations(self, surreal_memory):
        """Test fact insertion, search, and retrieval"""
        memory = surreal_memory
        
        # Test fact insertion
        test_fact = {
            'subject': 'test_user',
            'predicate': 'preference',
            'value': 'coffee',
            'fidelity': 4,
            'source_text': 'User mentioned they prefer coffee'
        }
        
        result = await memory.reinforce_or_insert(test_fact)
        assert result is False  # New insertion should return False
        
        # Test fact search
        facts = await memory.search_facts('coffee')
        assert len(facts) >= 1
        
        found_fact = facts[0]
        assert found_fact['subject'] == 'test_user'
        assert found_fact['predicate'] == 'preference'
        assert found_fact['value'] == 'coffee'
        
        # Test reinforcement
        result = await memory.reinforce_or_insert(test_fact)
        assert result is True  # Reinforcement should return True
        
        # Test top facts retrieval
        top_facts = await memory.get_top_facts(limit=5)
        assert len(top_facts) >= 1
    
    @pytest.mark.asyncio
    async def test_tape_operations(self, surreal_memory):
        """Test conversation tape operations"""
        memory = surreal_memory
        
        # Test tape entry addition
        await memory.add_entry(
            role='user',
            content='Hello, this is a test message for SurrealDB',
            speaker_id='test_user'
        )
        
        await memory.add_entry(
            role='assistant',
            content='Hello! I understand you are testing SurrealDB integration.',
            speaker_id='test_user'
        )
        
        # Test recent entries retrieval
        recent = await memory.get_recent(limit=5)
        assert len(recent) >= 2
        
        # Verify entries are in correct order (most recent first)
        assert recent[0]['role'] == 'assistant'
        assert recent[1]['role'] == 'user'
        
        # Test tape search
        search_results = await memory.search_tape('SurrealDB', limit=5)
        assert len(search_results) >= 1
        
        found_entry = search_results[0]
        assert 'SurrealDB' in found_entry['content']
    
    @pytest.mark.asyncio
    async def test_memory_system_compatibility(self):
        """Test compatibility with existing memory system interfaces"""
        # Set environment for SurrealDB
        os.environ['USE_SURREALDB'] = 'true'
        
        try:
            from memory import create_smart_memory_system
            
            memory_system = create_smart_memory_system()
            
            # Verify we got the SurrealDB adapter
            assert 'SurrealMemorySystemAdapter' in str(type(memory_system))
            
            # Test facts_graph interface
            await memory_system.facts_graph.reinforce_or_insert({
                'subject': 'compatibility_test',
                'predicate': 'status',
                'value': 'working',
                'fidelity': 3
            })
            
            facts = await memory_system.facts_graph.search_facts('compatibility_test')
            assert len(facts) >= 1
            
            # Test tape_store interface
            await memory_system.tape_store.add_entry(
                role='user',
                content='Testing compatibility interface',
                speaker_id='compatibility_test'
            )
            
            recent = await memory_system.tape_store.get_recent(limit=3)
            assert len(recent) >= 1
            
        finally:
            # Clean up environment
            if 'USE_SURREALDB' in os.environ:
                del os.environ['USE_SURREALDB']
    
    @pytest.mark.asyncio
    async def test_performance_characteristics(self, surreal_memory):
        """Test that SurrealDB operations meet performance requirements"""
        memory = surreal_memory
        
        # Test fact insertion performance
        start_time = time.time()
        for i in range(10):
            await memory.reinforce_or_insert({
                'subject': f'perf_test_{i}',
                'predicate': 'iteration',
                'value': str(i),
                'fidelity': 3
            })
        fact_insert_time = time.time() - start_time
        
        # Should complete 10 fact insertions in under 2 seconds
        assert fact_insert_time < 2.0
        
        # Test fact search performance
        start_time = time.time()
        for i in range(5):
            results = await memory.search_facts('perf_test')
        fact_search_time = time.time() - start_time
        
        # Should complete 5 searches in under 1 second
        assert fact_search_time < 1.0
        
        # Test tape operations performance
        start_time = time.time()
        for i in range(10):
            await memory.add_entry(
                role='user',
                content=f'Performance test message {i}',
                speaker_id='perf_test'
            )
        tape_insert_time = time.time() - start_time
        
        # Should complete 10 tape insertions in under 2 seconds
        assert tape_insert_time < 2.0
    
    @pytest.mark.asyncio
    async def test_data_persistence(self):
        """Test that data persists across connections"""
        # Create first connection and insert data
        memory1 = None
        try:
            from memory.surreal_memory import SurrealMemory
            
            memory1 = SurrealMemory(namespace="persistence_test", database="memory")
            await memory1.connect()
            
            # Insert unique test data
            test_id = f"persistence_test_{int(time.time())}"
            await memory1.reinforce_or_insert({
                'subject': test_id,
                'predicate': 'test',
                'value': 'persistent_data',
                'fidelity': 4
            })
            
            await memory1.close()
            
            # Create second connection and verify data exists
            memory2 = SurrealMemory(namespace="persistence_test", database="memory")
            await memory2.connect()
            
            facts = await memory2.search_facts(test_id)
            assert len(facts) >= 1
            
            found_fact = facts[0]
            assert found_fact['subject'] == test_id
            assert found_fact['value'] == 'persistent_data'
            
            await memory2.close()
            
        except Exception as e:
            if memory1:
                await memory1.close()
            raise e


def run_integration_tests():
    """Run SurrealDB integration tests"""
    if not SURREALDB_AVAILABLE:
        print("❌ SurrealDB client not available. Install with: pip install surrealdb")
        return False
    
    # Check if SurrealDB server is running
    import subprocess
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:8000/health'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            print("❌ SurrealDB server not running. Start with: ./scripts/start_surrealdb.sh")
            return False
    except subprocess.TimeoutExpired:
        print("❌ SurrealDB server not responding")
        return False
    
    print("🚀 Running SurrealDB integration tests...")
    
    # Run tests with pytest
    import subprocess
    import sys
    
    test_file = __file__
    result = subprocess.run([sys.executable, '-m', 'pytest', test_file, '-v'], 
                          cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print("✅ All SurrealDB integration tests passed!")
        return True
    else:
        print("❌ Some SurrealDB integration tests failed")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)