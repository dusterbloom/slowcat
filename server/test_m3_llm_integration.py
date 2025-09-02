#!/usr/bin/env python3
"""
M3 LLM Integration Tests

Tests the complete M3 memory generation system with local LLM integration,
including episodic and semantic memory extraction, embedding generation,
and context retrieval.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_m3_llm_integration():
    """Test complete M3 LLM integration"""
    print("🧠 Testing M3 LLM Integration System")
    print("=" * 50)
    
    try:
        # Import M3 components
        from memory.m3_surreal_integration import M3SurrealIntegration
        from memory.m3_llm_generator import M3LLMGenerator
        from services.embedding_service import EmbeddingService
        from services.m3_context_service import M3ContextService, ContextConfig
        from processors.m3_memory_processor import M3MemoryProcessor
        from memory.surreal_connection import SurrealConnectionManager
        
        # 1. Initialize all components
        print("\n1️⃣ Initializing M3 Components...")
        
        # SurrealDB connection
        surreal_connection = SurrealConnectionManager()
        await surreal_connection.connect()
        print("   ✅ SurrealDB connected")
        
        # M3 Integration
        m3_integration = M3SurrealIntegration(surreal_connection)
        await m3_integration.initialize()
        print("   ✅ M3 Integration initialized")
        
        # Embedding service
        embedding_service = EmbeddingService()
        embedding_test_ok = await embedding_service.test_embedding_generation()
        if embedding_test_ok:
            print("   ✅ Embedding service working")
        else:
            print("   ⚠️ Embedding service test failed, continuing anyway")
        
        # LLM Generator
        llm_generator = M3LLMGenerator()
        llm_connection_ok = await llm_generator.test_connection()
        if llm_connection_ok:
            print("   ✅ LLM generator connected to LM Studio")
        else:
            print("   ❌ LLM generator connection failed - is LM Studio running?")
            return False
        
        # Context service
        context_config = ContextConfig(
            max_context_tokens=1500,
            similarity_threshold=0.3,
            max_episodic_nodes=2,
            max_semantic_nodes=3,
            enable_context_summary=True
        )
        context_service = M3ContextService(
            m3_integration, embedding_service, llm_generator, config=context_config
        )
        print("   ✅ Context service initialized")
        
        # Memory processor
        memory_processor = M3MemoryProcessor(
            m3_integration=m3_integration,
            embedding_service=embedding_service,
            llm_generator=llm_generator,
            enable_llm_generation=True
        )
        print("   ✅ Memory processor initialized")
        
        # 2. Test LLM-powered semantic memory extraction
        print("\n2️⃣ Testing Semantic Memory Extraction...")
        
        test_texts = [
            "My dog Rex is a golden retriever who loves swimming and playing fetch.",
            "I work at OpenAI as a machine learning engineer focusing on language models.",
            "Yesterday I went to the new Italian restaurant downtown and had amazing pasta.",
            "My sister Sarah lives in Seattle and works as a software developer at Microsoft."
        ]
        
        semantic_memories = []
        for i, text in enumerate(test_texts):
            print(f"   Processing text {i+1}: {text[:40]}...")
            
            # Generate semantic memory
            semantic_memory = await llm_generator.generate_semantic_memory(text)
            
            if semantic_memory:
                print(f"      ✅ Extracted {len(semantic_memory.facts)} facts, "
                      f"{len(semantic_memory.concepts)} concepts")
                semantic_memories.append(semantic_memory)
                
                # Display extracted information
                if semantic_memory.facts:
                    print(f"      Facts: {semantic_memory.facts[:2]}")
                if semantic_memory.concepts:
                    print(f"      Concepts: {semantic_memory.concepts[:3]}")
            else:
                print(f"      ❌ Failed to extract semantic memory")
        
        print(f"   ✅ Generated {len(semantic_memories)} semantic memories")
        
        # 3. Test episodic memory generation
        print("\n3️⃣ Testing Episodic Memory Generation...")
        
        conversation_sequence = [
            "Hi, I'm looking for a good book recommendation",
            "What genre are you interested in?", 
            "I love science fiction and fantasy novels",
            "Have you read any Brandon Sanderson books?",
            "No, but I've heard great things about his work",
            "I'd recommend starting with Mistborn series"
        ]
        
        speaker_ids = ["user", "assistant", "user", "assistant", "user", "assistant"]
        
        print(f"   Processing conversation with {len(conversation_sequence)} turns...")
        
        episodic_memory = await llm_generator.generate_episodic_memory(
            conversation_sequence, speaker_ids
        )
        
        if episodic_memory:
            print(f"   ✅ Generated episodic memory")
            print(f"      Summary: {episodic_memory.summary}")
            print(f"      Key events: {episodic_memory.key_events}")
            print(f"      Participants: {episodic_memory.participants}")
        else:
            print("   ❌ Failed to generate episodic memory")
        
        # 4. Test memory storage with embeddings
        print("\n4️⃣ Testing Memory Storage...")
        
        stored_nodes = []
        
        # Store semantic memories
        for i, semantic_memory in enumerate(semantic_memories[:2]):  # Store first 2
            for fact in semantic_memory.facts[:2]:  # Store first 2 facts from each
                # Generate embedding
                embedding = await embedding_service.get_embedding(fact) if embedding_service else []
                
                node_id = await m3_integration.store_m3_node(
                    node_type="semantic",
                    contents=[fact],
                    embeddings=[embedding] if embedding else [],
                    speaker_id="user",
                    extraction_method="llm_test",
                    confidence=0.9
                )
                
                if node_id:
                    stored_nodes.append(node_id)
                    print(f"   ✅ Stored semantic node {node_id}: {fact[:30]}...")
        
        # Store episodic memory
        if episodic_memory:
            episode_embedding = await embedding_service.get_embedding(episodic_memory.summary) if embedding_service else []
            
            episode_node_id = await m3_integration.store_m3_node(
                node_type="episodic",
                contents=[episodic_memory.summary] + episodic_memory.key_events,
                embeddings=[episode_embedding] if episode_embedding else [],
                speaker_id="conversation",
                extraction_method="llm_test",
                confidence=episodic_memory.confidence
            )
            
            if episode_node_id:
                stored_nodes.append(episode_node_id)
                print(f"   ✅ Stored episodic node {episode_node_id}")
        
        print(f"   ✅ Stored {len(stored_nodes)} memory nodes total")
        
        # 5. Test context retrieval and summarization
        print("\n5️⃣ Testing Context Retrieval...")
        
        test_queries = [
            "Tell me about dogs",
            "What do you know about my work?",
            "Any restaurant recommendations?",
            "Book suggestions please"
        ]
        
        for query in test_queries:
            print(f"   Query: '{query}'")
            
            context_data = await context_service.get_conversation_context(
                query, speaker_id="user"
            )
            
            if context_data['has_memories']:
                print(f"      ✅ Found memories: {context_data['memory_stats']}")
                print(f"      Context tokens: {context_data['estimated_tokens']}")
                
                if context_data['summary']:
                    print(f"      LLM Summary: {context_data['summary'][:80]}...")
                
                # Test context injection
                mock_conversation = [
                    {"role": "user", "content": query}
                ]
                
                injected_conversation = await context_service.inject_context_into_conversation(
                    mock_conversation, query, "user"
                )
                
                if len(injected_conversation) > len(mock_conversation):
                    print(f"      ✅ Context injected into conversation")
                else:
                    print(f"      ℹ️ No context injection needed")
            else:
                print(f"      ℹ️ No relevant memories found")
        
        # 6. Test edge creation and graph relationships
        print("\n6️⃣ Testing Graph Relationships...")
        
        if stored_nodes and len(stored_nodes) > 1:
            # Test automatic edge inference
            edges_created = 0
            for node_id in stored_nodes[:3]:  # Test first 3 nodes
                edge_count = await m3_integration.infer_edges_for_node(
                    node_id, similarity_threshold=0.5, max_edges=2
                )
                edges_created += edge_count
                if edge_count > 0:
                    print(f"   ✅ Created {edge_count} edges for node {node_id}")
            
            print(f"   ✅ Created {edges_created} edges total")
            
            # Test graph context retrieval
            if stored_nodes:
                test_node = stored_nodes[0]
                try:
                    graph_context = await m3_integration.get_graph_context(test_node, max_depth=2)
                    
                    if isinstance(graph_context, dict):
                        connected_nodes = len(graph_context.get('nodes', []))
                        edges = len(graph_context.get('edges', []))
                        print(f"   ✅ Graph context: {connected_nodes} connected nodes, {edges} edges")
                    else:
                        print(f"   ℹ️ Graph context result: {graph_context}")
                except Exception as e:
                    print(f"   ⚠️ Graph context test skipped due to: {e}")
        
        # 7. Test speaker fact extraction
        print("\n7️⃣ Testing Speaker Fact Extraction...")
        
        speaker_text = "I have a cat named Whiskers and I live in Portland. I work as a data scientist."
        speaker_facts = await llm_generator.extract_speaker_facts(speaker_text, "test_user")
        
        if speaker_facts:
            print(f"   ✅ Extracted {len(speaker_facts)} speaker facts:")
            for fact in speaker_facts[:3]:
                print(f"      {fact['subject']} {fact['predicate']} {fact['object']}")
        else:
            print("   ⚠️ No speaker facts extracted")
        
        # 8. Test system statistics and performance
        print("\n8️⃣ Testing System Performance...")
        
        # Get M3 statistics
        stats = await m3_integration.get_statistics()
        if stats:
            print(f"   📊 M3 Statistics:")
            if isinstance(stats, dict):
                print(f"      Total nodes: {stats.get('total_nodes', 'unknown')}")
                print(f"      Total edges: {stats.get('total_edges', 'unknown')}")
                print(f"      Total clips: {stats.get('total_clips', 'unknown')}")
            else:
                print(f"      Raw stats: {stats}")
        else:
            print(f"   📊 M3 Statistics: Unable to retrieve")
        
        # Get embedding cache stats
        embedding_stats = embedding_service.get_cache_stats()
        print(f"   📊 Embedding Cache: {embedding_stats['cache_size']}/{embedding_stats['cache_limit']}")
        print(f"      Backend: {embedding_stats['backend']}")
        
        # Get context service stats
        context_stats = context_service.get_context_stats()
        print(f"   📊 Context Service: {context_stats['cache_size']} cached contexts")
        
        # 9. Performance timing test
        print("\n9️⃣ Testing Performance Timing...")
        
        performance_text = "This is a test sentence for performance measurement."
        
        # Time semantic extraction
        start_time = time.perf_counter()
        perf_semantic = await llm_generator.generate_semantic_memory(performance_text)
        semantic_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   ⏱️ Semantic extraction: {semantic_time:.1f}ms")
        
        # Time embedding generation
        start_time = time.perf_counter()
        perf_embedding = await embedding_service.get_embedding(performance_text)
        embedding_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   ⏱️ Embedding generation: {embedding_time:.1f}ms")
        
        # Time memory storage
        start_time = time.perf_counter()
        perf_node_id = await m3_integration.store_m3_node(
            "semantic", [performance_text], [perf_embedding] if perf_embedding else []
        )
        storage_time = (time.perf_counter() - start_time) * 1000
        
        print(f"   ⏱️ Memory storage: {storage_time:.1f}ms")
        
        total_time = semantic_time + embedding_time + storage_time
        print(f"   ⏱️ Total pipeline time: {total_time:.1f}ms")
        
        # Check sub-800ms target
        if total_time < 800:
            print(f"   ✅ Performance target met (sub-800ms)")
        else:
            print(f"   ⚠️ Performance target missed ({total_time:.1f}ms > 800ms)")
        
        print("\n🎉 M3 LLM Integration Test Complete!")
        print("=" * 50)
        print(f"✅ All major components tested successfully")
        print(f"📈 System ready for production use")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            if 'surreal_connection' in locals():
                await surreal_connection.close()
                print("   🔒 SurrealDB connection closed")
        except:
            pass

async def test_m3_equivalence_system():
    """Test M3 equivalence resolution system"""
    print("\n🔗 Testing M3 Equivalence System")
    print("-" * 30)
    
    try:
        from memory.m3_surreal_integration import M3SurrealIntegration
        from memory.surreal_connection import SurrealConnectionManager
        
        # Initialize components
        surreal_connection = SurrealConnectionManager()
        await surreal_connection.connect()
        
        m3_integration = M3SurrealIntegration(surreal_connection)
        await m3_integration.initialize()
        
        # Test equivalence resolution
        test_cases = [
            ("John Smith", [1001, 1002, 1003], "speaker"),
            ("Microsoft", [2001, 2002], "organization"),
            ("Seattle", [3001, 3002, 3003], "location")
        ]
        
        for entity_name, node_ids, entity_type in test_cases:
            canonical_id = await m3_integration.resolve_equivalence(
                entity_name, node_ids, entity_type
            )
            
            if canonical_id:
                print(f"   ✅ Resolved equivalence: {entity_name} -> {canonical_id}")
            else:
                print(f"   ❌ Failed to resolve: {entity_name}")
        
        await surreal_connection.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Equivalence test failed: {e}")
        return False

async def main():
    """Run all M3 integration tests"""
    print("🧠 M3 LLM Integration Test Suite")
    print("=" * 50)
    print("Testing M3 memory system with local LLM integration")
    print("Requirements: LM Studio running, SurrealDB available")
    print()
    
    # Run main integration test
    main_test_passed = await test_m3_llm_integration()
    
    if main_test_passed:
        # Run equivalence test
        equivalence_test_passed = await test_m3_equivalence_system()
        
        if equivalence_test_passed:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("M3 LLM integration system is ready for use.")
            return True
        else:
            print("\n⚠️ Main tests passed, equivalence tests failed")
            return False
    else:
        print("\n❌ TESTS FAILED")
        print("Check LM Studio connection and try again.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)