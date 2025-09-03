#!/usr/bin/env python3
"""
Real M3 Retrieval System Test with Actual Database
Tests the complete pipeline: retrieval + fact extraction + session awareness
"""

import asyncio
import os
import sys
import time
from typing import List, Dict, Any
from loguru import logger

# Import M3 components
from memory.m3_context_retriever import M3ContextRetriever, RetrievalStrategy, ContextType
from memory.m3_similarity_search import M3SimilaritySearch
from memory.m3_equivalence_resolver import M3EquivalenceResolver
from memory.surreal_connection import SurrealConnectionManager
from memory.m3_surreal_integration import M3SurrealIntegration
from services.embedding_service import EmbeddingService

class M3ConversationTester:
    """Test M3 retrieval system with real conversation scenarios"""
    
    def __init__(self):
        self.surreal_connection = None
        self.m3_integration = None
        self.m3_retriever = None
        self.current_speaker = "peppi"  # Use your actual speaker ID
        
    async def setup_m3_system(self):
        """Initialize M3 components with real database"""
        try:
            logger.info("🔧 Setting up M3 system for testing...")
            
            # Connect to SurrealDB using actual .env config
            self.surreal_connection = SurrealConnectionManager(
                url="ws://127.0.0.1:8000/rpc",
                namespace="slowcat",
                database="memory_graph"
            )
            
            await self.surreal_connection.connect()
            logger.info("✅ Connected to SurrealDB")
            
            # Initialize M3 integration
            self.m3_integration = M3SurrealIntegration(self.surreal_connection)
            await self.m3_integration.initialize()
            
            # Create M3 components
            similarity_search = M3SimilaritySearch(self.m3_integration)
            equivalence_resolver = M3EquivalenceResolver(self.m3_integration, similarity_search)
            
            # Initialize embedding service
            embedding_service = EmbeddingService()
            await embedding_service.test_embedding_generation()
            
            self.m3_retriever = M3ContextRetriever(
                self.m3_integration,
                similarity_search,
                equivalence_resolver,
                embedding_service=embedding_service
            )
            
            logger.info("✅ M3 system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup M3 system: {e}")
            return False
    
    async def get_real_conversation_queries(self) -> List[Dict[str, Any]]:
        """Get real conversation queries from the database"""
        try:
            # Get recent messages from the database
            messages_query = """
                SELECT content, role, speaker_id, timestamp, session_id
                FROM messages 
                WHERE speaker_id = $speaker_id 
                AND role = 'user'
                ORDER BY timestamp DESC 
                LIMIT 20
            """
            
            raw_messages = await self.m3_integration.query(messages_query, {
                "speaker_id": self.current_speaker
            })
            
            messages = self.m3_integration._normalize_query_result(raw_messages)
            
            # Create test scenarios from real data
            test_queries = []
            
            # Add some real queries if available
            for msg in messages[:5]:
                if isinstance(msg, dict) and msg.get('content'):
                    test_queries.append({
                        "query": msg['content'],
                        "description": f"Real query from {msg.get('session_id', 'unknown session')}",
                        "should_extract_facts": len(msg['content'].split()) > 3
                    })
            
            # Add our test scenarios
            test_scenarios = [
                {
                    "query": "continue talking from where we left off in the last session",
                    "description": "Last session continuation (should find previous session)",
                    "should_extract_facts": False
                },
                {
                    "query": "my dog's name is Potola and she is a golden retriever",
                    "description": "Pet information (should extract facts)",
                    "should_extract_facts": True
                },
                {
                    "query": "I work at Google as a software engineer", 
                    "description": "Job information (should extract facts)",
                    "should_extract_facts": True
                },
                {
                    "query": "what were we talking about before",
                    "description": "Context recall (should retrieve recent context)",
                    "should_extract_facts": False
                },
                {
                    "query": "I like pizza and Italian food",
                    "description": "Food preferences (should extract facts)",
                    "should_extract_facts": True
                }
            ]
            
            # Combine real and test queries
            test_queries.extend(test_scenarios)
            
            # Limit to 10 turns as requested
            return test_queries[:10]
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation queries: {e}")
            # Fallback to just test scenarios
            return [
                {"query": "continue from where we left off in the last session", "description": "Session continuation", "should_extract_facts": False},
                {"query": "my dog's name is Potola", "description": "Pet info", "should_extract_facts": True},
                {"query": "I work at Google", "description": "Job info", "should_extract_facts": True},
                {"query": "what were we discussing", "description": "Context recall", "should_extract_facts": False},
                {"query": "I like sushi", "description": "Food preference", "should_extract_facts": True},
                {"query": "hello how are you", "description": "Greeting", "should_extract_facts": False},
                {"query": "I live in San Francisco", "description": "Location info", "should_extract_facts": True},
                {"query": "resume our conversation", "description": "Continuation", "should_extract_facts": False},
                {"query": "my car is a Tesla Model 3", "description": "Vehicle info", "should_extract_facts": True},
                {"query": "what did we talk about last time", "description": "History recall", "should_extract_facts": False}
            ]
    
    async def test_retrieval_with_model(self, model_name: str, queries: List[Dict]) -> Dict[str, Any]:
        """Test M3 retrieval system with specific model configuration"""
        
        print(f"\n🧪 Testing with {model_name}")
        print("=" * 60)
        print(f"REL_MODEL: {os.getenv('DSPY_REL_MODEL', 'default')}")
        print(f"FACTS_MODEL: {os.getenv('DSPY_FACTS_MODEL', 'default')}")
        print()
        
        total_retrieval_time = 0
        total_extraction_time = 0
        successful_retrievals = 0
        total_facts_extracted = 0
        total_context_items = 0
        
        results = []
        
        for i, query_info in enumerate(queries, 1):
            query = query_info["query"]
            description = query_info["description"]
            should_extract = query_info["should_extract_facts"]
            
            print(f"🔍 Turn {i}/10: {description}")
            print(f"   Query: \"{query}\"")
            
            # Test M3 Context Retrieval
            retrieval_start = time.time()
            try:
                retrieved_context = await self.m3_retriever.retrieve_context(
                    query=query,
                    max_items=10,
                    strategy=RetrievalStrategy.HYBRID,
                    speaker_id=self.current_speaker
                )
                
                retrieval_time = time.time() - retrieval_start
                total_retrieval_time += retrieval_time
                
                context_items = len(retrieved_context.items)
                total_context_items += context_items
                
                print(f"   📊 Retrieved {context_items} context items in {retrieval_time:.2f}s")
                print(f"   🎯 Strategy: {retrieved_context.retrieval_strategy.value}")
                print(f"   ⭐ Total relevance: {retrieved_context.total_relevance:.2f}")
                
                # Show top context items
                for j, item in enumerate(retrieved_context.items[:3]):
                    content_preview = item.content[:50] + "..." if len(item.content) > 50 else item.content
                    print(f"      {j+1}. {content_preview} (score: {item.relevance_score:.2f})")
                
                if context_items > 0:
                    successful_retrievals += 1
                    
            except Exception as e:
                retrieval_time = time.time() - retrieval_start
                total_retrieval_time += retrieval_time
                print(f"   ❌ Retrieval failed: {e}")
                context_items = 0
            
            # Test Fact Extraction (if applicable)
            extraction_start = time.time()
            facts_extracted = 0
            
            if should_extract:
                try:
                    from memory.dspy_integration import extract_facts_from_text_dspy
                    facts = extract_facts_from_text_dspy(query)
                    
                    extraction_time = time.time() - extraction_start
                    total_extraction_time += extraction_time
                    
                    facts_extracted = len(facts)
                    total_facts_extracted += facts_extracted
                    
                    print(f"   🧠 Extracted {facts_extracted} facts in {extraction_time:.2f}s:")
                    for fact in facts:
                        subj = fact.get('subject', 'N/A')
                        pred = fact.get('predicate', 'N/A')
                        val = fact.get('value', 'N/A')
                        conf = fact.get('confidence', 0)
                        print(f"      • {subj} {pred} {val} (conf: {conf:.2f})")
                        
                except Exception as e:
                    extraction_time = time.time() - extraction_start
                    total_extraction_time += extraction_time
                    print(f"   ❌ Fact extraction failed: {e}")
            else:
                extraction_time = 0
                print(f"   🚫 Fact extraction skipped (meta-query)")
            
            results.append({
                'query': query,
                'description': description,
                'retrieval_time': retrieval_time,
                'extraction_time': extraction_time,
                'context_items': context_items,
                'facts_extracted': facts_extracted,
                'successful': context_items > 0 or not should_extract
            })
            
            print()
        
        # Calculate overall statistics
        avg_retrieval_time = total_retrieval_time / len(queries)
        avg_extraction_time = total_extraction_time / max(1, sum(1 for q in queries if q["should_extract_facts"]))
        success_rate = (successful_retrievals / len(queries)) * 100
        
        summary = {
            'model_name': model_name,
            'total_queries': len(queries),
            'successful_retrievals': successful_retrievals,
            'success_rate': success_rate,
            'avg_retrieval_time': avg_retrieval_time,
            'avg_extraction_time': avg_extraction_time,
            'total_context_items': total_context_items,
            'total_facts_extracted': total_facts_extracted,
            'avg_context_per_query': total_context_items / len(queries),
            'results': results
        }
        
        print(f"📊 {model_name} Summary:")
        print(f"   Success Rate: {successful_retrievals}/{len(queries)} ({success_rate:.1f}%)")
        print(f"   Avg Retrieval Time: {avg_retrieval_time:.2f}s")
        print(f"   Avg Extraction Time: {avg_extraction_time:.2f}s") 
        print(f"   Avg Context Items: {summary['avg_context_per_query']:.1f}")
        print(f"   Total Facts: {total_facts_extracted}")
        
        return summary
    
    async def test_chunk_processing_with_model(self, model_name: str, queries: List[Dict]) -> Dict[str, Any]:
        """Test M3 chunk processing with conversation buffering"""
        
        print(f"\n🧪 Testing with {model_name}")
        print("=" * 60)
        print(f"REL_MODEL: {os.getenv('DSPY_REL_MODEL', 'default')}")
        print(f"FACTS_MODEL: {os.getenv('DSPY_FACTS_MODEL', 'default')}")
        print()
        
        from memory.m3_conversation_buffer import M3ConversationBuffer
        from memory.dspy_integration import extract_facts_from_chunk_dspy
        
        # Create conversation buffer for M3-style processing
        buffer = M3ConversationBuffer(max_turns=5, max_seconds=30.0)
        
        total_retrieval_time = 0
        total_extraction_time = 0
        successful_retrievals = 0
        total_facts_extracted = 0
        total_context_items = 0
        chunks_processed = 0
        
        results = []
        
        for i, query_info in enumerate(queries, 1):
            query = query_info["query"]
            description = query_info["description"]
            should_extract = query_info["should_extract_facts"]
            
            print(f"🔍 Turn {i}/10: {description}")
            print(f"   Query: \"{query}\"")
            
            # Test M3 Context Retrieval (same as before)
            retrieval_start = time.time()
            try:
                retrieved_context = await self.m3_retriever.retrieve_context(
                    query=query,
                    max_items=10,
                    strategy=RetrievalStrategy.HYBRID,
                    speaker_id=self.current_speaker
                )
                
                retrieval_time = time.time() - retrieval_start
                total_retrieval_time += retrieval_time
                
                context_items = len(retrieved_context.items)
                total_context_items += context_items
                
                print(f"   📊 Retrieved {context_items} context items in {retrieval_time:.2f}s")
                print(f"   🎯 Strategy: {retrieved_context.retrieval_strategy.value}")
                print(f"   ⭐ Total relevance: {retrieved_context.total_relevance:.2f}")
                
                if context_items > 0:
                    successful_retrievals += 1
                    
            except Exception as e:
                retrieval_time = time.time() - retrieval_start
                total_retrieval_time += retrieval_time
                print(f"   ❌ Retrieval failed: {e}")
                context_items = 0
            
            # Add to conversation buffer
            should_process_chunk = buffer.add_turn(query, self.current_speaker) if should_extract else False
            facts_extracted = 0
            extraction_time = 0
            
            if should_process_chunk:
                # Process conversation chunk
                chunks_processed += 1
                extraction_start = time.time()
                
                try:
                    chunk = buffer.get_chunk()
                    if chunk:
                        print(f"   🔄 Processing chunk {chunks_processed} ({chunk.total_turns} turns)")
                        
                        # Get recent context for better extraction
                        recent_context = ""
                        if retrieved_context.items:
                            recent_context = " ".join([item.content[:50] for item in retrieved_context.items[:2]])
                        
                        facts = extract_facts_from_chunk_dspy(
                            chunk_text=chunk.combined_text,
                            previous_context=recent_context
                        )
                        
                        extraction_time = time.time() - extraction_start
                        total_extraction_time += extraction_time
                        
                        facts_extracted = len(facts)
                        total_facts_extracted += facts_extracted
                        
                        print(f"   🧠 Extracted {facts_extracted} facts from chunk in {extraction_time:.2f}s:")
                        for fact in facts:
                            subj = fact.get('subject', 'N/A')
                            pred = fact.get('predicate', 'N/A')
                            val = fact.get('value', 'N/A')
                            conf = fact.get('confidence', 0)
                            print(f"      • {subj} {pred} {val} (conf: {conf:.2f})")
                        
                        # Clear buffer after processing
                        buffer.clear_buffer()
                        
                except Exception as e:
                    extraction_time = time.time() - extraction_start
                    total_extraction_time += extraction_time
                    print(f"   ❌ Chunk extraction failed: {e}")
            else:
                if not should_extract:
                    print(f"   🚫 Fact extraction skipped (meta-query)")
                else:
                    print(f"   🔄 Added to buffer ({len(buffer.buffer)} turns)")
            
            results.append({
                'query': query,
                'description': description,
                'retrieval_time': retrieval_time,
                'extraction_time': extraction_time,
                'context_items': context_items,
                'facts_extracted': facts_extracted,
                'successful': context_items > 0 or not should_extract
            })
            
            print()
        
        # Process any remaining turns in buffer
        if buffer.buffer:
            chunks_processed += 1
            extraction_start = time.time()
            
            try:
                chunk = buffer.force_process()
                if chunk:
                    print(f"🔄 Final chunk {chunks_processed} ({chunk.total_turns} remaining turns)")
                    
                    facts = extract_facts_from_chunk_dspy(
                        chunk_text=chunk.combined_text,
                        previous_context=""
                    )
                    
                    extraction_time = time.time() - extraction_start
                    total_extraction_time += extraction_time
                    
                    facts_extracted = len(facts)
                    total_facts_extracted += facts_extracted
                    
                    print(f"🧠 Final chunk: {facts_extracted} facts in {extraction_time:.2f}s")
                    
            except Exception as e:
                print(f"❌ Final chunk extraction failed: {e}")
        
        # Calculate overall statistics
        avg_retrieval_time = total_retrieval_time / len(queries)
        extraction_queries = sum(1 for q in queries if q["should_extract_facts"])
        avg_extraction_time = total_extraction_time / max(1, chunks_processed)  # Per chunk, not per query
        success_rate = (successful_retrievals / len(queries)) * 100
        
        summary = {
            'model_name': model_name,
            'total_queries': len(queries),
            'successful_retrievals': successful_retrievals,
            'success_rate': success_rate,
            'avg_retrieval_time': avg_retrieval_time,
            'avg_extraction_time': avg_extraction_time,
            'total_context_items': total_context_items,
            'total_facts_extracted': total_facts_extracted,
            'chunks_processed': chunks_processed,
            'avg_context_per_query': total_context_items / len(queries),
            'facts_per_chunk': total_facts_extracted / max(1, chunks_processed),
            'results': results
        }
        
        print(f"📊 {model_name} Summary:")
        print(f"   Success Rate: {successful_retrievals}/{len(queries)} ({success_rate:.1f}%)")
        print(f"   Avg Retrieval Time: {avg_retrieval_time:.2f}s")
        print(f"   Chunks Processed: {chunks_processed}")
        print(f"   Avg Facts per Chunk: {summary['facts_per_chunk']:.1f}")
        print(f"   Total Facts: {total_facts_extracted}")
        
        return summary
    
    async def run_comparison_test(self):
        """Run the full comparison test between 0.5B and 4B models"""
        
        print("🚀 M3 Real Conversation System Test")
        print("=" * 80)
        
        # Setup M3 system
        if not await self.setup_m3_system():
            print("❌ Failed to setup M3 system - cannot run tests")
            return
        
        # Get test queries
        queries = await self.get_real_conversation_queries()
        print(f"📝 Testing with {len(queries)} conversation turns")
        
        # Test with current 4B models (our winners)
        os.environ['DSPY_REL_MODEL'] = 'qwen/qwen3-4b'
        os.environ['DSPY_FACTS_MODEL'] = 'qwen3-4b-instruct-2507'
        
        # Test 1: Individual message processing (old way)
        results_individual = await self.test_retrieval_with_model("Individual Processing (Old)", queries)
        
        # Test 2: Conversation chunk processing (M3 way)
        print("\n" + "=" * 80)
        print("🔄 Now testing with M3 conversation buffering...")
        results_chunk = await self.test_chunk_processing_with_model("M3 Chunk Processing (New)", queries)
        
        # Final comparison
        print("\n🏆 INDIVIDUAL vs CHUNK PROCESSING COMPARISON")
        print("=" * 80)
        print(f"{'Metric':<25} {'Individual (Old)':<20} {'Chunk M3 (New)':<20} {'Winner'}")
        print("-" * 80)
        
        metrics = [
            ('Success Rate', f"{results_individual['success_rate']:.1f}%", f"{results_chunk['success_rate']:.1f}%"),
            ('Avg Retrieval Time', f"{results_individual['avg_retrieval_time']:.2f}s", f"{results_chunk['avg_retrieval_time']:.2f}s"),
            ('Avg Extraction Time', f"{results_individual['avg_extraction_time']:.2f}s", f"{results_chunk['avg_extraction_time']:.2f}s"),
            ('Context Items/Query', f"{results_individual['avg_context_per_query']:.1f}", f"{results_chunk['avg_context_per_query']:.1f}"),
            ('Total Facts', f"{results_individual['total_facts_extracted']}", f"{results_chunk['total_facts_extracted']}"),
            ('Chunks Processed', "N/A (individual)", f"{results_chunk['chunks_processed']}"),
            ('Facts per Chunk', "N/A (individual)", f"{results_chunk['facts_per_chunk']:.1f}")
        ]
        
        for metric, val_individual, val_chunk in metrics:
            winner = self.determine_processing_winner(metric, results_individual, results_chunk)
            print(f"{metric:<25} {val_individual:<20} {val_chunk:<20} {winner}")
        
        # Analysis
        print("\n📊 ANALYSIS:")
        facts_improvement = results_chunk['total_facts_extracted'] - results_individual['total_facts_extracted']
        if facts_improvement > 0:
            improvement_pct = (facts_improvement / max(1, results_individual['total_facts_extracted'])) * 100
            print(f"✅ Chunk processing extracted {facts_improvement} more facts ({improvement_pct:.1f}% improvement)")
        elif facts_improvement == 0:
            print("⚡ Both approaches extracted the same number of facts")
        else:
            print(f"⚠️ Chunk processing extracted {abs(facts_improvement)} fewer facts")
        
        print(f"🔄 M3 principle: {results_chunk['chunks_processed']} chunks processed vs {len(queries)} individual messages")
        print(f"🧠 Contextual understanding: Facts extracted with conversation awareness")
        
        # Cleanup
        if self.surreal_connection:
            await self.surreal_connection.close()
    
    def determine_processing_winner(self, metric: str, results_individual: Dict, results_chunk: Dict) -> str:
        """Determine winner between individual vs chunk processing"""
        if metric == 'Success Rate':
            return "🥇 Chunk" if results_chunk['success_rate'] > results_individual['success_rate'] else "🥇 Individual" if results_individual['success_rate'] > results_chunk['success_rate'] else "🤝 Tie"
        elif 'Time' in metric:
            # For time metrics, lower is better
            individual_time = results_individual.get('avg_retrieval_time', 0) if 'Retrieval' in metric else results_individual.get('avg_extraction_time', 0)
            chunk_time = results_chunk.get('avg_retrieval_time', 0) if 'Retrieval' in metric else results_chunk.get('avg_extraction_time', 0)
            return "🥇 Individual" if individual_time < chunk_time else "🥇 Chunk" if chunk_time < individual_time else "🤝 Tie"
        elif metric == 'Total Facts':
            return "🥇 Chunk" if results_chunk['total_facts_extracted'] > results_individual['total_facts_extracted'] else "🥇 Individual" if results_individual['total_facts_extracted'] > results_chunk['total_facts_extracted'] else "🤝 Tie"
        elif 'Chunks' in metric or 'Facts per Chunk' in metric:
            return "🥇 M3 Advantage"  # These are unique to chunk processing
        else:
            # Context items per query
            return "🥇 Chunk" if results_chunk['avg_context_per_query'] > results_individual['avg_context_per_query'] else "🥇 Individual" if results_individual['avg_context_per_query'] > results_chunk['avg_context_per_query'] else "🤝 Tie"

    def determine_winner(self, metric: str, results_05b: Dict, results_4b: Dict) -> str:
        """Determine winner for each metric"""
        if metric == 'Success Rate':
            return "🥇 4B" if results_4b['success_rate'] > results_05b['success_rate'] else "🥇 0.5B" if results_05b['success_rate'] > results_4b['success_rate'] else "🤝 Tie"
        elif 'Time' in metric:
            return "🥇 0.5B" if results_05b['avg_retrieval_time'] < results_4b['avg_retrieval_time'] else "🥇 4B" if results_4b['avg_retrieval_time'] < results_05b['avg_retrieval_time'] else "🤝 Tie"
        elif metric == 'Total Facts':
            return "🥇 4B" if results_4b['total_facts_extracted'] > results_05b['total_facts_extracted'] else "🥇 0.5B" if results_05b['total_facts_extracted'] > results_4b['total_facts_extracted'] else "🤝 Tie"
        else:
            return "🥇 4B" if results_4b['avg_context_per_query'] > results_05b['avg_context_per_query'] else "🥇 0.5B" if results_05b['avg_context_per_query'] > results_4b['avg_context_per_query'] else "🤝 Tie"

async def main():
    """Run the comprehensive M3 test"""
    tester = M3ConversationTester()
    await tester.run_comparison_test()

if __name__ == "__main__":
    asyncio.run(main())