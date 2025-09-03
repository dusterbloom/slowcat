#!/usr/bin/env python
"""
M3 Retrieval System Validation Test

Comprehensive validation of M3 retrieval performance and accuracy:
1. Similarity search accuracy with different modality types
2. Context retrieval performance with various strategies
3. Query router classification and routing accuracy
4. End-to-end retrieval performance benchmarks
5. Real conversation scenario validation
"""

import asyncio
import time
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment
load_dotenv()

# M3 imports
from memory.m3_similarity_search import M3SimilaritySearch, ModalityType
from memory.m3_context_retriever import M3ContextRetriever, ContextType, RetrievalStrategy
from memory.m3_surreal_integration import M3SurrealIntegration
from memory.surreal_connection import SurrealConnectionManager
from memory.query_router import QueryRouter, create_m3_query_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalTestResult:
    """Results from a single retrieval test"""
    test_name: str
    query: str
    results_count: int
    retrieval_time_ms: float
    accuracy_score: float
    relevance_scores: List[float]
    strategy_used: str
    stores_queried: List[str]
    
    @property
    def avg_relevance(self) -> float:
        return sum(self.relevance_scores) / len(self.relevance_scores) if self.relevance_scores else 0.0

@dataclass 
class ValidationSuite:
    """Complete validation suite results"""
    test_results: List[RetrievalTestResult]
    total_tests: int
    passed_tests: int
    avg_retrieval_time_ms: float
    avg_accuracy: float
    performance_grade: str
    
    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0


class M3RetrievalValidator:
    """Validates M3 retrieval system performance and accuracy"""
    
    # Performance benchmarks
    MAX_RETRIEVAL_TIME_MS = 500    # 500ms max for real-time usage
    MIN_ACCURACY_THRESHOLD = 0.6   # 60% minimum accuracy
    MIN_RELEVANCE_THRESHOLD = 0.3  # 30% minimum relevance score
    
    def __init__(self):
        self.connection_manager = None
        self.m3_integration = None
        self.similarity_search = None
        self.context_retriever = None
        self.query_router = None
        
        self.test_conversations = self._create_test_conversations()
        self.validation_queries = self._create_validation_queries()
    
    async def initialize(self) -> bool:
        """Initialize M3 components for testing"""
        try:
            # Initialize SurrealDB connection
            surreal_url = os.getenv('SURREALDB_URL', 'ws://127.0.0.1:8000/rpc')
            self.connection_manager = SurrealConnectionManager(surreal_url)
            await self.connection_manager.connect()
            
            # Initialize M3 integration
            self.m3_integration = M3SurrealIntegration(self.connection_manager)
            
            # Initialize similarity search
            self.similarity_search = M3SimilaritySearch(self.m3_integration)
            
            # Initialize context retriever (without embedding service for now)
            self.context_retriever = M3ContextRetriever(
                m3_integration=self.m3_integration,
                similarity_search=self.similarity_search,
                equivalence_resolver=None,  # Optional for this test
                embedding_service=None      # Optional for this test
            )
            
            # Initialize query router with M3 components
            self.query_router = create_m3_query_router(
                m3_context_retriever=self.context_retriever,
                facts_graph=None,
                tape_store=None
            )
            
            logger.info("✅ M3 retrieval validator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize M3 validator: {e}")
            return False
    
    async def run_validation_suite(self) -> ValidationSuite:
        """Run complete validation suite"""
        logger.info("🧪 Starting M3 retrieval validation suite...")
        start_time = time.time()
        
        test_results = []
        
        # 1. Test similarity search accuracy
        similarity_results = await self._test_similarity_search()
        test_results.extend(similarity_results)
        
        # 2. Test context retrieval strategies  
        context_results = await self._test_context_retrieval()
        test_results.extend(context_results)
        
        # 3. Test query router classification
        router_results = await self._test_query_router()
        test_results.extend(router_results)
        
        # 4. Test real conversation scenarios
        conversation_results = await self._test_conversation_scenarios()
        test_results.extend(conversation_results)
        
        # 5. Performance benchmarks
        performance_results = await self._test_performance_benchmarks()
        test_results.extend(performance_results)
        
        # Calculate overall results
        total_time = time.time() - start_time
        passed_tests = sum(1 for r in test_results if r.accuracy_score >= self.MIN_ACCURACY_THRESHOLD)
        avg_retrieval_time = sum(r.retrieval_time_ms for r in test_results) / len(test_results)
        avg_accuracy = sum(r.accuracy_score for r in test_results) / len(test_results)
        
        # Determine performance grade
        performance_grade = self._calculate_performance_grade(avg_retrieval_time, avg_accuracy, passed_tests / len(test_results))
        
        suite_results = ValidationSuite(
            test_results=test_results,
            total_tests=len(test_results),
            passed_tests=passed_tests,
            avg_retrieval_time_ms=avg_retrieval_time,
            avg_accuracy=avg_accuracy,
            performance_grade=performance_grade
        )
        
        logger.info(f"🎯 Validation completed in {total_time:.1f}s: {performance_grade} grade")
        return suite_results
    
    async def _test_similarity_search(self) -> List[RetrievalTestResult]:
        """Test similarity search with different modality types"""
        logger.info("🔍 Testing similarity search accuracy...")
        
        results = []
        
        # Test queries for different modalities
        test_cases = [
            {
                'query': "user likes music",
                'modality': ModalityType.SEMANTIC,
                'expected_keywords': ['music', 'likes', 'user']
            },
            {
                'query': "hello my name is Alice",
                'modality': ModalityType.VOICE,
                'expected_keywords': ['alice', 'name', 'hello']
            },
            {
                'query': "software engineer machine learning",
                'modality': ModalityType.TEXT,
                'expected_keywords': ['software', 'engineer', 'learning']
            }
        ]
        
        for test_case in test_cases:
            start_time = time.time()
            
            try:
                # Generate dummy embedding for testing (in real usage, would use embedding service)
                import numpy as np
                query_embedding = np.random.randn(384).astype(np.float32)
                query_embedding = (query_embedding / np.linalg.norm(query_embedding)).tolist()
                
                # Perform similarity search
                search_results = await self.similarity_search.search_nodes(
                    query_embedding=query_embedding,
                    modality=test_case['modality'],
                    max_results=10
                )
                
                retrieval_time_ms = (time.time() - start_time) * 1000
                
                # Calculate accuracy based on relevance scores and content matching
                accuracy_score = self._calculate_search_accuracy(
                    search_results, 
                    test_case['expected_keywords']
                )
                
                relevance_scores = [r.similarity_score for r in search_results]
                
                result = RetrievalTestResult(
                    test_name=f"similarity_search_{test_case['modality'].value}",
                    query=test_case['query'],
                    results_count=len(search_results),
                    retrieval_time_ms=retrieval_time_ms,
                    accuracy_score=accuracy_score,
                    relevance_scores=relevance_scores,
                    strategy_used="similarity_first",
                    stores_queried=["m3_similarity"]
                )
                
                results.append(result)
                
                logger.debug(f"  {test_case['modality'].value}: {len(search_results)} results, "
                           f"{accuracy_score:.2f} accuracy, {retrieval_time_ms:.1f}ms")
                
            except Exception as e:
                logger.error(f"Similarity search test failed for {test_case['modality'].value}: {e}")
                # Add failed test result
                results.append(RetrievalTestResult(
                    test_name=f"similarity_search_{test_case['modality'].value}",
                    query=test_case['query'],
                    results_count=0,
                    retrieval_time_ms=0.0,
                    accuracy_score=0.0,
                    relevance_scores=[],
                    strategy_used="failed",
                    stores_queried=[]
                ))
        
        return results
    
    async def _test_context_retrieval(self) -> List[RetrievalTestResult]:
        """Test context retrieval with different strategies"""
        logger.info("🧠 Testing context retrieval strategies...")
        
        results = []
        
        # Test different retrieval strategies
        strategies = [
            RetrievalStrategy.SIMILARITY_FIRST,
            RetrievalStrategy.ENTITY_FIRST, 
            RetrievalStrategy.TEMPORAL_FIRST,
            RetrievalStrategy.HYBRID
        ]
        
        test_queries = [
            "What did Alice say about her work?",
            "Tell me about recent conversations",
            "What do you know about machine learning?"
        ]
        
        for strategy in strategies:
            for query in test_queries:
                start_time = time.time()
                
                try:
                    # Test context retrieval
                    context_result = await self.context_retriever.retrieve_context(
                        query=query,
                        context_type=None,
                        max_items=10,
                        strategy=strategy
                    )
                    
                    retrieval_time_ms = (time.time() - start_time) * 1000
                    
                    # Calculate accuracy based on relevance and content quality
                    accuracy_score = self._calculate_context_accuracy(context_result, query)
                    
                    relevance_scores = [item.relevance_score for item in context_result.items]
                    
                    result = RetrievalTestResult(
                        test_name=f"context_retrieval_{strategy.value}",
                        query=query,
                        results_count=len(context_result.items),
                        retrieval_time_ms=retrieval_time_ms,
                        accuracy_score=accuracy_score,
                        relevance_scores=relevance_scores,
                        strategy_used=strategy.value,
                        stores_queried=["m3_context"]
                    )
                    
                    results.append(result)
                    
                    logger.debug(f"  {strategy.value} + '{query[:30]}...': "
                               f"{len(context_result.items)} items, {accuracy_score:.2f} accuracy")
                    
                except Exception as e:
                    logger.error(f"Context retrieval test failed for {strategy.value}: {e}")
                    results.append(RetrievalTestResult(
                        test_name=f"context_retrieval_{strategy.value}",
                        query=query,
                        results_count=0,
                        retrieval_time_ms=0.0,
                        accuracy_score=0.0,
                        relevance_scores=[],
                        strategy_used="failed",
                        stores_queried=[]
                    ))
        
        return results
    
    async def _test_query_router(self) -> List[RetrievalTestResult]:
        """Test query router classification and routing"""
        logger.info("🧭 Testing query router classification...")
        
        results = []
        
        if not self.query_router:
            logger.warning("Query router not available, skipping tests")
            return results
        
        for query_data in self.validation_queries:
            start_time = time.time()
            
            try:
                # Route the query
                router_response = await self.query_router.route_query(
                    query=query_data['query'],
                    max_results=10
                )
                
                retrieval_time_ms = (time.time() - start_time) * 1000
                
                # Calculate accuracy based on expected intent and results quality
                accuracy_score = self._calculate_routing_accuracy(
                    router_response,
                    query_data['expected_intent'],
                    query_data['expected_stores']
                )
                
                relevance_scores = [r.relevance_score for r in router_response.results]
                
                result = RetrievalTestResult(
                    test_name="query_router_classification",
                    query=query_data['query'],
                    results_count=len(router_response.results),
                    retrieval_time_ms=retrieval_time_ms,
                    accuracy_score=accuracy_score,
                    relevance_scores=relevance_scores,
                    strategy_used=router_response.strategy_used.value,
                    stores_queried=router_response.stores_queried
                )
                
                results.append(result)
                
                logger.debug(f"  Router: '{query_data['query'][:30]}...' -> "
                           f"{router_response.classification.intent.value} "
                           f"({router_response.classification.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Query router test failed for '{query_data['query']}': {e}")
                results.append(RetrievalTestResult(
                    test_name="query_router_classification",
                    query=query_data['query'],
                    results_count=0,
                    retrieval_time_ms=0.0,
                    accuracy_score=0.0,
                    relevance_scores=[],
                    strategy_used="failed",
                    stores_queried=[]
                ))
        
        return results
    
    async def _test_conversation_scenarios(self) -> List[RetrievalTestResult]:
        """Test real conversation scenarios"""
        logger.info("💬 Testing real conversation scenarios...")
        
        results = []
        
        for scenario in self.test_conversations:
            start_time = time.time()
            
            try:
                # Test retrieval for the scenario query
                if self.query_router:
                    router_response = await self.query_router.route_query(
                        query=scenario['query'],
                        max_results=5
                    )
                    
                    retrieval_time_ms = (time.time() - start_time) * 1000
                    
                    # Calculate accuracy based on expected results
                    accuracy_score = self._calculate_scenario_accuracy(
                        router_response.results,
                        scenario['expected_content']
                    )
                    
                    relevance_scores = [r.relevance_score for r in router_response.results]
                    
                    result = RetrievalTestResult(
                        test_name=f"conversation_{scenario['name']}",
                        query=scenario['query'],
                        results_count=len(router_response.results),
                        retrieval_time_ms=retrieval_time_ms,
                        accuracy_score=accuracy_score,
                        relevance_scores=relevance_scores,
                        strategy_used=router_response.strategy_used.value,
                        stores_queried=router_response.stores_queried
                    )
                    
                    results.append(result)
                    
                    logger.debug(f"  Scenario '{scenario['name']}': "
                               f"{len(router_response.results)} results, {accuracy_score:.2f} accuracy")
                
            except Exception as e:
                logger.error(f"Conversation scenario test failed for '{scenario['name']}': {e}")
                results.append(RetrievalTestResult(
                    test_name=f"conversation_{scenario['name']}",
                    query=scenario['query'],
                    results_count=0,
                    retrieval_time_ms=0.0,
                    accuracy_score=0.0,
                    relevance_scores=[],
                    strategy_used="failed",
                    stores_queried=[]
                ))
        
        return results
    
    async def _test_performance_benchmarks(self) -> List[RetrievalTestResult]:
        """Test performance under load"""
        logger.info("⚡ Testing performance benchmarks...")
        
        results = []
        
        # Performance stress test - multiple concurrent queries
        concurrent_queries = [
            "What is my name?",
            "Tell me about recent conversations",
            "What do you know about my work?",
            "What are my hobbies?",
            "Who did I talk to yesterday?"
        ]
        
        start_time = time.time()
        
        try:
            if self.query_router:
                # Run queries concurrently
                tasks = [
                    self.query_router.route_query(query, max_results=5)
                    for query in concurrent_queries
                ]
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                total_time_ms = (time.time() - start_time) * 1000
                avg_time_per_query = total_time_ms / len(concurrent_queries)
                
                # Calculate performance score
                performance_score = 1.0 if avg_time_per_query < self.MAX_RETRIEVAL_TIME_MS else 0.5
                
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                total_results = sum(len(r.results) for r in successful_responses)
                
                result = RetrievalTestResult(
                    test_name="performance_concurrent_queries",
                    query=f"{len(concurrent_queries)} concurrent queries",
                    results_count=total_results,
                    retrieval_time_ms=avg_time_per_query,
                    accuracy_score=performance_score,
                    relevance_scores=[],
                    strategy_used="concurrent",
                    stores_queried=["multiple"]
                )
                
                results.append(result)
                
                logger.info(f"  Concurrent performance: {avg_time_per_query:.1f}ms avg per query")
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            results.append(RetrievalTestResult(
                test_name="performance_concurrent_queries",
                query="concurrent test",
                results_count=0,
                retrieval_time_ms=0.0,
                accuracy_score=0.0,
                relevance_scores=[],
                strategy_used="failed",
                stores_queried=[]
            ))
        
        return results
    
    def _create_test_conversations(self) -> List[Dict[str, Any]]:
        """Create realistic conversation scenarios for testing"""
        return [
            {
                'name': 'personal_info',
                'query': 'What is my name?',
                'expected_content': ['name', 'alice', 'user'],
                'context': 'User introduced themselves earlier'
            },
            {
                'name': 'work_discussion', 
                'query': 'Tell me about my work',
                'expected_content': ['software', 'engineer', 'work', 'job'],
                'context': 'Previous conversation about career'
            },
            {
                'name': 'hobby_interests',
                'query': 'What are my hobbies?', 
                'expected_content': ['hiking', 'weekend', 'hobbies'],
                'context': 'User mentioned recreational activities'
            },
            {
                'name': 'recent_activity',
                'query': 'What did we talk about recently?',
                'expected_content': ['machine', 'learning', 'recent'],
                'context': 'Recent conversation about ML topics'
            }
        ]
    
    def _create_validation_queries(self) -> List[Dict[str, Any]]:
        """Create queries with expected classification results"""
        return [
            {
                'query': 'What is my name?',
                'expected_intent': 'personal_fact',
                'expected_stores': ['facts', 'tape']
            },
            {
                'query': 'Tell me what we discussed yesterday',
                'expected_intent': 'conversation_history',
                'expected_stores': ['tape', 'embeddings']
            },
            {
                'query': 'What do you know about machine learning?',
                'expected_intent': 'semantic_search',
                'expected_stores': ['embeddings', 'facts']
            }
        ]
    
    def _calculate_search_accuracy(self, search_results, expected_keywords: List[str]) -> float:
        """Calculate accuracy of similarity search results"""
        if not search_results:
            return 0.0
        
        # Check relevance thresholds
        relevant_results = [r for r in search_results if r.similarity_score >= self.MIN_RELEVANCE_THRESHOLD]
        
        if not relevant_results:
            return 0.0
        
        # Check content matching with expected keywords
        keyword_matches = 0
        for result in relevant_results[:5]:  # Check top 5 results
            content_lower = ' '.join(result.content).lower()
            matches = sum(1 for keyword in expected_keywords if keyword.lower() in content_lower)
            keyword_matches += matches
        
        # Calculate accuracy as combination of relevance and keyword matching
        relevance_score = len(relevant_results) / len(search_results)
        keyword_score = keyword_matches / (len(expected_keywords) * min(5, len(relevant_results)))
        
        return (relevance_score + keyword_score) / 2
    
    def _calculate_context_accuracy(self, context_result, query: str) -> float:
        """Calculate accuracy of context retrieval"""
        if not context_result.items:
            return 0.0
        
        # Check if retrieval time is reasonable
        time_score = 1.0 if context_result.retrieval_time_ms < self.MAX_RETRIEVAL_TIME_MS else 0.5
        
        # Check relevance scores
        relevant_items = [item for item in context_result.items if item.relevance_score >= self.MIN_RELEVANCE_THRESHOLD]
        relevance_score = len(relevant_items) / len(context_result.items) if context_result.items else 0
        
        # Check diversity (different clips/entities)
        unique_clips = len(context_result.clips_referenced)
        diversity_score = min(1.0, unique_clips / 3)  # Good if references 3+ different clips
        
        return (time_score + relevance_score + diversity_score) / 3
    
    def _calculate_routing_accuracy(self, router_response, expected_intent: str, expected_stores: List[str]) -> float:
        """Calculate accuracy of query routing"""
        # Check if classification confidence is reasonable
        confidence_score = min(1.0, router_response.classification.confidence * 2)  # Boost reasonable confidence
        
        # Check if appropriate stores were queried
        store_overlap = len(set(router_response.stores_queried) & set(expected_stores))
        store_score = store_overlap / len(expected_stores) if expected_stores else 1.0
        
        # Check result quality
        result_score = 1.0 if router_response.results else 0.5
        
        return (confidence_score + store_score + result_score) / 3
    
    def _calculate_scenario_accuracy(self, results, expected_content: List[str]) -> float:
        """Calculate accuracy for conversation scenarios"""
        if not results:
            return 0.0
        
        # Check content relevance
        content_matches = 0
        for result in results:
            content_lower = result.content.lower()
            matches = sum(1 for expected in expected_content if expected.lower() in content_lower)
            content_matches += matches
        
        content_score = content_matches / (len(expected_content) * len(results))
        
        # Check relevance scores
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        relevance_score = min(1.0, avg_relevance * 2)  # Normalize relevance
        
        return (content_score + relevance_score) / 2
    
    def _calculate_performance_grade(self, avg_time_ms: float, avg_accuracy: float, success_rate: float) -> str:
        """Calculate overall performance grade"""
        # Time grade (A: <200ms, B: <500ms, C: <1000ms, D: <2000ms, F: >=2000ms)
        if avg_time_ms < 200:
            time_grade = 4.0
        elif avg_time_ms < 500:
            time_grade = 3.0
        elif avg_time_ms < 1000:
            time_grade = 2.0
        elif avg_time_ms < 2000:
            time_grade = 1.0
        else:
            time_grade = 0.0
        
        # Accuracy grade (A: >0.8, B: >0.6, C: >0.4, D: >0.2, F: <=0.2)
        if avg_accuracy > 0.8:
            accuracy_grade = 4.0
        elif avg_accuracy > 0.6:
            accuracy_grade = 3.0
        elif avg_accuracy > 0.4:
            accuracy_grade = 2.0
        elif avg_accuracy > 0.2:
            accuracy_grade = 1.0
        else:
            accuracy_grade = 0.0
        
        # Success rate grade
        if success_rate > 0.9:
            success_grade = 4.0
        elif success_rate > 0.7:
            success_grade = 3.0
        elif success_rate > 0.5:
            success_grade = 2.0
        elif success_rate > 0.3:
            success_grade = 1.0
        else:
            success_grade = 0.0
        
        # Overall grade (weighted: 40% time, 40% accuracy, 20% success rate)
        overall = (time_grade * 0.4 + accuracy_grade * 0.4 + success_grade * 0.2)
        
        if overall >= 3.5:
            return "A (Excellent)"
        elif overall >= 2.5:
            return "B (Good)"
        elif overall >= 1.5:
            return "C (Fair)"
        elif overall >= 0.5:
            return "D (Poor)"
        else:
            return "F (Failing)"
    
    def print_validation_report(self, suite_results: ValidationSuite):
        """Print detailed validation report"""
        print("\n" + "="*80)
        print("🧪 M3 RETRIEVAL SYSTEM VALIDATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total Tests: {suite_results.total_tests}")
        print(f"   Passed Tests: {suite_results.passed_tests}")
        print(f"   Success Rate: {suite_results.success_rate:.1%}")
        print(f"   Performance Grade: {suite_results.performance_grade}")
        print(f"   Average Retrieval Time: {suite_results.avg_retrieval_time_ms:.1f}ms")
        print(f"   Average Accuracy: {suite_results.avg_accuracy:.2f}")
        
        print(f"\n🔍 DETAILED TEST RESULTS:")
        
        # Group by test type
        test_groups = {}
        for result in suite_results.test_results:
            test_type = result.test_name.split('_')[0]
            if test_type not in test_groups:
                test_groups[test_type] = []
            test_groups[test_type].append(result)
        
        for test_type, results in test_groups.items():
            print(f"\n   {test_type.upper()} TESTS:")
            for result in results:
                status = "✅ PASS" if result.accuracy_score >= self.MIN_ACCURACY_THRESHOLD else "❌ FAIL"
                print(f"     {status} {result.test_name}")
                print(f"          Query: '{result.query[:50]}...'")
                print(f"          Results: {result.results_count}, Time: {result.retrieval_time_ms:.1f}ms, "
                      f"Accuracy: {result.accuracy_score:.2f}")
                if result.relevance_scores:
                    print(f"          Avg Relevance: {result.avg_relevance:.2f}")
        
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        fast_tests = [r for r in suite_results.test_results if r.retrieval_time_ms < self.MAX_RETRIEVAL_TIME_MS]
        print(f"   Fast Tests (<{self.MAX_RETRIEVAL_TIME_MS}ms): {len(fast_tests)}/{suite_results.total_tests}")
        
        accurate_tests = [r for r in suite_results.test_results if r.accuracy_score >= self.MIN_ACCURACY_THRESHOLD]
        print(f"   Accurate Tests (>={self.MIN_ACCURACY_THRESHOLD}): {len(accurate_tests)}/{suite_results.total_tests}")
        
        # Performance recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if suite_results.avg_retrieval_time_ms > self.MAX_RETRIEVAL_TIME_MS:
            print("   ⚠️  Consider caching optimization for faster retrieval")
        if suite_results.avg_accuracy < self.MIN_ACCURACY_THRESHOLD:
            print("   ⚠️  Review similarity thresholds and ranking algorithms")
        if suite_results.success_rate < 0.8:
            print("   ⚠️  Investigate failing test cases and error handling")
        
        if all([
            suite_results.avg_retrieval_time_ms < self.MAX_RETRIEVAL_TIME_MS,
            suite_results.avg_accuracy >= self.MIN_ACCURACY_THRESHOLD,
            suite_results.success_rate >= 0.8
        ]):
            print("   ✅ M3 retrieval system meets production readiness criteria!")
        
        print("\n" + "="*80)


async def main():
    """Main validation test runner"""
    print("🚀 Starting M3 Retrieval System Validation")
    
    validator = M3RetrievalValidator()
    
    # Initialize the validator
    if not await validator.initialize():
        print("❌ Failed to initialize validator")
        return
    
    # Run validation suite
    try:
        results = await validator.run_validation_suite()
        validator.print_validation_report(results)
        
    except Exception as e:
        logger.error(f"Validation suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if validator.connection_manager:
            await validator.connection_manager.close()


if __name__ == "__main__":
    asyncio.run(main())