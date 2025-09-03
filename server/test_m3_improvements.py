#!/usr/bin/env python3
"""
Test script for M3 retrieval improvements
"""

import asyncio
import sys
import os
from loguru import logger

# Test the meta-query filtering
def test_meta_query_filtering():
    """Test meta-query filtering in fact extraction"""
    try:
        from memory.dspy_integration import _is_meta_query
        
        # Test cases that should be filtered
        meta_queries = [
            "continue talking from where we left off in the last session",
            "what were we discussing",
            "hello",
            "hi there",
            "last session",
            "resume conversation"
        ]
        
        # Test cases that should NOT be filtered
        factual_queries = [
            "my dog's name is Potola and she is a golden retriever",
            "I work at Google as a software engineer",
            "I live in San Francisco California"
        ]
        
        print("🧪 Testing meta-query filtering...")
        
        for query in meta_queries:
            result = _is_meta_query(query)
            print(f"   Meta: '{query}' -> {result} {'✅' if result else '❌'}")
            assert result, f"Should filter meta-query: {query}"
        
        for query in factual_queries:
            result = _is_meta_query(query)
            print(f"   Fact: '{query}' -> {result} {'❌' if result else '✅'}")
            assert not result, f"Should NOT filter factual query: {query}"
        
        print("✅ Meta-query filtering test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Meta-query filtering test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_last_session_detection():
    """Test last session query detection"""
    try:
        # We need to create a minimal M3ContextRetriever to test this
        from memory.m3_context_retriever import M3ContextRetriever
        
        # Mock the dependencies
        class MockM3Integration:
            pass
        class MockSimilaritySearch:
            pass
        class MockEquivalenceResolver:
            pass
        
        retriever = M3ContextRetriever(
            MockM3Integration(),
            MockSimilaritySearch(),
            MockEquivalenceResolver()
        )
        
        # Test cases that should be detected as last session queries
        last_session_queries = [
            "continue from where we left off in the last session",
            "what were we talking about last time",
            "continue talking from previous session",
            "where we left off"
        ]
        
        # Test cases that should NOT be detected
        other_queries = [
            "what is the weather",
            "my dog name is Potola",
            "I like pizza"
        ]
        
        print("🧪 Testing last session detection...")
        
        for query in last_session_queries:
            result = retriever._is_last_session_query(query)
            print(f"   Session: '{query}' -> {result} {'✅' if result else '❌'}")
            assert result, f"Should detect last session query: {query}"
        
        for query in other_queries:
            result = retriever._is_last_session_query(query)
            print(f"   Other: '{query}' -> {result} {'❌' if result else '✅'}")
            assert not result, f"Should NOT detect as last session query: {query}"
        
        print("✅ Last session detection test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Last session detection test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_configuration():
    """Test larger model configuration"""
    try:
        # Test that environment variables are properly read
        old_rel_model = os.environ.get('DSPY_REL_MODEL')
        old_facts_model = os.environ.get('DSPY_FACTS_MODEL')
        
        # Test with larger models
        os.environ['DSPY_REL_MODEL'] = 'qwen2.5-3b-instruct'
        os.environ['DSPY_FACTS_MODEL'] = 'qwen2.5-3b-instruct'
        
        from memory.dspy_single_call_extractor import DSPySingleCallExtractor
        extractor = DSPySingleCallExtractor()
        
        print("🧪 Testing model configuration...")
        print(f"   Relation model: {extractor.model_rel}")
        print(f"   Facts model: {extractor.model_facts}")
        
        assert extractor.model_rel == 'qwen2.5-3b-instruct'
        assert extractor.model_facts == 'qwen2.5-3b-instruct'
        
        # Restore old values
        if old_rel_model:
            os.environ['DSPY_REL_MODEL'] = old_rel_model
        else:
            os.environ.pop('DSPY_REL_MODEL', None)
            
        if old_facts_model:
            os.environ['DSPY_FACTS_MODEL'] = old_facts_model
        else:
            os.environ.pop('DSPY_FACTS_MODEL', None)
        
        print("✅ Model configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Model configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all improvement tests"""
    
    print("🚀 Testing M3 Retrieval System Improvements")
    print("=" * 50)
    
    tests = [
        test_meta_query_filtering,
        test_last_session_detection,
        test_model_configuration
    ]
    
    results = []
    for test_func in tests:
        print()
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print()
    print("📊 Test Results:")
    print(f"   Passed: {sum(results)}/{len(results)}")
    print(f"   Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All M3 improvements working correctly!")
        return True
    else:
        print("💥 Some M3 improvements have issues!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)