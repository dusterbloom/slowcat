#!/usr/bin/env python3
"""
Comprehensive test suite for Advanced Accuracy Enhancement

Tests the accuracy enhancer against realistic STT errors including:
- Names and proper nouns
- URLs and domains  
- Technical terms
- Phone numbers and emails
- Multi-language scenarios

Compares performance with and without enhancement.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from dataclasses import asdict

# Import our advanced enhancer
from advanced_accuracy_enhancer import AdvancedAccuracyEnhancer, CorrectionResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test cases representing common STT errors
ACCURACY_TEST_CASES = [
    # Names and proper nouns
    {
        "original": "Hello my name is john smith and I work at goggle",
        "expected": "Hello my name is John Smith and I work at Google",
        "confidence": 0.8,
        "category": "names_companies",
        "description": "Proper noun capitalization and company name correction"
    },
    {
        "original": "I need to call marry johnson about the meeting with micro soft",
        "expected": "I need to call Mary Johnson about the meeting with Microsoft", 
        "confidence": 0.75,
        "category": "names_companies",
        "description": "Multiple name/company corrections"
    },
    
    # URLs and domains
    {
        "original": "Please visit bbb dot com slash news for more information",
        "expected": "Please visit bbb.com/news for more information",
        "confidence": 0.6,
        "category": "urls_domains", 
        "description": "URL format reconstruction"
    },
    {
        "original": "The article is at tech crunch dot com",
        "expected": "The article is at techcrunch.com",
        "confidence": 0.7,
        "category": "urls_domains",
        "description": "Domain name correction"
    },
    {
        "original": "Check out get hub dot com for the source code",
        "expected": "Check out github.com for the source code",
        "confidence": 0.65,
        "category": "urls_domains", 
        "description": "Popular domain correction"
    },
    
    # Technical terms
    {
        "original": "I'm using react j s with type script for the front end",
        "expected": "I'm using React.js with TypeScript for the frontend",
        "confidence": 0.8,
        "category": "technical_terms",
        "description": "Technical framework names"
    },
    {
        "original": "The dock her container is running on kuba net is",
        "expected": "The Docker container is running on Kubernetes",
        "confidence": 0.5,
        "category": "technical_terms",
        "description": "Cloud technology terms"  
    },
    
    # Contact information
    {
        "original": "My email is john dot smith at gee mail dot com",
        "expected": "My email is john.smith@gmail.com",
        "confidence": 0.7,
        "category": "contact_info",
        "description": "Email address reconstruction"
    },
    {
        "original": "Call me at five five five one two three four five six seven",
        "expected": "Call me at 555-123-4567", 
        "confidence": 0.6,
        "category": "contact_info",
        "description": "Phone number formatting"
    },
    
    # Mixed complexity
    {
        "original": "I work at amazon web services on the cloud formation team contact me at john at a w s dot com",
        "expected": "I work at Amazon Web Services on the CloudFormation team contact me at john@aws.com",
        "confidence": 0.4,
        "category": "complex_mixed",
        "description": "Complex technical + contact correction"
    },
    {
        "original": "The api endpoint is h t t p s colon slash slash a p i dot face book dot com slash graph",
        "expected": "The API endpoint is https://api.facebook.com/graph",
        "confidence": 0.3,
        "category": "complex_mixed", 
        "description": "Full URL reconstruction"
    },
    
    # Edge cases
    {
        "original": "I visited San Francisco last week",
        "expected": "I visited San Francisco last week",
        "confidence": 0.95,
        "category": "high_confidence",
        "description": "Should not change high confidence correct text"
    },
    {
        "original": "The quick brown fox jumps over the lazy dog",
        "expected": "The quick brown fox jumps over the lazy dog",
        "confidence": 0.98,
        "category": "high_confidence", 
        "description": "Common phrase should remain unchanged"
    }
]

class AccuracyTestSuite:
    """Comprehensive accuracy testing suite"""
    
    def __init__(self):
        self.enhancer = AdvancedAccuracyEnhancer()
        self.results = []
        
    async def run_comprehensive_test(self) -> Dict:
        """Run all accuracy tests and return results"""
        logger.info("Starting comprehensive accuracy enhancement tests...")
        
        total_start_time = time.time()
        category_results = {}
        
        for i, test_case in enumerate(ACCURACY_TEST_CASES):
            logger.info(f"Running test {i+1}/{len(ACCURACY_TEST_CASES)}: {test_case['description']}")
            
            # Test the enhancement
            result = await self.enhancer.enhance_accuracy(
                text=test_case['original'],
                confidence=test_case['confidence'],
                context="Test conversation"
            )
            
            # Analyze results
            test_result = self._analyze_test_result(test_case, result)
            self.results.append(test_result)
            
            # Group by category
            category = test_case['category']
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(test_result)
        
        total_time = time.time() - total_start_time
        
        # Generate comprehensive report
        report = self._generate_test_report(category_results, total_time)
        
        # Save results
        results_file = Path(__file__).parent / "results" / f"accuracy_test_{int(time.time())}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'test_results': [asdict(result.enhancement_result) for result in self.results],
                'analysis': report,
                'enhancer_stats': self.enhancer.get_stats()
            }, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        return report
    
    def _analyze_test_result(self, test_case: Dict, result: CorrectionResult) -> 'TestResult':
        """Analyze individual test result"""
        
        # Calculate accuracy metrics
        original_text = test_case['original']
        expected_text = test_case['expected'] 
        enhanced_text = result.corrected_text
        
        # Exact match score
        exact_match = enhanced_text.lower() == expected_text.lower()
        
        # Word-level accuracy using simple matching
        original_words = set(original_text.lower().split())
        expected_words = set(expected_text.lower().split())
        enhanced_words = set(enhanced_text.lower().split())
        
        # Calculate improvement
        original_correct = len(original_words & expected_words)
        enhanced_correct = len(enhanced_words & expected_words)
        total_expected = len(expected_words)
        
        original_accuracy = original_correct / total_expected if total_expected > 0 else 1.0
        enhanced_accuracy = enhanced_correct / total_expected if total_expected > 0 else 1.0
        
        improvement = enhanced_accuracy - original_accuracy
        
        return TestResult(
            test_case=test_case,
            enhancement_result=result,
            exact_match=exact_match,
            original_accuracy=original_accuracy,
            enhanced_accuracy=enhanced_accuracy,
            accuracy_improvement=improvement,
            processing_time_ms=result.processing_time_ms,
            corrections_applied=len(result.corrections_applied)
        )
    
    def _generate_test_report(self, category_results: Dict, total_time: float) -> Dict:
        """Generate comprehensive test report"""
        
        report = {
            'overall_stats': {
                'total_tests': len(self.results),
                'total_time_sec': total_time,
                'avg_processing_time_ms': sum(r.processing_time_ms for r in self.results) / len(self.results),
                'total_corrections': sum(r.corrections_applied for r in self.results)
            },
            'accuracy_metrics': {
                'exact_matches': sum(1 for r in self.results if r.exact_match),
                'exact_match_rate': sum(1 for r in self.results if r.exact_match) / len(self.results),
                'avg_original_accuracy': sum(r.original_accuracy for r in self.results) / len(self.results),
                'avg_enhanced_accuracy': sum(r.enhanced_accuracy for r in self.results) / len(self.results),
                'avg_improvement': sum(r.accuracy_improvement for r in self.results) / len(self.results),
                'tests_improved': sum(1 for r in self.results if r.accuracy_improvement > 0),
                'tests_degraded': sum(1 for r in self.results if r.accuracy_improvement < 0)
            },
            'category_breakdown': {}
        }
        
        # Category-specific analysis
        for category, results in category_results.items():
            category_stats = {
                'test_count': len(results),
                'exact_match_rate': sum(1 for r in results if r.exact_match) / len(results),
                'avg_improvement': sum(r.accuracy_improvement for r in results) / len(results),
                'avg_processing_time': sum(r.processing_time_ms for r in results) / len(results),
                'corrections_per_test': sum(r.corrections_applied for r in results) / len(results),
                'best_case': max(results, key=lambda r: r.accuracy_improvement),
                'worst_case': min(results, key=lambda r: r.accuracy_improvement)
            }
            
            report['category_breakdown'][category] = category_stats
        
        return report

from dataclasses import dataclass

@dataclass 
class TestResult:
    test_case: Dict
    enhancement_result: CorrectionResult
    exact_match: bool
    original_accuracy: float
    enhanced_accuracy: float  
    accuracy_improvement: float
    processing_time_ms: float
    corrections_applied: int

async def benchmark_performance():
    """Benchmark the performance impact of accuracy enhancement"""
    logger.info("Running performance benchmark...")
    
    # Test with varying text lengths
    test_texts = [
        "Hello john",  # Short
        "I work at goggle and need to visit face book dot com",  # Medium  
        "Please contact john smith at john dot smith at gee mail dot com about the project at get hub dot com slash user slash repo",  # Long
        " ".join(["This is a very long text with many words"] * 10)  # Very long
    ]
    
    enhancer = AdvancedAccuracyEnhancer()
    
    for i, text in enumerate(test_texts):
        iterations = 50 if len(text) < 100 else 10
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            result = await enhancer.enhance_accuracy(text, confidence=0.6)
            elapsed = (time.time() - start_time) * 1000
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"Text length {len(text):3d} chars: {avg_time:6.1f}ms avg ({min_time:5.1f}-{max_time:5.1f}ms)")

async def test_language_agnostic():
    """Test language-agnostic capabilities"""
    logger.info("Testing language-agnostic correction...")
    
    # These should work even without language-specific models
    test_cases = [
        "contact jean pierre at jean dot pierre at gee mail dot com",  # French name
        "visit stack overflow dot com for help",  # Technical site
        "my phone is three one zero five five five one two three four",  # Numbers
    ]
    
    enhancer = AdvancedAccuracyEnhancer()
    
    for text in test_cases:
        result = await enhancer.enhance_accuracy(text, confidence=0.5)
        logger.info(f"Input:  {text}")
        logger.info(f"Output: {result.corrected_text}")
        logger.info(f"Method: {result.method_used}")
        logger.info("")

def print_test_report(report: Dict):
    """Print formatted test report"""
    print("\n" + "="*80)
    print("ADVANCED ACCURACY ENHANCEMENT TEST RESULTS")
    print("="*80)
    
    overall = report['overall_stats']
    accuracy = report['accuracy_metrics']
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total Tests:          {overall['total_tests']}")
    print(f"  Total Time:           {overall['total_time_sec']:.2f}s")
    print(f"  Avg Processing:       {overall['avg_processing_time_ms']:.1f}ms per test")
    print(f"  Total Corrections:    {overall['total_corrections']}")
    
    print(f"\n🎯 ACCURACY METRICS")
    print(f"  Exact Matches:        {accuracy['exact_matches']}/{overall['total_tests']} ({accuracy['exact_match_rate']:.1%})")
    print(f"  Original Accuracy:    {accuracy['avg_original_accuracy']:.1%}")
    print(f"  Enhanced Accuracy:    {accuracy['avg_enhanced_accuracy']:.1%}")
    print(f"  Average Improvement:  {accuracy['avg_improvement']:+.1%}")
    print(f"  Tests Improved:       {accuracy['tests_improved']}")
    print(f"  Tests Degraded:       {accuracy['tests_degraded']}")
    
    print(f"\n📂 CATEGORY BREAKDOWN")
    for category, stats in report['category_breakdown'].items():
        print(f"\n  {category.upper().replace('_', ' ')}")
        print(f"    Tests:              {stats['test_count']}")
        print(f"    Exact Match Rate:   {stats['exact_match_rate']:.1%}")
        print(f"    Avg Improvement:    {stats['avg_improvement']:+.1%}")
        print(f"    Avg Time:           {stats['avg_processing_time']:.1f}ms")
        print(f"    Corrections/Test:   {stats['corrections_per_test']:.1f}")

async def main():
    """Main test runner"""
    print("🚀 Starting Advanced Accuracy Enhancement Tests...")
    
    # Run comprehensive accuracy test
    test_suite = AccuracyTestSuite()
    report = await test_suite.run_comprehensive_test()
    
    # Print results
    print_test_report(report)
    
    # Run performance benchmark
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)
    await benchmark_performance()
    
    # Test language-agnostic features
    print("\n" + "="*80) 
    print("LANGUAGE-AGNOSTIC TEST")
    print("="*80)
    await test_language_agnostic()
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())