#!/usr/bin/env python3
"""
Sherpa-ONNX Configuration Optimizer

This script analyzes benchmark results and provides recommendations
for optimal sherpa-onnx configurations based on your specific needs.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationProfile:
    """Configuration optimization profile for different use cases"""
    name: str
    priority_weights: Dict[str, float]  # Weight for each metric
    constraints: Dict[str, Any]  # Hard constraints
    description: str

class SherpaOptimizer:
    """Analyzes benchmark results and suggests optimal configurations"""
    
    def __init__(self):
        self.optimization_profiles = self._create_optimization_profiles()
    
    def _create_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Create different optimization profiles for various use cases"""
        profiles = {}
        
        # Real-time conversation profile
        profiles['realtime'] = OptimizationProfile(
            name='Real-time Conversation',
            priority_weights={
                'real_time_factor': 0.4,  # Must be < 1.0 for real-time
                'time_to_first_token_ms': 0.3,  # Low latency important
                'word_error_rate': 0.2,  # Accuracy still matters
                'entity_accuracy': 0.1   # Less critical for conversation
            },
            constraints={
                'real_time_factor': {'max': 0.8},  # Must be well under real-time
                'time_to_first_token_ms': {'max': 500}  # 500ms max TTFT
            },
            description='Optimized for low-latency real-time conversation'
        )
        
        # High accuracy profile  
        profiles['accuracy'] = OptimizationProfile(
            name='High Accuracy',
            priority_weights={
                'word_error_rate': 0.4,  # Primary focus on accuracy
                'entity_accuracy': 0.3,  # Entities very important
                'character_error_rate': 0.2,  # Character-level accuracy
                'real_time_factor': 0.1   # Speed less important
            },
            constraints={
                'word_error_rate': {'max': 0.05},  # Max 5% WER
                'entity_accuracy': {'min': 0.9}   # Min 90% entity accuracy
            },
            description='Optimized for maximum transcription accuracy'
        )
        
        # Balanced profile
        profiles['balanced'] = OptimizationProfile(
            name='Balanced',
            priority_weights={
                'word_error_rate': 0.3,
                'real_time_factor': 0.25,
                'entity_accuracy': 0.25,
                'time_to_first_token_ms': 0.2
            },
            constraints={
                'real_time_factor': {'max': 1.2},  # Allow slightly slower than real-time
                'word_error_rate': {'max': 0.1}    # Max 10% WER
            },
            description='Balanced performance across accuracy and speed'
        )
        
        # Resource constrained profile
        profiles['lightweight'] = OptimizationProfile(
            name='Lightweight',
            priority_weights={
                'peak_memory_mb': 0.3,   # Memory usage important
                'peak_cpu_percent': 0.3, # CPU usage important
                'model_load_time_ms': 0.2, # Fast startup
                'real_time_factor': 0.2   # Still need real-time
            },
            constraints={
                'peak_memory_mb': {'max': 500},    # Max 500MB memory
                'peak_cpu_percent': {'max': 80},   # Max 80% CPU
                'real_time_factor': {'max': 1.0}   # Real-time required
            },
            description='Optimized for resource-constrained environments'
        )
        
        return profiles
    
    def load_benchmark_results(self, results_file: Path) -> List[Dict[str, Any]]:
        """Load benchmark results from JSON file"""
        with open(results_file) as f:
            return json.load(f)
    
    def calculate_scores(self, results: List[Dict[str, Any]], profile: OptimizationProfile) -> Dict[str, float]:
        """Calculate optimization scores for each model based on profile"""
        model_scores = {}
        
        # Group results by model
        model_results = {}
        for result in results:
            model_name = result['model_name']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        # Calculate scores for each model
        for model_name, model_data in model_results.items():
            if any(r.get('error_message') for r in model_data):
                model_scores[model_name] = 0.0  # Failed models get 0 score
                continue
            
            # Calculate average metrics across all test files
            avg_metrics = {}
            for metric in profile.priority_weights.keys():
                values = [r.get(metric, 0) for r in model_data if r.get(metric) is not None]
                if values:
                    avg_metrics[metric] = np.mean(values)
                else:
                    avg_metrics[metric] = 0
            
            # Check hard constraints
            meets_constraints = True
            for constraint_metric, constraint_values in profile.constraints.items():
                metric_value = avg_metrics.get(constraint_metric, 0)
                
                if 'max' in constraint_values and metric_value > constraint_values['max']:
                    meets_constraints = False
                    break
                if 'min' in constraint_values and metric_value < constraint_values['min']:
                    meets_constraints = False
                    break
            
            if not meets_constraints:
                model_scores[model_name] = 0.0
                continue
            
            # Calculate weighted score (normalize metrics first)
            score = 0.0
            for metric, weight in profile.priority_weights.items():
                metric_value = avg_metrics.get(metric, 0)
                
                # Normalize different metrics (lower is better for most)
                if metric in ['word_error_rate', 'character_error_rate', 'real_time_factor', 
                             'time_to_first_token_ms', 'model_load_time_ms',
                             'peak_memory_mb', 'peak_cpu_percent']:
                    # Lower is better - invert the score
                    normalized_score = max(0, 1.0 - metric_value)
                else:
                    # Higher is better (entity_accuracy, etc.)
                    normalized_score = min(1.0, metric_value)
                
                score += weight * normalized_score
            
            model_scores[model_name] = score
        
        return model_scores
    
    def get_recommendations(self, results_file: Path, profile_name: str = 'balanced') -> Dict[str, Any]:
        """Get optimization recommendations for a specific profile"""
        if profile_name not in self.optimization_profiles:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(self.optimization_profiles.keys())}")
        
        profile = self.optimization_profiles[profile_name]
        results = self.load_benchmark_results(results_file)
        scores = self.calculate_scores(results, profile)
        
        # Sort models by score
        ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get detailed metrics for top models
        top_models = {}
        model_results = {}
        for result in results:
            model_name = result['model_name']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        for model_name, score in ranked_models[:3]:  # Top 3 models
            model_data = model_results[model_name]
            if not any(r.get('error_message') for r in model_data):
                # Calculate averages
                avg_metrics = {}
                for key in ['word_error_rate', 'entity_accuracy', 'real_time_factor', 
                           'time_to_first_token_ms', 'model_load_time_ms',
                           'peak_memory_mb', 'peak_cpu_percent']:
                    values = [r.get(key, 0) for r in model_data if r.get(key) is not None]
                    if values:
                        avg_metrics[key] = np.mean(values)
                
                top_models[model_name] = {
                    'score': score,
                    'metrics': avg_metrics,
                    'num_tests': len(model_data)
                }
        
        recommendations = {
            'profile': profile.name,
            'description': profile.description,
            'ranked_models': ranked_models,
            'top_models': top_models,
            'analysis': self._generate_analysis(top_models, profile)
        }
        
        return recommendations
    
    def _generate_analysis(self, top_models: Dict[str, Any], profile: OptimizationProfile) -> str:
        """Generate human-readable analysis"""
        if not top_models:
            return "No models met the specified constraints."
        
        analysis = []
        best_model = list(top_models.keys())[0]
        best_metrics = top_models[best_model]['metrics']
        
        analysis.append(f"RECOMMENDED MODEL: {best_model}")
        analysis.append(f"Optimization Score: {top_models[best_model]['score']:.3f}")
        analysis.append("")
        
        analysis.append("Key Metrics:")
        analysis.append(f"  • Word Error Rate: {best_metrics.get('word_error_rate', 0):.3f}")
        analysis.append(f"  • Entity Accuracy: {best_metrics.get('entity_accuracy', 0):.3f}")
        analysis.append(f"  • Real-time Factor: {best_metrics.get('real_time_factor', 0):.3f}")
        analysis.append(f"  • Time to First Token: {best_metrics.get('time_to_first_token_ms', 0):.1f}ms")
        analysis.append(f"  • Model Load Time: {best_metrics.get('model_load_time_ms', 0):.1f}ms")
        analysis.append(f"  • Peak Memory: {best_metrics.get('peak_memory_mb', 0):.1f}MB")
        analysis.append("")
        
        # Add specific recommendations based on profile
        if profile.name == 'Real-time Conversation':
            rtf = best_metrics.get('real_time_factor', 1.0)
            if rtf > 0.8:
                analysis.append("⚠️  Real-time factor is high. Consider using a smaller model.")
            else:
                analysis.append("✅ Good real-time performance.")
        
        elif profile.name == 'High Accuracy':
            wer = best_metrics.get('word_error_rate', 1.0)
            if wer > 0.05:
                analysis.append("⚠️  Word error rate could be improved with post-processing.")
            else:
                analysis.append("✅ Excellent accuracy performance.")
        
        # Compare top models
        if len(top_models) > 1:
            analysis.append("")
            analysis.append("Alternative Models:")
            for i, (model_name, data) in enumerate(list(top_models.items())[1:3], 2):
                analysis.append(f"  {i}. {model_name} (Score: {data['score']:.3f})")
        
        return "\n".join(analysis)
    
    def generate_config_suggestions(self, recommended_model: str, use_case: str = 'balanced') -> Dict[str, Any]:
        """Generate specific configuration suggestions for the recommended model"""
        config_suggestions = {}
        
        # Base configuration
        config_suggestions['chunk_size_ms'] = 200  # Default
        config_suggestions['enable_endpoint_detection'] = True
        config_suggestions['emit_partial_results'] = False
        
        # Adjust based on use case
        if use_case == 'realtime':
            config_suggestions['chunk_size_ms'] = 160  # Smaller chunks for lower latency
            config_suggestions['max_active_paths'] = 3   # Fewer paths for speed
            config_suggestions['emit_partial_results'] = True  # Enable partial results
            
        elif use_case == 'accuracy':
            config_suggestions['chunk_size_ms'] = 250  # Larger chunks for better context
            config_suggestions['max_active_paths'] = 6   # More paths for accuracy
            config_suggestions['temperature_scale'] = 1.5  # More conservative decoding
            
        elif use_case == 'lightweight':
            config_suggestions['chunk_size_ms'] = 300  # Larger chunks to reduce processing
            config_suggestions['max_active_paths'] = 2   # Fewer paths to save memory
            config_suggestions['num_threads'] = 1       # Single thread
        
        return config_suggestions

def main():
    parser = argparse.ArgumentParser(description='Sherpa-ONNX Configuration Optimizer')
    parser.add_argument('results_file', type=Path, help='Benchmark results JSON file')
    parser.add_argument('--profile', 
                       choices=['realtime', 'accuracy', 'balanced', 'lightweight'],
                       default='balanced',
                       help='Optimization profile')
    parser.add_argument('--output', type=Path, help='Output file for recommendations')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        logger.error(f"Results file not found: {args.results_file}")
        return
    
    optimizer = SherpaOptimizer()
    recommendations = optimizer.get_recommendations(args.results_file, args.profile)
    
    # Print recommendations
    print("="*80)
    print(f"SHERPA-ONNX OPTIMIZATION RECOMMENDATIONS")
    print(f"Profile: {recommendations['profile']}")
    print("="*80)
    print(recommendations['analysis'])
    print()
    
    if recommendations['top_models']:
        best_model = list(recommendations['top_models'].keys())[0]
        config = optimizer.generate_config_suggestions(best_model, args.profile)
        
        print("SUGGESTED CONFIGURATION:")
        print("-" * 40)
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nDetailed recommendations saved to: {args.output}")

if __name__ == "__main__":
    main()