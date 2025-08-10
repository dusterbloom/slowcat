#!/usr/bin/env python3
"""
Analyze Sherpa-ONNX Stress Test Results

Generate analysis and visualization of stress test results to identify:
1. Performance trends with longer audio
2. Memory usage patterns
3. CPU utilization patterns
4. Stability indicators
"""

import json
import argparse
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

def analyze_stress_results(results_file: Path):
    """Analyze stress test results from JSON file"""
    
    with open(results_file) as f:
        results = json.load(f)
    
    if not results:
        print("No results found in file")
        return
    
    print("="*80)
    print("SHERPA-ONNX STRESS TEST ANALYSIS")
    print("="*80)
    
    # Sort by duration for analysis
    results.sort(key=lambda x: x['audio_duration_sec'])
    
    # Performance Analysis
    print("\n📊 PERFORMANCE SCALING")
    print("-" * 40)
    
    durations = [r['audio_duration_sec'] for r in results]
    rtfs = [r['real_time_factor'] for r in results]
    
    for i, result in enumerate(results):
        duration = result['audio_duration_sec']
        rtf = result['real_time_factor']
        processing_time = result['total_processing_time_sec']
        
        print(f"{duration:6.1f}s: RTF={rtf:.3f} ({processing_time:.2f}s processing)")
    
    # Check for performance degradation
    if len(rtfs) > 1:
        rtf_trend = np.polyfit(durations, rtfs, 1)[0]  # Linear trend coefficient
        if abs(rtf_trend) > 0.001:
            trend_desc = "degrading" if rtf_trend > 0 else "improving"
            print(f"\n⚠️  Performance trend: {trend_desc} ({rtf_trend:+.6f} RTF per second)")
        else:
            print(f"\n✅ Performance remains stable across durations")
    
    # Memory Analysis
    print("\n🧠 MEMORY USAGE")
    print("-" * 40)
    
    for result in results:
        duration = result['audio_duration_sec']
        initial = result['initial_memory_mb']
        peak = result['peak_memory_mb']
        final = result['final_memory_mb']
        growth = result['memory_growth_mb']
        
        print(f"{duration:6.1f}s: {initial:.1f} → {peak:.1f} → {final:.1f} MB (Δ{growth:+.1f})")
    
    # Check for memory leaks
    memory_growths = [r['memory_growth_mb'] for r in results]
    if any(growth > 50 for growth in memory_growths):
        print("\n⚠️  Large memory growth detected (>50MB)")
    elif any(r['memory_leak_detected'] for r in results):
        print("\n⚠️  Memory leak indicators found")
    else:
        max_growth = max(memory_growths) if memory_growths else 0
        print(f"\n✅ Memory usage stable (max growth: {max_growth:.1f}MB)")
    
    # CPU Analysis
    print("\n💻 CPU UTILIZATION")
    print("-" * 40)
    
    for result in results:
        duration = result['audio_duration_sec']
        avg_cpu = result['avg_cpu_percent']
        peak_cpu = result['peak_cpu_percent']
        
        print(f"{duration:6.1f}s: {avg_cpu:.1f}% avg, {peak_cpu:.1f}% peak")
    
    # Check CPU efficiency
    avg_cpus = [r['avg_cpu_percent'] for r in results]
    if len(avg_cpus) > 1:
        cpu_trend = np.polyfit(durations, avg_cpus, 1)[0]
        if cpu_trend > 0.1:  # CPU usage increasing significantly
            print(f"\n⚠️  CPU usage trending up ({cpu_trend:+.2f}% per second)")
        else:
            print(f"\n✅ CPU usage remains efficient")
    
    # Streaming Analysis
    print("\n🎙️  STREAMING PERFORMANCE")
    print("-" * 40)
    
    for result in results:
        duration = result['audio_duration_sec']
        segments = result['total_segments']
        endpoints = result['endpoint_detections']
        avg_seg_len = result['avg_segment_length_sec']
        
        print(f"{duration:6.1f}s: {segments} segments, {endpoints} endpoints, {avg_seg_len:.1f}s avg length")
    
    # Error Analysis
    print("\n🚨 ERROR ANALYSIS")
    print("-" * 40)
    
    total_errors = 0
    for result in results:
        duration = result['audio_duration_sec']
        errors = len(result['errors_encountered'])
        total_errors += errors
        
        if errors > 0:
            print(f"{duration:6.1f}s: {errors} errors")
            for error in result['errors_encountered'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        else:
            print(f"{duration:6.1f}s: No errors")
    
    if total_errors == 0:
        print("\n✅ No errors detected across all tests")
    else:
        print(f"\n⚠️  Total errors: {total_errors}")
    
    # Overall Assessment
    print("\n🎯 OVERALL ASSESSMENT")
    print("-" * 40)
    
    max_duration = max(durations)
    stable_performance = all(r['real_time_factor'] < 1.0 for r in results)
    no_memory_leaks = not any(r['memory_leak_detected'] for r in results)
    no_perf_degradation = not any(r['performance_degradation'] for r in results)
    no_errors = total_errors == 0
    
    score = sum([stable_performance, no_memory_leaks, no_perf_degradation, no_errors])
    
    print(f"Maximum tested duration: {max_duration:.1f}s ({max_duration/60:.1f} minutes)")
    print(f"Real-time performance: {'✅' if stable_performance else '❌'}")
    print(f"Memory stability: {'✅' if no_memory_leaks else '❌'}")
    print(f"Performance stability: {'✅' if no_perf_degradation else '❌'}")
    print(f"Error-free operation: {'✅' if no_errors else '❌'}")
    
    if score == 4:
        print(f"\n🏆 EXCELLENT: Sherpa-ONNX shows excellent stability up to {max_duration/60:.1f} minutes")
    elif score >= 3:
        print(f"\n✅ GOOD: Sherpa-ONNX performs well with minor issues")
    elif score >= 2:
        print(f"\n⚠️  FAIR: Some performance concerns detected")
    else:
        print(f"\n❌ POOR: Significant stability issues found")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if stable_performance:
        best_rtf = min(rtfs)
        print(f"✅ Real-time performance confirmed (best RTF: {best_rtf:.3f})")
    else:
        worst_rtf = max(rtfs)
        print(f"⚠️  Consider reducing load or chunk size (worst RTF: {worst_rtf:.3f})")
    
    if max(memory_growths) > 10:  # >10MB growth
        print(f"⚠️  Monitor memory usage for long sessions (max growth: {max(memory_growths):.1f}MB)")
    else:
        print(f"✅ Memory usage is well-controlled")
    
    max_avg_cpu = max(avg_cpus) if avg_cpus else 0
    if max_avg_cpu > 50:
        print(f"⚠️  High CPU usage detected ({max_avg_cpu:.1f}% avg)")
    else:
        print(f"✅ CPU usage is efficient ({max_avg_cpu:.1f}% max avg)")
    
    print(f"\n📈 For production use:")
    print(f"   - Recommended max session length: {max_duration/60:.1f}+ minutes")
    print(f"   - Expected RTF: ~{np.mean(rtfs):.3f}")
    print(f"   - Expected memory usage: ~{np.mean([r['peak_memory_mb'] for r in results]):.0f}MB")

def main():
    parser = argparse.ArgumentParser(description='Analyze Sherpa-ONNX Stress Test Results')
    parser.add_argument('results_file', type=Path, help='JSON results file from stress test')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        logger.error(f"Results file not found: {args.results_file}")
        return
    
    try:
        analyze_stress_results(args.results_file)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()