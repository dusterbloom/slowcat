#!/usr/bin/env python3
"""
Ultra-comprehensive performance analyzer for Slowcat voice pipeline.
Identifies bottlenecks, measures timing, and suggests optimizations.

Usage:
    python performance_analyzer.py --profile-startup
    python performance_analyzer.py --benchmark-components
    python performance_analyzer.py --analyze-pipeline
"""

import asyncio
import time
import threading
import statistics
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import concurrent.futures

from loguru import logger

# Suppress noisy logs for cleaner profiling output
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

@dataclass
class PerformanceMetric:
    """Performance measurement data"""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    cpu_usage: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.operation}: {self.duration_ms:.2f}ms"


class PerformanceProfiler:
    """Context manager for measuring performance"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self._stack: List[Tuple[str, float]] = []
    
    @asynccontextmanager
    async def measure(self, operation: str, **metadata):
        """Measure async operation performance"""
        start_time = time.perf_counter()
        memory_before = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            memory_after = self._get_memory_usage()
            
            metric = PerformanceMetric(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                metadata=metadata
            )
            
            self.metrics.append(metric)
            logger.info(f"⚡ {metric}")
    
    def measure_sync(self, operation: str, **metadata):
        """Measure synchronous operation performance"""
        start_time = time.perf_counter()
        memory_before = self._get_memory_usage()
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000
                    memory_after = self._get_memory_usage()
                    
                    metric = PerformanceMetric(
                        operation=operation,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        memory_before_mb=memory_before,
                        memory_after_mb=memory_after,
                        metadata=metadata
                    )
                    
                    self.metrics.append(metric)
                    logger.info(f"⚡ {metric}")
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        # Group by operation
        operations = {}
        for metric in self.metrics:
            op = metric.operation
            if op not in operations:
                operations[op] = []
            operations[op].append(metric.duration_ms)
        
        # Calculate statistics
        summary = {
            "total_metrics": len(self.metrics),
            "operations": {}
        }
        
        for op, durations in operations.items():
            summary["operations"][op] = {
                "count": len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "avg_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "total_ms": sum(durations)
            }
            
            # Flag slow operations
            avg_ms = summary["operations"][op]["avg_ms"]
            if avg_ms > 1000:
                summary["operations"][op]["warning"] = "SLOW (>1s)"
            elif avg_ms > 500:
                summary["operations"][op]["warning"] = "MODERATE (>500ms)"
        
        # Find bottlenecks
        bottlenecks = []
        for op, stats in summary["operations"].items():
            if stats["avg_ms"] > 200:  # >200ms average is concerning for voice
                bottlenecks.append({
                    "operation": op,
                    "avg_ms": stats["avg_ms"],
                    "impact": "HIGH" if stats["avg_ms"] > 1000 else "MEDIUM"
                })
        
        summary["bottlenecks"] = sorted(bottlenecks, key=lambda x: x["avg_ms"], reverse=True)
        
        return summary


class ComponentBenchmarker:
    """Benchmark individual pipeline components"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
    
    async def benchmark_ml_loading(self):
        """Benchmark ML module loading times"""
        logger.info("🔬 Benchmarking ML module loading...")
        
        # Test MLX lock contention
        async with self.profiler.measure("mlx_lock_acquisition") as _:
            from utils.mlx_lock import MLX_GLOBAL_LOCK
            with MLX_GLOBAL_LOCK:
                await asyncio.sleep(0.001)  # Simulate brief ML operation
        
        # Test service factory initialization
        async with self.profiler.measure("service_factory_init") as _:
            from core.service_factory import ServiceFactory
            factory = ServiceFactory()
        
        # Test ML modules loading
        async with self.profiler.measure("ml_modules_loading") as _:
            await factory.get_service("ml_loader")
        
        # Test global analyzers
        async with self.profiler.measure("global_analyzers_init") as _:
            await factory.get_service("global_analyzers")
    
    async def benchmark_stt_performance(self):
        """Benchmark STT service performance"""
        logger.info("🔬 Benchmarking STT performance...")
        
        try:
            # Create mock audio data (1 second at 16kHz)
            import numpy as np
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)
            
            # Generate some test audio (sine wave)
            t = np.linspace(0, duration, samples, False)
            audio_np = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone
            audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
            
            # Benchmark STT processing
            async with self.profiler.measure("stt_service_creation") as _:
                from core.service_factory import service_factory
                stt_service = await service_factory._create_stt_service_for_language("en")
            
            # Multiple STT runs to measure consistency
            for i in range(3):
                async with self.profiler.measure(f"stt_transcription_run_{i+1}") as _:
                    results = []
                    async for frame in stt_service.run_stt(audio_bytes):
                        results.append(frame)
                        
        except Exception as e:
            logger.error(f"STT benchmark failed: {e}")
    
    async def benchmark_tts_performance(self):
        """Benchmark TTS service performance"""
        logger.info("🔬 Benchmarking TTS performance...")
        
        try:
            # Test TTS service creation
            async with self.profiler.measure("tts_service_creation") as _:
                from core.service_factory import service_factory
                tts_service = await service_factory._create_tts_service_for_language("en")
            
            # Test TTS generation with different text lengths
            test_texts = [
                "Hello",  # Short
                "This is a medium length sentence for testing.",  # Medium
                "This is a much longer sentence that should take more time to process and generate speech audio from, testing the scalability of the TTS system."  # Long
            ]
            
            for i, text in enumerate(test_texts):
                async with self.profiler.measure(f"tts_generation_len_{len(text)}_run_{i+1}") as _:
                    results = []
                    async for frame in tts_service.run_tts(text):
                        results.append(frame)
                        
        except Exception as e:
            logger.error(f"TTS benchmark failed: {e}")
    
    async def benchmark_llm_performance(self):
        """Benchmark LLM service performance"""
        logger.info("🔬 Benchmarking LLM performance...")
        
        try:
            # Test LLM service creation
            async with self.profiler.measure("llm_service_creation") as _:
                from core.service_factory import service_factory
                llm_service = await service_factory._create_llm_service_for_language("en", "qwen2.5-7b-instruct")
            
            # Test simple LLM inference
            async with self.profiler.measure("llm_simple_inference") as _:
                from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
                context = OpenAILLMContext([{"role": "user", "content": "Say hello briefly."}])
                
                # Mock the streaming
                response_text = ""
                async for chunk in llm_service._stream_chat_completions(context):
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            response_text += delta.content
                            
        except Exception as e:
            logger.error(f"LLM benchmark failed: {e}")
    
    async def benchmark_mcp_performance(self):
        """Benchmark MCP tool performance"""
        logger.info("🔬 Benchmarking MCP performance...")
        
        try:
            # Test MCP manager initialization
            async with self.profiler.measure("mcp_manager_init") as _:
                from services.simple_mcp_tool_manager import get_global_mcp_manager
                manager = get_global_mcp_manager("en")
            
            # Test tool discovery (cached)
            async with self.profiler.measure("mcp_cached_tools") as _:
                tools = manager.get_cached_tools_for_llm()
            
            # Test tool discovery (with HTTP)
            async with self.profiler.measure("mcp_fresh_discovery") as _:
                tools = await manager.get_tools_for_llm(force_refresh=True)
            
            # Test individual tool calls
            if tools:
                tool_name = tools[0]["function"]["name"]
                async with self.profiler.measure(f"mcp_tool_call_{tool_name}") as _:
                    result = await manager.call_tool(tool_name, {"query": "test"})
                    
        except Exception as e:
            logger.error(f"MCP benchmark failed: {e}")


class PipelineAnalyzer:
    """Analyze the complete pipeline performance"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
    
    async def analyze_startup_sequence(self):
        """Analyze the complete startup sequence"""
        logger.info("🔬 Analyzing startup sequence...")
        
        # Simulate the complete startup
        async with self.profiler.measure("total_startup") as _:
            # Service factory initialization
            async with self.profiler.measure("service_factory_creation") as _:
                from core.service_factory import ServiceFactory
                factory = ServiceFactory()
            
            # Wait for ML modules
            async with self.profiler.measure("ml_modules_wait") as _:
                await factory.wait_for_ml_modules()
            
            # Wait for global analyzers
            async with self.profiler.measure("global_analyzers_wait") as _:
                await factory.wait_for_global_analyzers()
            
            # Create services for language
            async with self.profiler.measure("services_creation") as _:
                services = await factory.create_services_for_language("en", "qwen2.5-7b-instruct")
            
            # Pipeline builder
            async with self.profiler.measure("pipeline_builder_creation") as _:
                from core.pipeline_builder import PipelineBuilder
                builder = PipelineBuilder(factory)
            
            # Mock WebRTC connection
            class MockWebRTC:
                def __init__(self):
                    self.pc_id = "mock"
                def get_answer(self):
                    return {"pc_id": self.pc_id, "sdp": "mock", "type": "answer"}
            
            # Build pipeline (this is the expensive part)
            async with self.profiler.measure("pipeline_build") as _:
                mock_webrtc = MockWebRTC()
                pipeline, task = await builder.build_pipeline(mock_webrtc, "en", "qwen2.5-7b-instruct")
    
    async def analyze_request_latency(self):
        """Analyze end-to-end request processing latency"""
        logger.info("🔬 Analyzing request latency...")
        
        # This would simulate a complete voice-to-voice cycle
        # For now, we'll measure the key components in sequence
        
        # STT processing
        async with self.profiler.measure("voice_to_text") as _:
            await asyncio.sleep(0.1)  # Simulate STT processing
        
        # LLM processing  
        async with self.profiler.measure("text_to_response") as _:
            await asyncio.sleep(0.3)  # Simulate LLM inference
        
        # TTS processing
        async with self.profiler.measure("text_to_voice") as _:
            await asyncio.sleep(0.2)  # Simulate TTS generation


class BottleneckDetector:
    """Detect and analyze performance bottlenecks"""
    
    @staticmethod
    def analyze_bottlenecks(metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze metrics to identify bottlenecks"""
        
        bottlenecks = {
            "critical": [],  # >1000ms
            "major": [],     # 500-1000ms  
            "minor": [],     # 200-500ms
            "recommendations": []
        }
        
        # Group metrics by operation
        operation_stats = {}
        for metric in metrics:
            op = metric.operation
            if op not in operation_stats:
                operation_stats[op] = []
            operation_stats[op].append(metric.duration_ms)
        
        # Analyze each operation
        for op, durations in operation_stats.items():
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            bottleneck_data = {
                "operation": op,
                "avg_ms": avg_duration,
                "max_ms": max_duration,
                "occurrences": len(durations)
            }
            
            # Classify bottleneck severity
            if avg_duration > 1000:
                bottlenecks["critical"].append(bottleneck_data)
                bottlenecks["recommendations"].append({
                    "operation": op,
                    "issue": f"Critical bottleneck: {avg_duration:.1f}ms average",
                    "suggestion": BottleneckDetector._get_optimization_suggestion(op, avg_duration)
                })
            elif avg_duration > 500:
                bottlenecks["major"].append(bottleneck_data)
                bottlenecks["recommendations"].append({
                    "operation": op,
                    "issue": f"Major bottleneck: {avg_duration:.1f}ms average",  
                    "suggestion": BottleneckDetector._get_optimization_suggestion(op, avg_duration)
                })
            elif avg_duration > 200:
                bottlenecks["minor"].append(bottleneck_data)
                bottlenecks["recommendations"].append({
                    "operation": op,
                    "issue": f"Minor bottleneck: {avg_duration:.1f}ms average",
                    "suggestion": BottleneckDetector._get_optimization_suggestion(op, avg_duration)
                })
        
        return bottlenecks
    
    @staticmethod
    def _get_optimization_suggestion(operation: str, avg_ms: float) -> str:
        """Get optimization suggestions based on operation type"""
        
        suggestions = {
            "ml_modules_loading": "Consider pre-loading ML modules during container startup or using model caching",
            "mlx_lock_acquisition": "CRITICAL: MLX global lock is serializing STT/TTS operations. Consider separate GPU contexts or model instances",
            "stt_transcription": "Consider using smaller Whisper models, audio preprocessing, or batch processing",
            "tts_generation": "Consider streaming TTS, voice model caching, or parallel generation",
            "llm_inference": "Consider smaller models, prompt optimization, or response streaming",
            "mcp_tool_call": "Consider caching tool responses, reducing HTTP timeouts, or local tool implementations",
            "mcp_fresh_discovery": "Consider longer TTL for tool discovery or startup-time discovery",
            "service_factory_creation": "Consider dependency injection optimization or lazy loading",
            "pipeline_build": "Consider pipeline caching or lazy processor initialization",
            "global_analyzers_init": "Consider pre-warming analyzers or shared instances"
        }
        
        # Match operation name to suggestion
        for key, suggestion in suggestions.items():
            if key in operation.lower():
                return suggestion
        
        # Generic suggestions based on timing
        if avg_ms > 1000:
            return "Consider async processing, caching, or breaking into smaller operations"
        elif avg_ms > 500:
            return "Consider performance optimization or parallel processing"
        else:
            return "Monitor for regression and consider minor optimizations"


async def main():
    """Main performance analysis orchestrator"""
    
    parser = argparse.ArgumentParser(description="Slowcat Performance Analyzer")
    parser.add_argument("--profile-startup", action="store_true", help="Profile startup sequence")
    parser.add_argument("--benchmark-components", action="store_true", help="Benchmark individual components")
    parser.add_argument("--analyze-pipeline", action="store_true", help="Analyze complete pipeline")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--output", default="performance_report.json", help="Output file for results")
    
    args = parser.parse_args()
    
    if not any([args.profile_startup, args.benchmark_components, args.analyze_pipeline, args.all]):
        parser.print_help()
        return
    
    logger.info("🚀 Starting Slowcat Performance Analysis")
    
    all_metrics = []
    
    # Component benchmarking
    if args.benchmark_components or args.all:
        logger.info("=" * 60)
        logger.info("🔬 COMPONENT BENCHMARKING")
        logger.info("=" * 60)
        
        benchmarker = ComponentBenchmarker()
        
        await benchmarker.benchmark_ml_loading()
        await benchmarker.benchmark_stt_performance()
        await benchmarker.benchmark_tts_performance()
        await benchmarker.benchmark_llm_performance()
        await benchmarker.benchmark_mcp_performance()
        
        all_metrics.extend(benchmarker.profiler.metrics)
    
    # Startup profiling
    if args.profile_startup or args.all:
        logger.info("=" * 60)
        logger.info("🔬 STARTUP SEQUENCE ANALYSIS")
        logger.info("=" * 60)
        
        analyzer = PipelineAnalyzer()
        await analyzer.analyze_startup_sequence()
        await analyzer.analyze_request_latency()
        
        all_metrics.extend(analyzer.profiler.metrics)
    
    # Pipeline analysis
    if args.analyze_pipeline or args.all:
        logger.info("=" * 60)
        logger.info("🔬 PIPELINE PERFORMANCE ANALYSIS")
        logger.info("=" * 60)
        
        # Additional pipeline-specific analysis would go here
        pass
    
    # Generate comprehensive report
    logger.info("=" * 60)
    logger.info("📊 PERFORMANCE ANALYSIS RESULTS")
    logger.info("=" * 60)
    
    if all_metrics:
        # Performance summary
        profiler = PerformanceProfiler()
        profiler.metrics = all_metrics
        summary = profiler.get_summary()
        
        # Bottleneck analysis
        bottlenecks = BottleneckDetector.analyze_bottlenecks(all_metrics)
        
        # Combined report
        report = {
            "timestamp": time.time(),
            "total_operations": len(all_metrics),
            "summary": summary,
            "bottlenecks": bottlenecks,
            "raw_metrics": [
                {
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "memory_delta_mb": (m.memory_after_mb - m.memory_before_mb) if (m.memory_after_mb and m.memory_before_mb) else None,
                    "metadata": m.metadata
                }
                for m in all_metrics
            ]
        }
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print key findings
        logger.info(f"📋 Performance Report Summary:")
        logger.info(f"   Total Operations: {len(all_metrics)}")
        logger.info(f"   Critical Bottlenecks: {len(bottlenecks['critical'])}")
        logger.info(f"   Major Bottlenecks: {len(bottlenecks['major'])}")
        logger.info(f"   Minor Bottlenecks: {len(bottlenecks['minor'])}")
        
        # Show top bottlenecks
        all_bottlenecks = bottlenecks['critical'] + bottlenecks['major'] + bottlenecks['minor']
        if all_bottlenecks:
            logger.info(f"\n🔥 Top Performance Issues:")
            for i, bottleneck in enumerate(sorted(all_bottlenecks, key=lambda x: x['avg_ms'], reverse=True)[:5]):
                logger.info(f"   {i+1}. {bottleneck['operation']}: {bottleneck['avg_ms']:.1f}ms avg")
        
        # Show recommendations
        if bottlenecks['recommendations']:
            logger.info(f"\n💡 Optimization Recommendations:")
            for rec in bottlenecks['recommendations'][:5]:
                logger.info(f"   • {rec['operation']}: {rec['suggestion']}")
        
        logger.info(f"\n📄 Full report saved to: {args.output}")
    
    else:
        logger.warning("No metrics collected!")


if __name__ == "__main__":
    asyncio.run(main())