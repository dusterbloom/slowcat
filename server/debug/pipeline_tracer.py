"""
Pipeline Frame Flow Tracer for Debugging

This module provides comprehensive frame flow tracing to debug pipeline issues.
Helps identify frame forwarding problems, StartFrame issues, and processor blocking.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path

from pipecat.frames.frames import (
    Frame, StartFrame, EndFrame, TextFrame, AudioRawFrame,
    LLMMessagesFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    TTSStartedFrame, TTSStoppedFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.pipeline.pipeline import Pipeline
from loguru import logger

@dataclass
class FrameEvent:
    """Single frame processing event"""
    timestamp: float
    processor_name: str
    processor_id: str
    frame_type: str
    frame_id: str
    direction: str
    event_type: str  # 'received', 'forwarded', 'blocked', 'error'
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict] = None

class PipelineTracer:
    """
    Comprehensive pipeline tracer that monitors all frame flow
    """
    
    def __init__(self, output_dir: str = "server/debug/traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Event tracking
        self.events: List[FrameEvent] = []
        self.processor_stats: Dict[str, Dict] = defaultdict(lambda: {
            'frames_received': 0,
            'frames_forwarded': 0,
            'frames_blocked': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'errors': [],
            'last_frame_time': 0,
            'is_blocked': False
        })
        
        # Frame flow tracking
        self.frame_lineage: Dict[str, List[str]] = {}  # frame_id -> processor_path
        self.pending_frames: Dict[str, Set[str]] = defaultdict(set)  # processor -> frame_ids
        self.start_frame_received: Set[str] = set()  # processors that got StartFrame
        
        # Pipeline state
        self.pipeline_start_time = time.time()
        self.last_activity_time = time.time()
        self.is_tracing = False
        
        # Alerts
        self.alerts: List[Dict] = []
        
        logger.info(f"Pipeline tracer initialized, output: {self.output_dir}")
    
    def start_tracing(self):
        """Start pipeline tracing"""
        self.is_tracing = True
        self.pipeline_start_time = time.time()
        logger.info("🔍 Pipeline tracing started")
    
    def stop_tracing(self):
        """Stop tracing and generate report"""
        self.is_tracing = False
        report_file = self.generate_report()
        logger.info(f"📊 Tracing stopped, report: {report_file}")
        return report_file
    
    def trace_frame_received(self, processor: FrameProcessor, frame: Frame, direction: FrameDirection) -> str:
        """Trace when a processor receives a frame"""
        if not self.is_tracing:
            return ""
            
        event_id = f"{time.time()}_{id(frame)}"
        processor_name = processor.__class__.__name__
        processor_id = f"{processor_name}#{id(processor)}"
        frame_type = frame.__class__.__name__
        frame_id = f"{frame_type}#{id(frame)}"
        
        # Track frame lineage
        if frame_id not in self.frame_lineage:
            self.frame_lineage[frame_id] = []
        self.frame_lineage[frame_id].append(processor_id)
        
        # Add to pending frames
        self.pending_frames[processor_id].add(frame_id)
        
        # Check if processor got StartFrame
        if isinstance(frame, StartFrame):
            self.start_frame_received.add(processor_id)
        elif processor_id not in self.start_frame_received:
            # Alert: processor getting frames before StartFrame
            self.add_alert(
                "CRITICAL",
                f"{processor_id} processing {frame_type} before receiving StartFrame",
                {"processor": processor_id, "frame": frame_id}
            )
        
        # Record event
        event = FrameEvent(
            timestamp=time.time(),
            processor_name=processor_name,
            processor_id=processor_id,
            frame_type=frame_type,
            frame_id=frame_id,
            direction=direction.name,
            event_type='received',
            metadata=self._extract_frame_metadata(frame)
        )
        
        self.events.append(event)
        self.processor_stats[processor_id]['frames_received'] += 1
        self.processor_stats[processor_id]['last_frame_time'] = time.time()
        self.last_activity_time = time.time()
        
        return event_id
    
    def trace_frame_forwarded(self, processor: FrameProcessor, frame: Frame, 
                            direction: FrameDirection, event_id: str, 
                            processing_start_time: float):
        """Trace when a processor forwards a frame"""
        if not self.is_tracing:
            return
            
        processor_id = f"{processor.__class__.__name__}#{id(processor)}"
        frame_id = f"{frame.__class__.__name__}#{id(frame)}"
        duration_ms = (time.time() - processing_start_time) * 1000
        
        # Remove from pending
        self.pending_frames[processor_id].discard(frame_id)
        
        # Record event
        event = FrameEvent(
            timestamp=time.time(),
            processor_name=processor.__class__.__name__,
            processor_id=processor_id,
            frame_type=frame.__class__.__name__,
            frame_id=frame_id,
            direction=direction.name,
            event_type='forwarded',
            duration_ms=duration_ms
        )
        
        self.events.append(event)
        stats = self.processor_stats[processor_id]
        stats['frames_forwarded'] += 1
        stats['total_processing_time'] += duration_ms
        stats['avg_processing_time'] = stats['total_processing_time'] / stats['frames_forwarded']
        stats['is_blocked'] = False
    
    def trace_frame_blocked(self, processor: FrameProcessor, frame: Frame, 
                          direction: FrameDirection, event_id: str, reason: str):
        """Trace when a frame gets blocked"""
        if not self.is_tracing:
            return
            
        processor_id = f"{processor.__class__.__name__}#{id(processor)}"
        frame_id = f"{frame.__class__.__name__}#{id(frame)}"
        
        # Record event
        event = FrameEvent(
            timestamp=time.time(),
            processor_name=processor.__class__.__name__,
            processor_id=processor_id,
            frame_type=frame.__class__.__name__,
            frame_id=frame_id,
            direction=direction.name,
            event_type='blocked',
            error=reason
        )
        
        self.events.append(event)
        self.processor_stats[processor_id]['frames_blocked'] += 1
        self.processor_stats[processor_id]['is_blocked'] = True
        
        # Add alert
        self.add_alert(
            "ERROR",
            f"Frame blocked in {processor_id}: {reason}",
            {"processor": processor_id, "frame": frame_id, "reason": reason}
        )
    
    def trace_error(self, processor: FrameProcessor, frame: Frame, 
                   direction: FrameDirection, error: Exception):
        """Trace processing errors"""
        if not self.is_tracing:
            return
            
        processor_id = f"{processor.__class__.__name__}#{id(processor)}"
        error_msg = f"{error.__class__.__name__}: {str(error)}"
        
        # Record event
        event = FrameEvent(
            timestamp=time.time(),
            processor_name=processor.__class__.__name__,
            processor_id=processor_id,
            frame_type=frame.__class__.__name__,
            frame_id=f"{frame.__class__.__name__}#{id(frame)}",
            direction=direction.name,
            event_type='error',
            error=error_msg
        )
        
        self.events.append(event)
        self.processor_stats[processor_id]['errors'].append(error_msg)
        
        # Add alert
        self.add_alert(
            "ERROR",
            f"Processing error in {processor_id}: {error_msg}",
            {"processor": processor_id, "error": error_msg}
        )
    
    def add_alert(self, level: str, message: str, metadata: Dict = None):
        """Add an alert"""
        alert = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            'metadata': metadata or {}
        }
        self.alerts.append(alert)
        
        if level in ['ERROR', 'CRITICAL']:
            logger.error(f"🚨 {level}: {message}")
        else:
            logger.warning(f"⚠️  {level}: {message}")
    
    def check_pipeline_health(self) -> Dict[str, Any]:
        """Check overall pipeline health"""
        now = time.time()
        
        health = {
            'status': 'healthy',
            'issues': [],
            'blocked_processors': [],
            'slow_processors': [],
            'frame_flow_broken': False,
            'runtime_seconds': now - self.pipeline_start_time,
            'idle_seconds': now - self.last_activity_time
        }
        
        # Check for blocked processors
        for processor_id, stats in self.processor_stats.items():
            if stats['is_blocked']:
                health['blocked_processors'].append(processor_id)
                health['status'] = 'degraded'
            
            # Check for slow processors (>100ms avg)
            if stats['avg_processing_time'] > 100:
                health['slow_processors'].append({
                    'processor': processor_id,
                    'avg_time_ms': stats['avg_processing_time']
                })
                if health['status'] == 'healthy':
                    health['status'] = 'degraded'
        
        # Check for pending frames (potential deadlock)
        total_pending = sum(len(frames) for frames in self.pending_frames.values())
        if total_pending > 50:  # Arbitrary threshold
            health['issues'].append(f"High pending frame count: {total_pending}")
            health['status'] = 'critical'
        
        # Check for pipeline inactivity
        if health['idle_seconds'] > 30:  # No activity for 30s
            health['issues'].append(f"Pipeline idle for {health['idle_seconds']:.1f}s")
            health['status'] = 'critical'
        
        # Check frame flow
        if not self._verify_frame_flow():
            health['frame_flow_broken'] = True
            health['status'] = 'critical'
        
        return health
    
    def _verify_frame_flow(self) -> bool:
        """Verify frames are flowing through pipeline properly"""
        # Check that frames are being forwarded
        recent_events = [e for e in self.events[-100:] if e.event_type == 'forwarded']
        if len(recent_events) == 0 and len(self.events) > 50:
            return False
        
        # Check for StartFrame propagation
        start_events = [e for e in self.events if e.frame_type == 'StartFrame']
        if len(start_events) == 0 and len(self.events) > 10:
            return False
        
        return True
    
    def _extract_frame_metadata(self, frame: Frame) -> Dict:
        """Extract useful metadata from frame"""
        metadata = {'type': frame.__class__.__name__}
        
        if hasattr(frame, 'text') and frame.text:
            metadata['text_preview'] = frame.text[:100] + ('...' if len(frame.text) > 100 else '')
        
        if hasattr(frame, 'messages') and frame.messages:
            metadata['message_count'] = len(frame.messages)
            if frame.messages:
                metadata['last_role'] = frame.messages[-1].get('role', 'unknown')
        
        if hasattr(frame, 'audio') and frame.audio is not None:
            metadata['audio_length'] = len(frame.audio)
        
        return metadata
    
    def generate_report(self) -> str:
        """Generate comprehensive trace report"""
        timestamp = int(time.time())
        report_file = self.output_dir / f"trace_report_{timestamp}.json"
        
        # Generate summary statistics
        summary = self._generate_summary()
        
        # Generate frame flow analysis
        flow_analysis = self._analyze_frame_flow()
        
        # Generate processor analysis
        processor_analysis = self._analyze_processors()
        
        # Create full report
        report = {
            'metadata': {
                'timestamp': timestamp,
                'total_events': len(self.events),
                'total_processors': len(self.processor_stats),
                'runtime_seconds': time.time() - self.pipeline_start_time,
                'trace_start': self.pipeline_start_time,
                'trace_end': time.time()
            },
            'summary': summary,
            'health': self.check_pipeline_health(),
            'flow_analysis': flow_analysis,
            'processor_analysis': processor_analysis,
            'alerts': self.alerts,
            'events': [asdict(event) for event in self.events[-1000:]]  # Last 1000 events
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable summary
        self._print_summary(report)
        
        return str(report_file)
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        total_events = len(self.events)
        if total_events == 0:
            return {'total_events': 0, 'message': 'No events recorded'}
        
        events_by_type = defaultdict(int)
        for event in self.events:
            events_by_type[event.event_type] += 1
        
        return {
            'total_events': total_events,
            'events_by_type': dict(events_by_type),
            'total_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a['level'] == 'CRITICAL']),
            'error_alerts': len([a for a in self.alerts if a['level'] == 'ERROR']),
            'processors_active': len(self.processor_stats),
            'start_frames_sent': len([e for e in self.events if e.frame_type == 'StartFrame'])
        }
    
    def _analyze_frame_flow(self) -> Dict:
        """Analyze frame flow patterns"""
        frame_types = defaultdict(int)
        broken_flows = []
        
        for frame_id, path in self.frame_lineage.items():
            frame_type = frame_id.split('#')[0]
            frame_types[frame_type] += 1
            
            # Check for broken flows (frames that didn't complete journey)
            received_events = [e for e in self.events if e.frame_id == frame_id and e.event_type == 'received']
            forwarded_events = [e for e in self.events if e.frame_id == frame_id and e.event_type == 'forwarded']
            
            if len(received_events) > len(forwarded_events):
                broken_flows.append({
                    'frame_id': frame_id,
                    'path': path,
                    'received': len(received_events),
                    'forwarded': len(forwarded_events)
                })
        
        return {
            'frame_types_processed': dict(frame_types),
            'broken_flows': broken_flows,
            'total_unique_frames': len(self.frame_lineage),
            'avg_path_length': sum(len(path) for path in self.frame_lineage.values()) / len(self.frame_lineage) if self.frame_lineage else 0
        }
    
    def _analyze_processors(self) -> Dict:
        """Analyze processor performance"""
        analysis = {}
        
        for processor_id, stats in self.processor_stats.items():
            processor_name = processor_id.split('#')[0]
            
            analysis[processor_id] = {
                'name': processor_name,
                'stats': stats,
                'efficiency': stats['frames_forwarded'] / max(stats['frames_received'], 1),
                'status': 'blocked' if stats['is_blocked'] else 'healthy',
                'got_start_frame': processor_id in self.start_frame_received
            }
        
        return analysis
    
    def _print_summary(self, report: Dict):
        """Print human-readable summary"""
        logger.info("=" * 80)
        logger.info("🔍 PIPELINE TRACE REPORT")
        logger.info("=" * 80)
        
        # Health status
        health = report['health']
        status_emoji = {"healthy": "✅", "degraded": "⚠️", "critical": "🚨"}
        logger.info(f"Status: {status_emoji.get(health['status'], '❓')} {health['status'].upper()}")
        
        # Key metrics
        summary = report['summary']
        logger.info(f"Events: {summary['total_events']} | Processors: {summary['processors_active']} | Alerts: {summary['total_alerts']}")
        
        # Issues
        if health['issues']:
            logger.warning("Issues found:")
            for issue in health['issues']:
                logger.warning(f"  - {issue}")
        
        # Blocked processors
        if health['blocked_processors']:
            logger.error("Blocked processors:")
            for proc in health['blocked_processors']:
                logger.error(f"  - {proc}")
        
        # Slow processors
        if health['slow_processors']:
            logger.warning("Slow processors (>100ms avg):")
            for proc in health['slow_processors']:
                logger.warning(f"  - {proc['processor']}: {proc['avg_time_ms']:.1f}ms")
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 60]
        if recent_alerts:
            logger.warning(f"Recent alerts ({len(recent_alerts)}):")
            for alert in recent_alerts[-5:]:  # Show last 5
                logger.warning(f"  - {alert['level']}: {alert['message']}")
        
        logger.info("=" * 80)


class TracingFrameProcessor(FrameProcessor):
    """
    Wrapper processor that adds tracing to any FrameProcessor
    """
    
    def __init__(self, wrapped_processor: FrameProcessor, tracer: PipelineTracer, **kwargs):
        super().__init__(**kwargs)
        self.wrapped = wrapped_processor
        self.tracer = tracer
        self._processor_name = wrapped_processor.__class__.__name__
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Start tracing
        event_id = self.tracer.trace_frame_received(self.wrapped, frame, direction)
        processing_start_time = time.time()
        
        try:
            # Call wrapped processor
            await self.wrapped.process_frame(frame, direction)
            
            # Trace successful forwarding
            self.tracer.trace_frame_forwarded(
                self.wrapped, frame, direction, event_id, processing_start_time
            )
            
        except Exception as e:
            # Trace error
            self.tracer.trace_error(self.wrapped, frame, direction, e)
            raise
        
        # Forward frame
        await self.push_frame(frame, direction)


def add_tracing_to_pipeline(pipeline: Pipeline, tracer: PipelineTracer) -> Pipeline:
    """Add tracing to all processors in a pipeline"""
    # This is a helper function to wrap existing processors
    # Implementation depends on your pipeline structure
    logger.info("🔍 Adding tracing to pipeline processors")
    return pipeline


def create_debug_pipeline_with_tracing(normal_pipeline_func, tracer: PipelineTracer):
    """
    Create a debug version of your pipeline with full tracing
    """
    def debug_pipeline_builder(*args, **kwargs):
        # Build normal pipeline
        pipeline = normal_pipeline_func(*args, **kwargs)
        
        # Add tracing
        tracer.start_tracing()
        pipeline = add_tracing_to_pipeline(pipeline, tracer)
        
        return pipeline
    
    return debug_pipeline_builder