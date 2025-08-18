"""
Lightweight Frame Flow Monitor

A simple decorator-based approach to monitor frame flow without heavy modifications.
"""

import time
import asyncio
from functools import wraps
from typing import Dict, Set, Optional
from collections import defaultdict, deque

from pipecat.frames.frames import Frame, StartFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from loguru import logger

class FrameFlowMonitor:
    """Lightweight frame flow monitoring"""
    
    def __init__(self):
        self.active = True
        self.frame_counts = defaultdict(int)
        self.processor_activity = defaultdict(lambda: {'last_seen': 0, 'frame_count': 0})
        self.blocked_processors = set()
        self.start_frame_processors = set()
        self.recent_events = deque(maxlen=100)
        
    def log_frame(self, processor_name: str, frame_type: str, direction: str, action: str):
        """Log a frame event"""
        if not self.active:
            return
            
        timestamp = time.time()
        event = f"{timestamp:.3f} | {processor_name:20} | {action:10} | {frame_type:20} | {direction}"
        
        self.recent_events.append(event)
        self.frame_counts[f"{processor_name}_{action}"] += 1
        self.processor_activity[processor_name]['last_seen'] = timestamp
        self.processor_activity[processor_name]['frame_count'] += 1
        
        # Track StartFrame reception
        if frame_type == 'StartFrame' and action == 'received':
            self.start_frame_processors.add(processor_name)
        
        # Check for potential issues
        if action == 'received' and frame_type != 'StartFrame':
            if processor_name not in self.start_frame_processors:
                logger.warning(f"🚨 {processor_name} processing {frame_type} before StartFrame!")
        
        logger.debug(event)
    
    def mark_blocked(self, processor_name: str, reason: str):
        """Mark a processor as blocked"""
        self.blocked_processors.add(processor_name)
        logger.error(f"🚫 {processor_name} BLOCKED: {reason}")
    
    def mark_unblocked(self, processor_name: str):
        """Mark a processor as unblocked"""
        self.blocked_processors.discard(processor_name)
        logger.info(f"✅ {processor_name} unblocked")
    
    def get_status(self) -> Dict:
        """Get current monitoring status"""
        now = time.time()
        
        return {
            'total_frames': sum(self.frame_counts.values()),
            'blocked_processors': list(self.blocked_processors),
            'active_processors': len(self.processor_activity),
            'recent_activity': len([
                p for p, data in self.processor_activity.items() 
                if now - data['last_seen'] < 5.0
            ]),
            'start_frame_coverage': len(self.start_frame_processors),
            'recent_events': list(self.recent_events)[-10:]
        }
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        logger.info("=" * 60)
        logger.info("📊 FRAME FLOW STATUS")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {status['total_frames']}")
        logger.info(f"Active processors: {status['active_processors']}")
        logger.info(f"Recent activity: {status['recent_activity']}")
        logger.info(f"StartFrame coverage: {status['start_frame_coverage']}")
        
        if status['blocked_processors']:
            logger.error(f"🚫 Blocked: {', '.join(status['blocked_processors'])}")
        
        logger.info("Recent events:")
        for event in status['recent_events']:
            logger.info(f"  {event}")
        logger.info("=" * 60)

# Global monitor instance
monitor = FrameFlowMonitor()

def monitor_frame_flow(processor_class):
    """
    Class decorator to add frame flow monitoring to any FrameProcessor
    
    Usage:
    @monitor_frame_flow
    class MyProcessor(FrameProcessor):
        # ... your implementation
    """
    
    original_process_frame = processor_class.process_frame
    original_init = processor_class.__init__
    
    @wraps(original_init)
    def monitored_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._monitor_name = f"{self.__class__.__name__}#{id(self)}"
        self._processing_start_times = {}
    
    @wraps(original_process_frame)
    async def monitored_process_frame(self, frame: Frame, direction: FrameDirection):
        processor_name = getattr(self, '_monitor_name', self.__class__.__name__)
        frame_type = frame.__class__.__name__
        direction_str = direction.name
        
        # Log frame received
        monitor.log_frame(processor_name, frame_type, direction_str, 'received')
        
        # Track processing time
        start_time = time.time()
        self._processing_start_times[id(frame)] = start_time
        
        try:
            # Call original process_frame
            result = await original_process_frame(self, frame, direction)
            
            # Log frame processed
            processing_time = (time.time() - start_time) * 1000
            monitor.log_frame(processor_name, frame_type, direction_str, f'processed({processing_time:.1f}ms)')
            
            # Check if frame was forwarded (this is a simple heuristic)
            # In a real implementation, you'd override push_frame too
            
            return result
            
        except Exception as e:
            # Log error
            monitor.log_frame(processor_name, frame_type, direction_str, f'ERROR:{str(e)[:50]}')
            monitor.mark_blocked(processor_name, str(e))
            raise
        finally:
            # Cleanup
            self._processing_start_times.pop(id(frame), None)
    
    # Replace methods
    processor_class.__init__ = monitored_init
    processor_class.process_frame = monitored_process_frame
    
    return processor_class


def monitor_push_frame(processor_class):
    """
    Decorator to monitor frame forwarding via push_frame
    """
    if hasattr(processor_class, 'push_frame'):
        original_push_frame = processor_class.push_frame
        
        @wraps(original_push_frame)
        async def monitored_push_frame(self, frame: Frame, direction: FrameDirection):
            processor_name = getattr(self, '_monitor_name', self.__class__.__name__)
            frame_type = frame.__class__.__name__
            direction_str = direction.name
            
            # Log frame forwarded
            monitor.log_frame(processor_name, frame_type, direction_str, 'forwarded')
            
            # Unblock processor if it was blocked
            if processor_name in monitor.blocked_processors:
                monitor.mark_unblocked(processor_name)
            
            return await original_push_frame(self, frame, direction)
        
        processor_class.push_frame = monitored_push_frame
    
    return processor_class


def full_monitor(processor_class):
    """
    Full monitoring decorator (combines frame flow and push frame monitoring)
    """
    return monitor_push_frame(monitor_frame_flow(processor_class))


# Context manager for monitoring sessions
class MonitoringSession:
    """Context manager for monitoring pipeline sessions"""
    
    def __init__(self, session_name: str = "pipeline"):
        self.session_name = session_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        monitor.active = True
        logger.info(f"🔍 Starting monitoring session: {self.session_name}")
        return monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"📊 Monitoring session '{self.session_name}' ended after {duration:.1f}s")
        monitor.print_status()
        
        if exc_type:
            logger.error(f"Session ended with exception: {exc_type.__name__}: {exc_val}")


# Async monitoring task
async def continuous_monitoring(interval: float = 5.0):
    """
    Continuous monitoring task that can run alongside your pipeline
    """
    logger.info(f"🔍 Starting continuous monitoring (interval: {interval}s)")
    
    while monitor.active:
        await asyncio.sleep(interval)
        
        # Check for stalled processors
        now = time.time()
        stalled = []
        
        for proc, data in monitor.processor_activity.items():
            if data['frame_count'] > 0 and now - data['last_seen'] > 10.0:
                stalled.append(proc)
        
        if stalled:
            logger.warning(f"⏱️  Stalled processors (>10s): {', '.join(stalled)}")
        
        # Print periodic status
        if int(now) % 30 == 0:  # Every 30 seconds
            monitor.print_status()
    
    logger.info("🔍 Continuous monitoring stopped")


# Helper functions for quick debugging
def start_monitoring():
    """Start monitoring"""
    monitor.active = True
    logger.info("🔍 Frame monitoring started")

def stop_monitoring():
    """Stop monitoring"""
    monitor.active = False
    logger.info("🔍 Frame monitoring stopped")

def get_monitoring_report():
    """Get current monitoring report"""
    return monitor.get_status()

def print_monitoring_status():
    """Print current monitoring status"""
    monitor.print_status()