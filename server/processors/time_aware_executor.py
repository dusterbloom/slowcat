"""
Time-aware task executor that can handle timed activities and track durations
"""

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from loguru import logger
import json

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    SystemFrame
)
from pipecat.processors.frame_processor import FrameProcessor


class TimedTask:
    """Represents a task with timing requirements"""
    def __init__(self, task_id: str, description: str, duration_seconds: int, 
                 output_file: Optional[str] = None, start_time: Optional[datetime] = None):
        self.task_id = task_id
        self.description = description
        self.duration_seconds = duration_seconds
        self.output_file = output_file
        self.start_time = start_time or datetime.now(timezone.utc)
        self.end_time = self.start_time + timedelta(seconds=duration_seconds)
        self.results: List[Dict[str, Any]] = []
        self.is_active = True
        self.completion_callback: Optional[Callable] = None


class TimeAwareExecutor(FrameProcessor):
    """
    Processor that adds time awareness and timed task execution capabilities
    """
    
    def __init__(
        self,
        *,
        base_output_dir: str = "./data",
        enable_auto_save: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.base_output_dir = Path(base_output_dir)
        self.enable_auto_save = enable_auto_save
        
        # Active tasks
        self.active_tasks: Dict[str, TimedTask] = {}
        self.task_counter = 0
        
        # Time tracking
        self.session_start = datetime.now(timezone.utc)
        
        # Task monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(f"TimeAwareExecutor initialized: base_dir={base_output_dir}")
    
    async def start_timed_task(self, description: str, duration_seconds: int, 
                              output_file: Optional[str] = None) -> str:
        """
        Start a new timed task
        
        Returns:
            Task ID for tracking
        """
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        # Create task
        task = TimedTask(
            task_id=task_id,
            description=description,
            duration_seconds=duration_seconds,
            output_file=output_file
        )
        
        self.active_tasks[task_id] = task
        
        # Start monitoring if not already running
        if not self._monitor_task or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_tasks())
        
        # Notify user
        end_time_str = task.end_time.strftime("%H:%M:%S")
        notification = f"⏰ Started timed task: {description}\n"
        notification += f"Duration: {duration_seconds // 60}m {duration_seconds % 60}s\n"
        notification += f"Will complete at: {end_time_str} UTC\n"
        notification += f"Task ID: {task_id}"
        
        await self.push_frame(TextFrame(notification))
        
        logger.info(f"Started timed task {task_id}: {description} for {duration_seconds}s")
        
        return task_id
    
    async def add_task_result(self, task_id: str, result: Dict[str, Any]):
        """Add a result to an active task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            result['timestamp'] = datetime.now(timezone.utc).isoformat()
            task.results.append(result)
            
            # Auto-save if enabled
            if self.enable_auto_save and task.output_file:
                await self._save_task_results(task)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a timed task"""
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
        
        task = self.active_tasks[task_id]
        now = datetime.now(timezone.utc)
        
        elapsed = int((now - task.start_time).total_seconds())
        remaining = max(0, int((task.end_time - now).total_seconds()))
        progress = min(100, int((elapsed / task.duration_seconds) * 100))
        
        return {
            "task_id": task_id,
            "description": task.description,
            "is_active": task.is_active,
            "elapsed_seconds": elapsed,
            "remaining_seconds": remaining,
            "progress_percent": progress,
            "results_count": len(task.results),
            "start_time": task.start_time.isoformat(),
            "end_time": task.end_time.isoformat()
        }
    
    async def stop_task(self, task_id: str) -> Dict[str, Any]:
        """Stop a task early"""
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
        
        task = self.active_tasks[task_id]
        task.is_active = False
        
        # Save final results
        if task.output_file:
            await self._save_task_results(task, final=True)
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        return {
            "task_id": task_id,
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "results_count": len(task.results)
        }
    
    async def _monitor_tasks(self):
        """Monitor active tasks and handle completions"""
        try:
            while self.active_tasks:
                now = datetime.now(timezone.utc)
                completed_tasks = []
                
                # Check each task
                for task_id, task in self.active_tasks.items():
                    if task.is_active and now >= task.end_time:
                        completed_tasks.append(task_id)
                
                # Complete tasks
                for task_id in completed_tasks:
                    await self._complete_task(task_id)
                
                # Sleep briefly
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.debug("Task monitor cancelled")
    
    async def _complete_task(self, task_id: str):
        """Complete a timed task"""
        task = self.active_tasks.get(task_id)
        if not task:
            return
        
        task.is_active = False
        
        # Save final results
        if task.output_file:
            await self._save_task_results(task, final=True)
        
        # Notify completion
        duration = task.duration_seconds
        notification = f"✅ Completed timed task: {task.description}\n"
        notification += f"Duration: {duration // 60}m {duration % 60}s\n"
        notification += f"Results: {len(task.results)} items"
        
        if task.output_file:
            notification += f"\nSaved to: {task.output_file}"
        
        await self.push_frame(TextFrame(notification))
        
        # Execute callback if set
        if task.completion_callback:
            await task.completion_callback(task)
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        logger.info(f"Completed task {task_id}: {task.description}")
    
    async def _save_task_results(self, task: TimedTask, final: bool = False):
        """Save task results to file"""
        if not task.output_file:
            return
        
        # Determine output path
        if task.output_file.startswith('/'):
            output_path = Path(task.output_file)
        else:
            output_path = self.base_output_dir / task.output_file
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare content based on file type
        if output_path.suffix == '.md':
            content = await self._format_results_markdown(task, final)
        elif output_path.suffix == '.json':
            content = json.dumps({
                "task": {
                    "id": task.task_id,
                    "description": task.description,
                    "duration_seconds": task.duration_seconds,
                    "start_time": task.start_time.isoformat(),
                    "end_time": task.end_time.isoformat(),
                    "is_complete": final
                },
                "results": task.results
            }, indent=2)
        else:
            # Default to text format
            content = await self._format_results_text(task, final)
        
        # Write to file
        output_path.write_text(content, encoding='utf-8')
        logger.debug(f"Saved task results to: {output_path}")
    
    async def _format_results_markdown(self, task: TimedTask, final: bool) -> str:
        """Format results as markdown"""
        now = datetime.now(timezone.utc)
        elapsed = int((now - task.start_time).total_seconds())
        
        content = f"# {task.description}\n\n"
        content += f"**Started**: {task.start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        
        if final:
            content += f"**Completed**: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        else:
            content += f"**Updated**: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        
        content += f"**Duration**: {elapsed // 60}m {elapsed % 60}s"
        if not final:
            remaining = max(0, task.duration_seconds - elapsed)
            content += f" (remaining: {remaining // 60}m {remaining % 60}s)"
        content += "\n\n---\n\n"
        
        # Add results
        if task.results:
            content += "## Results\n\n"
            for i, result in enumerate(task.results, 1):
                timestamp = result.get('timestamp', '')
                if timestamp:
                    dt = datetime.fromisoformat(timestamp)
                    rel_time = int((dt - task.start_time).total_seconds())
                    time_str = f"[{rel_time // 60}:{rel_time % 60:02d}]"
                else:
                    time_str = ""
                
                # Format based on result type
                if 'title' in result and 'snippet' in result:
                    # Search result format
                    content += f"### {i}. {result['title']} {time_str}\n"
                    content += f"{result['snippet']}\n"
                    if 'source' in result:
                        content += f"Source: {result['source']}\n"
                    content += "\n"
                elif 'content' in result:
                    # Generic content
                    content += f"### Result {i} {time_str}\n"
                    content += f"{result['content']}\n\n"
                else:
                    # Raw format
                    content += f"### Result {i} {time_str}\n"
                    content += f"```json\n{json.dumps(result, indent=2)}\n```\n\n"
        else:
            content += "*No results yet*\n"
        
        return content
    
    async def _format_results_text(self, task: TimedTask, final: bool) -> str:
        """Format results as plain text"""
        lines = []
        lines.append(f"Task: {task.description}")
        lines.append(f"Started: {task.start_time}")
        lines.append(f"Status: {'Completed' if final else 'In Progress'}")
        lines.append("-" * 40)
        
        for result in task.results:
            lines.append(json.dumps(result))
        
        return "\n".join(lines)
    
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames - mainly for forwarding"""
        # CRITICAL: Let parent class handle system frames first
        await super().process_frame(frame, direction)
        
        # Forward all frames
        await self.push_frame(frame, direction)
    
    def get_elapsed_session_time(self) -> int:
        """Get elapsed time since session start in seconds"""
        return int((datetime.now(timezone.utc) - self.session_start).total_seconds())
    
    def get_active_tasks_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active tasks"""
        summaries = []
        for task_id in self.active_tasks:
            summaries.append(asyncio.run(self.get_task_status(task_id)))
        return summaries