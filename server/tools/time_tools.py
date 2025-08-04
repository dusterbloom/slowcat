"""
Time-aware tools for the LLM to manage timed tasks
"""

from typing import Dict, Any, Optional
from loguru import logger

# Global time executor instance (will be set by pipeline)
_time_executor = None


def set_time_executor(executor):
    """Set the global time executor instance"""
    global _time_executor
    _time_executor = executor
    logger.info("Time executor configured for tools")


async def start_timed_task(description: str, duration_minutes: float, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Start a timed task that will run for a specified duration
    
    Args:
        description: What the task is doing (e.g., "Search for hotels in Paris")
        duration_minutes: How long to run the task in minutes
        output_file: Optional file path to save results (e.g., "searches/hotels.md")
    
    Returns:
        Task information including task_id
    """
    if not _time_executor:
        return {"error": "Time executor not configured"}
    
    try:
        duration_seconds = int(duration_minutes * 60)
        task_id = await _time_executor.start_timed_task(
            description=description,
            duration_seconds=duration_seconds,
            output_file=output_file
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "description": description,
            "duration_minutes": duration_minutes,
            "output_file": output_file or "Not specified",
            "message": f"Started timed task for {duration_minutes} minutes"
        }
    except Exception as e:
        logger.error(f"Error starting timed task: {e}")
        return {"error": str(e)}


async def check_task_status(task_id: str) -> Dict[str, Any]:
    """
    Check the status of a timed task
    
    Args:
        task_id: The task ID to check
    
    Returns:
        Task status including progress and time remaining
    """
    if not _time_executor:
        return {"error": "Time executor not configured"}
    
    try:
        status = await _time_executor.get_task_status(task_id)
        
        # Format for voice-friendly response
        if "error" not in status:
            remaining = status['remaining_seconds']
            status['remaining_formatted'] = f"{remaining // 60}m {remaining % 60}s"
            elapsed = status['elapsed_seconds']
            status['elapsed_formatted'] = f"{elapsed // 60}m {elapsed % 60}s"
        
        return status
    except Exception as e:
        logger.error(f"Error checking task status: {e}")
        return {"error": str(e)}


async def stop_timed_task(task_id: str) -> Dict[str, Any]:
    """
    Stop a timed task early
    
    Args:
        task_id: The task ID to stop
    
    Returns:
        Confirmation of task stopping
    """
    if not _time_executor:
        return {"error": "Time executor not configured"}
    
    try:
        result = await _time_executor.stop_task(task_id)
        return result
    except Exception as e:
        logger.error(f"Error stopping task: {e}")
        return {"error": str(e)}


async def add_to_timed_task(task_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a result to an active timed task
    
    Args:
        task_id: The task ID to add results to
        content: The content/result to add
    
    Returns:
        Confirmation of addition
    """
    if not _time_executor:
        return {"error": "Time executor not configured"}
    
    try:
        await _time_executor.add_task_result(task_id, content)
        return {
            "success": True,
            "task_id": task_id,
            "message": "Result added to task"
        }
    except Exception as e:
        logger.error(f"Error adding to task: {e}")
        return {"error": str(e)}


async def get_active_tasks() -> Dict[str, Any]:
    """
    Get summary of all active timed tasks
    
    Returns:
        List of active tasks with their status
    """
    if not _time_executor:
        return {"error": "Time executor not configured"}
    
    try:
        tasks = _time_executor.get_active_tasks_summary()
        return {
            "active_tasks": tasks,
            "count": len(tasks)
        }
    except Exception as e:
        logger.error(f"Error getting active tasks: {e}")
        return {"error": str(e)}