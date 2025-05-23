import threading
from typing import Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def start_task(self, task_id: str, action: str) -> None:
        """Register a new task"""
        with self._lock:
            self.tasks[task_id] = {
                'action': action,
                'start_time': time.time(),
                'stop_flag': threading.Event(),
                'status': 'running'
            }
    
    def stop_task(self, task_id: str) -> bool:
        """Signal a task to stop"""
        with self._lock:
            if task_id not in self.tasks:
                return False
            self.tasks[task_id]['stop_flag'].set()
            self.tasks[task_id]['status'] = 'stopping'
            return True
    
    def get_stop_flag(self, task_id: str) -> Optional[threading.Event]:
        """Get the stop flag for a task"""
        with self._lock:
            if task_id not in self.tasks:
                return None
            return self.tasks[task_id]['stop_flag']
    
    def cleanup_task(self, task_id: str) -> None:
        """Remove a completed task"""
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
    
    def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get the status of a task"""
        with self._lock:
            if task_id not in self.tasks:
                return None
            task = self.tasks[task_id]
            return {
                'task_id': task_id,
                'action': task['action'],
                'status': task['status'],
                'runtime': time.time() - task['start_time']
            }
    
    def list_tasks(self) -> Dict[str, dict]:
        """List all running tasks"""
        with self._lock:
            return {
                task_id: {
                    'task_id': task_id,
                    'action': info['action'],
                    'status': info['status'],
                    'runtime': time.time() - info['start_time']
                }
                for task_id, info in self.tasks.items()
            }
    
    def cleanup(self) -> None:
        """Stop all tasks during application shutdown"""
        logger.info("Cleaning up all tasks")
        with self._lock:
            task_ids = list(self.tasks.keys())
            for task_id in task_ids:
                try:
                    # Signal all tasks to stop
                    self.tasks[task_id]['stop_flag'].set()
                    self.tasks[task_id]['status'] = 'stopped'
                    logger.info(f"Signaled task {task_id} to stop during shutdown")
                except Exception as e:
                    logger.error(f"Error stopping task {task_id} during shutdown: {e}")
            
            # Clear the tasks dictionary
            self.tasks.clear()
            logger.info(f"Cleaned up {len(task_ids)} tasks")

# Global task manager instance
task_manager = TaskManager() 