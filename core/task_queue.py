"""
Task Queue - Priority-based queue with dependency tracking and retry support.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from core.logger import get_logger
from models import Task, TaskPriority

_PRIORITY_ORDER = {
    TaskPriority.CRITICAL: 0,
    TaskPriority.HIGH: 1,
    TaskPriority.MEDIUM: 2,
    TaskPriority.LOW: 3,
}


class TaskQueue:
    """
    Priority-based task queue with dependency management and retry support.
    Tasks are not returned until all their dependencies have completed.
    Uses a set for O(1) active-task membership checks.
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 5,
        retry_delay: float = 5.0,
        max_queue_size: int = 0,
    ):
        if max_queue_size < 0:
            raise ValueError(
                f"max_queue_size must be >= 0 (0 = unlimited), got {max_queue_size}"
            )
        self.max_concurrent_tasks = max_concurrent_tasks
        self.retry_delay = retry_delay
        self.max_queue_size = max_queue_size  # 0 = unlimited
        self.logger = get_logger()

        self._pending: List[Task] = []
        self._active_set: set = set()  # task_id set — O(1) lookup
        self._active: List[Task] = []  # ordered list for iteration
        self._completed: List[Task] = []
        self._failed: List[Task] = []
        self._completed_ids: set = set()  # fast dependency checks
        self._retry_counts: Dict[str, int] = {}

    def add_task(self, task: Task) -> None:
        """Add a task to the pending queue.

        Raises RuntimeError if max_queue_size (if configured) would be exceeded.
        """
        if self.max_queue_size and len(self._pending) >= self.max_queue_size:
            raise RuntimeError(
                f"[TaskQueue] Queue full ({self.max_queue_size} tasks); cannot add {task.task_id}"
            )
        self._pending.append(task)
        self.logger.info(
            f"[TaskQueue] Added task {task.task_id}: {task.description[:60]}"
        )

    def _dependencies_met(self, task: Task) -> bool:
        """Return True if all of this task's dependencies have completed."""
        if not task.dependencies:
            return True
        return all(dep_id in self._completed_ids for dep_id in task.dependencies)

    def get_next_task(self) -> Optional[Task]:
        """
        Return the highest-priority pending task whose dependencies are met.
        Respects max_concurrent_tasks limit. Returns None if no tasks are ready.
        """
        if len(self._active) >= self.max_concurrent_tasks:
            return None

        ready = [t for t in self._pending if self._dependencies_met(t)]
        if not ready:
            return None

        ready.sort(key=lambda t: _PRIORITY_ORDER.get(t.priority, 2))
        task = ready[0]
        self._pending.remove(task)
        self._active.append(task)
        self._active_set.add(task.task_id)
        return task

    def mark_completed(self, task: Task, result_data: Any = None) -> None:
        """Mark a task as successfully completed."""
        if task.task_id in self._active_set:
            self._active.remove(task)
            self._active_set.discard(task.task_id)
        self._completed.append(task)
        self._completed_ids.add(task.task_id)
        self.logger.info(f"[TaskQueue] Completed: {task.task_id}")

    def mark_failed(self, task: Task, error: str = "") -> None:
        """
        Mark a task as failed. Re-queues it if retries remain (uses
        task.max_retries if the attribute exists, otherwise defaults to 0).
        """
        if task.task_id in self._active_set:
            self._active.remove(task)
            self._active_set.discard(task.task_id)

        max_retries = getattr(task, "max_retries", 0)
        retry_count = self._retry_counts.get(task.task_id, 0)

        if retry_count < max_retries:
            new_count = retry_count + 1
            self._retry_counts[task.task_id] = new_count
            # Use Task's built-in method to properly reset status and timestamps
            task.increment_retry()
            self._pending.append(task)  # re-queue for retry
            self.logger.warning(
                f"[TaskQueue] Task {task.task_id} failed (attempt "
                f"{new_count}/{max_retries}), re-queuing. Error: {error}"
            )
        else:
            self._failed.append(task)
            self.logger.warning(
                f"[TaskQueue] Task {task.task_id} permanently failed "
                f"after {retry_count + 1} attempt(s). Error: {error}"
            )

    def is_empty(self) -> bool:
        """Return True when there are no more pending tasks."""
        return len(self._pending) == 0

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        total = len(self._completed) + len(self._failed)
        success_rate = len(self._completed) / total if total > 0 else 0.0
        return {
            "pending": len(self._pending),
            "active": len(self._active),
            "completed": len(self._completed),
            "failed": len(self._failed),
            "total_tasks": total,
            "success_rate": success_rate,
        }
