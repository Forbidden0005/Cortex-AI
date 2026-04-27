"""
Task Model - Represents a task in the AI system

This is the core data structure used to pass work through the system.
Agents, orchestrator, task queue, and memory manager all use this model.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Status of a task in the system"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"  # Waiting for dependencies


class TaskPriority(Enum):
    """Priority levels for tasks"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """
    Represents a single task in the AI system.

    Tasks flow through: User -> Planner -> Task Queue -> Agent Manager -> Agent -> Tools
    """

    # Core identification
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""

    # Task hierarchy and relationships
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)  # IDs of subtasks
    dependencies: List[str] = field(
        default_factory=list
    )  # IDs of tasks that must complete first

    # Execution details
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_agent: Optional[str] = None  # Which agent is handling this

    # Task parameters
    task_type: str = ""  # e.g., "file_operation", "code_generation", "web_search"
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional context for the task

    # Results and progress
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

    # Memory and learning
    save_to_memory: bool = True
    memory_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "parent_task_id": self.parent_task_id,
            "subtasks": self.subtasks,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_agent": self.assigned_agent,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "context": self.context,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "save_to_memory": self.save_to_memory,
            "memory_tags": self.memory_tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary"""
        task = cls(
            task_id=data.get("task_id", str(uuid.uuid4())),
            description=data.get("description", ""),
            parent_task_id=data.get("parent_task_id"),
            subtasks=data.get("subtasks", []),
            dependencies=data.get("dependencies", []),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 2)),
            assigned_agent=data.get("assigned_agent"),
            task_type=data.get("task_type", ""),
            parameters=data.get("parameters", {}),
            context=data.get("context", {}),
            result=data.get("result"),
            error=data.get("error"),
            progress=data.get("progress", 0.0),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            save_to_memory=data.get("save_to_memory", True),
            memory_tags=data.get("memory_tags", []),
        )

        # Handle datetime fields
        if data.get("created_at"):
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])

        return task

    def can_execute(self, completed_task_ids: set) -> bool:
        """Check if task can be executed based on dependencies"""
        return all(dep_id in completed_task_ids for dep_id in self.dependencies)

    def mark_started(self):
        """Mark task as started"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def mark_completed(self, result: Any = None):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 1.0
        if result is not None:
            self.result = result

    def mark_failed(self, error: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def should_retry(self) -> bool:
        """Check if task should be retried"""
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def increment_retry(self):
        """Increment retry counter and reset status"""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error = None


if __name__ == "__main__":
    # Test the task model
    task = Task(description="Test task", task_type="test", parameters={"key": "value"})
    print(f"Created task: {task.task_id}")
    print(f"Status: {task.status.value}")

    # Test serialization
    task_dict = task.to_dict()
    print(f"\nSerialized task: {task_dict}")

    # Test deserialization
    task2 = Task.from_dict(task_dict)
    print(f"\nDeserialized task: {task2.task_id}")
    print(f"Match: {task.task_id == task2.task_id}")
