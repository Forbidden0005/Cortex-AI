"""
Base Agent - Foundation for all agents in the system

All agents inherit from this class and must use:
- Task objects as input
- AgentResult objects as output
"""

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our models
sys.path.append(str(Path(__file__).parent.parent))
# Import infrastructure
from core.logger import get_logger
from models import AgentResult, Task


class BaseAgent(ABC):
    """
    Base class for all agents in the AI system.

    All specialized agents inherit from this and implement _do_work().

    This class handles:
    - Task lifecycle (mark started/completed/failed)
    - Error handling
    - Result formatting
    - Execution time tracking
    - Logging
    - Tool tracking
    """

    def __init__(
        self,
        name: str,
        agent_type: str,
        description: str = "",
        tools: Optional[List[str]] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent name (e.g., "FileAgent")
            agent_type: Agent type matching task_type (e.g., "file")
            description: What this agent does
            tools: List of tool names this agent can use
        """
        self.name = name
        self.agent_type = agent_type
        self.description = description
        self.tools = tools or []

        # Get logger
        self.logger = get_logger()

        # Track performance
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0

        # Files tracked during execution
        self._files_created: List[str] = []
        self._files_modified: List[str] = []
        self._files_read: List[str] = []
        self._tools_used: List[str] = []

    def execute(self, task: Task) -> AgentResult:
        """
        Main execution method - all agents use this interface.

        DO NOT OVERRIDE THIS METHOD.
        Override _do_work() instead.

        Args:
            task: Task object to execute

        Returns:
            AgentResult object
        """
        start_time = time.time()

        # Reset tracking
        self._files_created = []
        self._files_modified = []
        self._files_read = []
        self._tools_used = []

        try:
            # Log task start
            self.logger.log_agent_action(
                self.name,
                "task_started",
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "task_type": task.task_type,
                },
            )

            # Mark task as started
            task.mark_started()

            # Validate task
            self._validate_task(task)

            # Call subclass implementation
            result_data = self._do_work(task)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Update stats
            self.tasks_completed += 1
            self.total_execution_time += execution_time

            # Mark task completed
            task.mark_completed(result_data)

            # Log success
            self.logger.log_agent_action(
                self.name,
                "task_completed",
                {
                    "task_id": task.task_id,
                    "execution_time": execution_time,
                    "status": "success",
                },
            )

            # Return standardized success result
            return AgentResult.success(
                data=result_data,
                agent_name=self.name,
                agent_type=self.agent_type,
                task_id=task.task_id,
                task_description=task.description,
                message=f"Task completed successfully by {self.name}",
                execution_time=execution_time,
                tools_used=self._tools_used.copy(),
                files_created=self._files_created.copy(),
                files_modified=self._files_modified.copy(),
                files_read=self._files_read.copy(),
                save_to_memory=task.save_to_memory,
                memory_tags=task.memory_tags,
            )

        except Exception as e:
            # Calculate execution time even on failure
            execution_time = time.time() - start_time

            # Update stats
            self.tasks_failed += 1

            # Mark task failed
            task.mark_failed(str(e))

            # Log error
            self.logger.log_agent_action(
                self.name,
                "task_failed",
                {
                    "task_id": task.task_id,
                    "error": str(e),
                    "execution_time": execution_time,
                },
            )

            # Return error result
            return AgentResult.create_error(
                error=str(e),
                agent_name=self.name,
                agent_type=self.agent_type,
                task_id=task.task_id,
                task_description=task.description,
                execution_time=execution_time,
            )

    @abstractmethod
    def _do_work(self, task: Task) -> Any:
        """
        OVERRIDE THIS METHOD in subclasses.

        This is where you implement your agent's logic.

        Args:
            task: Task to execute

        Returns:
            Result data (any type - dict, string, list, etc.)

        Raises:
            Exception: If task fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _do_work()"
        )

    def _validate_task(self, task: Task):
        """
        Validate task before execution.
        Override in subclasses for custom validation.

        Args:
            task: Task to validate

        Raises:
            ValueError: If task is invalid
        """
        if not task.description:
            raise ValueError("Task description is required")

        if task.task_type != self.agent_type:
            raise ValueError(
                f"Task type '{task.task_type}' does not match agent type '{self.agent_type}'"
            )

    # Helper methods for subclasses

    def _track_file_created(self, filepath: str):
        """Track that a file was created"""
        self._files_created.append(filepath)

    def _track_file_modified(self, filepath: str):
        """Track that a file was modified"""
        self._files_modified.append(filepath)

    def _track_file_read(self, filepath: str):
        """Track that a file was read"""
        self._files_read.append(filepath)

    def _track_tool_used(self, tool_name: str):
        """Track that a tool was used"""
        if tool_name not in self._tools_used:
            self._tools_used.append(tool_name)

    def _log(self, message: str, level: str = "info"):
        """Log a message"""
        if level == "debug":
            self.logger.debug(f"[{self.name}] {message}")
        elif level == "info":
            self.logger.info(f"[{self.name}] {message}")
        elif level == "warning":
            self.logger.warning(f"[{self.name}] {message}")
        elif level == "error":
            self.logger.error(f"[{self.name}] {message}")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        total_tasks = self.tasks_completed + self.tasks_failed
        avg_time = self.total_execution_time / total_tasks if total_tasks > 0 else 0
        success_rate = self.tasks_completed / total_tasks if total_tasks > 0 else 0

        return {
            "name": self.name,
            "type": self.agent_type,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_tasks": total_tasks,
            "success_rate": success_rate,
            "avg_execution_time": avg_time,
            "total_execution_time": self.total_execution_time,
        }


if __name__ == "__main__":
    # Test the BaseAgent with a simple implementation

    class TestAgent(BaseAgent):
        """Simple test agent"""

        def __init__(self):
            super().__init__(
                name="TestAgent",
                agent_type="test",
                description="Test agent for validation",
            )

        def _do_work(self, task: Task):
            """Simple test implementation"""
            # Track that we used a tool
            self._track_tool_used("test_tool")

            # Return some data
            return {"message": "Test task completed", "input": task.description}

    # Create a test task
    from models import TaskPriority

    task = Task(
        description="Test task",
        task_type="test",
        priority=TaskPriority.MEDIUM,
        parameters={"test": "value"},
    )

    # Execute with agent
    agent = TestAgent()
    result = agent.execute(task)

    # Check result
    print(f"✅ Task Status: {result.status.value}")
    print(f"✅ Agent Name: {result.agent_name}")
    print(f"✅ Success: {result.is_successful()}")
    print(f"✅ Data: {result.data}")
    print(f"✅ Execution Time: {result.execution_time:.4f}s")
    print(f"✅ Tools Used: {result.tools_used}")

    # Check stats
    stats = agent.get_stats()
    print("\n📊 Agent Stats:")
    print(f"   Tasks Completed: {stats['tasks_completed']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")

    print("\n✅ BaseAgent test passed!")
