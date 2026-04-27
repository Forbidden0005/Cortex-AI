"""
Orchestrator - Central coordinator for the AI system

The orchestrator is the "brain" that:
- Receives user requests
- Breaks them into tasks (using PlanningAgent)
- Manages task execution through the queue
- Coordinates agents via AgentManager
- Stores results in memory
- Handles complex multi-step workflows
"""

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import models
sys.path.append(str(Path(__file__).parent.parent))
from core.agent_manager import AgentManager
from core.config_loader import get_config
from core.llm_interface import LLMInterface
# Import core components
from core.logger import get_logger
from core.memory_manager import MemoryManager
from core.task_queue import TaskQueue
from models import AgentResult, MemoryItem, MemoryType, Task, TaskPriority


class Orchestrator:
    """
    Central orchestrator for the AI system.

    Responsibilities:
    - Parse user requests into tasks
    - Break complex tasks into subtasks
    - Manage task queue and execution
    - Coordinate agents
    - Store results and learnings
    - Handle errors and retries
    """

    def __init__(self):
        """Initialize the orchestrator"""
        self.logger = get_logger()
        self.config = get_config()

        # Initialize components
        self.logger.info("Initializing Orchestrator...")

        self.memory = MemoryManager(storage_dir=self.config.memory.vector_db_path)

        self.llm = LLMInterface(self.config.model, use_mock=False)

        self.agent_manager = AgentManager()

        self.task_queue = TaskQueue(
            max_concurrent_tasks=self.config.task_queue.max_concurrent_tasks,
            retry_delay=self.config.task_queue.retry_delay,
        )

        # Conversation tracking
        self.current_conversation_id = str(uuid.uuid4())
        self.conversation_history: List[Dict[str, str]] = []

        # Execution tracking
        self.completed_workflows: List[str] = []
        self.active_workflows: Dict[str, Dict[str, Any]] = {}

        self.logger.log_event(
            "orchestrator_initialized", "Orchestrator started and ready"
        )

    def process_request(
        self, user_request: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user request from start to finish.

        Args:
            user_request: Natural language request from user
            context: Optional context information

        Returns:
            Dictionary with results and status
        """
        workflow_id = str(uuid.uuid4())

        self.logger.log_user_request(user_request)
        self.logger.info(f"[Orchestrator] Processing request (workflow: {workflow_id})")

        # Track workflow
        self.active_workflows[workflow_id] = {
            "request": user_request,
            "started_at": datetime.now(),
            "status": "in_progress",
            "tasks": [],
            "results": [],
        }

        try:
            # Step 1: Understand the request and create initial task
            initial_task = self._create_task_from_request(user_request, context)

            # Step 2: Determine if task needs planning
            if self._needs_planning(initial_task):
                # Break down into subtasks
                subtasks = self._plan_execution(initial_task)
                tasks_to_execute = subtasks
            else:
                # Single task, execute directly
                tasks_to_execute = [initial_task]

            # Step 3: Add tasks to queue
            for task in tasks_to_execute:
                self.task_queue.add_task(task)
                self.active_workflows[workflow_id]["tasks"].append(task.task_id)

            # Step 4: Execute tasks from queue
            results = self._execute_queued_tasks()

            # Step 5: Combine results
            final_result = self._combine_results(results)

            # Step 6: Store in memory
            self._store_workflow_memory(workflow_id, user_request, final_result)

            # Step 7: Update workflow tracking
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["completed_at"] = datetime.now()
            self.active_workflows[workflow_id]["results"] = results
            self.completed_workflows.append(workflow_id)

            self.logger.log_event(
                "workflow_completed",
                f"Workflow {workflow_id} completed successfully",
                workflow_id=workflow_id,
                num_tasks=len(tasks_to_execute),
            )

            return {
                "success": True,
                "workflow_id": workflow_id,
                "result": final_result,
                "tasks_executed": len(results),
                "message": "Request processed successfully",
            }

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {str(e)}")

            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)

            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e),
                "message": "Request processing failed",
            }

    def _create_task_from_request(
        self, request: str, context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """
        Convert user request into a Task object.

        Args:
            request: User's natural language request
            context: Optional context

        Returns:
            Task object
        """
        # Use LLM to determine task type and parameters
        prompt = f"""Analyze this user request and determine:
1. What type of task is it? (file, coding, web, data, planning, etc.)
2. What are the key parameters?

Request: {request}

Respond with just the task type and parameters."""

        # LLM response used implicitly to warm context; result not parsed here
        self.llm.ask(prompt)

        # For now, simple heuristic-based classification
        # In production, would parse LLM response properly
        task_type = self._classify_request(request)

        task = Task(
            description=request,
            task_type=task_type,
            priority=TaskPriority.MEDIUM,
            parameters=context or {},
            save_to_memory=True,
            memory_tags=["user_request", task_type],
        )

        self.logger.log_task(
            task.task_id,
            "created_from_request",
            {"request": request, "task_type": task_type},
        )

        return task

    def _classify_request(self, request: str) -> str:
        """
        Classify request to determine task type.

        Args:
            request: User request text

        Returns:
            Task type string
        """
        request_lower = request.lower()

        # Check for keywords
        if any(
            word in request_lower
            for word in ["plan", "organize", "break down", "steps"]
        ):
            return "planning"

        if any(
            word in request_lower
            for word in ["file", "folder", "read", "write", "save"]
        ):
            return "file"

        if any(
            word in request_lower for word in ["code", "script", "function", "program"]
        ):
            return "coding"

        if any(word in request_lower for word in ["search", "find", "lookup", "web"]):
            return "web"

        if any(
            word in request_lower for word in ["analyze", "data", "chart", "calculate"]
        ):
            return "data"

        # Default to planning for complex requests
        if len(request.split()) > 10:
            return "planning"

        return "file"  # Default fallback

    def _needs_planning(self, task: Task) -> bool:
        """
        Determine if a task needs to be broken down into subtasks.

        Args:
            task: Task to evaluate

        Returns:
            True if task should be planned
        """
        # Tasks explicitly marked as planning
        if task.task_type == "planning":
            return True

        # Complex tasks (long descriptions, multiple actions)
        if len(task.description.split()) > 15:
            return True

        # Tasks with multiple verbs (suggests multiple steps)
        verbs = ["create", "analyze", "generate", "move", "copy", "send", "update"]
        verb_count = sum(1 for verb in verbs if verb in task.description.lower())
        if verb_count > 1:
            return True

        return False

    def _plan_execution(self, task: Task) -> List[Task]:
        """
        Break down a complex task into subtasks.

        Args:
            task: Task to break down

        Returns:
            List of subtasks
        """
        self.logger.info(f"[Orchestrator] Planning execution for: {task.description}")

        # Create planning task
        planning_task = Task(
            description=f"Create execution plan for: {task.description}",
            task_type="planning",
            priority=task.priority,
            parameters={"original_task": task.to_dict()},
        )

        # Execute planning with PlanningAgent
        result = self.agent_manager.execute_task(planning_task)

        if result.is_successful() and result.data:
            subtasks = result.data.get("subtasks", [])
            self.logger.info(f"[Orchestrator] Generated {len(subtasks)} subtasks")
            return subtasks
        else:
            # Planning failed, return original task
            self.logger.warning(
                "[Orchestrator] Planning failed, executing original task"
            )
            return [task]

    def _execute_queued_tasks(self) -> List[AgentResult]:
        """
        Execute all tasks in the queue.

        Returns:
            List of agent results
        """
        results = []

        self.logger.info("[Orchestrator] Starting task execution")

        while not self.task_queue.is_empty():
            # Get next ready task
            task = self.task_queue.get_next_task()

            if not task:
                # No tasks ready (dependencies not met)
                break

            self.logger.info(f"[Orchestrator] Executing: {task.description}")

            # Execute with agent manager
            result = self.agent_manager.execute_task(task)

            # Update queue based on result
            if result.is_successful():
                self.task_queue.mark_completed(task, result.data)
                results.append(result)

                self.logger.info(
                    f"[Orchestrator] Task completed by {result.agent_name} "
                    f"in {result.execution_time:.3f}s"
                )
            else:
                self.task_queue.mark_failed(task, result.error or "Unknown error")
                results.append(result)

                self.logger.warning(f"[Orchestrator] Task failed: {result.error}")

        self.logger.info(f"[Orchestrator] Completed {len(results)} tasks")

        return results

    def _combine_results(self, results: List[AgentResult]) -> Dict[str, Any]:
        """
        Combine multiple agent results into final output.

        Args:
            results: List of agent results

        Returns:
            Combined result dictionary
        """
        if not results:
            return {"message": "No results"}

        # If single result, return its data
        if len(results) == 1:
            return results[0].data

        # Multiple results - combine them
        combined = {
            "summary": f"Completed {len(results)} tasks",
            "results": [],
            "success_rate": sum(1 for r in results if r.is_successful()) / len(results),
        }

        for i, result in enumerate(results, 1):
            combined["results"].append(
                {
                    "step": i,
                    "agent": result.agent_name,
                    "status": result.status.value,
                    "data": result.data,
                    "time": result.execution_time,
                }
            )

        return combined

    def _store_workflow_memory(self, workflow_id: str, request: str, result: Any):
        """
        Store workflow execution in memory for learning.

        Args:
            workflow_id: Workflow identifier
            request: Original user request
            result: Final result
        """
        memory = MemoryItem(
            content=f"User requested: {request}. System completed with result: {str(result)[:200]}",
            memory_type=MemoryType.CONVERSATION,
            related_conversation_id=self.current_conversation_id,
            tags=["workflow", "completed"],
            source="orchestrator",
            metadata={
                "workflow_id": workflow_id,
                "request": request,
                "result_summary": str(result)[:500],
            },
        )

        self.memory.save(memory)

        self.logger.log_reflection(
            "workflow_completion",
            f"Workflow completed: {request}",
            {"workflow_id": workflow_id, "success": True},
        )

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow status or None
        """
        return self.active_workflows.get(workflow_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "total_workflows": len(self.completed_workflows)
            + len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "active_workflows": len(self.active_workflows),
            "agent_stats": self.agent_manager.get_performance_stats(),
            "queue_stats": self.task_queue.get_stats(),
            "memory_stats": self.memory.get_stats(),
        }


if __name__ == "__main__":
    # Test the orchestrator
    print("=" * 60)
    print("ORCHESTRATOR TEST")
    print("=" * 60)

    # Import agents for testing
    from agents.file_agent import FileAgent
    from agents.planning_agent import PlanningAgent

    # Create orchestrator
    print("\n[1/3] Initializing Orchestrator...")
    orchestrator = Orchestrator()

    # Register some agents
    print("\n[2/3] Registering Agents...")
    orchestrator.agent_manager.register_agent(PlanningAgent(orchestrator.llm))
    orchestrator.agent_manager.register_agent(FileAgent())
    print(f"✓ Registered {len(orchestrator.agent_manager.list_agents())} agents")

    # Test simple request
    print("\n[3/3] Processing Test Request...")
    request1 = "List the files in the current directory"
    print(f"\nRequest: '{request1}'")

    result1 = orchestrator.process_request(request1, {"filepath": "."})

    print("\nResult:")
    print(f"  Success: {result1['success']}")
    print(f"  Workflow ID: {result1['workflow_id']}")
    print(f"  Tasks Executed: {result1.get('tasks_executed', 0)}")
    if result1["success"]:
        print(f"  Result: {str(result1['result'])[:200]}")

    # Test complex request (requires planning)
    print("\n" + "-" * 60)
    request2 = "Scan the project files, categorize them, and create a summary report"
    print(f"\nRequest: '{request2}'")

    result2 = orchestrator.process_request(request2)

    print("\nResult:")
    print(f"  Success: {result2['success']}")
    print(f"  Workflow ID: {result2['workflow_id']}")
    print(f"  Tasks Executed: {result2.get('tasks_executed', 0)}")
    if result2["success"]:
        summary = result2["result"]
        if isinstance(summary, dict):
            print(f"  Summary: {summary.get('summary', 'N/A')}")
            print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")

    # Get stats
    print("\n" + "=" * 60)
    print("ORCHESTRATOR STATISTICS")
    print("=" * 60)

    stats = orchestrator.get_stats()
    print("\nWorkflows:")
    print(f"  Total: {stats['total_workflows']}")
    print(f"  Completed: {stats['completed_workflows']}")
    print(f"  Active: {stats['active_workflows']}")

    print("\nTask Queue:")
    queue_stats = stats["queue_stats"]
    print(f"  Total Tasks: {queue_stats['total_tasks']}")
    print(f"  Completed: {queue_stats['completed']}")
    print(f"  Success Rate: {queue_stats['success_rate']:.1%}")

    print("\n" + "=" * 60)
    print("✅ ORCHESTRATOR TEST COMPLETE!")
    print("=" * 60)
