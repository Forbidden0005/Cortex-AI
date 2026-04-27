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
from typing import Any, Callable, Dict, List, Optional

# Import models
sys.path.append(str(Path(__file__).parent.parent))
from core.agent_manager import AgentManager
from core.config_loader import get_config
from core.llm_interface import LLMInterface
# Import core components
from core.logger import get_logger
from core.loop_controller import LoopController, LoopReport
from core.memory_manager import MemoryManager
from core.session_store import SessionStore
from core.task_classifier import (
    LONG_REQUEST_WORD_THRESHOLD,
    SHORT_MESSAGE_WORD_THRESHOLD,
    classify_task_type,
)
from core.task_queue import TaskQueue
from models import AgentResult, MemoryItem, MemoryType, Task, TaskPriority

# Maximum characters stored in memory for a single workflow result
_MEMORY_CONTENT_MAX_CHARS = 200
_MEMORY_SUMMARY_MAX_CHARS = 500

# Minimum verb matches that suggest a task needs multi-step planning
_MULTI_VERB_THRESHOLD = 1

# Word count above which a task description is considered complex
_COMPLEX_TASK_WORD_THRESHOLD = 15


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
        self.logger.info("Initializing Cortex...")

        self.memory = MemoryManager(storage_dir=self.config.memory.vector_db_path)

        self.llm = LLMInterface(self.config.model, use_mock=False)

        self.agent_manager = AgentManager()

        self.task_queue = TaskQueue(
            max_concurrent_tasks=self.config.task_queue.max_concurrent_tasks,
            retry_delay=self.config.task_queue.retry_delay,
        )

        self.session_store = SessionStore()

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
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        _task_type: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user request from start to finish.

        Args:
            user_request: Natural language request from user
            context:      Optional context information
            _task_type:   Pre-classified type (skips classification when provided,
                          avoiding a redundant LLM call from the caller).
            on_status:    Optional callback for live progress updates — called
                          by LoopController before/after each task step.

        Returns:
            Dictionary with results and status
        """
        workflow_id = str(uuid.uuid4())

        self.logger.log_user_request(user_request)

        # Fast path: conversational / general requests skip the full pipeline.
        # Accept a pre-classified type so the caller doesn't have to classify twice.
        task_type = _task_type if _task_type is not None else self._classify_request(user_request)
        if task_type == "general":
            return self._respond_directly(user_request)

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
            initial_task = self._create_task_from_request(user_request, context, task_type)

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

            # Step 4: Execute tasks via production loop controller
            controller = LoopController(session_id=workflow_id, on_status=on_status)
            loop_report = controller.run(self.task_queue, self.agent_manager)
            results = loop_report.results

            # Step 5: Combine results
            final_result = self._combine_results(results)

            # Step 6: Store in memory
            self._store_workflow_memory(workflow_id, user_request, final_result, loop_report)

            # Step 7: Update workflow tracking
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["completed_at"] = datetime.now()
            self.active_workflows[workflow_id]["results"] = results
            self.active_workflows[workflow_id]["loop_report"] = loop_report.to_dict()
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
                "success_rate": loop_report.success_rate,
                "retried": loop_report.retried,
                "escalated": loop_report.escalated,
                "frozen": loop_report.frozen,
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
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
    ) -> Task:
        """
        Convert a natural-language user request into a Task object.

        Args:
            request:   User's natural language request.
            context:   Optional context dict.
            task_type: Pre-classified type (skips classification if provided).

        Returns:
            Task object
        """
        if task_type is None:
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

    def _respond_directly(self, user_request: str) -> Dict[str, Any]:
        """
        Fast path for conversational / general requests.

        Calls the LLM directly — no task queue, no agents, no memory write,
        no session checkpoint. Returns in the same shape as process_request
        so callers don't need to branch.

        Args:
            user_request: The user's message.

        Returns:
            Result dict with success=True and a "response" key.
        """
        _system = "You are Cortex, a helpful AI assistant. Reply naturally and concisely."
        try:
            response = self.llm.ask(
                user_request, system_prompt=_system, max_tokens=300
            ).strip()
        except Exception as exc:  # noqa: BLE001
            response = f"Sorry, I couldn't generate a response: {exc}"

        return {
            "success": True,
            "workflow_id": None,
            "result": {"response": response},
            "tasks_executed": 0,
            "message": "Direct response",
        }

    def _classify_request(self, request: str) -> str:
        """
        Determine the task type for a natural-language request.

        Two-stage process:
          1. Keyword classifier — fast O(n) pass; returns a specific type
             immediately when confident keywords are found.
          2. LLM classifier — invoked only when keywords yield no clear
             match ("general"), so conversational phrasing and synonyms
             are handled correctly.

        Long requests that remain "general" after both passes are promoted
        to "planning" so the PlanningAgent can break them down.

        Args:
            request: User request text.

        Returns:
            Task type string (e.g. "coding", "file", "general", "planning").
        """
        # Stage 1: fast keyword match
        task_type = classify_task_type(request)

        # Stage 2: short messages that didn't match any specific keyword are
        # treated as conversational without invoking the LLM at all.
        if task_type == "general" and len(request.split()) <= SHORT_MESSAGE_WORD_THRESHOLD:
            return "general"

        # Stage 3: LLM disambiguation only for longer ambiguous requests
        if task_type == "general":
            task_type = self._classify_with_llm(request)

        # Promote long unclassified requests to planning
        if task_type == "general" and len(request.split()) > LONG_REQUEST_WORD_THRESHOLD:
            return "planning"

        return task_type

    def _classify_with_llm(self, request: str) -> str:
        """
        Use the LLM to classify a request when keyword matching fails.

        Asks the LLM to pick the single best task type from the known list.
        Parses the first word of the response and validates it against the
        allowed types; falls back to "general" if the reply is unexpected.

        Args:
            request: User request text.

        Returns:
            Task type string from the known set, or "general".
        """
        _valid_types = {
            "file", "coding", "web", "data", "planning",
            "automation", "security", "memory", "vision",
            "audio", "qa", "general",
        }

        prompt = (
            "You are a task router. Classify the following user request into "
            "exactly ONE of these task types:\n"
            "file, coding, web, data, planning, automation, security, "
            "memory, vision, audio, qa, general\n\n"
            "Rules:\n"
            "- Use 'general' for greetings, questions, conversation, or anything "
            "that doesn't clearly fit another category.\n"
            "- Reply with ONLY the single task type word, nothing else.\n\n"
            f"Request: {request}\n\n"
            "Task type:"
        )

        try:
            raw = self.llm.ask(prompt, max_tokens=5).strip().lower().split()[0]
            # Strip punctuation that the LLM might append
            task_type = raw.rstrip(".,;:")
            if task_type in _valid_types:
                self.logger.debug(
                    f"[Orchestrator] LLM classified {request!r:.40} -> {task_type}"
                )
                return task_type
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(f"[Orchestrator] LLM classification failed: {exc}")

        return "general"

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

        # Complex tasks (long descriptions)
        if len(task.description.split()) > _COMPLEX_TASK_WORD_THRESHOLD:
            return True

        # Tasks with multiple action verbs (suggests multiple steps)
        _action_verbs = ["create", "generate", "move", "copy", "send", "update"]
        verb_count = sum(1 for verb in _action_verbs if verb in task.description.lower())
        if verb_count > _MULTI_VERB_THRESHOLD:
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

        # If single result, return its data (or a meaningful error on failure)
        if len(results) == 1:
            r = results[0]
            if not r.is_successful():
                return {
                    "response": (
                        r.error
                        or r.message
                        or "I wasn't able to complete that task."
                    )
                }
            return r.data

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

    def _store_workflow_memory(
        self,
        workflow_id: str,
        request: str,
        result: Any,
        loop_report: Optional["LoopReport"] = None,
    ):
        """
        Store workflow execution in memory for learning.

        Args:
            workflow_id: Workflow identifier
            request: Original user request
            result: Final result
            loop_report: Optional LoopReport with execution stats
        """
        metadata: Dict[str, Any] = {
            "workflow_id": workflow_id,
            "request": request,
            "result_summary": str(result)[:_MEMORY_SUMMARY_MAX_CHARS],
        }

        if loop_report is not None:
            metadata["loop_success_rate"] = loop_report.success_rate
            metadata["loop_retried"] = loop_report.retried
            metadata["loop_escalated"] = loop_report.escalated
            metadata["loop_frozen"] = loop_report.frozen

        memory = MemoryItem(
            content=f"User requested: {request}. System completed with result: {str(result)[:_MEMORY_CONTENT_MAX_CHARS]}",
            memory_type=MemoryType.CONVERSATION,
            related_conversation_id=self.current_conversation_id,
            tags=["workflow", "completed"],
            source="orchestrator",
            metadata=metadata,
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
    print("\n[1/3] Initializing Cortex...")
    orchestrator = Orchestrator()

    # Register some agents
    print("\n[2/3] Registering Agents...")
    orchestrator.agent_manager.register_agent(PlanningAgent(orchestrator.llm))
    orchestrator.agent_manager.register_agent(FileAgent())
    print(f"[OK] Registered {len(orchestrator.agent_manager.list_agents())} agents")

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
    print("[PASS] ORCHESTRATOR TEST COMPLETE!")
    print("=" * 60)
