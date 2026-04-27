"""
Loop Controller - Production-grade task execution loop.

Replaces the bare while-loop in Orchestrator._execute_queued_tasks with
a control loop that applies four safeguards on every iteration:

  1. Quality gate   -- validate each result before accepting it
  2. Eval harness   -- on failure: RETRY / ESCALATE / ABORT decision
  3. Session store  -- checkpoint state to disk after every task
  4. Churn guard    -- abort if failure ratio or task-cap exceeded

Usage:
    controller = LoopController(session_id="my-session")
    report = controller.run(task_queue, agent_manager)
    print(report.success_rate)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from core.eval_harness import EvalDecision, EvalHarness
from core.quality_gate import QualityGate
from core.session_store import SessionStore
from models import AgentResult, Task

_logger = logging.getLogger("Cortex")

# ------------------------------------------------------------------
# Tunable limits
# ------------------------------------------------------------------

# Fraction of tasks that may fail before the loop pauses
# (only enforced once at least _MIN_TASKS_FOR_RATIO_CHECK have run)
_MAX_FAILURE_RATIO = 0.5
_MIN_TASKS_FOR_RATIO_CHECK = 4

# Hard cap on tasks processed per run (prevents runaway loops)
_MAX_TASKS_PER_RUN = 500


# ------------------------------------------------------------------
# Report dataclass
# ------------------------------------------------------------------

@dataclass
class LoopReport:
    """Summary of a completed loop run."""

    session_id: str
    results: List[AgentResult] = field(default_factory=list)
    completed: int = 0
    failed: int = 0
    retried: int = 0
    escalated: int = 0
    frozen: List[str] = field(default_factory=list)
    aborted_early: bool = False
    abort_reason: str = ""

    @property
    def total(self) -> int:
        return self.completed + self.failed

    @property
    def success_rate(self) -> float:
        return self.completed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "completed": self.completed,
            "failed": self.failed,
            "retried": self.retried,
            "escalated": self.escalated,
            "frozen": self.frozen,
            "success_rate": self.success_rate,
            "aborted_early": self.aborted_early,
            "abort_reason": self.abort_reason,
        }


# ------------------------------------------------------------------
# LoopController
# ------------------------------------------------------------------

class LoopController:
    """
    Runs a TaskQueue to completion with quality gates, eval-driven
    retries, session persistence, and churn detection.

    Instantiate one controller per workflow run (each gets its own
    session_id so checkpoints and event logs stay separate).
    """

    def __init__(
        self,
        session_id: str,
        sessions_dir: str = "",
        on_status: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            session_id:   Unique identifier for this run (e.g. workflow_id).
            sessions_dir: Override directory for session storage.
            on_status:    Optional callback invoked with a short status string
                          before and after each task. Use this to drive a live
                          progress display in the UI layer.
        """
        self.session_id = session_id
        self.quality_gate = QualityGate()
        self.eval_harness = EvalHarness()
        self.session_store = SessionStore(sessions_dir)
        self._on_status: Callable[[str], None] = on_status or (lambda _: None)

    def run(self, task_queue: Any, agent_manager: Any) -> LoopReport:
        """
        Execute all tasks in the queue with full production safeguards.

        Args:
            task_queue:    TaskQueue instance with pending work.
            agent_manager: AgentManager that routes tasks to agents.

        Returns:
            LoopReport summarising results, retries, escalations, and
            any frozen tasks.
        """
        report = LoopReport(session_id=self.session_id)
        iterations = 0

        _logger.info(f"[LoopController] Session {self.session_id} started")
        self.session_store.log_event(
            self.session_id, "loop_started", {"session_id": self.session_id}
        )

        while not task_queue.is_empty():
            # Safety cap — prevent runaway loops
            if iterations >= _MAX_TASKS_PER_RUN:
                self._abort(
                    report,
                    f"Reached max tasks per run ({_MAX_TASKS_PER_RUN})",
                )
                break

            # Failure ratio guard (only after enough data)
            if self._failure_ratio_exceeded(report):
                self._abort(
                    report,
                    f"Failure ratio exceeded {_MAX_FAILURE_RATIO:.0%} "
                    f"({report.failed}/{report.total} tasks failed)",
                )
                break

            task = task_queue.get_next_task()
            if task is None:
                # No tasks ready — dependencies unmet, nothing to do
                break

            iterations += 1
            _logger.info(
                f"[LoopController] [{iterations}] Executing: {task.description[:70]}"
            )

            desc = task.description[:55] + "..." if len(task.description) > 55 else task.description
            self._on_status(f"Step {iterations} — {desc}")

            result = agent_manager.execute_task(task)

            if result.is_successful():
                self._on_status(
                    f"Step {iterations} done [{result.agent_name}] — {desc}"
                )
                self._handle_success(task, result, task_queue, report, agent_manager)
            else:
                error_short = (result.error or "unknown error")[:40]
                self._on_status(f"Step {iterations} failed ({error_short}) — retrying...")
                self._handle_failure(task, result, task_queue, report, agent_manager)

            # Checkpoint after every task
            self._checkpoint(task_queue, report)

        self._finalize(report)
        return report

    # ------------------------------------------------------------------
    # Success path
    # ------------------------------------------------------------------

    def _handle_success(
        self,
        task: Task,
        result: AgentResult,
        task_queue: Any,
        report: LoopReport,
        agent_manager: Any,
    ) -> None:
        """Run quality gate; accept or soft-fail the result."""
        gate = self.quality_gate.check(result, task)

        if gate.warnings:
            for w in gate.warnings:
                _logger.warning(f"[QualityGate] {task.task_id}: {w}")

        if gate.passed:
            task_queue.mark_completed(task, result.data)
            report.results.append(result)
            report.completed += 1
            self.eval_harness.reset(task.task_id)

            _logger.info(
                f"[LoopController] Accepted {task.task_id} "
                f"(gate: {gate.score:.2f})"
            )
            self.session_store.log_event(
                self.session_id,
                "task_completed",
                {"task_id": task.task_id, "gate_score": gate.score},
            )
        else:
            # Gate rejected a nominally-successful result — treat as failure
            _logger.warning(
                f"[QualityGate] Rejected {task.task_id}: {gate.reason}"
            )
            gate_failed_result = AgentResult.create_error(
                error=f"Quality gate failed: {gate.reason}",
                agent_name=result.agent_name,
                task_id=task.task_id,
                task_description=task.description,
            )
            self._handle_failure(
                task, gate_failed_result, task_queue, report, agent_manager
            )

    # ------------------------------------------------------------------
    # Failure path
    # ------------------------------------------------------------------

    def _handle_failure(
        self,
        task: Task,
        result: AgentResult,
        task_queue: Any,
        report: LoopReport,
        agent_manager: Optional[Any],
    ) -> None:
        """Run eval harness; retry, escalate, or freeze the task."""
        error = result.error or "Unknown error"

        decision = self.eval_harness.evaluate_failure(
            task_id=task.task_id,
            task_description=task.description,
            error=error,
            task_type=task.task_type,
        )

        _logger.info(
            f"[EvalHarness] {task.task_id} -> {decision.value} "
            f"| error: {error[:80]}"
        )
        self.session_store.log_event(
            self.session_id,
            "task_eval",
            {
                "task_id": task.task_id,
                "decision": decision.value,
                "error": error[:200],
            },
        )

        if decision == EvalDecision.RETRY:
            task_queue.mark_failed(task, error)
            report.retried += 1
            _logger.info(f"[LoopController] Re-queued {task.task_id} for retry")

        elif decision == EvalDecision.ESCALATE and agent_manager is not None:
            self._escalate_to_planning(task, error, task_queue, report)

        else:
            # ABORT — permanently discard (bypass TaskQueue retry logic)
            task_queue.mark_abandoned(task, error)
            report.results.append(result)
            report.failed += 1
            report.frozen.append(task.task_id)
            _logger.warning(
                f"[LoopController] Frozen {task.task_id}: {error[:100]}"
            )
            self.session_store.log_event(
                self.session_id,
                "task_frozen",
                {"task_id": task.task_id, "error": error[:200]},
            )

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def _escalate_to_planning(
        self,
        task: Task,
        error: str,
        task_queue: Any,
        report: LoopReport,
    ) -> None:
        """
        Create a re-planning task to decompose the failed task differently.
        The original task is marked failed and the new plan task is queued.
        """
        from models import Task as T  # local import avoids circular dependency

        plan_task = T(
            description=f"Re-plan failed task: {task.description}",
            task_type="planning",
            priority=task.priority,
            parameters={
                "original_task": task.description,
                "failure_reason": error[:300],
            },
        )
        task_queue.add_task(plan_task)
        task_queue.mark_abandoned(task, f"escalated to planning: {error[:100]}")
        report.escalated += 1

        _logger.info(
            f"[LoopController] Escalated {task.task_id} -> "
            f"planning task {plan_task.task_id}"
        )
        self.session_store.log_event(
            self.session_id,
            "task_escalated",
            {
                "original_task_id": task.task_id,
                "plan_task_id": plan_task.task_id,
                "error": error[:200],
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _checkpoint(self, task_queue: Any, report: LoopReport) -> None:
        """Save current loop state to disk."""
        stats = task_queue.get_stats()
        self.session_store.save_checkpoint(
            session_id=self.session_id,
            pending_count=stats["pending"],
            completed_count=report.completed,
            failed_count=report.failed,
            frozen_tasks=report.frozen,
            extra={
                "retried": report.retried,
                "escalated": report.escalated,
                "success_rate": report.success_rate,
            },
        )

    def _failure_ratio_exceeded(self, report: LoopReport) -> bool:
        """Return True if too many tasks have failed to continue safely."""
        if report.total < _MIN_TASKS_FOR_RATIO_CHECK:
            return False
        return (report.failed / report.total) > _MAX_FAILURE_RATIO

    def _abort(self, report: LoopReport, reason: str) -> None:
        """Record an early abort reason."""
        report.aborted_early = True
        report.abort_reason = reason
        _logger.warning(f"[LoopController] Aborting early: {reason}")
        self.session_store.log_event(
            self.session_id,
            "loop_aborted",
            {"reason": reason},
        )

    def _finalize(self, report: LoopReport) -> None:
        """Log final summary and write closing event."""
        _logger.info(
            f"[LoopController] Session {self.session_id} finished — "
            f"completed={report.completed} failed={report.failed} "
            f"retried={report.retried} escalated={report.escalated} "
            f"frozen={len(report.frozen)} "
            f"success_rate={report.success_rate:.1%}"
        )
        self.session_store.log_event(
            self.session_id,
            "loop_finished",
            report.to_dict(),
        )
