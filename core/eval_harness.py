"""
Eval Harness - Classifies task failures and decides the recovery action.

When a task fails (agent error or quality gate rejection) the harness
analyses the failure history and returns one of three decisions:

  RETRY     - re-queue for another attempt (transient / recoverable error)
  ESCALATE  - hand to PlanningAgent for re-decomposition (complex failure)
  ABORT     - freeze the task and move on (churn detected or limits reached)

Churn is detected when the same error string appears _CHURN_THRESHOLD
times in a row for the same task — a strong signal that retrying will
not help.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ------------------------------------------------------------------
# Tunable thresholds
# ------------------------------------------------------------------

# Retries before promoting to ESCALATE
_MAX_RETRIES_BEFORE_ESCALATE = 2

# Escalations before giving up entirely
_MAX_ESCALATIONS_BEFORE_ABORT = 1

# Identical consecutive errors that signal a stuck loop
_CHURN_THRESHOLD = 3

# Error substrings that look transient and are worth retrying
_RETRYABLE_KEYWORDS = [
    "timeout",
    "connection",
    "network",
    "temporary",
    "rate limit",
    "busy",
    "unavailable",
    "try again",
    "timed out",
]


class EvalDecision(Enum):
    """Recovery decision returned by the harness."""

    RETRY = "retry"
    ESCALATE = "escalate"
    ABORT = "abort"


@dataclass
class EvalRecord:
    """Tracks all failure attempts for a single task."""

    task_id: str
    errors: List[str] = field(default_factory=list)
    retry_count: int = 0
    escalation_count: int = 0


class EvalHarness:
    """
    Classifies task failures and returns a recovery decision.

    Maintains a per-task EvalRecord so it can detect repeated failures
    and churn patterns across multiple loop iterations.
    """

    def __init__(self):
        self._records: Dict[str, EvalRecord] = {}

    def evaluate_failure(
        self,
        task_id: str,
        task_description: str,
        error: str,
        task_type: str = "",
    ) -> EvalDecision:
        """
        Decide what to do with a failed task.

        Decision logic (in priority order):
        1. Churn detected (same error N times) → ABORT
        2. Escalation limit reached → ABORT
        3. Transient error + retries remaining → RETRY
        4. Retry limit reached + escalatable task type → ESCALATE
        5. Default → ABORT

        Args:
            task_id:          Unique task identifier.
            task_description: Human-readable description (for logging).
            error:            Error message from agent or quality gate.
            task_type:        Agent type — planning tasks skip escalation.

        Returns:
            EvalDecision enum value.
        """
        record = self._records.setdefault(
            task_id, EvalRecord(task_id=task_id)
        )
        record.errors.append(error)

        # 1. Churn guard — same error repeating identically
        if self._is_churn(record):
            return EvalDecision.ABORT

        # 2. Escalation budget exhausted
        if record.escalation_count >= _MAX_ESCALATIONS_BEFORE_ABORT:
            return EvalDecision.ABORT

        # 3. Transient error with retries left
        if self._is_retryable(error) and record.retry_count < _MAX_RETRIES_BEFORE_ESCALATE:
            record.retry_count += 1
            return EvalDecision.RETRY

        # 4. Retry budget spent — try escalating to planning
        if record.retry_count >= _MAX_RETRIES_BEFORE_ESCALATE:
            # Planning tasks cannot escalate to themselves
            if task_type != "planning":
                record.escalation_count += 1
                return EvalDecision.ESCALATE

        return EvalDecision.ABORT

    def get_record(self, task_id: str) -> Optional[EvalRecord]:
        """Return the failure record for a task, or None if not tracked."""
        return self._records.get(task_id)

    def reset(self, task_id: str) -> None:
        """Clear the failure record after a task succeeds."""
        self._records.pop(task_id, None)

    def summary(self) -> Dict[str, int]:
        """Return aggregate counts across all tracked tasks."""
        return {
            "tasks_tracked": len(self._records),
            "total_retries": sum(r.retry_count for r in self._records.values()),
            "total_escalations": sum(r.escalation_count for r in self._records.values()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_retryable(self, error: str) -> bool:
        """Return True if the error string looks like a transient failure."""
        error_lower = error.lower()
        return any(kw in error_lower for kw in _RETRYABLE_KEYWORDS)

    def _is_churn(self, record: EvalRecord) -> bool:
        """Return True if the last N errors are all identical."""
        if len(record.errors) < _CHURN_THRESHOLD:
            return False
        recent = record.errors[-_CHURN_THRESHOLD:]
        return len(set(recent)) == 1
