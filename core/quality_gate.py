"""
Quality Gate - Validates AgentResult objects before marking tasks complete.

Runs a series of checks against each result to ensure it meets minimum
quality standards. A failed gate re-routes the result to the eval harness
for retry or escalation rather than silently accepting bad output.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from models import AgentResult

# Execution time above this (seconds) triggers a slow-result warning
_SLOW_RESULT_THRESHOLD_S = 30.0

# Minimum quality score required to pass (0.0–1.0)
_PASS_THRESHOLD = 0.5

# Substrings in result data that hint at a silent failure
_ERROR_KEYWORDS = ("error", "failed", "exception", "traceback")


@dataclass
class GateResult:
    """Outcome of a quality gate check."""

    passed: bool
    reason: str
    score: float  # 0.0 = total failure, 1.0 = perfect
    warnings: List[str] = field(default_factory=list)


class QualityGate:
    """
    Validates AgentResult objects against a set of quality checks.

    Checks applied (in order):
    1. Result data is not None.
    2. No error keywords hiding inside result data strings.
    3. Execution time is within a reasonable bound.
    4. Result is a non-empty dict or non-empty string.

    A result must score >= _PASS_THRESHOLD to be accepted. Warnings
    are recorded but do not fail the gate on their own.
    """

    def check(self, result: AgentResult, task: Optional[Any] = None) -> GateResult:
        """
        Run all quality checks against a result.

        Args:
            result: The AgentResult to validate.
            task:   Optional Task for context (reserved for future type-specific checks).

        Returns:
            GateResult with pass/fail, reason, score, and any warnings.
        """
        warnings: List[str] = []
        score = 1.0

        # Check 1: data must not be None
        if result.data is None:
            return GateResult(
                passed=False,
                reason="Result data is None",
                score=0.0,
            )

        # Check 2: no hidden error indicators in string fields
        hidden = self._find_hidden_error(result.data)
        if hidden:
            score -= 0.3
            warnings.append(f"Possible hidden error in result: {hidden[:80]}")

        # Check 3: execution time sanity
        if result.execution_time and result.execution_time > _SLOW_RESULT_THRESHOLD_S:
            score -= 0.1
            warnings.append(
                f"Slow result: {result.execution_time:.1f}s "
                f"(threshold: {_SLOW_RESULT_THRESHOLD_S}s)"
            )

        # Check 4: result must be non-empty
        if isinstance(result.data, dict) and not result.data:
            score -= 0.2
            warnings.append("Result data is an empty dict")
        elif isinstance(result.data, str) and not result.data.strip():
            score -= 0.2
            warnings.append("Result data is an empty string")

        passed = score >= _PASS_THRESHOLD
        if passed and not warnings:
            reason = "All checks passed"
        elif warnings:
            reason = warnings[0]
        else:
            reason = f"Score {score:.2f} below threshold {_PASS_THRESHOLD}"

        return GateResult(passed=passed, reason=reason, score=score, warnings=warnings)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_hidden_error(self, data: Any) -> str:
        """
        Recursively scan result data for error-indicator substrings.

        Returns the first suspicious string found (truncated to 100 chars),
        or an empty string if nothing suspicious is detected.
        """
        if isinstance(data, str):
            lower = data.lower()
            for keyword in _ERROR_KEYWORDS:
                if keyword in lower:
                    return data[:100]
            return ""

        if isinstance(data, dict):
            for value in data.values():
                found = self._find_hidden_error(value)
                if found:
                    return found

        if isinstance(data, list):
            for item in data:
                found = self._find_hidden_error(item)
                if found:
                    return found

        return ""
